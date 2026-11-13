"""Accuracy and scratchpad diagnostics, per task and per split.

A task is evaluated the way it was trained: with the lookup table on, or off
for a task that was trained with it off. The other setting is reported
separately, to show what depends on the table.
"""

from contextlib import contextmanager

import mlx.core as mx
import numpy as np

from .tasks import PAD


@contextmanager
def eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _batches(n, size):
    return [slice(i, i + size) for i in range(0, n, size)]


def predict(model, questions, steps=None, batch_size=1024, slot_override=None, table=True):
    """The K predicted tokens for each question: (n, K).

    ``table=False`` switches the lookup off, leaving only the core.
    """
    out = []
    for rows in _batches(len(questions), batch_size):
        batch = mx.array(questions[rows])
        override = None if slot_override is None else mx.array(slot_override[rows])
        table_on = None if table else mx.zeros((batch.shape[0],))
        logits = model(batch, steps=steps, slot_override=override, table_on=table_on)
        out.append(np.array(mx.argmax(logits, axis=-1)))
    return np.concatenate(out)


def exact(predicted, target):
    """Share of rows where every token, padding included, is right."""
    return float((predicted == target).all(axis=1).mean())


def accuracy_by_split(model, task, steps=None, table=True):
    return {
        name: exact(predict(model, items.question, steps, table=table), items.answer)
        for name, items in task.eval_sets.items()
    }


def table_usage(model, suite, table_off=(), sample=4000):
    """Share of table entries a sample of training questions retrieves,
    over the tasks that use the table."""
    cfg = model.cfg
    tasks = [t for t in suite.tasks if t.name not in table_off]
    if cfg.ff != "lookup" or not tasks:
        return None
    rng = np.random.default_rng(0)
    counts = np.zeros(cfg.n_subkeys**2, dtype=np.int64)
    for task in tasks:
        questions = task.sample(rng, sample // 2, sample // 2).question
        for rows in _batches(len(questions), 1024):
            _, trace = model(mx.array(questions[rows]), return_trace=True)
            for index in trace["retrieved"]:
                counts += np.bincount(np.array(index).ravel(), minlength=len(counts))
    p = counts[counts > 0] / counts.sum()
    return {
        "entries": int(len(counts)),
        "used_frac": float((counts > 0).mean()),
        "effective_frac": float(np.exp(-(p * np.log(p)).sum()) / len(counts)),
    }


def scratchpad_report(model, task, items, table=True, batch_size=1024):
    """What step 1 wrote for two-hop questions, and whether step 2 uses it.

    ``bridge_written`` is the share of chains whose scratchpad holds exactly
    the bridge after step 1. The interventions replace only what step 2 reads:
    the learned empty scratchpad, another question's scratchpad, or the true
    bridge. The last uses the answer key, so it is a diagnostic, never a
    score.
    """
    questions = items.question
    rng = np.random.default_rng(0)
    written, slots = [], []
    for rows in _batches(len(questions), batch_size):
        batch = mx.array(questions[rows])
        table_on = None if table else mx.zeros((batch.shape[0],))
        _, trace = model(batch, steps=2, return_trace=True, table_on=table_on)
        written.append(np.array(mx.argmax(trace["step_logits"][0], axis=-1)))
        slots.append(np.array(trace["slots"][0]))
    written, slots = np.concatenate(written), np.concatenate(slots)
    embed = np.array(model.embed.weight)
    empty = np.broadcast_to(np.array(model.slot_empty), slots.shape)
    conditions = {
        "normal": None,
        "cleared": empty,
        "shuffled": slots[rng.permutation(len(slots))],
        "true_bridge": embed[items.bridge],
    }
    return {
        "bridge_written": exact(written, items.bridge),
        "interventions": {
            name: exact(
                predict(model, questions, 2, batch_size, override, table), items.answer
            )
            for name, override in conditions.items()
        },
    }


def two_call(model, task, items, steps, table=True):
    """Answer a chain as two separate one-step questions.

    The second question is about the entity the model predicted for the
    first, not the true bridge. ``both_known`` counts chains whose two facts
    the model answers correctly on their own; ``native_on_both_known`` is how
    many of those it also answers as a single two-hop question. Returns None
    for a task that is not built from one-step questions.
    """
    if items.first is None:
        return None

    def ask(questions, depth=None):
        return predict(model, questions, depth, table=table)

    native = (ask(items.question) == items.answer).all(axis=1)
    first = ask(items.first, steps)
    answer = ask(task.second_questions(first, items), steps)
    oracle = ask(task.second_questions(items.bridge, items), steps)
    known = (first == items.bridge).all(axis=1) & (oracle == items.answer).all(axis=1)
    return {
        "n": len(items),
        "native": float(native.mean()),
        "two_call": exact(answer, items.answer),
        "both_known": int(known.sum()),
        "native_on_both_known": int((native & known).sum()),
    }


def periodic_evaluation(model, suite, table_off=()):
    with eval_mode(model):
        return {
            task.name: accuracy_by_split(model, task, table=task.name not in table_off)
            for task in suite.tasks
        }


def full_evaluation(model, suite, eval_steps, table_off=()):
    """Everything reported at the end of a run, per task.

    ``table_off`` names the tasks trained with the lookup switched off.
    """
    cfg = model.cfg
    depths = eval_steps if cfg.scratchpad else (1,)
    call_depths = (1, 2) if cfg.scratchpad else (1,)
    result = {"default_steps": cfg.default_steps, "tasks": {}}
    with eval_mode(model):
        for task in suite.tasks:
            table = task.name not in table_off  # how this task was trained
            sets = task.eval_sets
            chains = {name: items for name, items in sets.items() if name.startswith("chain")}
            on = accuracy_by_split(model, task, table=True)
            off = accuracy_by_split(model, task, table=False)
            entry = {
                "trained_with_table": table,
                "accuracy": on if table else off,
                "accuracy_table_on": on,
                "accuracy_table_off": off,
                "accuracy_by_steps": {
                    str(d): accuracy_by_split(model, task, d, table) for d in depths
                },
                "majority_baseline": {
                    name: task.majority_baseline(items) for name, items in sets.items()
                },
                "two_call": {
                    name: {str(d): two_call(model, task, items, d, table) for d in call_depths}
                    for name, items in chains.items()
                    if items.first is not None
                },
                "scratchpad": None,
            }
            if cfg.scratchpad:
                entry["scratchpad"] = {
                    name: scratchpad_report(model, task, items, table)
                    for name, items in chains.items()
                }
            result["tasks"][task.name] = entry
        result["table"] = table_usage(model, suite, table_off)
    return result
