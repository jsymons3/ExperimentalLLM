"""Accuracy by split, and what the table, the adapters and the note contribute.

An answer counts as right when the model's single most likely next token,
out of its whole vocabulary, is the answer's token.

Two-step questions are answered the way the run was trained (``chain_mode``):

* "note": the question produces a one-token note, the note is written into a
  fresh one-step question, and the model runs again from scratch.
* "latent": one pass; the answer is read after Ouro's last loop.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from phrase_scratchpad.evaluate import eval_mode

from .data import ORDINARY

BATCH = 64


def loop_predictions(model, tokens, **switches):
    """The top next token after each loop, for each question: (n, loops)."""
    out = []
    for i in range(0, len(tokens), BATCH):
        states = model.states(mx.array(tokens[i : i + BATCH]), **switches)
        tops = [mx.argmax(model.read(state[:, -1]), axis=-1) for state in states]
        out.append(np.array(mx.stack(tops, axis=1)))
    return np.concatenate(out)


def predict(model, tokens, **switches):
    """The model's answer to each question: (n,)."""
    return loop_predictions(model, tokens, **switches)[:, -1]


def answer_chains(model, world, ids, mode, notes=None, **switches):
    """(answers, notes) for two-step questions. ``notes`` replaces the notes
    the model would write itself; in "latent" mode there are none."""
    if mode == "latent":
        return predict(model, world.chain_tokens[ids], **switches), None
    if notes is None:
        notes = predict(model, world.chain_tokens[ids], **switches)
    return predict(model, world.second_questions(ids, notes), **switches), notes


def eval_ids(world, size):
    """({split: fact ids}, {split: chain ids}), without the empty splits."""
    facts = {name: ids[:size] for name, ids in world.eval_facts.items() if len(ids)}
    chains = {name: ids[:size] for name, ids in world.eval_chains.items() if len(ids)}
    return facts, chains


def accuracy_by_split(model, world, sets, mode, **switches):
    facts, chains = sets
    out = {}
    for name, ids in facts.items():
        guess = predict(model, world.fact_tokens[ids], **switches)
        out[name] = float((guess == world.facts(ids).answer).mean())
    for name, ids in chains.items():
        guess, _ = answer_chains(model, world, ids, mode, **switches)
        out[name] = float((guess == world.chains(ids).answer).mean())
    return out


def chain_detail(model, world, chains, mode):
    """Per group of two-step questions: what limits the accuracy.

    both_facts_known       share where each fact is recalled as a one-step question
    accuracy_when_both_known
    two_call               the two one-step questions asked in turn, the second
                           about whatever the first answered: what chaining
                           could reach with the facts as they are recalled
    by_loop                how often the state after each loop reads as the
                           intermediate answer, and as the final answer
    In "note" mode also:
    note_right             the written note is the intermediate answer
    shuffled_notes         accuracy with notes swapped between questions
    correct_notes          accuracy with the true intermediate answer supplied
    """
    out = {}
    for name, ids in chains.items():
        target = world.chains(ids)
        first = predict(model, world.fact_tokens[world.chain_first[ids]])
        second_known = predict(model, world.fact_tokens[world.chain_second[ids]]) == target.answer
        known = (first == target.bridge) & second_known
        tops = loop_predictions(model, world.chain_tokens[ids])
        if mode == "note":
            notes = tops[:, -1]
            answers, _ = answer_chains(model, world, ids, mode, notes=notes)
        else:
            notes, answers = None, tops[:, -1]
        right = answers == target.answer
        two_call, _ = answer_chains(model, world, ids, "note", notes=first)
        entry = {
            "both_facts_known": float(known.mean()),
            "accuracy_when_both_known": float(right[known].mean()) if known.any() else None,
            "two_call": float((two_call == target.answer).mean()),
            "by_loop": {
                "bridge": (tops == target.bridge[:, None]).mean(axis=0).tolist(),
                "answer": (tops == target.answer[:, None]).mean(axis=0).tolist(),
            },
        }
        if mode == "note":
            # Neighbouring questions often share a note, so shuffle properly.
            order = np.random.default_rng(0).permutation(len(notes))
            swapped, _ = answer_chains(model, world, ids, mode, notes=notes[order])
            entry.update(
                note_right=float((notes == target.bridge).mean()),
                shuffled_notes=float((swapped == target.answer).mean()),
                # With the true note, step two is the second fact's own question.
                correct_notes=float(second_known.mean()),
            )
        out[name] = entry
    return out


def table_usage(model, tokens):
    """Share of each table's entries retrieved at least once on these questions."""
    seen = [set() for _ in model.tables]
    for i in range(0, len(tokens), BATCH):
        _, retrieved = model.states(mx.array(tokens[i : i + BATCH]), return_retrieved=True)
        # One item per lookup: loops in order and, within a loop, tables in layer order.
        for k, index in enumerate(retrieved):
            seen[k % len(model.tables)].update(np.unique(np.array(index)).tolist())
    return [len(s) / table.fire.shape[0] for s, table in zip(seen, model.tables)]


def ordinary_text_loss(model, world, **switches):
    """Mean next-token loss on everyday sentences."""
    total, count = 0.0, 0
    for sentence in ORDINARY:
        ids = [world.start] + list(world.encode(sentence))
        logits = model.read(model.states(mx.array(ids)[None], **switches)[-1][0, :-1])
        total += nn.losses.cross_entropy(logits, mx.array(ids[1:]), reduction="sum").item()
        count += len(ids) - 1
    return total / count


def periodic_evaluation(model, world, mode, size):
    """A quick look during training: accuracy on a sample of each split."""
    sets = eval_ids(world, size)
    with eval_mode(model):
        accuracy = accuracy_by_split(model, world, sets, mode)
        if model.tables and "one_hop" in sets[0]:
            ids = sets[0]["one_hop"]
            guess = predict(model, world.fact_tokens[ids], table=False)
            accuracy["one_hop_table_off"] = float((guess == world.facts(ids).answer).mean())
    return accuracy


def full_evaluation(model, world, mode, size):
    sets = eval_ids(world, size)
    facts, chains = sets
    has_table, has_adapters = bool(model.tables), bool(model.adapters)
    with eval_mode(model):
        out = {
            "chain_mode": mode,
            "accuracy": accuracy_by_split(model, world, sets, mode),
            # The frozen model with nothing added: what Ouro knew beforehand.
            "accuracy_original": accuracy_by_split(
                model, world, sets, mode, table=False, adapters=False
            ),
            "guess_rate": {
                **{name: world.guess_rate(world.facts(ids)) for name, ids in facts.items()},
                **{name: world.guess_rate(world.chains(ids)) for name, ids in chains.items()},
            },
            "chains": chain_detail(model, world, chains, mode),
            "ordinary_text_loss": {
                "original": ordinary_text_loss(model, world, table=False, adapters=False),
                "with_additions": ordinary_text_loss(model, world),
            },
        }
        if has_table and has_adapters:
            out["accuracy_table_off"] = accuracy_by_split(model, world, sets, mode, table=False)
            out["accuracy_adapters_off"] = accuracy_by_split(
                model, world, sets, mode, adapters=False
            )
        if has_table:
            out["table_used"] = table_usage(model, world.fact_tokens[: 4 * size])
    return out
