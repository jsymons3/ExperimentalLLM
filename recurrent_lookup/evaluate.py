"""Accuracy, table usage, retrieval overlap and scratchpad diagnostics.

"Depth" means the number of passes for the original model and the number of
outer steps for a scratchpad model.
"""

from contextlib import contextmanager

import mlx.core as mx
import numpy as np

TWO_HOP_SPLITS = (
    "two_hop_train",
    "two_hop_id",
    "two_hop_held_first",
    "two_hop_held_second",
    "two_hop_held_both",
)


@contextmanager
def eval_mode(model):
    """Freeze the query-normalisation statistics while evaluating."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def uses_slots(model):
    return bool(getattr(getattr(model, "cfg", None), "num_slots", 0))


def forward(model, tokens, depth=None, **kwargs):
    """Run the model at a depth: passes, or outer steps with a scratchpad."""
    if uses_slots(model):
        return model(tokens, steps=depth, **kwargs)
    return model(tokens, n_passes=depth, **kwargs)


def default_depth(model):
    cfg = model.cfg
    return cfg.default_steps if cfg.num_slots else cfg.n_passes


def _batches(tokens, batch_size):
    for i in range(0, len(tokens), batch_size):
        yield tokens[i : i + batch_size]


def predict(model, tokens, depth=None, batch_size=1024):
    """Top entity for each row of tokens."""
    return np.concatenate(
        [
            np.array(mx.argmax(forward(model, mx.array(batch), depth), axis=-1))
            for batch in _batches(tokens, batch_size)
        ]
    )


def accuracy(model, data, rows, depth=None, batch_size=1024):
    """Exact-match accuracy on one-hop (n, 2) or two-hop (n, 3) rows."""
    tokens, targets = data.encode(rows)
    return float((predict(model, tokens, depth, batch_size) == targets).mean())


def accuracy_by_split(model, data, depth=None):
    return {
        name: accuracy(model, data, rows, depth)
        for name, rows in data.eval_sets.items()
        if len(rows)
    }


def table_usage(model, data, depth=None, sample=20000, batch_size=1024):
    """How much of the table a sample of training questions touches.

    ``used_frac`` is the share of entries retrieved at least once.
    ``effective_frac`` is exp(entropy of the retrieval distribution) divided by
    the table size: it drops when a few entries take most of the traffic.
    Returns None for models without a lookup.
    """
    cfg = model.cfg
    if cfg.ff != "lookup":
        return None
    rng = np.random.default_rng(0)
    counts = np.zeros(cfg.n_entries, dtype=np.int64)
    for rows in (data.facts, data.two_hop_train):
        rows = rows[rng.permutation(len(rows))[:sample]]
        tokens, _ = data.encode(rows)
        for batch in _batches(tokens, batch_size):
            _, retrieved = forward(model, mx.array(batch), depth, return_retrieval=True)
            for index in retrieved:
                if index is not None:
                    flat = np.array(index).ravel()
                    counts += np.bincount(flat, minlength=len(counts))
    p = counts[counts > 0] / counts.sum()
    return {
        "entries": int(len(counts)),
        "used_frac": float((counts > 0).mean()),
        "effective_frac": float(np.exp(-(p * np.log(p)).sum()) / len(counts)),
        "retrievals": int(counts.sum()),
    }


def _answer_position_sets(model, tokens, depth, batch_size=1024):
    """sets[d][i]: entries retrieved at the last position of row i at depth d.

    The last position is the answer position: the final question token, or the
    work position of a scratchpad model. For a scratchpad model the entries of
    all the block applications inside a step are pooled. sets[d] is None when
    no lookup ran at that depth.
    """
    group = model.cfg.n_passes if uses_slots(model) else 1
    parts = [[] for _ in range(depth)]
    for batch in _batches(tokens, batch_size):
        _, retrieved = forward(model, mx.array(batch), depth, return_retrieval=True)
        for d in range(depth):
            found = [
                np.array(index[:, -1]).reshape(len(batch), -1)
                for index in retrieved[d * group : (d + 1) * group]
                if index is not None
            ]
            if found:
                parts[d].append(np.concatenate(found, axis=1))
    return [
        [set(row) for row in np.concatenate(p).tolist()] if p else None for p in parts
    ]


def _mean_jaccard(a, b):
    if a is None or b is None:
        return None
    return float(np.mean([len(x & y) / len(x | y) for x, y in zip(a, b)]))


def retrieval_overlap(model, data, chains, depth=None):
    """Does the second hop fetch the entries a direct question would fetch?

    For each chain (h, r1, r2) with bridge b = tail[h, r1], compare the entries
    retrieved at the answer position of the two-hop question with those
    retrieved at the answer position of the direct question [b, r2].

    ``same[p][q]`` is the mean Jaccard overlap between depth p+1 of the
    two-hop question and depth q+1 of the direct question (passes, or outer
    steps with a scratchpad). ``control[p][q]`` is the same for the direct
    question [b', r2] about a different entity b'. If ``same`` is well above
    ``control``, the model is reusing the stored fact about b rather than a
    separate memory of the chain.
    Returns None for models without a lookup.
    """
    cfg = model.cfg
    if cfg.ff != "lookup" or len(chains) == 0:
        return None
    depth = depth or max(2, default_depth(model))
    rng = np.random.default_rng(0)
    h, r1, r2 = chains[:, 0], chains[:, 1], chains[:, 2]
    bridge = data.tail[h, r1]
    other = bridge[rng.permutation(len(bridge))]
    clash = other == bridge
    other[clash] = (other[clash] + 1) % data.n_entities

    def sets(tokens):
        return _answer_position_sets(model, tokens, depth)

    two = sets(data.encode_two_hop(chains)[0])
    direct = sets(data.encode_one_hop(np.stack([bridge, r2], axis=1))[0])
    wrong = sets(data.encode_one_hop(np.stack([other, r2], axis=1))[0])
    return {
        "same": [[_mean_jaccard(a, b) for b in direct] for a in two],
        "control": [[_mean_jaccard(a, b) for b in wrong] for a in two],
    }


def decode_by_depth(model, data, chains, max_depth, batch_size=1024):
    """What the answer position holds after each pass (or outer step).

    Reads the answer position through the output head at every depth of a
    two-hop question and reports how often the top entity is the bridge b and
    how often it is the answer c.

    For the original model, ``first_hop_position_bridge`` is the same reading
    at the first-hop position (the r1 token). Attention is causal, so that
    position computes exactly what the one-hop question [h, r1] computes.
    """
    h, r1 = chains[:, 0], chains[:, 1]
    bridge = data.tail[h, r1]
    tokens, answer = data.encode_two_hop(chains)
    result = {"depth": list(range(1, max_depth + 1)), "bridge": [], "answer": []}
    if uses_slots(model):
        hits = np.zeros((max_depth, 2))
        for start in range(0, len(tokens), batch_size):
            rows = slice(start, start + batch_size)
            _, trace = model(mx.array(tokens[rows]), steps=max_depth, return_trace=True)
            for d, logits in enumerate(trace["step_logits"]):
                top = np.array(mx.argmax(logits, axis=-1))
                hits[d] += [(top == bridge[rows]).sum(), (top == answer[rows]).sum()]
        result["bridge"] = (hits[:, 0] / len(tokens)).tolist()
        result["answer"] = (hits[:, 1] / len(tokens)).tolist()
        return result

    first_hop, _ = data.encode_one_hop(chains[:, :2])
    result["first_hop_position_bridge"] = []
    for d in result["depth"]:
        top = predict(model, tokens, d, batch_size)
        result["bridge"].append(float((top == bridge).mean()))
        result["answer"].append(float((top == answer).mean()))
        at_r1 = predict(model, first_hop, d, batch_size)
        result["first_hop_position_bridge"].append(float((at_r1 == bridge).mean()))
    return result


def slot_contents(model, data, chains, batch_size=1024):
    """What a scratchpad model wrote after the first step of two-hop questions.

    ``bridge_in_slot`` is the share of chains where some slot is closest (by
    cosine similarity over entity embeddings) to the bridge entity.
    ``bridge_in_readout`` is the share where the step-1 readout's top entity
    is the bridge. ``write_weight`` is the mean total weight written to slots.
    """
    cfg = model.cfg
    bridge = data.tail[chains[:, 0], chains[:, 1]]
    tokens, _ = data.encode_two_hop(chains)
    entities = model.embed.weight[: cfg.n_entities]
    entities = entities / mx.linalg.norm(entities, axis=-1, keepdims=True)
    in_slot = in_readout = 0
    written = 0.0
    for start in range(0, len(tokens), batch_size):
        b = bridge[start : start + batch_size]
        _, trace = model(
            mx.array(tokens[start : start + batch_size]), steps=1, return_trace=True
        )
        slots = trace["slots"][0]  # (B, K, d)
        nearest = np.array(mx.argmax(slots @ entities.T, axis=-1))  # (B, K)
        in_slot += (nearest == b[:, None]).any(axis=1).sum()
        in_readout += (np.array(mx.argmax(trace["step_logits"][0], axis=-1)) == b).sum()
        written += float(trace["write"][0][:, : cfg.num_slots].sum())
    n = len(tokens)
    return {
        "bridge_in_slot": float(in_slot / n),
        "bridge_in_readout": float(in_readout / n),
        "write_weight": written / n,
    }


def two_hop_sets(data, limit=None):
    return {
        name: data.eval_sets[name][:limit]
        for name in TWO_HOP_SPLITS
        if len(data.eval_sets.get(name, ()))
    }


def slot_interventions(model, data, chains, batch_size=512):
    """Replace only the slot passed from step 1 to step 2, at evaluation time.

    Oracle conditions use the true bridge solely as a diagnostic. The normal
    training and inference paths never receive it. Shuffling preserves stored
    vectors but assigns every question a different question's slot.
    """
    if not uses_slots(model) or default_depth(model) < 2:
        raise ValueError("slot interventions need at least two outer steps")
    tokens, target = data.encode_two_hop(chains)
    n = len(tokens)
    bridge = data.tail[chains[:, 0], chains[:, 1]]
    stored, writes = [], []
    with eval_mode(model):
        for start in range(0, n, batch_size):
            _, trace = model(mx.array(tokens[start:start + batch_size]), steps=1, return_trace=True)
            stored.append(np.array(trace["slots"][0]))
            writes.append(np.array(trace["write"][0][:, :model.cfg.num_slots, None]))
        stored = np.concatenate(stored)
        writes = mx.array(np.concatenate(writes))
        order = np.random.default_rng(0).permutation(n)
        permutation = np.empty(n, dtype=np.int32)
        permutation[order] = np.roll(order, 1)
        true_content = model.embed.weight[mx.array(bridge)][:, None, :]
        empty = mx.broadcast_to(model.slot_empty, stored.shape)
        overrides = {
            "normal": None,
            "cleared": empty,
            "shuffled": mx.array(stored[permutation]),
            "oracle_write": (1 - writes) * empty + writes * true_content,
            "oracle_exact": mx.broadcast_to(true_content, stored.shape),
        }
        result = {}
        for name, slots in overrides.items():
            correct = 0
            for start in range(0, n, batch_size):
                end = start + batch_size
                kwargs = {} if slots is None else {"slot_override": slots[start:end]}
                pred = np.array(mx.argmax(model(mx.array(tokens[start:end]), **kwargs), axis=-1))
                correct += int((pred == target[start:end]).sum())
            result[name] = {"correct": correct, "n": n, "accuracy": correct / n}
    return result


def full_evaluation(model, data, depths, overlap_chains=500):
    """Everything reported at the end of a run."""
    # Imported here because two_call imports this module.
    from .two_call import diagnose as two_call_diagnose

    cfg = model.cfg
    slots = uses_slots(model)
    depth = default_depth(model)
    by_depth = "accuracy_by_steps" if slots else "accuracy_by_passes"
    # One-hop calls at half the normal depth match the normal compute; calls
    # at the full depth are the ceiling.
    call_depths = sorted({max(1, depth // 2), depth})
    with eval_mode(model):
        result = {
            "depth_unit": "steps" if slots else "passes",
            "eval_depth": depth,
            "accuracy": accuracy_by_split(model, data),
            by_depth: {str(d): accuracy_by_split(model, data, d) for d in depths},
            "table": table_usage(model, data),
            "overlap": None,
            "decode": {
                name: decode_by_depth(model, data, rows, max(depth, 4) if slots else depth)
                for name, rows in two_hop_sets(data).items()
            },
            "slots": None,
        }
        if cfg.ff == "lookup":
            result["overlap"] = {
                name: retrieval_overlap(model, data, rows)
                for name, rows in two_hop_sets(data, overlap_chains).items()
            }
        if slots:
            result["slots"] = {
                name: slot_contents(model, data, rows)
                for name, rows in two_hop_sets(data).items()
            }
            if depth >= 2:
                result["slot_interventions"] = {
                    name: slot_interventions(model, data, rows)
                    for name, rows in two_hop_sets(data).items()
                }
    result["two_call"] = two_call_diagnose(model, data, call_depths, (depth,))
    return result
