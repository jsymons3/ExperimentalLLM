"""Accuracy, table usage and retrieval overlap."""

from contextlib import contextmanager

import mlx.core as mx
import numpy as np


@contextmanager
def eval_mode(model):
    """Freeze the query-normalisation statistics while evaluating."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _batches(tokens, batch_size):
    for i in range(0, len(tokens), batch_size):
        yield tokens[i : i + batch_size]


def accuracy(model, data, rows, n_passes=None, batch_size=1024):
    """Exact-match accuracy on one-hop (n, 2) or two-hop (n, 3) rows."""
    tokens, targets = data.encode(rows)
    preds = [
        np.array(mx.argmax(model(mx.array(batch), n_passes=n_passes), axis=-1))
        for batch in _batches(tokens, batch_size)
    ]
    return float((np.concatenate(preds) == targets).mean())


def accuracy_by_split(model, data, n_passes=None):
    return {
        name: accuracy(model, data, rows, n_passes)
        for name, rows in data.eval_sets.items()
        if len(rows)
    }


def table_usage(model, data, n_passes=None, sample=20000, batch_size=1024):
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
            _, retrieved = model(
                mx.array(batch), n_passes=n_passes, return_retrieval=True
            )
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


def _answer_position_sets(model, tokens, n_passes, batch_size=1024):
    """sets[p][i]: entries retrieved at the last position of row i on pass p.

    sets[p] is None for a pass without a lookup.
    """
    parts = [[] for _ in range(n_passes)]
    for batch in _batches(tokens, batch_size):
        _, retrieved = model(mx.array(batch), n_passes=n_passes, return_retrieval=True)
        for p, index in enumerate(retrieved):
            if index is not None:
                parts[p].append(np.array(index[:, -1]).reshape(len(batch), -1))
    return [
        [set(row) for row in np.concatenate(p).tolist()] if p else None for p in parts
    ]


def _mean_jaccard(a, b):
    if a is None or b is None:
        return None
    return float(np.mean([len(x & y) / len(x | y) for x, y in zip(a, b)]))


def retrieval_overlap(model, data, chains, n_passes=None):
    """Does the second hop fetch the entries a direct question would fetch?

    For each chain (h, r1, r2) with bridge b = tail[h, r1], compare the entries
    retrieved at the answer position of the two-hop question with those
    retrieved at the answer position of the direct question [b, r2].

    ``same[p][q]`` is the mean Jaccard overlap between pass p+1 of the two-hop
    question and pass q+1 of the direct question. ``control[p][q]`` is the same
    for the direct question [b', r2] about a different entity b'. If ``same``
    is well above ``control``, the model is reusing the stored fact about b
    rather than a separate memory of the chain.
    Returns None for models without a lookup.
    """
    cfg = model.cfg
    if cfg.ff != "lookup" or len(chains) == 0:
        return None
    n_passes = n_passes or cfg.n_passes
    rng = np.random.default_rng(0)
    h, r1, r2 = chains[:, 0], chains[:, 1], chains[:, 2]
    bridge = data.tail[h, r1]
    other = bridge[rng.permutation(len(bridge))]
    clash = other == bridge
    other[clash] = (other[clash] + 1) % data.n_entities

    def sets(tokens):
        return _answer_position_sets(model, tokens, n_passes)

    two = sets(data.encode_two_hop(chains)[0])
    direct = sets(data.encode_one_hop(np.stack([bridge, r2], axis=1))[0])
    wrong = sets(data.encode_one_hop(np.stack([other, r2], axis=1))[0])
    return {
        "same": [[_mean_jaccard(a, b) for b in direct] for a in two],
        "control": [[_mean_jaccard(a, b) for b in wrong] for a in two],
    }


def full_evaluation(model, data, eval_passes, overlap_chains=500):
    """Everything reported at the end of a run."""
    cfg = model.cfg
    with eval_mode(model):
        result = {
            "accuracy": accuracy_by_split(model, data),
            "accuracy_by_passes": {
                str(p): accuracy_by_split(model, data, p) for p in eval_passes
            },
            "table": table_usage(model, data),
            "overlap": None,
        }
        if cfg.ff == "lookup":
            result["overlap"] = {
                name: retrieval_overlap(model, data, rows[:overlap_chains])
                for name, rows in data.eval_sets.items()
                if name.startswith("two_hop") and len(rows)
            }
    return result
