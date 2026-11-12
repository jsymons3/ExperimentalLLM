"""Checks on the splits: a leak here would fake the main result."""

import numpy as np

from recurrent_lookup.data import GraphConfig, MultiHopData


def make():
    return MultiHopData(GraphConfig(n_entities=60, n_relations=6, eval_size=100000))


def as_set(rows):
    return {tuple(r) for r in rows.tolist()}


def test_every_fact_is_a_one_hop_training_example():
    data = make()
    assert len(data.facts) == 60 * 6
    assert len(as_set(data.facts)) == 60 * 6
    assert int(data.held.sum()) == round(0.05 * 60 * 6)


def test_training_chains_never_use_a_held_out_fact():
    data = make()
    h, r1, r2 = data.two_hop_train.T
    assert not data.held[h, r1].any()
    assert not data.held[data.tail[h, r1], r2].any()


def test_two_hop_splits_are_disjoint_and_cover_all_chains():
    data = make()
    names = [n for n in data.eval_sets if n.startswith("two_hop") and n != "two_hop_train"]
    parts = [as_set(data.two_hop_train)] + [as_set(data.eval_sets[n]) for n in names]
    assert sum(len(p) for p in parts) == len(set().union(*parts)) == 60 * 6 * 6
    assert as_set(data.eval_sets["two_hop_train"]) <= as_set(data.two_hop_train)


def test_held_out_splits_hold_out_the_named_hop():
    data = make()
    expect = {
        "two_hop_id": (False, False),
        "two_hop_held_first": (True, False),
        "two_hop_held_second": (False, True),
        "two_hop_held_both": (True, True),
    }
    for name, (first, second) in expect.items():
        h, r1, r2 = data.eval_sets[name].T
        assert (data.held[h, r1] == first).all(), name
        assert (data.held[data.tail[h, r1], r2] == second).all(), name


def test_encoding_and_targets():
    data = make()
    chains = data.eval_sets["two_hop_held_second"]
    tokens, targets = data.encode(chains)
    assert tokens.shape == (len(chains), 3)
    assert (tokens[:, 0] < 60).all() and (tokens[:, 1:] >= 60).all()
    assert tokens.max() < data.vocab_size
    for (h, r1, r2), t in zip(chains.tolist(), targets.tolist()):
        assert t == data.tail[data.tail[h, r1], r2]
    tokens, targets = data.encode(data.facts)
    assert tokens.shape == (360, 2)
    assert (targets == data.tail[data.facts[:, 0], data.facts[:, 1]]).all()


def test_same_seed_same_graph():
    a, b = make(), make()
    assert (a.tail == b.tail).all() and (a.held == b.held).all()
    assert (a.two_hop_train == b.two_hop_train).all()
