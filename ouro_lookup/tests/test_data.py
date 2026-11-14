import numpy as np
import pytest

from ouro_lookup.data import NAMES, World, WordTokenizer


def world(**kw):
    kw = {"n_people": 30, "n_relations": 3, "seed": 1, **kw}
    return World(WordTokenizer().encode, **kw)


def test_sizes_and_question_shapes():
    w = world()
    assert len(w.fact_answer) == 90 and len(w.chain_first) == 270
    assert w.fact_tokens.shape == (90, 6)  # start, The, relation, of, person, is
    assert w.chain_tokens.shape == (270, 9)
    assert (w.fact_tokens[:, 0] == w.start).all() and (w.chain_tokens[:, 0] == w.start).all()
    assert len(set(w.names)) == 30 and len(set(w.name_token.tolist())) == 30


def test_nobody_is_their_own_answer():
    w = world()
    assert (w.fact_answer != w.fact_person).all()


def test_a_chain_is_two_facts_that_meet():
    w = world()
    first, second = w.chain_first, w.chain_second
    # The first fact's answer is the person the second fact is about.
    assert (w.fact_answer[first] == w.fact_person[second]).all()
    ids = np.arange(len(first))
    chains = w.chains(ids)
    assert (chains.bridge == w.name_token[w.fact_answer[first]]).all()
    assert (chains.answer == w.name_token[w.fact_answer[second]]).all()
    # The question names the first fact's person and both relations.
    assert (w.chain_tokens[:, -2] == w.name_token[w.fact_person[first]]).all()


def test_second_question_about_the_true_note_is_the_second_fact():
    w = world()
    ids = np.arange(len(w.chain_first))
    asked = w.second_questions(ids, w.chains(ids).bridge)
    assert (asked == w.fact_tokens[w.chain_second]).all()
    other = w.second_questions(ids[:1], np.array([12345]))
    assert other[0, w.person_position] == 12345


def test_held_out_facts_never_appear_in_training_chains():
    w = world(n_people=40)
    held = ~w.practised
    assert held.sum() > 0
    assert not held[w.chain_first[w.train_chains]].any()
    assert not held[w.chain_second[w.train_chains]].any()
    in_train = set(w.train_chains.tolist())
    for name, ids in w.groups.items():
        assert not in_train & set(ids.tolist())
        a, b = w.practised[w.chain_first[ids]], w.practised[w.chain_second[ids]]
        want = {
            "chain_unseen": (True, True), "chain_held_first": (False, True),
            "chain_held_second": (True, False), "chain_held_both": (False, False),
        }[name]
        assert (a == want[0]).all() and (b == want[1]).all()


def test_training_samples_use_training_chains_only():
    w = world()
    one, chain = w.sample(np.random.default_rng(0), 50, 50)
    assert len(one) == 50 and len(chain) == 50
    train_rows = {tuple(row) for row in w.chain_tokens[w.train_chains]}
    assert all(tuple(row) in train_rows for row in chain.tokens)
    assert (one.bridge == 0).all()


def test_same_seed_same_world():
    a, b = world(seed=3), world(seed=3)
    assert (a.fact_answer == b.fact_answer).all() and (a.train_chains == b.train_chains).all()
    assert (world(seed=4).fact_answer != a.fact_answer).any()


def test_names_must_be_single_tokens():
    def splits_names(text):  # a tokenizer that breaks every name in two
        words = text.split()
        return [i for w in words for i in ([1, 2] if w in NAMES else [3])]

    with pytest.raises(ValueError, match="single-token names"):
        World(splits_names, n_people=10, n_relations=2)


def test_guess_rate_is_low():
    w = world(n_people=40)
    assert w.guess_rate(w.facts(np.arange(len(w.fact_answer)))) < 0.2
