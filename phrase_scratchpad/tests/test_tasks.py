"""Checks on the tasks: a leak or a wrong label here would fake a result."""

import hashlib
import json

import numpy as np
import pytest

from phrase_scratchpad.clutrr import StoryTask
from phrase_scratchpad.tasks import (
    FACTS_FILE,
    PAD,
    Suite,
    compact,
    facts_task,
    load_facts,
    maths_task,
)


def number(task, tokens):
    """The value of a lowest-digit-first number."""
    digits = [int(task.vocab[t]) for t in tokens if t != PAD]
    return sum(d * 10**i for i, d in enumerate(digits))


def test_compact_moves_padding_to_the_end_and_keeps_order():
    joined = compact(np.array([[5, 0, 0], [1, 2, 0]]), np.array([[7, 8], [0, 0]]))
    assert joined.tolist() == [[5, 7, 8, 0, 0], [1, 2, 0, 0, 0]]


def test_arithmetic_is_right_and_never_negative():
    task = maths_task(max_operand=12, eval_size=50)
    op = {task.vocab.index("+"): 1, task.vocab.index("-"): -1}
    width = task.answer_len
    for split, items in task.eval_sets.items():
        for q, bridge, answer, chain in zip(items.question, items.bridge, items.answer, items.chain):
            x, sign1, y = number(task, q[:width]), op[q[width]], number(task, q[width + 1 : 2 * width + 1])
            first = x + sign1 * y
            assert first >= 0
            if chain:
                rest = q[2 * width + 1 :]
                final = first + op[rest[0]] * number(task, rest[1 : width + 1])
                assert number(task, bridge) == first, split
                assert number(task, answer) == final >= 0, split
            else:
                assert number(task, answer) == first and (bridge == PAD).all(), split
    # 47 is written lowest digit first.
    big = maths_task(max_operand=99, eval_size=5)
    row = np.flatnonzero((big.subject == [8, 5, 1]).all(axis=1))[0]  # tokens for 7, 4, 0
    assert number(big, big.subject[row]) == 47


@pytest.mark.parametrize("make", [lambda: maths_task(max_operand=12, eval_size=10**6),
                                  lambda: facts_task(n_people=80, eval_size=10**6)])
def test_split_by_what_training_chains_use(make):
    task = make()
    first, second = task.chain_first, task.chain_second
    # The bridge of a chain is the subject of its second fact.
    assert (task.answer[first] == task.subject[second]).all()
    used = np.zeros(len(task.subject), dtype=bool)
    used[first[task.train_chains]] = True
    used[second[task.train_chains]] = True
    assert (used == task.practised).all() and (~used).any()

    expect = {
        "chain_unseen": (True, True),
        "chain_held_first": (False, True),
        "chain_held_second": (True, False),
        "chain_held_both": (False, False),
    }
    seen = set(task.train_chains.tolist())
    for name, (a, b) in expect.items():
        ids = task.eval_chains[name]
        assert len(ids), name
        assert (used[first[ids]] == a).all() and (used[second[ids]] == b).all(), name
        assert not seen & set(ids.tolist()), name
        seen |= set(ids.tolist())
    assert len(seen) == len(first)  # every chain is in exactly one group
    assert not used[task.eval_facts["one_hop_held"]].any()
    assert used[task.eval_facts["one_hop"]].all()


def test_facts_file_matches_its_recorded_hash():
    meta = json.loads(FACTS_FILE.with_suffix(".meta.json").read_text())
    assert hashlib.sha256(FACTS_FILE.read_bytes()).hexdigest() == meta["sha256"]
    rows = load_facts()
    assert len(rows) == meta["facts"]
    assert len({(s, r) for s, r, _ in rows}) == len(rows)  # one object per subject and relation
    assert all(1 <= len(name.split()) <= 4 for s, _, o in rows for name in (s, o))


def test_fact_chains_read_as_expected():
    task = facts_task(n_people=80, eval_size=20)
    rows = {(s, f"[{r}]"): o for s, r, o in load_facts()}
    items = task.eval_sets["chain_held_second"]
    assert items.answer.max() < task.n_output and items.bridge.max() < task.n_output
    for q, bridge, answer, first in zip(items.question, items.bridge, items.answer, items.first):
        words = task.decode(q).split(" [")
        subject, r1, r2 = words[0], "[" + words[1], "[" + words[2]
        assert rows[(subject, r1)] == task.decode(bridge)
        assert rows[(task.decode(bridge), r2)] == task.decode(answer)
        assert task.decode(first) == f"{subject} {r1}"
    again = facts_task(n_people=80, eval_size=20)
    assert (again.eval_sets["chain_held_second"].question == items.question).all()
    second = task.second_questions(items.bridge, items)
    assert task.decode(second[0]) == f"{task.decode(items.bridge[0])} {task.decode(items.suffix[0])}"


def test_guessing_baseline_is_between_zero_and_one_and_not_trivial():
    task = facts_task(n_people=80, eval_size=200)
    baseline = task.majority_baseline(task.eval_sets["chain_unseen"])
    assert 0 < baseline < 0.9


def stories():
    def story(split, text, people, relations, answer):
        return {"split": split, "story": text, "people": people, "relations": relations, "answer": answer}

    return [
        story("train", "[Ann]'s daughter, [Bea], sang. [Cal]'s sister [Bea] left.", ["Ann", "Bea", "Cal"], ["daughter", "brother"], "son"),
        story("train", "[Dan] and his sister [Eve] ran. [Eve]'s son [Fay] slept.", ["Dan", "Eve", "Fay"], ["sister", "son"], "nephew"),
        story("test", "[Gus] called his mother [Ann]. [Ann]'s husband [Hal] cooked.", ["Gus", "Ann", "Hal"], ["mother", "husband"], "father"),
        # A test story that repeats a training story must be dropped.
        story("test", "[Dan] and his sister [Eve] ran. [Eve]'s son [Fay] slept.", ["Dan", "Eve", "Fay"], ["sister", "son"], "nephew"),
        story("train", "long " * 60, ["Ann", "Bea", "Cal"], ["daughter", "brother"], "son"),  # too long
    ]


def test_story_questions_and_their_intermediate_answer():
    task = StoryTask(stories(), eval_size=10)
    assert task.n_stories == {"train": 2, "test": 1}
    two = task.train_two
    assert task.decode(two.question[0]).endswith("? [Ann] [Cal]")
    assert task.decode(two.bridge[0]) == "daughter [Bea]"  # the first link and the middle person
    assert task.decode(two.answer[0]) == "son"
    one = task.train_one
    assert task.decode(one.question[0]).endswith("? [Ann] [Bea]") and task.decode(one.answer[0]) == "daughter"
    assert task.decode(one.question[1]).endswith("? [Bea] [Cal]") and task.decode(one.answer[1]) == "brother"
    assert two.chain.all() and not one.chain.any()
    unseen = task.eval_sets["chain_unseen"]
    assert len(unseen) == 1 and task.decode(unseen.answer[0]) == "father"
    assert max(two.answer.max(), two.bridge.max(), one.answer.max()) < task.n_output
    batch = task.sample(np.random.default_rng(0), 3, 2)
    assert batch.question.shape == (5, task.question_len) and batch.chain.tolist() == [0, 0, 0, 1, 1]


def test_tasks_share_a_vocabulary_without_sharing_tokens():
    parts = [maths_task(max_operand=12, eval_size=20), StoryTask(stories(), eval_size=10),
             facts_task(n_people=60, eval_size=20)]
    before = [(t.decode(t.sample(np.random.default_rng(1), 1, 1).question[1]),
               t.decode(t.sample(np.random.default_rng(1), 1, 1).bridge[1])) for t in parts]
    suite = Suite(parts)
    assert suite.answer_len == 4 and suite.question_len == max(t.question_len for t in suite.tasks)
    assert len(set(suite.vocab)) == len(suite.vocab)
    for task, (question, bridge) in zip(suite.tasks, before):
        items = task.sample(np.random.default_rng(1), 1, 1)
        strip = lambda text: " ".join(w.split(":", 1)[1] for w in text.split())
        assert strip(task.decode(items.question[1])) == question
        assert strip(task.decode(items.bridge[1])) == bridge
        assert items.answer.shape[1] == items.bridge.shape[1] == 4
        assert max(items.answer.max(), items.bridge.max()) < suite.n_output
        assert all(suite.vocab[t].startswith(task.name + ":") for t in items.question[1] if t != PAD)
