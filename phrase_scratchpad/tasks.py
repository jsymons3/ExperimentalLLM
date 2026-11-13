"""Arithmetic and real-name facts, and how several tasks share one model.

Every task presents the same things (questions, answers, an intermediate
answer for two-step questions, and named test splits), so the model and the
training loop do not know which one they are running. The family-story task
is in ``clutrr.py``.

Arithmetic and facts are both built from one-step questions:

* A *fact* is a one-step question: a subject of up to K tokens, a suffix of up
  to P tokens, and an answer of up to K tokens.

      arithmetic   subject 47               suffix + 85         answer 132
      facts        subject "Jaws"           suffix [director]   answer "Steven Spielberg"

* A *chain* is two facts where the first answer is the second subject. Its
  question is the first subject followed by both suffixes, and the first
  answer (the bridge) is what the scratchpad should hold after step one.

      arithmetic   47 + 85 - 20                      bridge 132     answer 112
      facts        "Jaws [director] [date of birth]"  bridge "Steven Spielberg"

Numbers are written with a fixed number of digits, one token each, lowest
digit first: 47 is the tokens "7 4 0". Attention is causal, so this order
lets the position working out the tens see the position working out the
ones, which is the direction a carry travels.

Splits follow the graph experiment. Every fact is trained as a one-step
question. Some facts are kept out of every training chain. Test chains are
grouped by whether their first fact, second fact, or both never appeared in
a training chain.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAD = 0
FACTS_FILE = Path(__file__).parent / "facts" / "people_facts.tsv"
CHAIN_GROUPS = ("chain_unseen", "chain_held_first", "chain_held_second", "chain_held_both")


@dataclass
class Items:
    """A set of questions as token arrays."""

    question: np.ndarray  # (n, L), padded at the end
    answer: np.ndarray  # (n, K)
    bridge: np.ndarray  # (n, K): the intermediate answer; padding if none
    chain: np.ndarray  # (n,) bool: True for two-step questions
    # Where a task is built from one-step questions: the one-step question
    # for the first hop, and the question's last suffix (the relation or
    # operation that produces the answer). Used by the two-call diagnostic
    # and the guessing baseline.
    first: np.ndarray = None  # (n, L)
    suffix: np.ndarray = None  # (n, P)
    # How many tokens at the start of a two-step question are its subject,
    # which training may hide after the first step. 0 where that is not
    # meaningful.
    subject_len: np.ndarray = None  # (n,)

    def __post_init__(self):
        if self.subject_len is None:
            self.subject_len = np.zeros(len(self.question), dtype=np.int32)

    def __len__(self):
        return len(self.question)


def join(a, b):
    """Two sets of items, one after the other."""

    def cat(x, y):
        return None if x is None or y is None else np.concatenate([x, y])

    return Items(*(cat(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__))


def compact(*parts):
    """Join token arrays side by side and move the padding to the end."""
    joined = np.concatenate(parts, axis=1)
    order = np.argsort(joined == PAD, axis=1, kind="stable")
    return np.take_along_axis(joined, order, axis=1)


def widen(tokens, width):
    """Right-pad a token array to a given width."""
    extra = width - tokens.shape[1]
    return np.pad(tokens, ((0, 0), (0, extra))) if extra else tokens


class Task:
    """Facts, the chains between them, and their train/test split."""

    def __init__(
        self,
        name,
        vocab,
        n_output,
        subject,
        suffix,
        answer,
        chain_first,
        chain_second,
        held_frac,
        train_frac,
        eval_size,
        seed,
    ):
        self.name = name
        self.vocab = list(vocab)  # id -> token; id 0 is padding
        self.n_output = n_output  # answers only use ids below this
        self.subject = subject.astype(np.int32)  # (facts, K)
        self.suffix = suffix.astype(np.int32)  # (facts, P)
        self.answer = answer.astype(np.int32)  # (facts, K)
        self.chain_first = chain_first.astype(np.int64)  # fact id of the first hop
        self.chain_second = chain_second.astype(np.int64)
        self.answer_len = subject.shape[1]
        self.suffix_len = suffix.shape[1]
        assert answer.shape[1] == self.answer_len and self.answer.max() < n_output
        # A chain is valid only if the first answer is the second subject.
        assert (self.answer[self.chain_first] == self.subject[self.chain_second]).all()
        self._split(held_frac, train_frac, eval_size, np.random.default_rng(seed))

    # -- splits -----------------------------------------------------------

    def _split(self, held_frac, train_frac, eval_size, rng):
        n_facts, n_chains = len(self.subject), len(self.chain_first)
        first, second = self.chain_first, self.chain_second

        # Facts that are kept out of every training chain.
        held = np.zeros(n_facts, dtype=bool)
        held[rng.permutation(n_facts)[: int(round(held_frac * n_facts))]] = True
        clean = np.flatnonzero(~held[first] & ~held[second])
        clean = clean[rng.permutation(len(clean))]
        self.train_chains = np.sort(clean[: int(round(train_frac * len(clean)))])

        # A fact counts as practised if any training chain uses it, in either
        # position. Test chains are grouped by which of their facts were not.
        practised = np.zeros(n_facts, dtype=bool)
        practised[first[self.train_chains]] = True
        practised[second[self.train_chains]] = True
        self.practised = practised
        in_train = np.zeros(n_chains, dtype=bool)
        in_train[self.train_chains] = True
        rest = np.flatnonzero(~in_train)
        a, b = practised[first[rest]], practised[second[rest]]
        groups = {
            "chain_unseen": rest[a & b],
            "chain_held_first": rest[~a & b],
            "chain_held_second": rest[a & ~b],
            "chain_held_both": rest[~a & ~b],
        }
        self.group_sizes = {name: len(idx) for name, idx in groups.items()}

        def sample(idx):
            return np.sort(idx[rng.permutation(len(idx))[:eval_size]])

        self.eval_facts = {
            "one_hop": sample(np.flatnonzero(practised)),
            "one_hop_held": sample(np.flatnonzero(~practised)),
        }
        self.eval_chains = {"chain_train": sample(self.train_chains)}
        self.eval_chains.update({name: sample(idx) for name, idx in groups.items()})

    # -- items ------------------------------------------------------------

    @property
    def question_len(self):
        return self.answer_len + 2 * self.suffix_len

    def fact_items(self, ids):
        n = len(ids)
        blank = np.zeros((n, self.suffix_len), dtype=np.int32)
        return Items(
            question=compact(self.subject[ids], self.suffix[ids], blank),
            answer=self.answer[ids],
            bridge=np.zeros((n, self.answer_len), dtype=np.int32),
            chain=np.zeros(n, dtype=bool),
            first=compact(self.subject[ids], self.suffix[ids], blank),
            suffix=self.suffix[ids],
        )

    def chain_items(self, ids):
        first, second = self.chain_first[ids], self.chain_second[ids]
        blank = np.zeros((len(ids), self.suffix_len), dtype=np.int32)
        return Items(
            question=compact(self.subject[first], self.suffix[first], self.suffix[second]),
            answer=self.answer[second],
            bridge=self.answer[first],
            chain=np.ones(len(ids), dtype=bool),
            first=compact(self.subject[first], self.suffix[first], blank),
            suffix=self.suffix[second],
            subject_len=(self.subject[first] != PAD).sum(axis=1).astype(np.int32),
        )

    @property
    def eval_sets(self):
        """{split name: Items}, skipping empty splits."""
        sets = {name: self.fact_items(ids) for name, ids in self.eval_facts.items()}
        sets.update({name: self.chain_items(ids) for name, ids in self.eval_chains.items()})
        return {name: items for name, items in sets.items() if len(items)}

    def sample(self, rng, n_one, n_chain):
        """Random training questions: every fact, and the training chains."""
        one = self.fact_items(rng.integers(0, len(self.subject), n_one))
        picked = self.train_chains[rng.integers(0, len(self.train_chains), n_chain)]
        return join(one, self.chain_items(picked))

    def second_questions(self, bridge, items):
        """The one-step question for the second hop, about a given bridge."""
        return widen(compact(bridge, items.suffix), self.question_len)

    def rebased(self, remap, answer_len):
        """A copy using another vocabulary's token ids and a wider answer."""
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.subject = widen(remap[self.subject], answer_len)
        clone.answer = widen(remap[self.answer], answer_len)
        clone.suffix = remap[self.suffix]
        clone.answer_len = answer_len
        return clone

    def majority_baseline(self, items):
        """Accuracy of always giving the commonest training answer for the
        question's last suffix. Shows how far guessing gets."""
        keys, inverse = np.unique(self.suffix, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        best = {}
        for k, key in enumerate(keys):
            answers, counts = np.unique(self.answer[inverse == k], axis=0, return_counts=True)
            best[key.tobytes()] = answers[counts.argmax()]
        nothing = np.zeros(self.answer_len, dtype=np.int32)
        guesses = np.stack([best.get(row.tobytes(), nothing) for row in items.suffix])
        return float((guesses == items.answer).all(axis=1).mean())

    def decode(self, tokens):
        """Token ids back to text, for inspection."""
        return " ".join(self.vocab[t] for t in tokens if t != PAD)

    def summary(self):
        out = {
            "task": self.name,
            "vocabulary": len(self.vocab),
            "answer_tokens": self.n_output,
            "answer_len": self.answer_len,
            "question_len": self.question_len,
            "facts": len(self.subject),
            "facts_never_in_a_training_chain": int((~self.practised).sum()),
            "chains": len(self.chain_first),
            "training_chains": len(self.train_chains),
        }
        out.update({f"all_{name}": size for name, size in self.group_sizes.items()})
        return out


# -- arithmetic -------------------------------------------------------------


def maths_task(max_operand=99, held_frac=0.05, train_frac=0.2, eval_size=2000, seed=0):
    """Two-step addition and subtraction on whole numbers.

    A one-step problem is ``x + y`` or ``x - y`` with x up to 2 * max_operand,
    y up to max_operand, and no negative results. A chain is
    ``(a op b) op c`` with a, b, c up to max_operand. Numbers are written
    with a fixed number of digits, lowest digit first, one token each.
    """
    m = max_operand
    width = len(str(3 * m))  # digits in the largest possible result
    vocab = ["<pad>"] + [str(d) for d in range(10)] + ["+", "-"]
    plus, minus = 11, 12

    def digits(values):
        powers = 10 ** np.arange(width)  # lowest digit first
        return (values[:, None] // powers) % 10 + 1

    # One-step problems, with an id for each valid (op, x, y).
    op, x, y = np.meshgrid(np.arange(2), np.arange(2 * m + 1), np.arange(m + 1), indexing="ij")
    valid = (op == 0) | (x >= y)
    fact_id = np.full(valid.shape, -1, dtype=np.int64)
    fact_id[valid] = np.arange(valid.sum())
    op, x, y = op[valid], x[valid], y[valid]
    result = np.where(op == 0, x + y, x - y)
    subject = digits(x)
    suffix = np.concatenate([np.where(op == 0, plus, minus)[:, None], digits(y)], axis=1)
    answer = digits(result)

    # Chains (a op1 b) op2 c.
    op1, op2, a, b, c = np.meshgrid(
        np.arange(2), np.arange(2), np.arange(m + 1), np.arange(m + 1), np.arange(m + 1),
        indexing="ij",
    )
    mid = np.where(op1 == 0, a + b, a - b)
    ok = ((op1 == 0) | (a >= b)) & ((op2 == 0) | (mid >= c))
    chain_first = fact_id[op1[ok], a[ok], b[ok]]
    chain_second = fact_id[op2[ok], mid[ok], c[ok]]
    return Task(
        "maths", vocab, 11, subject, suffix, answer, chain_first, chain_second,
        held_frac, train_frac, eval_size, seed,
    )


# -- facts ------------------------------------------------------------------


def load_facts(path=FACTS_FILE):
    rows = [line.split("\t") for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert all(len(row) == 3 for row in rows)
    return rows


def facts_task(
    n_people=1500, held_frac=0.1, train_frac=0.7, eval_size=2000, seed=0, path=FACTS_FILE
):
    """Real-name facts about people, and chains that go through a person.

    Facts come from ``facts/people_facts.tsv``: "X [director] person" style
    first hops and "person [date of birth] value" style second hops. Names are
    split into words, one token each, and each relation is one token.
    ``n_people`` picks a random subset of the people, with all their facts.
    """
    from .prepare_facts import ABOUT_PERSON, MAX_WORDS, TO_PERSON

    rows = load_facts(path)
    rng = np.random.default_rng(seed)
    incoming, outgoing = {}, {}
    for i, (subject, relation, obj) in enumerate(rows):
        if relation in TO_PERSON:
            incoming.setdefault(obj, []).append(i)
        if relation in ABOUT_PERSON:
            outgoing.setdefault(subject, []).append(i)
    people = sorted(p for p in incoming if p in outgoing)
    if n_people and n_people < len(people):
        people = sorted(people[i] for i in rng.permutation(len(people))[:n_people])

    kept = sorted({i for p in people for i in incoming[p] + outgoing[p]})
    renumber = {old: new for new, old in enumerate(kept)}
    facts = [rows[i] for i in kept]
    chain_first, chain_second = [], []
    for p in people:
        for i in incoming[p]:
            for j in outgoing[p]:
                chain_first.append(renumber[i])
                chain_second.append(renumber[j])

    # Vocabulary: padding, then every word that can appear in an answer,
    # then relations and words that only appear in questions.
    answer_words = sorted({w for _, _, obj in facts for w in obj.split()})
    relations = sorted({f"[{relation}]" for _, relation, _ in facts})
    other_words = sorted({w for s, _, _ in facts for w in s.split()} - set(answer_words))
    vocab = ["<pad>"] + answer_words + relations + other_words
    token = {w: i for i, w in enumerate(vocab)}

    def encode(names):
        out = np.zeros((len(names), MAX_WORDS), dtype=np.int32)
        for i, name in enumerate(names):
            words = name.split()
            out[i, : len(words)] = [token[w] for w in words]
        return out

    subject = encode([s for s, _, _ in facts])
    answer = encode([o for _, _, o in facts])
    suffix = np.array([[token[f"[{r}]"]] for _, r, _ in facts], dtype=np.int32)
    return Task(
        "facts", vocab, 1 + len(answer_words), subject, suffix, answer,
        np.array(chain_first), np.array(chain_second), held_frac, train_frac, eval_size, seed,
    )


# -- one or several tasks ---------------------------------------------------


class Suite:
    """The task or tasks one model is trained on.

    Each task keeps its own questions and splits. With more than one task the
    token ids are moved into a joint vocabulary (answer tokens first, and no
    token shared between tasks) and answers are padded to the longest task's,
    so one model can read and write every task. Questions keep their own
    length per task.
    """

    def __init__(self, tasks):
        self.name = "+".join(t.name for t in tasks)
        if len(tasks) == 1:
            self.tasks = list(tasks)
            self.vocab, self.n_output = tasks[0].vocab, tasks[0].n_output
        else:
            answers, others = ["<pad>"], []
            for t in tasks:
                answers += [f"{t.name}:{w}" for w in t.vocab[1 : t.n_output]]
                others += [f"{t.name}:{w}" for w in t.vocab[t.n_output :]]
            self.vocab, self.n_output = answers + others, len(answers)
            answer_len = max(t.answer_len for t in tasks)
            self.tasks = []
            next_answer, next_other = 1, self.n_output
            for t in tasks:
                remap = np.zeros(len(t.vocab), dtype=np.int32)
                n_answers, n_others = t.n_output - 1, len(t.vocab) - t.n_output
                remap[1 : t.n_output] = next_answer + np.arange(n_answers)
                remap[t.n_output :] = next_other + np.arange(n_others)
                next_answer, next_other = next_answer + n_answers, next_other + n_others
                clone = t.rebased(remap, answer_len)
                clone.vocab, clone.n_output = self.vocab, self.n_output
                self.tasks.append(clone)
        self.answer_len = self.tasks[0].answer_len
        self.question_len = max(t.question_len for t in self.tasks)

    def summary(self):
        return {
            "vocabulary": len(self.vocab),
            "answer_tokens": self.n_output,
            "answer_len": self.answer_len,
            "question_len": self.question_len,
            "tasks": [t.summary() for t in self.tasks],
        }


PRESETS = {
    # Small enough for tests and a quick look.
    "smoke": {"maths": {"max_operand": 9}, "facts": {"n_people": 60}, "clutrr": {"max_stories": 300}},
    "pilot": {"maths": {"max_operand": 39}, "facts": {"n_people": 400}, "clutrr": {"max_stories": 2000}},
    "full": {"maths": {"max_operand": 99}, "facts": {"n_people": 1500}, "clutrr": {}},
}


def build_suite(names, preset="full", eval_size=2000, seed=0, overrides=None):
    """The tasks named, built at a preset size. ``overrides`` maps a task name
    to keyword arguments that replace the preset's."""
    from .clutrr import clutrr_task

    makers = {"maths": maths_task, "facts": facts_task, "clutrr": clutrr_task}
    tasks = []
    for name in names:
        kwargs = {**PRESETS[preset][name], **(overrides or {}).get(name, {})}
        tasks.append(makers[name](eval_size=eval_size, seed=seed, **kwargs))
    return Suite(tasks)
