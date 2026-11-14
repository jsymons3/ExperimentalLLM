"""Made-up facts about people, and two-step questions that chain them.

Each person is known by one first name. For every person and every relation
(mentor, manager, ...) there is one fact, drawn at random:

    The mentor of Daniel is Alice.

A two-step question chains two facts:

    The doctor of the mentor of Daniel is ...     (the doctor of Alice)

Its intermediate answer (Alice) can be written into a fresh one-step question
("The doctor of Alice is"), which is how the "note" runs answer it.

The names are real first names, but the facts are random, so no pretrained
model can know them. Every name and relation is a single token of the model's
tokenizer, so an answer is one token and every question of a kind has the
same length.

The split is the one used throughout this repository. Every fact is taught as
a one-step question. A share of the facts is kept out of every two-step
training question, and two-step test questions are grouped by which of their
two facts were kept out.
"""

from dataclasses import dataclass

import numpy as np

# First names that are single tokens in Ouro's tokenizer when they follow a
# space. Checked again when a world is built, against the tokenizer in use.
NAMES = """
Aaron Abel Abraham Adam Adrian Alan Albert Alex Alexander Alfred Allen Andrew
Andy Anthony Antonio Arnold Arthur Barry Ben Benjamin Bernard Bob Bobby Brad
Bradley Brian Bruce Bryan Calvin Carl Carlos Cecil Charles Charlie Chris
Christopher Clarence Clark Claude Clifford Colin Craig Curtis Dale Dan Daniel
Dave David Dean Dennis Diego Don Donald Douglas Duncan Earl Edgar Edmund Edward
Edwin Eli Elijah Elliott Eric Ernest Ethan Eugene Felix Fernando Francis
Francisco Franklin Fred Frederick Gabriel Gary Gene Geoffrey George Gerald
Gilbert Glenn Gordon Graham Greg Gregory Harold Harry Harvey Henry Herbert
Herman Howard Hugh Hugo Ian Isaac Ivan Jack Jacob Jake James Jamie Jason Jay
Jeff Jeffrey Jeremy Jerome Jerry Jesse Jim Jimmy Joe Joel John Johnny Jon
Jonathan Jorge Jose Joseph Josh Joshua Juan Julian Justin Karl Keith Ken
Kenneth Kevin Larry Lawrence Lee Leo Leon Leonard Lewis Lloyd Logan Louis Lucas
Luis Luke Malcolm Manuel Marcus Mario Martin Matt Matthew Maurice Max Michael
Miguel Mike Miles Mitchell Morgan Nathan Neil Nelson Nicholas Nick Noah Norman
Oliver Oscar Owen Pablo Patrick Paul Pedro Perry Peter Phil Philip Phillip
Pierre Ralph Raymond Rex Richard Rick Robert Robin Roger Roland Ron Ronald Ross
Roy Russell Ryan Sam Samuel Scott Sean Sebastian Seth Sidney Simon Stanley
Stephen Steve Steven Stuart Ted Terry Theodore Thomas Tim Timothy Todd Tom
Tommy Tony Troy Victor Vincent Walter Warren Wayne Wesley William Zach Abigail
Ada Agnes Alice Alison Amanda Amelia Amy Andrea Angela Ann Anna Anne Annie
Barbara Bella Beth Betty Carol Caroline Catherine Christina Christine Claire
Clara Dana Deborah Diana Diane Dorothy Edith Eleanor Elizabeth Ella Ellen Emily
Emma Erin Esther Eva Frances Hannah Harriet Heather Helen Holly Isabel Isabella
Jackie Jane Janet Jean Jennifer Jenny Jessica Jill Joan Joyce Judith Judy Julia
Julie Karen Kate Katherine Kathleen Katie Kelly Kim Laura Lauren Leslie Linda
Lisa Lorraine Louise Lucy Lynn Margaret Maria Marie Marion Martha Mary Melissa
Michelle Molly Monica Nancy Naomi Nicole Nina Olivia Patricia Penny Rachel
Rebecca Rosa Ruth Sally Samantha Sandra Sandy Sara Sarah Sharon Shirley Sophia
Sophie Stephanie Susan Tara Teresa Vera Wendy Ahmed Ali Hassan Omar Ibrahim Raj
Wei Ming Chen Boris Vladimir Hans Fritz Otto Heinrich Lars Erik Jacques Henri
Giovanni Marco Carlo Sofia Andre Anton Bruno Ernst Franz Gustav Johann Josef
Kurt Lorenzo Ludwig Mikhail Rudolf Salvador Stefan Wolfgang Mei Kai
""".split()

RELATIONS = ["mentor", "manager", "doctor", "lawyer", "landlord", "coach", "editor", "dentist"]

# Ordinary sentences, to check that the additions leave everyday text alone.
ORDINARY = [
    "The train to the coast leaves every morning at nine and takes about two hours.",
    "She put the kettle on, found two clean cups and waited for the water to boil.",
    "Most birds build their nests in spring, when food is easy to find.",
    "He had never been good at chess, but he enjoyed playing with his grandfather.",
    "The museum is closed on Mondays and opens at ten on every other day.",
    "After the storm passed, the streets were covered in leaves and broken branches.",
    "Water freezes at zero degrees Celsius and boils at one hundred.",
    "The children walked to school together and talked about the football match.",
    "If you add two and three, the answer is five.",
    "The capital of France is Paris, and the capital of Italy is Rome.",
    "She wrote a long letter to her sister and posted it the next day.",
    "The recipe needs flour, butter, two eggs and a little sugar.",
    "Many people learn to swim when they are young.",
    "The library on the corner lends books, films and music for free.",
    "It rained all night, so the river was high in the morning.",
    "He fixed the bicycle, pumped up the tyres and rode into town.",
]

PRESETS = {
    # For tests and the smallest checks.
    "smoke": dict(n_people=24, n_relations=3),
    # A quick check that the real model can learn from the table at all.
    "check": dict(n_people=16, n_relations=4),
    "pilot": dict(n_people=96, n_relations=4),
    # 2,160 facts and 12,960 possible two-step questions.
    "full": dict(n_people=360, n_relations=6),
}


@dataclass
class Batch:
    """Questions of one kind, all the same length."""

    tokens: np.ndarray  # (n, T): start token, then the question
    answer: np.ndarray  # (n,): the answer's token
    bridge: np.ndarray  # (n,): the intermediate answer's token; 0 for one-step

    def __len__(self):
        return len(self.tokens)


class World:
    """The people, the facts about them, the chains and the train/test split.

    ``encode`` turns text into a list of token ids; ``start`` is the token
    that begins every sequence.
    """

    def __init__(
        self, encode, n_people, n_relations, start=0, held_frac=0.1, train_frac=0.7,
        eval_size=500, seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.encode = encode
        self.start = start

        def single(word):
            ids = encode(" " + word)
            return ids[0] if len(ids) == 1 else None

        usable = [(name, single(name)) for name in NAMES]
        usable = [(name, token) for name, token in usable if token is not None]
        if len({token for _, token in usable}) < n_people:
            raise ValueError(f"only {len(usable)} single-token names; asked for {n_people} people")
        picked = rng.permutation(len(usable))[:n_people]
        self.names = [usable[i][0] for i in picked]
        self.name_token = np.array([usable[i][1] for i in picked], dtype=np.int32)
        if n_relations > len(RELATIONS):
            raise ValueError(f"at most {len(RELATIONS)} relations")
        self.relations = RELATIONS[:n_relations]
        if any(single(r) is None for r in self.relations):
            raise ValueError("a relation word is not a single token in this tokenizer")
        P, R = n_people, n_relations
        self.n_people, self.n_relations = P, R

        # Fact f = person * R + relation; its answer is another person.
        answer = rng.integers(0, P - 1, size=(P, R))
        answer += answer >= np.arange(P)[:, None]  # never the person themselves
        self.fact_person = np.repeat(np.arange(P), R)
        self.fact_relation = np.tile(np.arange(R), P)
        self.fact_answer = answer.reshape(-1)

        # Chain c: a first fact, then a relation of that fact's answer.
        first = np.repeat(np.arange(P * R), R)
        second = self.fact_answer[first] * R + np.tile(np.arange(R), P * R)
        self.chain_first, self.chain_second = first, second

        one = [self.fact_text(f) for f in range(P * R)]
        two = [self.chain_text(c) for c in range(len(first))]
        self.fact_tokens = self._tokenise(one)
        self.chain_tokens = self._tokenise(two)
        self._split(held_frac, train_frac, eval_size, rng)

    # -- text -------------------------------------------------------------

    def fact_text(self, f):
        person, relation = self.names[self.fact_person[f]], self.relations[self.fact_relation[f]]
        return f"The {relation} of {person} is"

    def chain_text(self, c):
        f, g = self.chain_first[c], self.chain_second[c]
        person = self.names[self.fact_person[f]]
        return (
            f"The {self.relations[self.fact_relation[g]]} of "
            f"the {self.relations[self.fact_relation[f]]} of {person} is"
        )

    def _tokenise(self, texts):
        rows = [[self.start] + list(self.encode(text)) for text in texts]
        if len({len(row) for row in rows}) != 1:
            raise ValueError("questions of one kind should all have the same length")
        return np.array(rows, dtype=np.int32)

    # -- split ------------------------------------------------------------

    def _split(self, held_frac, train_frac, eval_size, rng):
        n_facts, n_chains = len(self.fact_answer), len(self.chain_first)
        first, second = self.chain_first, self.chain_second

        # Facts that are kept out of every two-step training question.
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
        self.groups = {
            "chain_unseen": rest[a & b],
            "chain_held_first": rest[~a & b],
            "chain_held_second": rest[a & ~b],
            "chain_held_both": rest[~a & ~b],
        }

        def sample(idx):
            return np.sort(idx[rng.permutation(len(idx))[:eval_size]])

        self.eval_facts = {
            "one_hop": sample(np.flatnonzero(practised)),
            "one_hop_held": sample(np.flatnonzero(~practised)),
        }
        self.eval_chains = {"chain_train": sample(self.train_chains)}
        self.eval_chains.update({name: sample(idx) for name, idx in self.groups.items()})

    # -- questions --------------------------------------------------------

    def facts(self, ids):
        ids = np.asarray(ids)
        return Batch(
            tokens=self.fact_tokens[ids],
            answer=self.name_token[self.fact_answer[ids]],
            bridge=np.zeros(len(ids), dtype=np.int32),
        )

    def chains(self, ids):
        ids = np.asarray(ids)
        first, second = self.chain_first[ids], self.chain_second[ids]
        return Batch(
            tokens=self.chain_tokens[ids],
            answer=self.name_token[self.fact_answer[second]],
            bridge=self.name_token[self.fact_answer[first]],
        )

    @property
    def eval_sets(self):
        """{split name: Batch}, skipping empty splits."""
        sets = {name: self.facts(ids) for name, ids in self.eval_facts.items()}
        sets.update({name: self.chains(ids) for name, ids in self.eval_chains.items()})
        return {name: batch for name, batch in sets.items() if len(batch)}

    def sample(self, rng, n_one, n_chain):
        """Random training questions: any fact, and the training chains."""
        one = self.facts(rng.integers(0, len(self.fact_answer), n_one))
        picked = self.train_chains[rng.integers(0, len(self.train_chains), n_chain)]
        return one, self.chains(picked)

    def second_questions(self, chain_ids, bridge_tokens):
        """One-step questions for each chain's second hop, asked about the
        given person tokens in place of the true intermediate answer."""
        tokens = self.fact_tokens[self.chain_second[np.asarray(chain_ids)]].copy()
        tokens[:, self.person_position] = bridge_tokens
        return tokens

    @property
    def person_position(self):
        """Where the person's token sits in a one-step question."""
        return self.fact_tokens.shape[1] - 2  # "... of <person> is"

    def guess_rate(self, batch):
        """Accuracy of always giving the commonest answer in the world."""
        tokens, counts = np.unique(self.name_token[self.fact_answer], return_counts=True)
        return float((batch.answer == tokens[counts.argmax()]).mean())

    def summary(self):
        out = {
            "people": self.n_people,
            "relations": self.relations,
            "facts": len(self.fact_answer),
            "facts_never_in_a_training_chain": int((~self.practised).sum()),
            "chains": len(self.chain_first),
            "training_chains": len(self.train_chains),
            "one_step_tokens": int(self.fact_tokens.shape[1]),
            "two_step_tokens": int(self.chain_tokens.shape[1]),
            "example_one_step": self.fact_text(0),
            "example_two_step": self.chain_text(0),
        }
        out.update({f"all_{name}": len(idx) for name, idx in self.groups.items()})
        return out


class WordTokenizer:
    """A stand-in tokenizer for tests: one id per space-separated word."""

    def __init__(self):
        self.ids = {"<start>": 0}

    def encode(self, text):
        return [self.ids.setdefault(word, len(self.ids)) for word in text.split()]

    @property
    def vocab_size(self):
        return len(self.ids)
