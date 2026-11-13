"""CLUTRR: short family stories, and how two people in them are related.

    python -m phrase_scratchpad.clutrr      # download the data once

From "Ashley's daughter, Lillian, asked her mom to read her a story.
Nicholas's sister Lillian asked him for some help planting her garden." the
question "Ashley ? Nicholas" has the answer "son". Nothing has to be stored
to answer it: the two relations are in the story and have to be read and
combined (a daughter's brother is a son).

* A two-step question asks about the two ends of the story. Its intermediate
  answer is the first stated link: the relation and the person in the
  middle, here "daughter Lillian".
* A one-step question asks about two people the story links directly.

Stories are natural sentences written by crowd workers. Training uses the
dataset's training stories and testing its validation stories, which the
model has never seen. Only two-relation stories are used.

The data is CC BY-NC 4.0 (https://github.com/facebookresearch/clutrr), so it
is downloaded to ``data/clutrr`` and not kept in this repository.
"""

import ast
import csv
import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np

from .tasks import PAD, Items, join, widen

SOURCE = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/"
FILES = {
    "gen_train234_test2to10/train.csv": "a6123f6e305cc528f0cd9fe441014fc244cc4b4a1e13a56c381624a7ca26c392",
    "gen_train234_test2to10/validation.csv": "414390d3322b9fd27ca00d99e442df63b497c9c05918a09f72843c926556cfaa",
    "rob_train_clean_23_test_all_23/train.csv": "7654d8b27ccabac4bd176f60749874b40b16c64585591e15a1e0828dd8fa8a5f",
    "rob_train_clean_23_test_all_23/validation.csv": "95c85d4e6f5051052d854f8ec7229f1d697d3e2bb7708a364bdb2955833736bf",
}
DATA_DIR = Path("data/clutrr")
TOKEN = re.compile(r"\[[^\]]+\]|[A-Za-z]+(?:-[A-Za-z]+)*|'s|[^\sA-Za-z]")
ASK = "?"


def tokenize(story):
    """Words and punctuation; names keep their brackets, other words are
    lower-cased."""
    return [t if t.startswith("[") else t.lower() for t in TOKEN.findall(story)]


def download(data_dir=DATA_DIR):
    """Fetch the four source files if they are missing, and check them."""
    paths = []
    for name, expected in FILES.items():
        path = Path(data_dir) / name.replace("/", "_")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {SOURCE + name}")
            request = urllib.request.Request(SOURCE + name, headers={"User-Agent": "clutrr-fetch/0.1"})
            with urllib.request.urlopen(request) as response:
                path.write_bytes(response.read())
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found != expected:
            raise ValueError(
                f"{path} is not the file this code was written against "
                f"(sha256 {found}); delete it to download again"
            )
        paths.append(path)
    return paths


def read_stories(data_dir=DATA_DIR):
    """Two-relation stories as dicts, from the downloaded files."""
    missing = [n for n in FILES if not (Path(data_dir) / n.replace("/", "_")).exists()]
    if missing:
        raise FileNotFoundError(
            f"CLUTRR is not in {data_dir}. Run: python -m phrase_scratchpad.clutrr"
        )
    stories = []
    for path in download(data_dir):
        split = "train" if path.name.endswith("train.csv") else "test"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["task_name"] != "task_1.2":
                    continue
                stories.append(
                    {
                        "split": split,
                        "story": row["clean_story"],
                        "people": [p.split(":")[0] for p in row["genders"].split(",")],
                        "relations": ast.literal_eval(row["edge_types"]),
                        "answer": row["target_text"],
                    }
                )
    return stories


class StoryTask:
    """Questions about stories. Presents the same things as tasks.Task."""

    name = "clutrr"

    def __init__(self, stories, max_story=44, max_stories=None, eval_size=2000, seed=0):
        rng = np.random.default_rng(seed)
        for story in stories:
            story["tokens"] = tokenize(story["story"])
        train_text = {s["story"] for s in stories if s["split"] == "train"}
        kept, seen = [], set()
        for s in stories:
            repeat = s["story"] in seen or (s["split"] == "test" and s["story"] in train_text)
            if len(s["tokens"]) <= max_story and not repeat:
                kept.append(s)
                seen.add(s["story"])
        train = [s for s in kept if s["split"] == "train"]
        test = [s for s in kept if s["split"] == "test"]
        if max_stories and max_stories < len(train):
            train = [train[i] for i in sorted(rng.permutation(len(train))[:max_stories])]
            test = test[: max(1, max_stories // 4)]

        # Vocabulary: padding, then everything that can be an answer or an
        # intermediate answer (relation words and names), then other words.
        used = train + test
        relations = sorted({r for s in used for r in s["relations"] + [s["answer"]]})
        names = sorted({f"[{p}]" for s in used for p in s["people"]})
        answers = relations + names
        others = sorted({t for s in used for t in s["tokens"]} - set(answers) | {ASK})
        self.vocab = ["<pad>"] + answers + others
        self.n_output = 1 + len(answers)
        self.answer_len = 2  # an intermediate answer is a relation and a name
        self.question_len = max_story + 3  # the story, "?", and two names
        self.n_stories = {"train": len(train), "test": len(test)}
        token = {w: i for i, w in enumerate(self.vocab)}

        def build(part):
            """(one-step items, two-step items) for a list of stories."""
            one_q, one_a, two_q, two_a, two_b = [], [], [], [], []
            for s in part:
                text = [token[t] for t in s["tokens"]] + [token[ASK]]
                a, b, c = (token[f"[{p}]"] for p in s["people"])
                first, second = (token[r] for r in s["relations"])
                one_q += [text + [a, b], text + [b, c]]
                one_a += [[first, PAD], [second, PAD]]
                two_q.append(text + [a, c])
                two_a.append([token[s["answer"]], PAD])
                two_b.append([first, b])  # "daughter Lillian"

            def pad(rows):
                out = np.zeros((len(rows), self.question_len), dtype=np.int32)
                for i, row in enumerate(rows):
                    out[i, : len(row)] = row
                return out

            blank = np.zeros((len(one_q), 2), dtype=np.int32)
            one = Items(pad(one_q), np.array(one_a, np.int32), blank, np.zeros(len(one_q), bool))
            two = Items(
                pad(two_q), np.array(two_a, np.int32), np.array(two_b, np.int32),
                np.ones(len(two_q), bool),
            )
            return one, two

        self.train_one, self.train_two = build(train)
        test_one, test_two = build(test)

        def sample(items):
            idx = np.sort(rng.permutation(len(items))[:eval_size])
            return _take(items, idx)

        self.eval_sets = {
            "one_hop": sample(test_one),
            "chain_train": sample(self.train_two),
            "chain_unseen": sample(test_two),
        }

    def sample(self, rng, n_one, n_chain):
        one = _take(self.train_one, rng.integers(0, len(self.train_one), n_one))
        two = _take(self.train_two, rng.integers(0, len(self.train_two), n_chain))
        return join(one, two)

    def majority_baseline(self, items):
        """Accuracy of always giving the commonest training answer."""
        source = self.train_two if items.chain.all() else self.train_one
        answers, counts = np.unique(source.answer, axis=0, return_counts=True)
        return float((items.answer == answers[counts.argmax()]).all(axis=1).mean())

    def rebased(self, remap, answer_len):
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)

        def move(items):
            return Items(
                remap[items.question],
                widen(remap[items.answer], answer_len),
                widen(remap[items.bridge], answer_len),
                items.chain,
            )

        clone.train_one, clone.train_two = move(self.train_one), move(self.train_two)
        clone.eval_sets = {name: move(items) for name, items in self.eval_sets.items()}
        clone.answer_len = answer_len
        return clone

    def decode(self, tokens):
        return " ".join(self.vocab[t] for t in tokens if t != PAD)

    def summary(self):
        return {
            "task": self.name,
            "vocabulary": len(self.vocab),
            "answer_tokens": self.n_output,
            "answer_len": self.answer_len,
            "question_len": self.question_len,
            "training_stories": self.n_stories["train"],
            "test_stories": self.n_stories["test"],
            "one_step_training_questions": len(self.train_one),
            "two_step_training_questions": len(self.train_two),
        }


def _take(items, idx):
    return Items(items.question[idx], items.answer[idx], items.bridge[idx], items.chain[idx])


def clutrr_task(data_dir=DATA_DIR, **kwargs):
    return StoryTask(read_stories(data_dir), **kwargs)


if __name__ == "__main__":
    download()
    task = clutrr_task()
    print(task.summary())
