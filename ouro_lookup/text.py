"""Ordinary text for the take-over experiment: WikiText-103.

    python -m ouro_lookup.text            # download, check and tokenise

Downloads the first half of raw WikiText-103 and its validation part from
Hugging Face (about 160 MB of Wikipedia articles, CC BY-SA), checks them
against known checksums, undoes the spaced-out punctuation, and writes the
text as Ouro token ids:

    data/wikitext-103/train.npy   the first part of the training articles
    data/wikitext-103/valid.npy   the validation articles: never trained on

Nothing here is committed to the repository. ``--text-file`` uses any plain
text file of your own instead of the download; its last tenth becomes the
held-out part.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from .download import DEFAULT_DIR as DEFAULT_WEIGHTS
from .download import open_url, sha256

REPO = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SOURCE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/wikitext-103-raw-v1"
FILES = {
    "train-00000-of-00002.parquet": "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212",
    "validation-00000-of-00001.parquet": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
}
TRAIN, VALID = FILES
DEFAULT_DIR = "data/wikitext-103"
TRAIN_CHARS = 120_000_000  # of 540 million: 25.7 million tokens
MIN_CHARS = 200  # shorter paragraphs are mostly captions and stubs

_QUOTED = re.compile(r'" (.*?) "')
_RULES = [
    (re.compile(r" @([-,.])@ "), r"\1"),  # 1 @,@ 000 -> 1,000 ; well @-@ known -> well-known
    (re.compile(r" ([,.;:!?%)\]])"), r"\1"),
    (re.compile(r"([(\[$£]) "), r"\1"),
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),
    (re.compile(r"(\w) ' (s)\b"), r"\1'\2"),
]


def clean(line):
    """A WikiText line as ordinary text: its punctuation is spaced out."""
    line = _QUOTED.sub(r'"\1"', line.strip())
    for pattern, replacement in _RULES:
        line = pattern.sub(replacement, line)
    return line


def paragraphs(lines, max_chars=None):
    """Cleaned paragraphs from WikiText lines, without headings and stubs."""
    used = 0
    for line in lines:
        used += len(line)
        if max_chars and used > max_chars:
            return
        text = line.strip()
        if text.startswith("=") or len(text) < MIN_CHARS:
            continue
        yield clean(text)


def fetch(name, directory=DEFAULT_DIR):
    """Download one file if it is missing and verify it. Returns its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if not target.exists():
        part = directory / (name + ".part")
        print(f"downloading {name}")
        with open_url(urllib.request.Request(f"{SOURCE}/{name}")) as response, open(part, "wb") as f:
            while block := response.read(1 << 22):
                f.write(block)
        found = sha256(part)
        if found != FILES[name]:
            part.unlink()
            raise SystemExit(f"{name}: checksum {found} does not match the expected {FILES[name]}")
        part.rename(target)
    return target


def read_lines(path):
    """The lines of text in one of the downloaded files."""
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=["text"]).column("text").to_pylist()


def tokenise(texts, weights):
    """Paragraphs as one stream of token ids, each followed by a line break."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(weights) / "tokenizer.json"))
    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("token ids are stored in 16 bits")
    out = []
    texts = list(texts)
    for at in range(0, len(texts), 4096):
        batch = [text + "\n" for text in texts[at : at + 4096]]
        out += [np.array(e.ids, dtype=np.uint16) for e in tokenizer.encode_batch(batch)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint16)


def prepare(weights=DEFAULT_WEIGHTS, directory=DEFAULT_DIR, text_file=None, train_chars=TRAIN_CHARS):
    """Write train.npy and valid.npy unless they are already there."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "meta.json"
    source = str(text_file) if text_file else SOURCE
    wanted = {"source": source, "weights": str(weights), "train_chars": train_chars}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in wanted.items()):
            return meta
    if text_file:
        lines = Path(text_file).read_text(encoding="utf-8").splitlines()
        cut = len(lines) - max(1, len(lines) // 10)
        train = [line.strip() for line in lines[:cut] if line.strip()]
        valid = [line.strip() for line in lines[cut:] if line.strip()]
    else:
        train = list(paragraphs(read_lines(fetch(TRAIN, directory)), train_chars))
        valid = list(paragraphs(read_lines(fetch(VALID, directory))))
    print(f"tokenising {len(train):,} training and {len(valid):,} held-out paragraphs")
    arrays = {"train": tokenise(train, weights), "valid": tokenise(valid, weights)}
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array)
    meta = {**wanted, **{f"{name}_tokens": int(len(a)) for name, a in arrays.items()},
            "example": train[0][:200] if train else ""}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


class Text:
    """Windows of ordinary text, each begun with the start token."""

    def __init__(self, train, valid, start, length):
        if len(train) < length or len(valid) < length:
            raise ValueError("not enough text for one window")
        self.train, self.valid = train, valid
        self.start, self.length = start, length

    @classmethod
    def load(cls, directory, start, length):
        directory = Path(directory)
        if not (directory / "train.npy").exists():
            raise SystemExit(
                f"no text in {directory}; run: python -m ouro_lookup.text --out {directory}"
            )
        train = np.load(directory / "train.npy", mmap_mode="r")
        valid = np.load(directory / "valid.npy", mmap_mode="r")
        return cls(train, valid, start, length)

    def _windows(self, stream, at):
        body = np.stack([stream[a : a + self.length - 1] for a in at]).astype(np.int32)
        first = np.full((len(at), 1), self.start, dtype=np.int32)
        return np.concatenate([first, body], axis=1)

    def batch(self, rng, n):
        """(n, length) random training windows."""
        return self._windows(self.train, rng.integers(0, len(self.train) - self.length + 2, n))

    def held_out(self, n):
        """Up to n windows of the held-out text, one after another, the same
        ones every time."""
        step = self.length - 1
        at = np.arange(0, len(self.valid) - step + 1, step)[:n]
        return self._windows(self.valid, at)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="directory with Ouro's tokenizer")
    p.add_argument("--out", default=DEFAULT_DIR)
    p.add_argument("--text-file", help="a plain text file to use instead of the download")
    p.add_argument("--train-chars", type=int, default=TRAIN_CHARS)
    args = p.parse_args(argv)
    meta = prepare(args.weights, args.out, args.text_file, args.train_chars)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
