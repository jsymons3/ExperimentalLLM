"""Prepare byte corpora; preserve source splits and document boundaries."""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def articles(lines):
    result, current = [], []
    for line in lines:
        # Top-level WikiText article titles have exactly one '=' on each side.
        if re.match(r"^\s*= [^=].*[^=] =\s*$", line) and current:
            result.append("".join(current))
            current = []
        current.append(line + "\n")
    if current:
        result.append("".join(current))
    return [x for x in result if x.strip()]


def prepare(splits, out, source):
    out = Path(out)
    if (out / "manifest.json").exists():
        raise ValueError(f"{out} already contains a corpus; use a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    seen, meta = set(), {}
    for split, docs in splits.items():
        clean, hashes, local_seen = [], [], set()
        duplicates = 0
        for doc in docs:
            if not doc.strip():
                continue
            key = digest(doc.strip())
            if key in seen:
                raise ValueError("duplicate document across splits")
            if key in local_seen:
                duplicates += 1
                continue
            local_seen.add(key)
            hashes.append(key)
            clean.append(
                np.concatenate(
                    [
                        np.frombuffer(doc.encode("utf-8"), dtype=np.uint8).astype(
                            np.uint16
                        ),
                        [256],
                    ]
                ).astype(np.uint16)
            )
        if not clean:
            raise ValueError(f"empty {split} split")
        offsets = np.concatenate([[0], np.cumsum([len(x) for x in clean])])
        tokens = np.concatenate(clean)
        np.savez(out / f"{split}.npz", tokens=tokens, offsets=offsets)
        meta[split] = {
            "documents": len(clean),
            "tokens": len(tokens),
            "document_hashes": hashes,
            "duplicates_removed_within_split": duplicates,
            "file_sha256": hashlib.sha256(
                (out / f"{split}.npz").read_bytes()
            ).hexdigest(),
        }
        seen.update(local_seen)
    manifest = {"encoding": "UTF-8 bytes, EOS=256", "source": source, "splits": meta}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class Corpus:
    def __init__(self, root, split, context):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        path = root / f"{split}.npz"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["splits"][split]["file_sha256"]
        ):
            raise ValueError("corpus content differs from its manifest")
        with np.load(path) as data:
            self.tokens = data["tokens"]
            self.offsets = data["offsets"]
        self.context = context
        counts = np.maximum(np.diff(self.offsets) - context, 0)
        self.cumulative = np.cumsum(counts)
        if not self.cumulative[-1]:
            raise ValueError("no document is long enough for this context")

    def sample(self, rng, batch):
        # Uniform over valid window starts. No window crosses a document boundary.
        choice = rng.integers(0, self.cumulative[-1], size=batch)
        docs = np.searchsorted(self.cumulative, choice, side="right")
        previous = np.where(docs > 0, self.cumulative[np.maximum(docs - 1, 0)], 0)
        starts = self.offsets[docs] + choice - previous
        rows = self.tokens[starts[:, None] + np.arange(self.context + 1)].astype(
            np.int32
        )
        return rows[:, :-1], rows[:, 1:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--wikitext-2", action="store_true")
    p.add_argument("--train", nargs="+", type=Path)
    p.add_argument("--valid", nargs="+", type=Path)
    p.add_argument("--test", nargs="+", type=Path)
    a = p.parse_args()
    if a.wikitext_2:
        if a.train or a.valid or a.test:
            p.error("choose WikiText or local files, not both")
        import pyarrow.parquet as pq  # optional download dependency

        # Resolve the remote revision once, then pin every downloaded split.
        with urllib.request.urlopen(
            "https://huggingface.co/api/datasets/Salesforce/wikitext", timeout=60
        ) as r:
            revision = json.load(r)["sha"]
        splits = {}
        cache = Path(a.out) / "_source"
        cache.mkdir(parents=True, exist_ok=True)
        for remote, local in [
            ("train", "train"),
            ("validation", "valid"),
            ("test", "test"),
        ]:
            url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/wikitext-2-raw-v1/{remote}-00000-of-00001.parquet"
            path = cache / f"{remote}.parquet"
            with urllib.request.urlopen(url, timeout=120) as r, path.open("wb") as f:
                f.write(r.read())
            splits[local] = articles(
                pq.read_table(path, columns=["text"])["text"].to_pylist()
            )
        source = {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "revision": revision,
            "url": "https://huggingface.co/datasets/Salesforce/wikitext",
            "license": ["CC-BY-SA-3.0", "GFDL"],
        }
    else:
        if not a.train or not a.valid:
            p.error("--train and --valid files are required")
        splits = {
            name: [f.read_text() for f in files]
            for name, files in [
                ("train", a.train),
                ("valid", a.valid),
                ("test", a.test),
            ]
            if files
        }
        source = {
            "local_files": {
                name: [str(f.resolve()) for f in files]
                for name, files in [
                    ("train", a.train),
                    ("valid", a.valid),
                    ("test", a.test),
                ]
                if files
            }
        }
    manifest = prepare(splits, a.out, source)
    print(
        json.dumps(
            {
                k: {q: v[q] for q in ("documents", "tokens")}
                for k, v in manifest["splits"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
