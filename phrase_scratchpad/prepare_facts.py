"""Build the facts file from 2WikiMultihopQA's annotated evidence triples.

    python -m phrase_scratchpad.prepare_facts

Only needed to regenerate ``facts/people_facts.tsv``, which is committed. It
downloads the dataset's train and dev splits (about 375 MB) from a pinned
revision and needs ``pyarrow``.

Every question in that dataset lists the (subject, relation, object) triples
needed to answer it. This script pools those triples and keeps a
person-centred part of the graph: facts that lead to a person (a film's
director, someone's father) and facts about that person (date of birth,
citizenship, spouse). A two-hop chain goes through the person.

Simplifications, which make this a controlled fact table and not a faithful
copy of Wikipedia:

* A trailing parenthetical such as "(director)" or "(1932 film)" is removed
  from names, so different entities that share a name are merged.
* Each (subject, relation) keeps one object, the most frequently listed one.
* Names longer than MAX_WORDS words are dropped.
"""

import argparse
import collections
import hashlib
import json
import re
import urllib.request
from pathlib import Path

DATASET = "xanhho/2WikiMultihopQA"
REVISION = "612bc5039a457880d9e7d84c3b0a4cf154b70e4f"
SPLITS = ("train", "dev")
MAX_WORDS = 4
MIN_IN, MIN_OUT = 3, 3

# Relations whose object is a person: the first hop of a chain.
TO_PERSON = (
    "director", "performer", "composer", "father", "mother", "spouse", "producer",
    "child", "sibling", "founded by", "creator", "editor", "presenter",
)
# Relations about a person: the second hop.
ABOUT_PERSON = (
    "country of citizenship", "date of birth", "date of death", "place of birth",
    "place of death", "educated at", "cause of death", "place of burial",
    "employer", "award received", "father", "mother", "spouse", "child", "sibling",
)
HERE = Path(__file__).parent


def canonical(name):
    name = re.sub(r"\s*\([^()]*\)\s*$", "", name)
    return re.sub(r"\s+", " ", name).strip()


def download(cache):
    paths = []
    for split in SPLITS:
        path = cache / f"{split}.parquet"
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{split}.parquet"
            print(f"downloading {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "facts-builder/0.1"})
            with urllib.request.urlopen(request) as response, open(path, "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
        paths.append(path)
    return paths


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_triples(paths):
    import pyarrow.parquet as pq

    counts = collections.Counter()
    for path in paths:
        for evidences in pq.read_table(path, columns=["evidences"]).column(0).to_pylist():
            for subject, relation, obj in json.loads(evidences):
                counts[(canonical(subject), relation, canonical(obj))] += 1
    return counts


def select(counts):
    """Return the kept facts as sorted (subject, relation, object) rows."""
    objects = collections.defaultdict(collections.Counter)
    for (subject, relation, obj), n in counts.items():
        objects[(subject, relation)][obj] += n
    # One object per (subject, relation): the most listed, ties by name.
    fact = {
        key: min(c.items(), key=lambda item: (-item[1], item[0]))[0]
        for key, c in objects.items()
    }

    def short(name):
        return 1 <= len(name.split()) <= MAX_WORDS

    incoming = collections.defaultdict(list)
    outgoing = collections.defaultdict(list)
    for (subject, relation), obj in fact.items():
        if not (short(subject) and short(obj)) or subject == obj:
            continue
        if relation in TO_PERSON:
            incoming[obj].append((subject, relation, obj))
        if relation in ABOUT_PERSON:
            outgoing[subject].append((subject, relation, obj))
    kept = set()
    for person, rows in incoming.items():
        if len(rows) >= MIN_IN and len(outgoing.get(person, ())) >= MIN_OUT:
            kept.update(rows)
            kept.update(outgoing[person])
    return sorted(kept)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", type=Path, default=Path("data/2wikimultihopqa"))
    p.add_argument("--out", type=Path, default=HERE / "facts" / "people_facts.tsv")
    args = p.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    paths = download(args.cache)
    rows = select(read_triples(paths))
    if any("\t" in field or "\n" in field for row in rows for field in row):
        raise ValueError("a name contains a tab or newline")
    text = "".join("\t".join(row) + "\n" for row in rows)
    args.out.write_text(text, encoding="utf-8")
    meta = {
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "source_license": "Apache-2.0",
        "source_sha256": {path.name: sha256(path) for path in paths},
        "max_words_per_name": MAX_WORDS,
        "min_first_hop_facts_per_person": MIN_IN,
        "min_second_hop_facts_per_person": MIN_OUT,
        "first_hop_relations": list(TO_PERSON),
        "second_hop_relations": list(ABOUT_PERSON),
        "facts": len(rows),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(rows)} facts to {args.out}")


if __name__ == "__main__":
    main()
