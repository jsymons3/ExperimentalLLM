"""Summarise finished runs as a markdown report.

    python -m recurrent_lookup.report runs/overnight
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .configs import VARIANTS

SPLIT_COLUMNS = [
    ("one_hop", "1-hop"),
    ("two_hop_train", "2-hop train"),
    ("two_hop_id", "2-hop unseen"),
    ("two_hop_held_first", "held-out 1st"),
    ("two_hop_held_second", "held-out 2nd"),
    ("two_hop_held_both", "held-out both"),
]
MAIN_SPLIT = "two_hop_held_second"


def load_runs(root):
    """{variant: [final.json dicts]} for every finished run under root."""
    runs = defaultdict(list)
    for path in sorted(Path(root).glob("*/final.json")):
        final = json.loads(path.read_text())
        runs[final["variant"]].append(final)
    return {v: runs[v] for v in VARIANTS if v in runs}


def spread(values):
    """Mean, with the range when there is more than one seed."""
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f} ({min(values):.3f}-{max(values):.3f})"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def overlap_pair(final, split, kind):
    """Overlap of two-hop pass 2 with the direct question's pass 1."""
    matrix = ((final.get("overlap") or {}).get(split) or {}).get(kind)
    if not matrix or len(matrix) < 2:
        return None
    return matrix[1][0]


def overlap_best(final, split):
    """Largest same-entity overlap over all pass pairs, with its control."""
    entry = (final.get("overlap") or {}).get(split)
    if not entry:
        return None, None
    best = (None, None)
    for same_row, control_row in zip(entry["same"], entry["control"]):
        for same, control in zip(same_row, control_row):
            if same is not None and (best[0] is None or same > best[0]):
                best = (same, control)
    return best


def build_report(root):
    runs = load_runs(root)
    if not runs:
        return f"No finished runs (final.json) under {root}."
    out = [f"# Recurrent lookup results: {root}", ""]

    def acc(variant, split):
        return mean([f["accuracy"].get(split) for f in runs.get(variant, [])])

    rows = []
    for variant, finals in runs.items():
        first = finals[0]
        usage = [f["table"]["used_frac"] if f["table"] else None for f in finals]
        rows.append(
            [
                variant,
                len(finals),
                f"{first['params']['total']:,}",
                first["steps"],
                f"{mean([f['sec_per_step'] for f in finals]) * 1000:.1f}",
                *[spread([f["accuracy"].get(s) for f in finals]) for s, _ in SPLIT_COLUMNS],
                spread(usage),
            ]
        )
    out += [
        "## Accuracy at the trained number of passes",
        "",
        table(
            ["run", "seeds", "params", "steps", "ms/step"]
            + [label for _, label in SPLIT_COLUMNS]
            + ["table used"],
            rows,
        ),
        "",
        "Every fact is trained as a one-hop question. \"held-out\" chains need a "
        "fact that never appeared inside a two-hop training chain, in the first "
        "hop, the second hop, or both. Held-out 2nd and both are the main "
        "result: chance is 1 / number of entities. Values are means over seeds, "
        "with the range in brackets.",
        "",
    ]

    passes = sorted({int(p) for fs in runs.values() for f in fs for p in f["accuracy_by_passes"]})
    rows = [
        [variant]
        + [
            spread([f["accuracy_by_passes"].get(str(p), {}).get(MAIN_SPLIT) for f in finals])
            for p in passes
        ]
        for variant, finals in runs.items()
    ]
    out += [
        "## Held-out 2nd-hop accuracy by number of passes at test time",
        "",
        table(["run"] + [f"{p} pass" + "es" * (p != 1) for p in passes], rows),
        "",
    ]

    rows = []
    for variant, finals in runs.items():
        if not finals[0].get("overlap"):
            continue
        for split in ("two_hop_train", MAIN_SPLIT):
            bests = [overlap_best(f, split) for f in finals]
            rows.append(
                [
                    variant,
                    split.replace("two_hop_", ""),
                    spread([overlap_pair(f, split, "same") for f in finals]),
                    spread([overlap_pair(f, split, "control") for f in finals]),
                    spread([b[0] for b in bests]),
                    spread([b[1] for b in bests]),
                ]
            )
    if rows:
        out += [
            "## Retrieval overlap",
            "",
            table(
                ["run", "chains", "pass 2 vs direct", "control", "best pair", "its control"],
                rows,
            ),
            "",
            "For a chain a -> b -> c: overlap (Jaccard) between the table entries "
            "retrieved at the answer position of the two-hop question and those "
            "retrieved by the direct question about b. \"pass 2 vs direct\" compares "
            "pass 2 of the two-hop question with pass 1 of the direct question; "
            "\"best pair\" is the largest overlap over all pass pairs. The control "
            "asks the direct question about a different entity with the same "
            "relation. Reuse of stored facts shows up as overlap well above control.",
            "",
        ]

    out += ["## Readout", ""]

    def line(text, *values):
        if all(v is not None for v in values):
            out.append("- " + text.format(*values))

    c = acc("C", MAIN_SPLIT)
    line("Held-out 2nd hop, lookup every pass (C): {:.3f}.", c)
    line("Same block without the table (A): {:.3f}.", acc("A", MAIN_SPLIT))
    line("Dense block with C's parameter count (B): {:.3f}.", acc("B", MAIN_SPLIT))
    line("Lookup on the first pass only (D): {:.3f}.", acc("D", MAIN_SPLIT))
    line(
        "Table size, small / C / large: held-out 2nd hop {:.3f} / {:.3f} / {:.3f}, "
        "1-hop recall {:.3f} / {:.3f} / {:.3f}.",
        acc("E_small", MAIN_SPLIT),
        c,
        acc("E_large", MAIN_SPLIT),
        acc("E_small", "one_hop"),
        acc("C", "one_hop"),
        acc("E_large", "one_hop"),
    )
    line(
        "C on trained chains {:.3f} versus held-out 2nd hop {:.3f}. High on "
        "trained chains and near chance on held-out means it built a case table.",
        acc("C", "two_hop_train"),
        c,
    )
    out += [
        "",
        "What would count as success: C clearly above A and B on the held-out "
        "columns, D well below C, and retrieval overlap well above its control. "
        "One seed per run for B, D and E, so treat small gaps as noise.",
    ]
    return "\n".join(out) + "\n"


def write_report(root):
    text = build_report(root)
    (Path(root) / "report.md").write_text(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", help="directory holding the run directories")
    args = p.parse_args(argv)
    print(write_report(args.root))


if __name__ == "__main__":
    main()
