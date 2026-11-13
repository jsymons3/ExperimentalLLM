"""Summarise finished runs as a markdown report.

    python -m recurrent_lookup.report runs/overnight
    python -m recurrent_lookup.report runs/slots runs/overnight

Several directories are merged into one report. If the same variant and seed
appears twice, the run with more training steps is kept.
"""

import argparse
import json
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
TEST_GROUPS = SPLIT_COLUMNS[2:]
MAIN_SPLIT = "two_hop_held_second"


def load_runs(roots):
    """{variant: [final.json dicts]} for every finished run under the roots."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    best = {}
    for root in roots:
        for path in sorted(Path(root).glob("*/final.json")):
            final = json.loads(path.read_text())
            final["_dir"] = path.parent
            key = (final["variant"], final["seed"])
            if key not in best or final["steps"] > best[key]["steps"]:
                best[key] = final
    runs = {v: [] for v in VARIANTS}
    for (variant, _), final in sorted(best.items(), key=lambda item: item[0][1]):
        runs.setdefault(variant, []).append(final)
    return {v: finals for v, finals in runs.items() if finals}


def spread(values, fmt="{:.3f}"):
    """Mean, with the range when there is more than one seed."""
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    average = fmt.format(sum(values) / len(values))
    if len(values) == 1:
        return average
    return f"{average} ({fmt.format(min(values))}-{fmt.format(max(values))})"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def table(header, rows):
    lines = ["| " + " | ".join(str(c) for c in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def depth_label(final):
    unit = final.get("depth_unit", "passes")
    return f"{final.get('eval_depth', final.get('trained_passes'))} {unit}"


def overlap_pair(final, split, kind):
    """Overlap of depth 2 of the chain with depth 1 of the direct question."""
    matrix = ((final.get("overlap") or {}).get(split) or {}).get(kind)
    if not matrix or len(matrix) < 2:
        return None
    return matrix[1][0]


def overlap_best(final, split):
    """Largest same-entity overlap over all depth pairs, with its control."""
    entry = (final.get("overlap") or {}).get(split)
    if not entry:
        return None, None
    best = (None, None)
    for same_row, control_row in zip(entry["same"], entry["control"]):
        for same, control in zip(same_row, control_row):
            if same is not None and (best[0] is None or same > best[0]):
                best = (same, control)
    return best


def two_call_numbers(final, split):
    """(normal, two calls at matched compute, two calls at full depth).

    Read from final.json, or from a diagnostics file written later for a run
    that finished before these were recorded.
    """
    normal_depth = final.get("eval_depth", final.get("trained_passes"))
    result = final.get("two_call")
    if result is None:
        for name, key in (("diagnostics.json", "two_call"), ("two-call-diagnostic.json", "splits")):
            path = final["_dir"] / name
            if path.exists():
                result = json.loads(path.read_text())[key]
                break
    if not result or split not in result:
        return None, None, None
    by_depth = result[split]
    depths = sorted(by_depth, key=int)
    native = by_depth[depths[0]]["native"].get(str(normal_depth), {}).get("accuracy")
    return (
        native,
        by_depth[depths[0]]["two_call"]["accuracy"],
        by_depth[depths[-1]]["two_call"]["accuracy"],
    )


def build_report(roots):
    runs = load_runs(roots)
    names = roots if isinstance(roots, (str, Path)) else ", ".join(str(r) for r in roots)
    if not runs:
        return f"No finished runs (final.json) under {names}."
    out = [f"# Recurrent lookup results: {names}", ""]
    slot_runs = {v: f for v, f in runs.items() if f[0].get("depth_unit") == "steps"}
    pass_runs = {v: f for v, f in runs.items() if v not in slot_runs}

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
                depth_label(first),
                f"{mean([f['sec_per_step'] for f in finals]) * 1000:.1f}",
                *[spread([f["accuracy"].get(s) for f in finals]) for s, _ in SPLIT_COLUMNS],
                spread(usage),
            ]
        )
    out += [
        "## Accuracy",
        "",
        table(
            ["run", "seeds", "params", "steps", "evaluated at", "ms/step"]
            + [label for _, label in SPLIT_COLUMNS]
            + ["table used"],
            rows,
        ),
        "",
        "Every fact is trained as a one-hop question. \"held-out\" chains need a "
        "fact that never appeared inside a two-hop training chain, in the first "
        "hop, the second hop, or both. Held-out 2nd and both are the main "
        "result: chance is 1 / number of entities. Values are means over seeds, "
        "with the range in brackets. Original runs are evaluated at their "
        "trained number of passes. Scratchpad runs are evaluated at 2 outer "
        "steps of 2 passes, the same number of block applications as 4 passes.",
        "",
    ]

    if pass_runs:
        passes = sorted(
            {int(p) for fs in pass_runs.values() for f in fs for p in f["accuracy_by_passes"]}
        )
        rows = [
            [variant]
            + [
                spread(
                    [f["accuracy_by_passes"].get(str(p), {}).get(MAIN_SPLIT) for f in finals]
                )
                for p in passes
            ]
            for variant, finals in pass_runs.items()
        ]
        out += [
            "## Held-out 2nd-hop accuracy by number of passes at test time",
            "",
            table(["run"] + [f"{p} pass" + "es" * (p != 1) for p in passes], rows),
            "",
        ]

    if slot_runs:
        steps = sorted(
            {int(t) for fs in slot_runs.values() for f in fs for t in f["accuracy_by_steps"]}
        )
        rows = [
            [variant, label]
            + [
                spread([f["accuracy_by_steps"].get(str(t), {}).get(split) for f in finals])
                for t in steps
            ]
            for variant, finals in slot_runs.items()
            for split, label in (SPLIT_COLUMNS[0], SPLIT_COLUMNS[1], SPLIT_COLUMNS[4])
        ]
        out += [
            "## Scratchpad runs: accuracy by number of outer steps at test time",
            "",
            table(["run", "questions"] + [f"{t} step" + "s" * (t != 1) for t in steps], rows),
            "",
            "Training samples 2, 3 or 4 steps per batch. One-hop answers have to "
            "survive the extra steps.",
            "",
        ]

        measures = [
            ("bridge_in_slot", "bridge in slot"),
            ("bridge_in_readout", "bridge in step-1 readout"),
            ("write_weight", "write weight"),
        ]
        groups = [SPLIT_COLUMNS[1]] + TEST_GROUPS
        rows = [
            [variant, label]
            + [
                spread([(f["slots"].get(split) or {}).get(key) for f in finals])
                for split, _ in groups
            ]
            for variant, finals in slot_runs.items()
            for key, label in measures
        ]
        out += [
            "## Scratchpad runs: what is in the slot after step 1",
            "",
            table(["run", "measure"] + [label for _, label in groups], rows),
            "",
            "\"bridge in slot\" is the share of two-hop questions where a slot is "
            "closest, among entity embeddings, to the bridge entity. \"write "
            "weight\" is the mean weight the write head put on the slots.",
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
                ["run", "chains", "2nd vs direct", "control", "best pair", "its control"],
                rows,
            ),
            "",
            "For a chain a -> b -> c: overlap (Jaccard) between the table entries "
            "retrieved at the answer position of the two-hop question and those "
            "retrieved by the direct question about b. \"2nd vs direct\" compares "
            "pass 2 of the two-hop question with pass 1 of the direct question "
            "(step 2 with step 1 for scratchpad runs, pooling the passes inside "
            "a step). \"best pair\" is the largest overlap over all pairs. The "
            "control asks the direct question about a different entity with the "
            "same relation. Reuse of stored facts shows up as overlap well above "
            "control.",
            "",
        ]

    rows = []
    for variant, finals in runs.items():
        cells = []
        for split, _ in TEST_GROUPS:
            numbers = [two_call_numbers(f, split) for f in finals]
            cells.append(
                " / ".join(spread([n[i] for n in numbers], "{:.1%}") for i in range(3))
            )
        if any(c != "- / - / -" for c in cells):
            rows.append([variant] + cells)
    if rows:
        out += [
            "## Two-call ceiling",
            "",
            table(["run"] + [label for _, label in TEST_GROUPS], rows),
            "",
            "Each cell is: normal run / two calls at the same total compute / two "
            "calls at full depth each. A two-call answer asks the first hop as a "
            "one-hop question, then asks the second hop about the entity the "
            "model predicted. It is the ceiling set by what the model has stored.",
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
    if "slot_soft" in runs:
        soft = runs["slot_soft"]
        ceiling = mean([two_call_numbers(f, MAIN_SPLIT)[2] for f in soft])
        in_slot = mean([(f["slots"].get(MAIN_SPLIT) or {}).get("bridge_in_slot") for f in soft])
        line(
            "Token-form slot (slot_soft): held-out 2nd hop {:.3f}, its two-call "
            "ceiling {:.3f}, bridge in slot on those chains {:.3f}.",
            acc("slot_soft", MAIN_SPLIT),
            ceiling,
            in_slot,
        )
        line("Raw-vector slot (slot_raw): held-out 2nd hop {:.3f}.", acc("slot_raw", MAIN_SPLIT))
        line(
            "Token-form slot with the bridge loss (slot_soft_prefix): held-out "
            "2nd hop {:.3f}.",
            acc("slot_soft_prefix", MAIN_SPLIT),
        )
        line("Three slots (slot_soft_k3): held-out 2nd hop {:.3f}.", acc("slot_soft_k3", MAIN_SPLIT))
    out += [
        "",
        "Chance is 1 / number of entities. Variants with one seed cannot "
        "support conclusions from small gaps.",
    ]
    return "\n".join(out) + "\n"


def write_report(roots, out_dir=None):
    text = build_report(roots)
    if out_dir is None:
        out_dir = roots if isinstance(roots, (str, Path)) else roots[0]
    (Path(out_dir) / "report.md").write_text(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("roots", nargs="+", help="directories holding run directories")
    args = p.parse_args(argv)
    print(write_report(args.roots))


if __name__ == "__main__":
    main()
