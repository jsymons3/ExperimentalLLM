"""Summarise finished runs as a markdown report.

    python -m phrase_scratchpad.report runs/phrase-scratchpad
"""

import argparse
import json
from pathlib import Path

from recurrent_lookup.report import table

from .configs import VARIANTS

SPLITS = [
    ("one_hop", "1-step"),
    ("one_hop_held", "1-step, never in a chain"),
    ("chain_train", "trained chains"),
    ("chain_unseen", "unseen chains"),
    ("chain_held_first", "held-out 1st"),
    ("chain_held_second", "held-out 2nd"),
    ("chain_held_both", "held-out both"),
]
MAIN = "chain_held_second"
LABELS = dict(SPLITS)


def hardest(entry):
    """The hardest test split a task has: held-out second step where the
    task has stored facts to hold out, otherwise unseen two-step questions."""
    return MAIN if MAIN in entry["accuracy"] else "chain_unseen"


def pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def load_runs(root):
    runs = {}
    for path in sorted(Path(root).glob("*/final.json")):
        final = json.loads(path.read_text())
        runs[final["variant"]] = final
    return {v: runs[v] for v in VARIANTS if v in runs}


def build_report(root):
    runs = load_runs(root)
    if not runs:
        return f"No finished runs (final.json) under {root}."
    out = [f"# Multi-token scratchpad results: {root}", ""]
    rows_by_task = [(v, name, entry) for v, f in runs.items() for name, entry in f["tasks"].items()]

    rows = [
        [v, name, runs[v]["steps"]]
        + [pct(entry["accuracy"].get(split)) for split, _ in SPLITS]
        + [pct(entry["majority_baseline"].get(MAIN))]
        for v, name, entry in rows_by_task
    ]
    out += [
        "## Accuracy",
        "",
        table(
            ["run", "task", "steps"] + [label for _, label in SPLITS] + ["guessing, held-out 2nd"],
            rows,
        ),
        "",
        "An answer counts only if every token is right. Every one-step question "
        "is trained. A held-out chain needs a fact that never appeared in a "
        "training chain, as its first step, its second step, or both. "
        "\"Guessing\" always gives the commonest training answer for the final "
        "relation or operation, and is the floor to compare against. Each task "
        "is scored the way it was trained, with the table on or off.",
        "",
    ]

    rows = []
    for v, name, entry in rows_by_task:
        split = hardest(entry)
        report = (entry.get("scratchpad") or {}).get(split)
        if report:
            i = report["interventions"]
            rows.append(
                [v, name, LABELS[split], pct(report["bridge_written"])]
                + [pct(i[k]) for k in ("normal", "cleared", "shuffled", "true_bridge")]
            )
    if rows:
        out += [
            "## Is the scratchpad used?",
            "",
            table(
                ["run", "task", "questions", "bridge written at step 1", "normal",
                 "scratchpad cleared", "scratchpad shuffled", "true bridge supplied"],
                rows,
            ),
            "",
            "\"Bridge written\" is the share of chains whose scratchpad holds "
            "exactly the intermediate answer after step 1. The other columns are "
            "accuracy when what step 2 reads is replaced. If the answer goes "
            "through the scratchpad, clearing or shuffling it should destroy "
            "accuracy. Supplying the true bridge uses the answer key and is a "
            "diagnostic, not a score.",
            "",
        ]

    rows = []
    for v, name, entry in rows_by_task:
        calls = entry["two_call"].get(MAIN)
        if calls:
            last = calls[max(calls, key=int)]
            rows.append(
                [v, name, pct(last["native"])]
                + [pct(calls[d]["two_call"]) if d in calls else "-" for d in ("1", "2")]
                + [f"{last['both_known']:,} of {last['n']:,}", f"{last['native_on_both_known']:,}"]
            )
    if rows:
        out += [
            "## Two-call ceiling (held-out 2nd chains)",
            "",
            table(
                ["run", "task", "one question", "two calls, 1 step each", "two calls, 2 steps each",
                 "both facts known", "of those, answered as one question"],
                rows,
            ),
            "",
            "Two calls ask the first step as its own question, then ask the "
            "second step about whatever the model answered. It shows what the "
            "model has stored, whether or not it can chain it. A run without a "
            "scratchpad has no outer steps, so it has one kind of call.",
            "",
        ]

    rows = [
        [v, name, "on" if entry["trained_with_table"] else "off"]
        + [
            f"{pct(entry['accuracy_table_on'].get(s))} / {pct(entry['accuracy_table_off'].get(s))}"
            for s in ("one_hop", "chain_unseen")
        ]
        for v, name, entry in rows_by_task
    ]
    out += [
        "## What depends on the table",
        "",
        table(
            ["run", "task", "trained with table", "1-step: on / off", "unseen chains: on / off"],
            rows,
        ),
        "",
        "Each cell is accuracy with the lookup table on, then off. A split "
        "between a reasoning core and a fact table would show arithmetic and "
        "the story questions surviving with the table off, and facts collapsing.",
        "",
    ]

    steps = sorted({int(d) for _, _, e in rows_by_task for d in e["accuracy_by_steps"]})
    rows = [
        [v, name, LABELS[hardest(entry)]]
        + [pct(entry["accuracy_by_steps"].get(str(d), {}).get(hardest(entry))) for d in steps]
        for v, name, entry in rows_by_task
    ]
    out += [
        "## Accuracy by number of outer steps at test time",
        "",
        table(["run", "task", "questions"] + [f"{d} step" + "s" * (d != 1) for d in steps], rows),
        "",
    ]

    def acc(variant, task, split=MAIN):
        entry = runs.get(variant, {}).get("tasks", {}).get(task)
        return None if entry is None else entry["accuracy"].get(split)

    def written(variant, task, split="chain_unseen"):
        entry = runs.get(variant, {}).get("tasks", {}).get(task) or {}
        return ((entry.get("scratchpad") or {}).get(split) or {}).get("bridge_written")

    out += ["## Readout", ""]

    def line(text, *values):
        if all(v is not None for v in values):
            out.append("- " + text.format(*(pct(v) for v in values)))

    line("Stage 1, arithmetic with the intermediate result taught: held-out 2nd {}.", acc("maths", "maths"))
    line("Arithmetic without a scratchpad: held-out 2nd {}.", acc("maths_plain", "maths"))
    line(
        "Stage 1, family stories with the intermediate link taught: unseen stories {}.",
        acc("clutrr", "clutrr", "chain_unseen"),
    )
    line(
        "Family stories without a scratchpad: unseen stories {}.",
        acc("clutrr_plain", "clutrr", "chain_unseen"),
    )
    line("Stage 2, facts with the intermediate answer taught: held-out 2nd {}.", acc("facts", "facts"))
    line("Facts without a scratchpad: held-out 2nd {}.", acc("facts_plain", "facts"))
    line(
        "Stage 3 control, facts alone with no intermediate answer taught: unseen "
        "chains {}, held-out 2nd {}, bridge written on unseen chains {}.",
        acc("facts_unsupervised", "facts", "chain_unseen"),
        acc("facts_unsupervised", "facts"),
        written("facts_unsupervised", "facts"),
    )
    for variant, label in (
        ("mix_split", "table off on arithmetic and stories"),
        ("mix", "table always on"),
    ):
        line(
            "Stage 3 (" + label + "), fact chains with the intermediate taught only "
            "on arithmetic and stories: unseen chains {}, held-out 2nd {}, bridge "
            "written on unseen chains {}.",
            acc(variant, "facts", "chain_unseen"),
            acc(variant, "facts"),
            written(variant, "facts"),
        )
    out += [
        "",
        "Stage 3 worked if its fact chains beat the control and the bridge is "
        "written for fact chains, where it was never taught. One seed per run: "
        "small gaps are noise.",
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
