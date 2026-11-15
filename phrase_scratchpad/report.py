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


def spread(values):
    """The mean of a few runs' values, with their range if there are several."""
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    mean = pct(sum(values) / len(values))
    if len(values) == 1:
        return mean
    return f"{mean} ({100 * min(values):.1f} to {100 * max(values):.1f})"


def load_runs(root):
    """{label: final.json contents}, in the order of VARIANTS and then seed.

    The label is the run's name, with its seed added when the directory
    holds more than seed 0.
    """
    finals = [json.loads(path.read_text()) for path in sorted(Path(root).glob("*/final.json"))]
    finals = [f for f in finals if f["variant"] in VARIANTS]
    order = list(VARIANTS)
    finals.sort(key=lambda f: (order.index(f["variant"]), f["seed"]))
    seeded = any(f["seed"] != 0 for f in finals)
    return {
        (f"{f['variant']}, seed {f['seed']}" if seeded else f["variant"]): f for f in finals
    }


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

    rows, extra = [], False
    for v, name, entry in rows_by_task:
        split = hardest(entry)
        report = (entry.get("scratchpad") or {}).get(split)
        if report:
            i = report["interventions"]
            extra = extra or "fresh_question" in report or "subject_hidden" in report
            rows.append(
                [v, name, LABELS[split], pct(report["bridge_written"])]
                + [pct(i[k]) for k in ("normal", "cleared", "shuffled", "true_bridge")]
                + [pct(report.get("answer_written")), pct(report.get("subject_hidden")),
                   pct(report.get("fresh_question"))]
            )
    if rows:
        head = ["run", "task", "questions", "bridge written at step 1", "normal",
                "scratchpad cleared", "scratchpad shuffled", "true bridge supplied"]
        if extra:
            head += ["final answer written at step 1", "subject hidden after step 1",
                     "own note in a fresh question"]
        else:
            rows = [row[:-3] for row in rows]
        out += [
            "## Is the scratchpad used?",
            "",
            table(head, rows),
            "",
            "\"Bridge written\" is the share of chains whose scratchpad holds "
            "exactly the intermediate answer after step 1. The other columns are "
            "accuracy when what step 2 reads is replaced. If the answer goes "
            "through the scratchpad, clearing or shuffling it should destroy "
            "accuracy. Supplying the true bridge uses the answer key and is a "
            "diagnostic, not a score.",
            "",
        ]
        if extra:
            out += [
                "\"Final answer written at step 1\" is the share of chains whose "
                "scratchpad already holds the final answer after step 1: the chain "
                "answered in one step, with the scratchpad only carrying the result "
                "forward. \"Subject hidden\" hides the question's subject from step 2 on, as "
                "half the training fact chains do. \"Own note in a fresh question\" "
                "takes the note the model wrote, writes it into a new one-step "
                "question about the remaining relation, and asks that on its own: "
                "the handoff done by rewriting the question in code, not by "
                "reading the scratchpad.",
                "",
            ]

    rows = []
    for v, name, entry in rows_by_task:
        for split, label in (("one_hop", "used in training chains"),
                             ("one_hop_held", "never in a training chain")):
            r = (entry.get("reader") or {}).get(split)
            if r:
                rows.append(
                    [v, name, label, f"{r['n']:,}", pct(r["as_a_question"]),
                     pct(r["through_note"]), pct(r["through_note_question_shown"])]
                )
    if rows:
        out += [
            "## Reading a known fact from the note",
            "",
            table(
                ["run", "task", "facts", "count", "asked as a question",
                 "asked through the note", "through the note, question in view"],
                rows,
            ),
            "",
            "One-step facts that can be the second step of a chain. \"Through the "
            "note\" puts the fact's subject in the scratchpad of an unrelated "
            "two-step question and asks for the fact's answer, first with that "
            "question's subject hidden after step 1 and then with it in view. A "
            "held-out second step needs a fact from the \"never in a training "
            "chain\" rows to be read this way.",
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

    by_variant = {}
    for final in runs.values():
        by_variant.setdefault(final["variant"], []).append(final)

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def entries(variant, task):
        return [f["tasks"][task] for f in by_variant.get(variant, []) if task in f["tasks"]]

    def acc(variant, task, split=MAIN):
        return mean([e["accuracy"].get(split) for e in entries(variant, task)])

    def notes(variant, task, split):
        return [(e.get("scratchpad") or {}).get(split) or {} for e in entries(variant, task)]

    def written(variant, task, split="chain_unseen"):
        return mean([n.get("bridge_written") for n in notes(variant, task, split)])

    reading = [v for v in by_variant if v.endswith("_reader")]
    if reading:
        rows = []
        for variant in by_variant:
            if not entries(variant, "facts"):
                continue
            pad = notes(variant, "facts", MAIN)
            held = [(e.get("reader") or {}).get("one_hop_held") or {} for e in entries(variant, "facts")]
            rows.append(
                [
                    variant,
                    len(by_variant[variant]),
                    spread([e["accuracy"].get(MAIN) for e in entries(variant, "facts")]),
                    spread([n.get("bridge_written") for n in pad]),
                    spread([(n.get("interventions") or {}).get("true_bridge") for n in pad]),
                    spread([(n.get("interventions") or {}).get("shuffled") for n in pad]),
                    spread([n.get("subject_hidden") for n in pad]),
                    spread([n.get("fresh_question") for n in pad]),
                    spread([h.get("through_note") for h in held]),
                ]
            )
        out += [
            "## The reading test",
            "",
            table(
                ["run", "seeds", "held-out 2nd", "bridge written", "true bridge supplied",
                 "scratchpad shuffled", "subject hidden after step 1",
                 "own note in a fresh question", "held-out fact read through the note"],
                rows,
            ),
            "",
            "Fact chains whose second fact never appeared in a training chain; the "
            "mean over seeds, with the range. A `_reader` run is the run above it "
            "with half of its one-step questions about second-step facts asked "
            "through the note instead. It sees each fact as often and sees no "
            "extra chain. The last column is the direct check that the teaching "
            "took: one-step facts never used in a training chain, asked through "
            "the note.",
            "",
            "- Reading is fixed if, with the intermediate answer taught, "
            "`facts_reader` raises \"true bridge supplied\" and held-out 2nd "
            "toward \"own note in a fresh question\".",
            "- If `facts` scores far higher with the subject hidden than without, "
            "its note is read correctly when nothing else is in view, and the loss "
            "comes from the question still being visible at step 2.",
            "- The note appears by itself if `facts_unsupervised_reader` writes "
            "the bridge, where no run was ever taught to, and its held-out 2nd "
            "rises above `facts_unsupervised` while shuffling still destroys it. A "
            "high \"true bridge supplied\" with the bridge not written means step 2 "
            "can read a name and step 1 does not supply one.",
            "",
        ]

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
    for variant, label in (
        ("facts_reader", "the intermediate answer taught"),
        ("facts_unsupervised_reader", "no intermediate answer taught"),
    ):
        line(
            "Reading test, facts with " + label + " and known facts also asked "
            "through the note: unseen chains {}, held-out 2nd {}, bridge written on "
            "unseen chains {}.",
            acc(variant, "facts", "chain_unseen"),
            acc(variant, "facts"),
            written(variant, "facts"),
        )
    several = any(len(finals) > 1 for finals in by_variant.values())
    out += [
        "",
        "Stage 3 worked if its fact chains beat the control and the bridge is "
        "written for fact chains, where it was never taught. "
        + ("These lines give the mean over seeds." if several
           else "One seed per run: small gaps are noise."),
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
