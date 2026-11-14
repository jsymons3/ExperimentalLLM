"""Summarise finished runs as a markdown report.

    python -m ouro_lookup.report runs/ouro-lookup
"""

import argparse
import json
from pathlib import Path

from recurrent_lookup.report import table

from .configs import VARIANTS

SPLITS = [
    ("one_hop", "one-step"),
    ("chain_train", "two-step, trained"),
    ("chain_unseen", "two-step, unseen"),
    ("chain_held_first", "held-out 1st"),
    ("chain_held_second", "held-out 2nd"),
    ("chain_held_both", "held-out both"),
]
TEST_GROUPS = [s for s in SPLITS if s[0] not in ("one_hop", "chain_train")]


def pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def load_runs(root):
    runs = {}
    for path in sorted(Path(root).glob("*/final.json")):
        if path.parent.name.startswith("_"):
            continue  # the quick check and the timing runs
        final = json.loads(path.read_text())
        runs[final["variant"]] = final
    return {v: runs[v] for v in VARIANTS if v in runs}


def build_report(root):
    root = Path(root)
    runs = load_runs(root)
    if not runs:
        return f"No finished runs (final.json) under {root}."
    out = [f"# Ouro with a lookup table: {root}", ""]

    check_path = root / "_check" / "verdict.json"
    if check_path.exists():
        check = json.loads(check_path.read_text())
        out += [
            f"Quick check before the runs ({check['facts']} facts, {check['steps']} steps): "
            f"one-step recall {pct(check['table_on'])} with the table, "
            f"{pct(check['table_off'])} without. "
            + ("Passed." if check["passed"] else "**Failed**: read the runs below with care."),
            "",
        ]

    facts = next(iter(runs.values()))["facts"]
    out += [
        f"{facts['facts']} made-up facts about {facts['people']} people "
        f"(for example \"{facts['example_one_step']} ...\"), "
        f"{facts['training_chains']} two-step training questions "
        f"(for example \"{facts['example_two_step']} ...\"). "
        f"{facts['facts_never_in_a_training_chain']} facts never appear in a two-step "
        "training question.",
        "",
        "## Runs",
        "",
        table(
            ["run", "what it is", "two-step questions", "steps", "minutes", "trained numbers"],
            [
                [
                    v, f["about"],
                    "written note" if f["chain_mode"] == "note" else "one pass",
                    f["steps"], f"{f['wall_minutes']:.0f}", f"{f['params']['trained']:,}",
                ]
                for v, f in runs.items()
            ],
        ),
        "",
        "## Accuracy",
        "",
        table(
            ["run"] + [label for _, label in SPLITS] + ["guessing"],
            [
                [v]
                + [pct(f["accuracy"].get(split)) for split, _ in SPLITS]
                + [pct(f["guess_rate"].get("one_hop"))]
                for v, f in runs.items()
            ],
        ),
        "",
        "An answer counts only if it is the model's single most likely next token. "
        "Every fact is trained as a one-step question. A held-out two-step question "
        "needs a fact that never appeared in a two-step training question, as its "
        "first step, its second step, or both. \"Guessing\" always gives the "
        "commonest answer.",
        "",
        "## What each addition contributes",
        "",
    ]
    rows = []
    for v, f in runs.items():
        for split, label in (("one_hop", "one-step"), ("chain_unseen", "two-step, unseen")):
            rows.append(
                [
                    v, label,
                    pct(f["accuracy"].get(split)),
                    pct(f.get("accuracy_table_off", {}).get(split)),
                    pct(f.get("accuracy_adapters_off", {}).get(split)),
                    pct(f["accuracy_original"].get(split)),
                ]
            )
    out += [
        table(["run", "questions", "everything on", "table off", "adapters off", "original Ouro"], rows),
        "",
        "\"Original Ouro\" is the frozen model with nothing added: what it knew "
        "beforehand, which should be nothing. For a run with only one addition, "
        "that column is also the run with its addition switched off.",
        "",
        "## What limits two-step accuracy",
        "",
    ]
    rows = []
    for v, f in runs.items():
        for split, label in TEST_GROUPS:
            entry = f["chains"].get(split)
            if entry:
                rows.append(
                    [
                        v, label,
                        pct(f["accuracy"].get(split)),
                        pct(entry["both_facts_known"]),
                        pct(entry["accuracy_when_both_known"]),
                        pct(entry["two_call"]),
                    ]
                )
    out += [
        table(
            ["run", "questions", "accuracy", "both facts known", "accuracy when both known",
             "two separate questions"],
            rows,
        ),
        "",
        "\"Both facts known\" is the share where each fact is answered correctly as a "
        "one-step question. \"Two separate questions\" asks the first step outright, then "
        "asks the second step about whatever came back: what chaining could reach with "
        "the facts as they are recalled.",
        "",
    ]

    rows = []
    for v, f in runs.items():
        if f["chain_mode"] != "note":
            continue
        for split, label in TEST_GROUPS:
            entry = f["chains"].get(split)
            if entry:
                rows.append(
                    [
                        v, label,
                        pct(entry["note_right"]),
                        pct(f["accuracy"].get(split)),
                        pct(entry["shuffled_notes"]),
                        pct(entry["correct_notes"]),
                    ]
                )
    if rows:
        out += [
            "## The written note",
            "",
            table(
                ["run", "questions", "note is right", "own notes", "shuffled notes",
                 "correct notes supplied"],
                rows,
            ),
            "",
            "The model writes one name as a note, and a fresh one-step question is asked "
            "about that name. Shuffled notes are swapped between questions; if accuracy "
            "holds up with them, the note is not being used.",
            "",
        ]

    rows = []
    for v, f in runs.items():
        entry = f["chains"].get("chain_unseen")
        if entry:
            loops = entry["by_loop"]
            rows.append([v, "intermediate answer"] + [pct(x) for x in loops["bridge"]])
            rows.append([v, "final answer"] + [pct(x) for x in loops["answer"]])
    if rows:
        n_loops = len(rows[0]) - 2
        out += [
            "## What the state reads as after each loop",
            "",
            table(["run", "reads as"] + [f"loop {i + 1}" for i in range(n_loops)], rows),
            "",
            "Unseen two-step questions. After every loop Ouro's state can be read out as a "
            "next token; this is how often that token is the intermediate answer, and how "
            "often the final one.",
            "",
        ]

    out += [
        "## Side effects and cost",
        "",
        table(
            ["run", "loss on ordinary text, original", "with additions", "table entries used",
             "ms per step", "peak memory"],
            [
                [
                    v,
                    f"{f['ordinary_text_loss']['original']:.3f}",
                    f"{f['ordinary_text_loss']['with_additions']:.3f}",
                    ", ".join(pct(x) for x in f["table_used"]) if "table_used" in f else "-",
                    f"{1000 * f['sec_per_step']:.0f}",
                    f"{f['peak_memory_gb']:.1f} GB",
                ]
                for v, f in runs.items()
            ],
        ),
        "",
        "Loss on ordinary text is the mean next-token loss on sixteen everyday "
        "sentences; a rise means the additions disturb what Ouro already did.",
        "",
    ]
    return "\n".join(out)


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
