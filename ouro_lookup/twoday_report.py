"""Summarise the two-day run as a markdown report.

    python -m ouro_lookup.twoday_report runs/ouro-twoday
"""

import argparse
import json
from pathlib import Path

from phrase_scratchpad.report import spread
from recurrent_lookup.report import table

from . import flex, takeover


def pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def num(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def share(part, whole):
    return pct(part / whole if whole else None)


def load(root):
    """(take-over runs, retention runs, editing runs), each in the order of
    the variants and then by seed."""
    finals = []
    for path in sorted(Path(root).glob("*/final.json")):
        if not path.parent.name.startswith("_"):
            finals.append(json.loads(path.read_text()))
    take = {f["variant"]: f for f in finals if f.get("kind") == "takeover"}
    take = [take[v] for v in takeover.VARIANTS if v in take]

    def group(mode):
        by_variant = {}
        for f in finals:
            if f.get("kind") == "flex" and f["mode"] == mode:
                by_variant.setdefault(f["variant"], []).append(f)
        return {
            v: sorted(by_variant[v], key=lambda f: f["seed"]) for v in flex.VARIANTS if v in by_variant
        }

    return take, group("retain"), group("edit")


def takeover_section(runs):
    first = runs[0]
    held, removed = first["held_out"], first["removed"]
    known = first["probes"]["trained"]["known"]
    out = [
        "## Taking over the removed layers",
        "",
        f"The MLP of layers {removed['layers'][0]} to {removed['layers'][-1]} is removed: "
        f"{removed['numbers']:,} numbers, {pct(removed['share_of_model'])} of the model. "
        "Each stand-in is trained to imitate the original model on ordinary text. Held-out "
        f"loss is on {held['tokens']:,} tokens of articles never trained on. Known facts "
        f"are {known} sentence starts, of {first['probes']['listed']} tried, that the "
        "original model completes correctly. Nothing is trained on them.",
        "",
    ]
    rows = [
        ["original Ouro", "-", "100%", "-", num(held["original"]), "-", f"{known}", share(known, known)],
        ["nothing put back", "0", "0%", "-", num(held["nothing"]), "0.0%",
         f"{first['probes']['nothing']['right']}", share(first["probes"]["nothing"]["right"], known)],
    ]
    for f in runs:
        facts = f["probes"]["trained"]
        rows.append([
            f["variant"], f"{f['params']['trained']:,}", pct(f["cost"]["share"]),
            f"{f['tokens_trained'] / 1e6:.1f}M", num(f["held_out"]["trained"]),
            pct(f["held_out"]["gap_recovered"]), f"{facts['right']}",
            share(facts["right"], facts["known"]),
        ])
    out += [
        table(
            ["what stands in", "trained numbers", "computation, as a share of the removed MLPs",
             "tokens trained on", "held-out loss", "share of the lost ground recovered",
             "known facts right", "share of known facts"],
            rows,
        ),
        "",
        "\"Lost ground\" is the rise in held-out loss when the MLPs are removed and nothing "
        "is put back. Computation counts multiplications per token for what stands in for "
        "one removed MLP; adapters add none once folded into the weights.",
        "",
    ]
    for f in runs:
        out.append(f"- `{f['variant']}`: {f['about']}.")
    out.append("")

    categories = list(first["probes"]["trained"]["by_category"])
    rows = [["original Ouro"] + [first["probes"]["trained"]["by_category"][c]["known"] for c in categories]]
    rows.append(
        ["nothing put back"] + [first["probes"]["nothing"]["by_category"][c]["right"] for c in categories]
    )
    for f in runs:
        rows.append(
            [f["variant"]] + [f["probes"]["trained"]["by_category"][c]["right"] for c in categories]
        )
    out += [
        "### Known facts by kind",
        "",
        table(["what stands in"] + categories, rows),
        "",
        "Counts of sentence starts completed correctly. \"Sequences\" are patterns to "
        "continue (\"Monday, Tuesday, Wednesday,\"), which need no stored fact.",
        "",
        "### Side measures",
        "",
        table(
            ["what stands in", "distance from the original's predictions", "loss on sixteen sentences",
             "table entries used", "ms per step", "peak memory", "hours"],
            [
                ["nothing put back", num(held["kl_nothing"]), num(first["sentences"]["nothing"]),
                 "-", "-", "-", "-"]
            ] + [
                [
                    f["variant"], num(f["held_out"]["kl_trained"]), num(f["sentences"]["trained"]),
                    pct(sum(f["table_used"]) / len(f["table_used"])) if f["table_used"] else "-",
                    f"{1000 * f['sec_per_step']:.0f}", f"{f['peak_memory_gb']:.1f} GB",
                    f"{f['wall_minutes'] / 60:.1f}",
                ]
                for f in runs
            ],
        ),
        "",
        "Distance is the mean divergence of the next-token distribution from the original "
        f"model's, on the held-out text. The original scores {num(first['sentences']['original'])} "
        "on the sixteen sentences.",
        "",
    ]
    return out


def retain_section(groups):
    out = ["## Adding facts batch by batch", ""]
    any_run = next(iter(groups.values()))[0]
    n = len(any_run["phases"])
    out += [
        f"{any_run['facts']['facts']} made-up facts about {any_run['facts']['people']} people, "
        f"split by person into {n} batches. Each batch is trained alone until it is recalled, "
        "and no earlier batch is shown again.",
        "",
    ]
    rows = []
    for variant, runs in groups.items():
        s = [f["summary"] for f in runs]
        rows.append([
            variant, len(runs),
            spread([x["learned"] for x in s]),
            spread([x["kept_at_end"] for x in s]),
            spread([x["first_batch_at_end"] for x in s]),
            spread([x["all_at_end"] for x in s]),
            f"{sum(sum(x['steps']) for x in s) / sum(len(x['steps']) for x in s):.0f}",
            "yes" if all(all(x["reached_target"]) for x in s) else "no",
            num(sum(f["ordinary_text_loss"]["at_end"] for f in runs) / len(runs)),
        ])
    out += [
        table(
            ["run", "seeds", "recall of a batch just after learning it",
             "earlier batches at the end", "first batch at the end", "all facts at the end",
             "steps per batch", "every batch learned", "loss on sixteen sentences at the end"],
            rows,
        ),
        "",
        "The mean over seeds, with the range. \"Earlier batches at the end\" is recall, after "
        "the last batch, of every batch but the last: what survives later learning. The "
        f"original model scores {num(any_run['ordinary_text_loss']['original'])} on the "
        "sixteen sentences.",
        "",
    ]
    for variant, runs in groups.items():
        out.append(f"- `{variant}`: {runs[0]['about']}.")
    out += ["", "### Recall of each batch after each phase", ""]
    for variant, runs in groups.items():
        f = runs[0]
        rows = [
            [f"after batch {p['phase'] + 1}", p["steps"]] + [pct(x) for x in p["recall"]]
            for p in f["phases"]
        ]
        out += [
            f"`{variant}`, seed {f['seed']}:",
            "",
            table(["", "steps"] + [f"batch {i + 1}" for i in range(n)], rows),
            "",
        ]
    return out


def edit_section(groups):
    any_run = next(iter(groups.values()))[0]
    e = any_run["edit"]
    out = [
        "## Changing facts",
        "",
        f"All the facts are trained together. Then {e['changed']} of them get a new answer and "
        f"only those are trained, for {e['at_end']['step']} steps. The other {e['untouched']} "
        "are not shown again.",
        "",
    ]
    rows = []
    for variant, runs in groups.items():
        took = [f["edit"]["when_changed"] for f in runs]
        done = [t for t in took if t]
        rows.append([
            variant, len(runs),
            spread([f["joint"]["recall"] for f in runs]),
            f"{sum(t['step'] for t in done) / len(done):.0f}" if len(done) == len(took) else "not reached",
            spread([t["untouched"] for t in done]) if len(done) == len(took) else "-",
            spread([f["edit"]["at_end"]["recall"] for f in runs]),
            spread([f["edit"]["at_end"]["old_answer"] for f in runs]),
            spread([f["edit"]["at_end"]["untouched"] for f in runs]),
            num(sum(f["ordinary_text_loss"]["at_end"] for f in runs) / len(runs)),
        ])
    out += [
        table(
            ["run", "seeds", "all facts before the change", "steps until 95% of new answers",
             "untouched facts kept at that point", "new answers given at the end",
             "old answers still given", "untouched facts kept at the end",
             "loss on sixteen sentences at the end"],
            rows,
        ),
        "",
        "The mean over seeds, with the range. \"At that point\" is the first check at which "
        "95% of the changed facts give their new answer.",
        "",
    ]
    return out


def build_report(root):
    take, retain, edit = load(root)
    if not (take or retain or edit):
        return f"No finished runs (final.json) under {root}."
    out = [f"# Ouro, two-day run: {root}", ""]
    check_path = Path(root) / "_check" / "verdict.json"
    if check_path.exists():
        check = json.loads(check_path.read_text())
        out += [
            "Quick check before the runs: "
            + ("passed." if check["passed"] else "**failed**: read the runs below with care."),
            "",
        ]
    if take:
        out += takeover_section(take)
    if retain:
        out += retain_section(retain)
    if edit:
        out += edit_section(edit)
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
