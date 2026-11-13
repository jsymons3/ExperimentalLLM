"""Summarize completed natural-text runs without semantic claims from loss gaps."""

import argparse
import json
from pathlib import Path


def report(root):
    lines = [
        "# Core and memory experiment",
        "",
        "| Run | Variant | Steps | Train off/on NLL | Valid off | Valid on | Valid gated | Valid gap | Calls/token |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path in sorted(Path(root).glob("*/final.json")):
        if not (path.parent / "config.json").exists():
            continue
        config = json.loads((path.parent / "config.json").read_text())
        if "model" not in config or "lookup_budget" not in config["model"]:
            continue
        result = json.loads(path.read_text())
        train = result["train"]
        valid = result["valid"]
        values = [
            path.parent.name,
            config["variant"],
            str(result["step"]),
            f"{train['modes']['off']['nll']:.4f} / {train['modes']['on']['nll']:.4f}",
            *[f"{valid['modes'][m]['nll']:.4f}" for m in ("off", "on", "gated")],
            f"{valid['off_minus_on_nll']:.4f}",
            f"{valid['gated_calls_per_token']:.3f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "Lower NLL is better. Gap = off minus on; positive means memory helps.",
        "Core-only runs have untrained memory/gates: use their off score.",
        "Calls count memory contributions, not actual avoided computation.",
        "Only compare runs with matching corpus, context, width, seeds, training exposure and evaluation settings.",
        "A train-only advantage can reflect memorization. Even a validation advantage does not establish fact/abstraction separation.",
        "The core control has fewer active parameters; it is not a parameter-matched baseline.",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    result = report(a.root)
    (a.root / "report.md").write_text(result)
    print(result)


if __name__ == "__main__":
    main()
