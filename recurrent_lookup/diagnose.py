"""Read-only checks on trained runs.

    python -m recurrent_lookup.diagnose runs/overnight/B_s0 runs/overnight/C_s0

For each run directory this loads the checkpoint, changes nothing, and reports:

1. Whether the checkpoint reproduces the accuracies stored in final.json.
2. The two-call diagnostic: a normal run against two separate one-hop calls,
   where the second call is given the first call's predicted entity.
3. What the answer position holds after each pass (or outer step): how often
   its top entity is the bridge, and how often it is the answer.

The tables are printed and saved as diagnostics.md and diagnostics.json in the
run directory.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from .evaluate import (
    accuracy_by_split,
    decode_by_depth,
    default_depth,
    eval_mode,
    slot_contents,
    two_hop_sets,
    uses_slots,
)
from .loading import load_run
from .report import table
from .two_call import diagnose as two_call_diagnose

GROUP_LABELS = {
    "two_hop_train": "trained chain",
    "two_hop_id": "unseen chain",
    "two_hop_held_first": "first fact held out",
    "two_hop_held_second": "second fact held out",
    "two_hop_held_both": "both facts held out",
}


def pct(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def check_reproduces(run_dir, model, data, step):
    """Compare accuracies recomputed from the checkpoint with final.json."""
    final_path = Path(run_dir) / "final.json"
    if not final_path.exists():
        return {"status": "skipped", "reason": "no final.json in the run directory"}
    final = json.loads(final_path.read_text())
    if final["steps"] != step:
        return {
            "status": "skipped",
            "reason": f"checkpoint is at step {step}, final.json at step {final['steps']}",
        }
    with eval_mode(model):
        recomputed = accuracy_by_split(model, data)
    stored = final["accuracy"]
    worst = max(abs(recomputed[k] - stored[k]) for k in recomputed)
    return {
        "status": "identical" if worst == 0 else "differs",
        "max_abs_difference": worst,
        "recomputed": recomputed,
        "stored": stored,
    }


def diagnose_run(run_dir):
    cfg, data, model, step = load_run(run_dir)
    slots = uses_slots(model)
    depth = default_depth(model)
    unit = "steps" if slots else "passes"
    call_depths = sorted({max(1, depth // 2), depth})
    result = {
        "run": str(run_dir),
        "variant": cfg.variant,
        "seed": cfg.seed,
        "step": step,
        "depth_unit": unit,
        "normal_depth": depth,
        "call_depths": call_depths,
        "device": str(mx.default_device()),
        "reproduces_final": check_reproduces(run_dir, model, data, step),
        "two_call": two_call_diagnose(model, data, call_depths, (depth,)),
    }
    with eval_mode(model):
        result["decode"] = {
            name: decode_by_depth(model, data, rows, max(depth, 4) if slots else depth)
            for name, rows in two_hop_sets(data).items()
        }
        if slots:
            result["slots"] = {
                name: slot_contents(model, data, rows)
                for name, rows in two_hop_sets(data).items()
            }
    return result


def to_markdown(result):
    unit, depth = result["depth_unit"], result["normal_depth"]
    out = [
        f"## {result['run']} ({result['variant']}, seed {result['seed']}, "
        f"step {result['step']})",
        "",
    ]

    check = result["reproduces_final"]
    if check["status"] == "skipped":
        out.append(f"Checkpoint against final.json: not compared ({check['reason']}).")
    elif check["status"] == "identical":
        out.append("Checkpoint against final.json: every accuracy is identical.")
    else:
        out.append(
            "Checkpoint against final.json: DIFFERS, by up to "
            f"{100 * check['max_abs_difference']:.2f} points."
        )
    out.append("")

    matched = str(result["call_depths"][0])
    rows = []
    for name, by_depth in result["two_call"].items():
        known = by_depth[matched]["both_facts_known"]
        chained = by_depth[matched]["on_both_known"]["native"][str(depth)]
        rows.append(
            [GROUP_LABELS[name], pct(by_depth[matched]["native"][str(depth)]["accuracy"])]
            + [pct(by_depth[str(d)]["two_call"]["accuracy"]) for d in result["call_depths"]]
            + [f"{known['correct']:,} of {known['n']:,}", f"{chained['correct']:,}"]
        )
    out += [
        "### Two-call diagnostic",
        "",
        table(
            ["chain type", f"normal, {depth} {unit}"]
            + [f"two calls, {d} {unit} each" for d in result["call_depths"]]
            + [f"both facts recalled ({matched} {unit} each)", "of those, chained in a normal run"],
            rows,
        ),
        "",
        "The second call is given the first call's predicted entity, not the "
        "true one.",
        "",
    ]

    rows = []
    for name, decode in result["decode"].items():
        for i, d in enumerate(decode["depth"]):
            row = [GROUP_LABELS[name], d, pct(decode["bridge"][i]), pct(decode["answer"][i])]
            if "first_hop_position_bridge" in decode:
                row.append(pct(decode["first_hop_position_bridge"][i]))
            rows.append(row)
    header = ["chain type", unit[:-2] if unit == "passes" else "step"]
    header += ["answer position: bridge", "answer position: answer"]
    if not result.get("slots"):
        header.append("first-hop position: bridge")
    out += [
        f"### What the answer position holds after each {header[1]}",
        "",
        table(header, rows),
        "",
        "Share of chains whose top entity, read through the output head, is the "
        "bridge b or the answer c."
        + (
            ""
            if result.get("slots")
            else " The first-hop position is the r1 token, which computes "
            "exactly what the one-hop question [h, r1] computes."
        ),
        "",
    ]

    if result.get("slots"):
        rows = [
            [GROUP_LABELS[name], pct(s["bridge_in_slot"]), pct(s["bridge_in_readout"]), f"{s['write_weight']:.2f}"]
            for name, s in result["slots"].items()
        ]
        out += [
            "### Slot contents after step 1",
            "",
            table(["chain type", "bridge in slot", "bridge in step-1 readout", "write weight"], rows),
            "",
        ]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", type=Path, help="run directories")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    if args.cpu:
        mx.set_default_device(mx.cpu)
    for run_dir in args.runs:
        result = diagnose_run(run_dir)
        text = to_markdown(result)
        (run_dir / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
        (run_dir / "diagnostics.md").write_text(text)
        print(text, flush=True)


if __name__ == "__main__":
    main()
