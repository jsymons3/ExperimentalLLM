"""Run a set of training runs one after another.

The first comparison (runs A-E), sized to a time budget:

    python -m recurrent_lookup.sweep --preset full --hours 7.5 --out runs/overnight

The scratchpad runs, with the step count of the first comparison:

    python -m recurrent_lookup.sweep --runs slots --preset full --steps 128200 \
        --baseline runs/overnight/C_s0 --out runs/slots

1. Times a few hundred steps of each variant and prints how long every run
   should take. ``--estimate-only`` stops there.
2. Uses one step count for every run: ``--steps`` if given, otherwise the
   largest count that fits ``--hours``. Equal steps keeps runs comparable.
3. Runs each configuration in its own process, most important first, skipping
   any that already finished. Re-run the same command to pick up where it
   stopped.
4. Writes report.md.

Arguments this script does not recognise are passed through to every training
run, for example ``--lr 3e-4`` or ``--top-k 16``.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .configs import PRESETS, load_run_config, model_config
from .report import write_report
from .train import config_from_args, parse_args

# In priority order: if the night ends early, the key comparison is done.
RUNS = [
    ("A", 0),
    ("C", 0),
    ("D", 0),
    ("B", 0),
    ("E_small", 0),
    ("E_large", 0),
    ("A", 1),
    ("C", 1),
    ("A", 2),
    ("C", 2),
]
SLOT_RUNS = [
    ("slot_soft", 0),
    ("slot_raw", 0),
    ("slot_soft_prefix", 0),
    ("slot_soft", 1),
    ("slot_soft", 2),
    ("slot_soft_k3", 0),
]
RUN_SETS = {"overnight": RUNS, "slots": SLOT_RUNS}
EVALS_PER_RUN = 40

# Settings that must match the baseline run for results to be comparable.
COMPARED_TRAINING = (
    "steps",
    "batch_size",
    "two_hop_frac",
    "lr",
    "warmup",
    "weight_decay",
    "table_weight_decay",
    "grad_clip",
)
COMPARED_MODEL = (
    "d_model",
    "n_heads",
    "ff",
    "n_subkeys",
    "lookup_heads",
    "top_k",
    "d_key",
    "query_norm",
    "lookup_passes",
    "tie_embeddings",
)


def train_command(preset, variant, seed, out, extra):
    return [
        sys.executable,
        "-m",
        "recurrent_lookup.train",
        "--preset",
        preset,
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--out",
        str(out),
        *extra,
    ]


def run_logged(cmd, log_path):
    """Run cmd, echoing its output and appending it to log_path."""
    with open(log_path, "a") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        interrupted = False
        while True:
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                break
            except KeyboardInterrupt:
                # Ctrl-C reaches the child too; let it save its checkpoint.
                interrupted = True
        code = proc.wait()
        return 130 if interrupted else code


def pilot(preset, out, pilot_steps, extra, runs=RUNS):
    """Time each variant once. Returns {variant: timing dict}."""
    timings = {}
    for variant in dict.fromkeys(v for v, _ in runs):
        pilot_dir = out / "_pilot" / variant
        timing_path = pilot_dir / "timing.json"
        if not timing_path.exists():
            print(f"\n=== timing {variant} ===", flush=True)
            pilot_dir.mkdir(parents=True, exist_ok=True)
            cmd = train_command(preset, variant, 0, pilot_dir, extra)
            code = run_logged(
                cmd + ["--time-only", str(pilot_steps)], pilot_dir / "train.log"
            )
            if code != 0:
                raise SystemExit(f"timing run for {variant} failed (exit {code})")
        timings[variant] = json.loads(timing_path.read_text())
    return timings


def run_minutes(timing, steps):
    """Estimated minutes for one run of this variant."""
    seconds = (
        steps * timing["sec_per_step"]
        + EVALS_PER_RUN * timing["eval_sec"]
        + timing["final_eval_sec"]
    )
    return seconds / 60


def plan_steps(timings, budget_sec, runs=RUNS):
    """Largest common step count whose total estimated time fits the budget."""
    fixed = sum(
        EVALS_PER_RUN * timings[v]["eval_sec"] + timings[v]["final_eval_sec"]
        for v, _ in runs
    )
    per_step = sum(timings[v]["sec_per_step"] for v, _ in runs)
    steps = int((budget_sec - fixed) / per_step)
    return max(0, steps // 100 * 100)


def baseline_differences(base, new):
    """Settings where a planned run differs from the baseline run.

    Compares the graph, the training settings and the model settings that are
    not part of the variant itself. Returns a list of readable differences.
    """
    found = []
    if base.graph != new.graph:
        found.append(f"graph: baseline {base.graph}, this run {new.graph}")
    for name in COMPARED_TRAINING:
        if getattr(base, name) != getattr(new, name):
            found.append(
                f"{name}: baseline {getattr(base, name)}, this run {getattr(new, name)}"
            )
    base_model, new_model = model_config(base), model_config(new)
    for name in COMPARED_MODEL:
        if getattr(base_model, name) != getattr(new_model, name):
            found.append(
                f"model {name}: baseline {getattr(base_model, name)}, "
                f"this run {getattr(new_model, name)}"
            )
    return found


def planned_config(preset, variant, extra, steps):
    """The configuration a run of this variant would be trained with."""
    argv = train_command(preset, variant, 0, "unused", extra)[3:]
    cfg = config_from_args(parse_args(argv + ["--steps", str(steps)]))
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", choices=list(RUN_SETS), default="overnight")
    p.add_argument("--preset", choices=list(PRESETS), default="full")
    p.add_argument("--out", help="default: runs/<name of the run set>")
    p.add_argument("--hours", type=float, default=7.5, help="total time budget")
    p.add_argument("--steps", type=int, help="fix the step count for every run")
    p.add_argument("--pilot-steps", type=int, default=200)
    p.add_argument("--no-timing", action="store_true", help="skip the timing runs")
    p.add_argument(
        "--estimate-only", action="store_true", help="time, print the estimate, stop"
    )
    p.add_argument(
        "--baseline",
        type=Path,
        help="a finished run directory (for example runs/overnight/C_s0) whose "
        "graph, optimizer, schedule, batch and step count these runs must match",
    )
    p.add_argument(
        "--allow-mismatch", action="store_true", help="run even if --baseline differs"
    )
    args, extra = p.parse_known_args(argv)

    runs = RUN_SETS[args.runs]
    out = Path(args.out or f"runs/{args.runs}")
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if args.runs == "slots" and not args.steps and not plan_path.exists():
        raise SystemExit(
            "the scratchpad runs need --steps, set to the step count of the runs "
            "they are compared with (128200 for the first overnight sweep)"
        )
    if args.no_timing and not args.steps:
        raise SystemExit("--no-timing needs --steps")

    if plan_path.exists() and not args.steps:
        plan = json.loads(plan_path.read_text())
        print(f"reusing {plan_path}: {plan['steps']} steps per run")
    else:
        started = time.perf_counter()
        timings = {} if args.no_timing else pilot(args.preset, out, args.pilot_steps, extra, runs)
        if args.steps:
            plan = {"steps": args.steps, "source": "--steps"}
        else:
            budget = args.hours * 3600 - (time.perf_counter() - started)
            steps = plan_steps(timings, budget, runs)
            if steps < 1000:
                raise SystemExit(
                    f"a {args.hours}h budget allows only {steps} steps per run; "
                    "raise --hours, pick a smaller --preset, or pass --steps"
                )
            plan = {"steps": steps, "source": f"{args.hours}h budget"}
        if timings:
            plan["estimated_minutes"] = {
                v: run_minutes(t, plan["steps"]) for v, t in timings.items()
            }
            plan["timings"] = timings
    plan_path.write_text(json.dumps(plan, indent=2))

    steps = plan["steps"]
    print(f"\nplan: {steps} steps for each of {len(runs)} runs ({plan['source']})")
    estimates = plan.get("estimated_minutes", {})
    for variant, minutes in estimates.items():
        count = sum(v == variant for v, _ in runs)
        print(f"  {variant:17s} about {minutes:6.1f} min per run, {count} run(s)")
    if estimates:
        total = sum(estimates[v] for v, _ in runs)
        print(f"  total             about {total / 60:6.1f} hours, run in the order listed above")

    eval_every = max(1, steps // EVALS_PER_RUN)
    if args.baseline:
        base = load_run_config(args.baseline)
        problems = [
            f"{variant}: {difference}"
            for variant in dict.fromkeys(v for v, _ in runs)
            for difference in baseline_differences(
                base, planned_config(args.preset, variant, extra, steps)
            )
        ]
        if problems:
            print(f"\nsettings that differ from the baseline {args.baseline}:")
            for problem in problems:
                print(f"  {problem}")
            if not args.allow_mismatch:
                raise SystemExit(
                    "stopping so the runs stay comparable; "
                    "fix the flags or pass --allow-mismatch"
                )
        else:
            print(f"\nsettings match the baseline {args.baseline}")
    if args.estimate_only:
        return

    failed = []
    for variant, seed in runs:
        run_dir = out / f"{variant}_s{seed}"
        if (run_dir / "final.json").exists():
            print(f"\n=== {run_dir.name}: already done, skipping ===")
            continue
        print(f"\n=== {run_dir.name} ===", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = train_command(args.preset, variant, seed, run_dir, extra)
        cmd += ["--steps", str(steps), "--eval-every", str(eval_every)]
        if (run_dir / "checkpoint.safetensors").exists():
            cmd.append("--resume")
        code = run_logged(cmd, run_dir / "train.log")
        if code != 0:
            failed.append(run_dir.name)
            print(f"{run_dir.name} exited with code {code}; continuing")
            if code == 130:  # interrupted by the user
                break

    print()
    roots = [out] + ([args.baseline.parent] if args.baseline else [])
    print(write_report(roots, out))
    if failed:
        raise SystemExit(f"runs that did not finish: {', '.join(failed)}")


if __name__ == "__main__":
    main()
