"""Run the ten-run comparison (A-E) inside a time budget.

    python -m recurrent_lookup.sweep --preset full --hours 7.5 --out runs/overnight

1. Times a few hundred steps of each variant.
2. Picks one step count, the same for every run, so the whole sweep fits the
   budget. Equal steps keeps the runs comparable; slower variants simply take
   a larger share of the night.
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

from .configs import PRESETS
from .report import write_report

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
EVALS_PER_RUN = 40


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


def pilot(preset, out, pilot_steps, extra):
    """Time each variant once. Returns {variant: timing dict}."""
    timings = {}
    for variant in dict.fromkeys(v for v, _ in RUNS):
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


def plan_steps(timings, budget_sec):
    """Largest common step count whose total estimated time fits the budget."""
    fixed = sum(
        EVALS_PER_RUN * timings[v]["eval_sec"] + timings[v]["final_eval_sec"]
        for v, _ in RUNS
    )
    per_step = sum(timings[v]["sec_per_step"] for v, _ in RUNS)
    steps = int((budget_sec - fixed) / per_step)
    return max(0, steps // 100 * 100)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", choices=list(PRESETS), default="full")
    p.add_argument("--out", default="runs/overnight")
    p.add_argument("--hours", type=float, default=7.5, help="total time budget")
    p.add_argument("--steps", type=int, help="fix the step count; skips timing")
    p.add_argument("--pilot-steps", type=int, default=200)
    args, extra = p.parse_known_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"

    if args.steps:
        plan = {"steps": args.steps, "source": "--steps"}
    elif plan_path.exists():
        plan = json.loads(plan_path.read_text())
        print(f"reusing {plan_path}: {plan['steps']} steps per run")
    else:
        started = time.perf_counter()
        timings = pilot(args.preset, out, args.pilot_steps, extra)
        budget = args.hours * 3600 - (time.perf_counter() - started)
        steps = plan_steps(timings, budget)
        if steps < 1000:
            raise SystemExit(
                f"a {args.hours}h budget allows only {steps} steps per run; "
                "raise --hours, pick a smaller --preset, or pass --steps"
            )
        plan = {
            "steps": steps,
            "source": f"{args.hours}h budget",
            "estimated_minutes": {
                v: (
                    steps * t["sec_per_step"]
                    + EVALS_PER_RUN * t["eval_sec"]
                    + t["final_eval_sec"]
                )
                / 60
                for v, t in timings.items()
            },
            "timings": timings,
        }
    plan_path.write_text(json.dumps(plan, indent=2))

    steps = plan["steps"]
    print(f"\nplan: {steps} steps for each of {len(RUNS)} runs ({plan['source']})")
    for variant, minutes in plan.get("estimated_minutes", {}).items():
        print(f"  {variant:8s} about {minutes:5.1f} min per run")

    failed = []
    for variant, seed in RUNS:
        run_dir = out / f"{variant}_s{seed}"
        if (run_dir / "final.json").exists():
            print(f"\n=== {run_dir.name}: already done, skipping ===")
            continue
        print(f"\n=== {run_dir.name} ===", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = train_command(args.preset, variant, seed, run_dir, extra)
        cmd += ["--steps", str(steps), "--eval-every", str(max(1, steps // EVALS_PER_RUN))]
        if (run_dir / "checkpoint.safetensors").exists():
            cmd.append("--resume")
        code = run_logged(cmd, run_dir / "train.log")
        if code != 0:
            failed.append(run_dir.name)
            print(f"{run_dir.name} exited with code {code}; continuing")
            if code == 130:  # interrupted by the user
                break

    print()
    print(write_report(out))
    if failed:
        raise SystemExit(f"runs that did not finish: {', '.join(failed)}")


if __name__ == "__main__":
    main()
