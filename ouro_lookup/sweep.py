"""Run the stage 1 runs one after another.

    python -m ouro_lookup.sweep --hours 8 --out runs/ouro-lookup

1. A quick check: can the frozen model learn 64 facts through the table in a
   few hundred steps? Takes minutes. If it fails, the sweep says so and
   carries on, since the longer runs may still learn.
2. Times a few steps of each run and prints how long each should take.
   ``--estimate-only`` stops there.
3. Uses one step count for every run: ``--steps`` if given, otherwise the
   largest count that fits ``--hours`` (never more than ``--max-steps``).
4. Trains each run in its own process, in the order below, skipping any that
   already finished. Re-run the same command to pick up where it stopped.
5. Writes report.md after every finished run.

Arguments this script does not recognise are passed through to every training
run, for example ``--lr 3e-4`` or ``--weights data/ouro-tiny``.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from recurrent_lookup.sweep import EVALS_PER_RUN, plan_steps, run_logged, run_minutes

from .configs import VARIANTS
from .data import PRESETS
from .report import write_report

# In priority order.
RUNS = [
    "table",  # frozen Ouro plus the table, written note
    "table_adapters",  # the same with adapters
    "latent",  # two-step questions in one pass through Ouro's own loops
    "adapters",  # control: no table
]
EXTRAS = [v for v in VARIANTS if v not in RUNS]
CHECK_STEPS = 300


def train_command(preset, variant, out, extra):
    return [
        sys.executable, "-m", "ouro_lookup.train",
        "--preset", preset, "--variant", variant, "--out", str(out), *extra,
    ]


def quick_check(out, extra, steps):
    """Train the table on a handful of facts and report whether it took."""
    check_dir = out / "_check"
    verdict_path = check_dir / "verdict.json"
    if verdict_path.exists():
        return json.loads(verdict_path.read_text())
    print("\n=== quick check: can the frozen model learn a few facts through the table? ===", flush=True)
    check_dir.mkdir(parents=True, exist_ok=True)
    cmd = train_command("check", "table", check_dir, extra)
    cmd += ["--steps", str(steps), "--eval-every", str(max(1, steps // 6))]
    cmd += ["--two-hop-frac", "0", "--warmup", "50"]  # one-step facts only, and a short ramp
    code = run_logged(cmd, check_dir / "train.log")
    if code != 0:
        raise SystemExit(f"the quick check did not run (exit {code}); see {check_dir / 'train.log'}")
    final = json.loads((check_dir / "final.json").read_text())
    verdict = {
        "facts": final["facts"]["facts"],
        "steps": final["steps"],
        "table_on": final["accuracy"]["one_hop"],
        "table_off": final["accuracy_original"]["one_hop"],
    }
    verdict["passed"] = verdict["table_on"] >= 0.9 and verdict["table_off"] <= 0.1
    verdict_path.write_text(json.dumps(verdict, indent=2))
    return verdict


def pilot(preset, out, pilot_steps, extra, runs):
    timings = {}
    for variant in runs:
        pilot_dir = out / "_pilot" / variant
        timing_path = pilot_dir / "timing.json"
        if not timing_path.exists():
            print(f"\n=== timing {variant} ===", flush=True)
            pilot_dir.mkdir(parents=True, exist_ok=True)
            cmd = train_command(preset, variant, pilot_dir, extra)
            code = run_logged(cmd + ["--time-only", str(pilot_steps)], pilot_dir / "train.log")
            if code != 0:
                raise SystemExit(f"timing run for {variant} failed (exit {code})")
        timings[variant] = json.loads(timing_path.read_text())
    return timings


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", choices=list(PRESETS), default="full")
    p.add_argument("--out", default="runs/ouro-lookup")
    p.add_argument("--hours", type=float, default=8.0, help="total time budget")
    p.add_argument("--steps", type=int, help="fix the step count for every run")
    p.add_argument("--max-steps", type=int, default=20000, help="cap for a time-budgeted plan")
    p.add_argument("--min-steps", type=int, default=1000, help="refuse plans shorter than this")
    p.add_argument("--pilot-steps", type=int, default=30)
    p.add_argument("--only", nargs="+", metavar="RUN", help="run only these, in this order")
    p.add_argument("--skip-check", action="store_true", help="skip the quick check")
    p.add_argument("--check-steps", type=int, default=CHECK_STEPS)
    p.add_argument("--check-only", action="store_true", help="run the quick check and stop")
    p.add_argument("--estimate-only", action="store_true", help="time, print the estimate, stop")
    args, extra = p.parse_known_args(argv)

    runs = args.only or RUNS
    unknown = [r for r in runs if r not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown runs {unknown}; choose from {RUNS + EXTRAS}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Hold this for the sweep's lifetime so the same sweep cannot start twice.
    lock = (out / "sweep.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another sweep is already using {out}") from None
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()

    started = time.perf_counter()
    if not args.skip_check:
        verdict = quick_check(out, extra, args.check_steps)
        print(
            f"\nquick check: one-step recall {100 * verdict['table_on']:.0f}% with the table, "
            f"{100 * verdict['table_off']:.0f}% without, on {verdict['facts']} facts after "
            f"{verdict['steps']} steps: " + ("passed" if verdict["passed"] else "FAILED")
        )
        if not verdict["passed"]:
            print(
                "The frozen model did not learn a few facts through the table in the time "
                "given. The runs below go ahead, but the 'table' run may not learn either; "
                "watch its first evaluations."
            )
    if args.check_only:
        return

    plan_path = out / "plan.json"
    signature = {
        "preset": args.preset,
        "runs": runs,
        "extra": extra,
        "budget": {"hours": args.hours, "steps": args.steps, "max_steps": args.max_steps},
    }
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["signature"] != signature:
            raise SystemExit(
                f"{plan_path} was made with different settings; use a new --out directory"
            )
        print(f"reusing {plan_path}: {plan['steps']} steps per run")
    else:
        timings = pilot(args.preset, out, args.pilot_steps, extra, runs)
        pairs = [(variant, 0) for variant in runs]
        if args.steps:
            plan = {"steps": args.steps, "source": "--steps"}
        else:
            budget = args.hours * 3600 - (time.perf_counter() - started)
            steps = min(plan_steps(timings, budget, pairs), args.max_steps)
            if steps < args.min_steps:
                raise SystemExit(
                    f"a {args.hours}h budget allows only {steps} steps per run; raise --hours, "
                    "run fewer runs with --only, or pass --steps"
                )
            plan = {"steps": steps, "source": f"{args.hours}h budget, at most {args.max_steps}"}
        plan["estimated_minutes"] = {v: run_minutes(t, plan["steps"]) for v, t in timings.items()}
        plan["timings"] = timings
        plan["signature"] = signature
        plan_path.write_text(json.dumps(plan, indent=2))

    steps = plan["steps"]
    print(f"\nplan: {steps} steps for each of {len(runs)} runs ({plan['source']})")
    for variant in runs:
        timing = plan["timings"][variant]
        print(
            f"  {variant:16s} about {plan['estimated_minutes'][variant]:6.1f} min "
            f"({timing['tokens_per_sec']:.0f} tokens/s, peak {timing['peak_memory_gb']:.1f} GB)"
        )
    total = sum(plan["estimated_minutes"][v] for v in runs)
    print(f"  total            about {total / 60:6.1f} hours, in the order listed")
    if args.estimate_only:
        return

    failed = []
    for variant in runs:
        run_dir = out / variant
        if (run_dir / "final.json").exists():
            print(f"\n=== {variant}: already done, skipping ===")
            continue
        print(f"\n=== {variant} ===", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = train_command(args.preset, variant, run_dir, extra)
        cmd += ["--steps", str(steps), "--eval-every", str(max(1, steps // EVALS_PER_RUN))]
        if (run_dir / "checkpoint.safetensors").exists():
            cmd.append("--resume")
        code = run_logged(cmd, run_dir / "train.log")
        if code != 0:
            failed.append(variant)
            print(f"{variant} exited with code {code}; continuing")
            if code == 130:  # interrupted by the user
                break
        write_report(out)

    print()
    print(write_report(out))
    if failed:
        raise SystemExit(f"runs that did not finish: {', '.join(failed)}")


if __name__ == "__main__":
    main()
