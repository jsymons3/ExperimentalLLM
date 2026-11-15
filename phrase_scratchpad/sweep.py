"""Run the three stages and their controls one after another.

    python -m phrase_scratchpad.sweep --preset full --hours 8 --out runs/phrase-scratchpad

1. Times a few hundred steps of each run and prints how long each should take.
   ``--estimate-only`` stops there.
2. Uses one step count for every run: ``--steps`` if given, otherwise the
   largest count that fits ``--hours`` (never more than ``--max-steps``).
3. Trains each run in its own process, in the order below, skipping any that
   already finished. Re-run the same command to pick up where it stopped.
4. Writes report.md after every finished run.

``--reading`` runs the reading test instead of the three stages, and
``--seeds 0 1`` repeats every run with each seed (all runs with the first seed
come first).

Arguments this script does not recognise are passed through to every training
run, for example ``--lr 3e-4`` or ``--n-people 800``.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from recurrent_lookup.sweep import EVALS_PER_RUN, plan_steps, run_logged, run_minutes

from .report import write_report
from .tasks import PRESETS

# In priority order. Stages 1 and 2 first: each task alone, with the
# scratchpad that worked on the graph task. Then stage 3: all three together,
# with the intermediate answer taught for arithmetic and stories only.
RUNS = [
    "maths",  # stage 1: arithmetic
    "clutrr",  # stage 1: family stories
    "facts",  # stage 2
    "mix_split",  # stage 3, table switched off on arithmetic and stories
    "facts_unsupervised",  # control: the same facts on their own
]
# Left out by default, because every extra run shortens all of them.
EXTRAS = ["mix"]  # stage 3 with the table always on: does the switch matter?
BASELINES = ["maths_plain", "clutrr_plain", "facts_plain"]  # no scratchpad
# The reading test: the taught-note run and the untaught control, each next
# to the same run with known facts also asked through the note. The new runs
# come first in each pair.
READING = ["facts_reader", "facts", "facts_unsupervised_reader", "facts_unsupervised"]
KNOWN = list(dict.fromkeys(RUNS + EXTRAS + BASELINES + READING))


def run_name(variant, seed, seeds):
    """The directory of one run. A sweep with seed 0 alone keeps bare names."""
    return variant if list(seeds) == [0] else f"{variant}_s{seed}"


def train_command(preset, variant, out, extra):
    return [
        sys.executable, "-m", "phrase_scratchpad.train",
        "--preset", preset, "--variant", variant, "--out", str(out), *extra,
    ]


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
    p.add_argument("--out", default="runs/phrase-scratchpad")
    p.add_argument("--hours", type=float, default=8.0, help="total time budget")
    p.add_argument("--steps", type=int, help="fix the step count for every run")
    p.add_argument("--max-steps", type=int, default=64000, help="cap for a time-budgeted plan")
    p.add_argument("--pilot-steps", type=int, default=200)
    p.add_argument("--only", nargs="+", metavar="RUN", help="run only these, in this order")
    p.add_argument("--with-mix", action="store_true", help="add stage 3 with the table always on")
    p.add_argument("--with-baselines", action="store_true", help="add the no-scratchpad runs")
    p.add_argument("--reading", action="store_true", help="run the reading test instead")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], metavar="SEED")
    p.add_argument("--estimate-only", action="store_true", help="time, print the estimate, stop")
    args, extra = p.parse_known_args(argv)

    runs = args.only or (
        READING
        if args.reading
        else RUNS + (EXTRAS if args.with_mix else []) + (BASELINES if args.with_baselines else [])
    )
    unknown = [r for r in runs if r not in KNOWN]
    if unknown:
        raise SystemExit(f"unknown runs {unknown}; choose from {KNOWN}")
    seeds = list(dict.fromkeys(args.seeds))
    # Every run with the first seed, then every run with the next.
    pairs = [(variant, seed) for seed in seeds for variant in runs]
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

    plan_path = out / "plan.json"
    signature = {
        "preset": args.preset,
        "runs": runs,
        "extra": extra,
        "budget": {"hours": args.hours, "steps": args.steps, "max_steps": args.max_steps},
    }
    if seeds != [0]:
        signature["seeds"] = seeds
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["signature"] != signature:
            raise SystemExit(
                f"{plan_path} was made with different settings; use a new --out directory"
            )
        print(f"reusing {plan_path}: {plan['steps']} steps per run")
    else:
        started = time.perf_counter()
        timings = pilot(args.preset, out, args.pilot_steps, extra, runs)
        if args.steps:
            plan = {"steps": args.steps, "source": "--steps"}
        else:
            budget = args.hours * 3600 - (time.perf_counter() - started)
            steps = min(plan_steps(timings, budget, pairs), args.max_steps)
            if steps < 1000:
                raise SystemExit(
                    f"a {args.hours}h budget allows only {steps} steps per run; "
                    "raise --hours, pick a smaller --preset, or pass --steps"
                )
            plan = {"steps": steps, "source": f"{args.hours}h budget, at most {args.max_steps}"}
        plan["estimated_minutes"] = {v: run_minutes(t, plan["steps"]) for v, t in timings.items()}
        plan["timings"] = timings
        plan["signature"] = signature
        plan_path.write_text(json.dumps(plan, indent=2))

    steps = plan["steps"]
    print(f"\nplan: {steps} steps for each of {len(pairs)} runs ({plan['source']})")
    each = f" for each of seeds {seeds}" if len(seeds) > 1 else ""
    for variant in runs:
        print(f"  {variant:26s} about {plan['estimated_minutes'][variant]:6.1f} min{each}")
    total = sum(plan["estimated_minutes"][v] for v, _ in pairs)
    print(f"  {'total':26s} about {total / 60:6.1f} hours")
    if args.estimate_only:
        return

    failed = []
    for variant, seed in pairs:
        name = run_name(variant, seed, seeds)
        run_dir = out / name
        if (run_dir / "final.json").exists():
            print(f"\n=== {name}: already done, skipping ===")
            continue
        print(f"\n=== {name} ===", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = train_command(args.preset, variant, run_dir, extra)
        cmd += ["--seed", str(seed)]
        cmd += ["--steps", str(steps), "--eval-every", str(max(1, steps // EVALS_PER_RUN))]
        if (run_dir / "checkpoint.safetensors").exists():
            cmd.append("--resume")
        code = run_logged(cmd, run_dir / "train.log")
        if code != 0:
            failed.append(name)
            print(f"{name} exited with code {code}; continuing")
            if code == 130:  # interrupted by the user
                break
        write_report(out)

    print()
    print(write_report(out))
    if failed:
        raise SystemExit(f"runs that did not finish: {', '.join(failed)}")


if __name__ == "__main__":
    main()
