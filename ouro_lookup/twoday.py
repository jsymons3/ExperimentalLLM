"""The two-day run: take over eight layers, then test flexibility.

    python -m ouro_lookup.twoday --hours 44 --out runs/ouro-twoday

1. A quick check of both kinds of run at small size (minutes). ``--check-only``
   stops there.
2. Times a few steps of each kind of run. ``--estimate-only`` stops after
   printing the plan.
3. Gives each take-over run the same number of steps: as many as fit in
   ``--takeover-hours``, or fewer if the whole plan would exceed ``--hours``.
   The flexibility runs stop by themselves once their facts are learned, so
   the plan counts their longest possible time.
4. Trains each run in its own process, in the order below, skipping any that
   already finished. Re-run the same command to pick up where it stopped.
5. Writes report.md after every finished run.

The order puts the main comparison of each part first, so that an
interrupted run still has it.
"""

import argparse
import fcntl
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from recurrent_lookup.sweep import run_logged

from . import flex
from .twoday_report import write_report

EVALS_PER_RUN = 40
FLEX_MAIN = ["table", "adapters"]
FLEX_MORE = ["adapters_matched", "table_fixed_keys"]
# Not in the default list; ``--only takeover_table_large`` runs one.
EXTRA_TAKEOVERS = ["table_adapters", "table_large"]
MIN_TAKEOVER_STEPS = 2000


@dataclass(frozen=True)
class Run:
    kind: str  # "takeover", "retain" or "edit"
    variant: str
    seed: int = 0

    @property
    def name(self):
        return f"{self.kind}_{self.variant}" + ("" if self.kind == "takeover" else f"_s{self.seed}")

    @property
    def timing_key(self):
        """Runs that cost the same per step share one timing."""
        if self.kind == "takeover":
            return f"takeover_{self.variant}"
        return "flex_" + self.variant.replace("table_fixed_keys", "table")


def run_list(seeds):
    """Every run, most informative first."""
    first, rest = seeds[0], seeds[1:]

    def flex_runs(variants, seed):
        return [Run(kind, v, seed) for kind in ("retain", "edit") for v in variants]

    runs = [Run("takeover", v) for v in ("table", "dense", "adapters")]
    runs += flex_runs(FLEX_MAIN, first) + flex_runs(FLEX_MORE, first)
    runs += [Run("takeover", v) for v in ("table_small", "dense_wide")]
    for seed in rest:
        runs += flex_runs(FLEX_MAIN, seed) + flex_runs(FLEX_MORE, seed)
    return runs


def command(run, out, common, extra):
    module = "ouro_lookup.takeover" if run.kind == "takeover" else "ouro_lookup.flex"
    cmd = [sys.executable, "-m", module, "--variant", run.variant, "--out", str(out)]
    if run.kind != "takeover":
        cmd += ["--mode", run.kind]
    return cmd + ["--seed", str(run.seed), *common, *extra[run.kind != "takeover"]]


def quick_check(out, common, extra):
    """A short take-over run and a small retention run on the real model."""
    check_dir = out / "_check"
    verdict_path = check_dir / "verdict.json"
    if verdict_path.exists():
        return json.loads(verdict_path.read_text())
    print("\n=== quick check: does each kind of run train on this machine? ===", flush=True)
    jobs = {
        "takeover": command(Run("takeover", "table"), check_dir / "takeover", common, extra) + [
            "--steps", "40", "--batch-size", "4", "--seq-len", "48", "--warmup", "10",
            "--eval-every", "20", "--log-every", "10", "--eval-windows", "32",
            "--periodic-windows", "8",
        ],
        "retain": command(Run("retain", "table"), check_dir / "retain", common, extra) + [
            "--n-people", "24", "--batches", "2", "--phase-steps", "400", "--warmup", "50",
        ],
    }
    for name, cmd in jobs.items():
        (check_dir / name).mkdir(parents=True, exist_ok=True)
        code = run_logged(cmd, check_dir / name / "train.log")
        if code != 0:
            raise SystemExit(
                f"the quick check of the {name} run failed (exit {code}); "
                f"see {check_dir / name / 'train.log'}"
            )
    for checkpoint in check_dir.glob("*/checkpoint.safetensors"):
        checkpoint.unlink()  # the full-size table is 2 GB and is not needed again
    records = [
        json.loads(line)
        for line in (check_dir / "takeover" / "metrics.jsonl").read_text().splitlines()
    ]
    held = json.loads((check_dir / "takeover" / "final.json").read_text())["held_out"]
    retain = json.loads((check_dir / "retain" / "final.json").read_text())["summary"]
    verdict = {
        "takeover_loss_first": records[0]["loss"],
        "takeover_loss_last": records[-1]["loss"],
        "held_out": {k: held[k] for k in ("original", "nothing", "trained")},
        "retain_learned": retain["learned"],
        "retain_kept": retain["kept_at_end"],
    }
    # The take-over loss should fall and the removal should hurt; the small
    # table should learn two batches of 72 facts.
    verdict["passed"] = bool(
        verdict["takeover_loss_last"] < verdict["takeover_loss_first"]
        and held["nothing"] > held["original"]
        and retain["learned"] >= 0.9
    )
    verdict_path.write_text(json.dumps(verdict, indent=2))
    return verdict


def pilot(out, runs, common, extra, steps):
    timings = {}
    for run in runs:
        key = run.timing_key
        if key in timings:
            continue
        pilot_dir = out / "_pilot" / key
        timing_path = pilot_dir / "timing.json"
        if not timing_path.exists():
            print(f"\n=== timing {key} ===", flush=True)
            pilot_dir.mkdir(parents=True, exist_ok=True)
            cmd = command(run, pilot_dir, common, extra) + ["--time-only", str(steps)]
            code = run_logged(cmd, pilot_dir / "train.log")
            if code != 0:
                raise SystemExit(f"timing run for {key} failed (exit {code})")
        timings[key] = json.loads(timing_path.read_text())
    return timings


def flex_limits(extra):
    """The step limits the flexibility runs will use, from their defaults and
    any overrides passed through."""
    args = flex.parse_args(["--out", "x", *extra])
    cfg = flex.config_from_args(args)
    return cfg


def longest_minutes(run, timing, limits):
    """The longest a flexibility run can take: every phase to its step limit."""
    # ``eval_sec`` is the time to test every fact once.
    if run.kind == "retain":
        steps = limits.batches * limits.phase_steps
        # A check tests one batch; after each phase every batch is tested.
        tests = steps / limits.check_every / limits.batches + limits.batches
    else:
        steps = limits.joint_steps + limits.edit_steps
        tests = limits.joint_steps / (2 * limits.check_every) + 1.2 * limits.edit_steps / limits.edit_check
    return (steps * timing["sec_per_step"] + tests * timing["eval_sec"]) / 60


def takeover_minutes(timing, steps):
    seconds = steps * timing["sec_per_step"] + EVALS_PER_RUN * timing["eval_sec"]
    return (seconds + timing["final_eval_sec"]) / 60


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="runs/ouro-twoday")
    p.add_argument("--hours", type=float, default=44.0, help="total time budget")
    p.add_argument(
        "--takeover-hours", type=float, default=4.5, help="most time one take-over run may get"
    )
    p.add_argument("--steps", type=int, help="fix the step count of every take-over run")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="for the flexibility runs")
    p.add_argument("--only", nargs="+", metavar="RUN", help="run only these, in this order")
    p.add_argument("--pilot-steps", type=int, default=10)
    p.add_argument("--skip-check", action="store_true", help="skip the quick check")
    p.add_argument("--check-only", action="store_true", help="run the quick check and stop")
    p.add_argument("--estimate-only", action="store_true", help="time, print the plan, stop")
    p.add_argument("--weights", help="directory with Ouro's files")
    p.add_argument("--text", help="directory with the prepared text")
    p.add_argument("--dtype", help="precision Ouro runs in")
    p.add_argument("--takeover-extra", default="", help="more arguments for every take-over run")
    p.add_argument("--flex-extra", default="", help="more arguments for every flexibility run")
    args = p.parse_args(argv)

    seeds = list(dict.fromkeys(args.seeds))
    runs = run_list(seeds)
    if args.only:
        by_name = {run.name: run for run in runs + [Run("takeover", v) for v in EXTRA_TAKEOVERS]}
        unknown = [name for name in args.only if name not in by_name]
        if unknown:
            raise SystemExit(f"unknown runs {unknown}; choose from {list(by_name)}")
        runs = [by_name[name] for name in args.only]
    common = []
    if args.weights:
        common += ["--weights", args.weights]
    if args.dtype:
        common += ["--dtype", args.dtype]
    text = ["--text", args.text] if args.text else []
    # Index 0: arguments for take-over runs; 1: for flexibility runs.
    extra = (text + shlex.split(args.takeover_extra), shlex.split(args.flex_extra))

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
        verdict = quick_check(out, common, extra)
        held = verdict["held_out"]
        print(
            f"\nquick check: take-over loss {verdict['takeover_loss_first']:.3f} -> "
            f"{verdict['takeover_loss_last']:.3f}; held-out text loss {held['original']:.3f} "
            f"original, {held['nothing']:.3f} with nothing put back; a small table learned "
            f"{100 * verdict['retain_learned']:.0f}% of two batches of facts: "
            + ("passed" if verdict["passed"] else "FAILED")
        )
        if not verdict["passed"]:
            print("The runs below go ahead, but read their first evaluations with care.")
    if args.check_only:
        return

    plan_path = out / "plan.json"
    signature = {
        "runs": [run.name for run in runs],
        "extra": [common, *extra],
        "budget": {"hours": args.hours, "takeover_hours": args.takeover_hours, "steps": args.steps},
    }
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["signature"] != signature:
            raise SystemExit(f"{plan_path} was made with different settings; use a new --out directory")
        print(f"reusing {plan_path}: {plan['steps']} steps per take-over run")
    else:
        timings = pilot(out, runs, common, extra, args.pilot_steps)
        limits = flex_limits(extra[1])
        minutes = {
            run.name: longest_minutes(run, timings[run.timing_key], limits)
            for run in runs if run.kind != "takeover"
        }
        takeovers = [run for run in runs if run.kind == "takeover"]
        steps, source = 0, "no take-over runs"
        if takeovers and args.steps:
            steps, source = args.steps, "--steps"
        elif takeovers:
            spent = time.perf_counter() - started
            left = args.hours * 3600 - spent - 60 * sum(minutes.values())
            each = min(args.takeover_hours * 3600, left / len(takeovers))
            # One step count for all, set by the slowest of them.
            fits = []
            for run in takeovers:
                t = timings[run.timing_key]
                fixed = EVALS_PER_RUN * t["eval_sec"] + t["final_eval_sec"]
                fits.append(int((each - fixed) / t["sec_per_step"]))
            steps = max(0, min(fits)) // 100 * 100
            source = f"{each / 3600:.1f} hours for the slowest take-over run"
            if steps < MIN_TAKEOVER_STEPS:
                raise SystemExit(
                    f"the budget allows only {steps} steps per take-over run; raise --hours, "
                    "run fewer runs with --only, or pass --steps"
                )
        for run in takeovers:
            minutes[run.name] = takeover_minutes(timings[run.timing_key], steps)
        plan = {
            "steps": steps, "source": source, "estimated_minutes": minutes,
            "timings": timings, "signature": signature,
        }
        plan_path.write_text(json.dumps(plan, indent=2))

    steps = plan["steps"]
    print(f"\nplan: {len(runs)} runs; {steps} steps for each take-over run ({plan['source']})")
    for run in runs:
        timing = plan["timings"][run.timing_key]
        note = "" if run.kind == "takeover" else " at most"
        print(
            f"  {run.name:34s}{note:8s} {plan['estimated_minutes'][run.name]:6.0f} min "
            f"({timing['tokens_per_sec']:.0f} tokens/s, peak {timing['peak_memory_gb']:.1f} GB)"
        )
    total = sum(plan["estimated_minutes"][run.name] for run in runs)
    print(f"  {'total':34s} at most {total / 60:6.1f} hours, in the order listed")
    if args.estimate_only:
        return

    failed = []
    for run in runs:
        run_dir = out / run.name
        if (run_dir / "final.json").exists():
            print(f"\n=== {run.name}: already done, skipping ===")
            continue
        print(f"\n=== {run.name} ===", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = command(run, run_dir, common, extra)
        if run.kind == "takeover":
            cmd += ["--steps", str(steps), "--eval-every", str(max(1, steps // EVALS_PER_RUN))]
            cmd += ["--checkpoint-minutes", "30"]
        if (run_dir / "checkpoint.safetensors").exists():
            cmd.append("--resume")
        code = run_logged(cmd, run_dir / "train.log")
        if code != 0:
            failed.append(run.name)
            print(f"{run.name} exited with code {code}; continuing")
            if code == 130:  # interrupted by the user
                break
        write_report(out)

    print()
    print(write_report(out))
    if failed:
        raise SystemExit(f"runs that did not finish: {', '.join(failed)}")


if __name__ == "__main__":
    main()
