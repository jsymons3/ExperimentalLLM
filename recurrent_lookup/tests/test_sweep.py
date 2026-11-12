"""Checks on the sweep planning and the report."""

import json

from recurrent_lookup.report import build_report
from recurrent_lookup.sweep import EVALS_PER_RUN, RUNS, plan_steps


def test_plan_fits_the_budget():
    variants = {v for v, _ in RUNS}
    timings = {
        v: {"sec_per_step": 0.02 + 0.01 * i, "eval_sec": 2.0, "final_eval_sec": 20.0}
        for i, v in enumerate(sorted(variants))
    }
    budget = 3600.0
    steps = plan_steps(timings, budget)
    total = sum(
        steps * timings[v]["sec_per_step"]
        + EVALS_PER_RUN * timings[v]["eval_sec"]
        + timings[v]["final_eval_sec"]
        for v, _ in RUNS
    )
    assert steps > 0 and steps % 100 == 0
    assert total <= budget
    assert plan_steps(timings, 60.0) == 0


def test_ten_runs_with_three_seeds_for_a_and_c():
    assert len(RUNS) == len(set(RUNS)) == 10
    assert sum(v == "A" for v, _ in RUNS) == sum(v == "C" for v, _ in RUNS) == 3


def fake_final(variant, seed, held_second, lookup):
    splits = ["one_hop", "two_hop_train", "two_hop_id", "two_hop_held_first",
              "two_hop_held_second", "two_hop_held_both"]
    acc = {s: 0.9 for s in splits}
    acc["two_hop_held_second"] = held_second
    matrix = [[0.1, 0.1], [0.8, 0.2]]
    return {
        "variant": variant, "seed": seed, "steps": 1000, "sec_per_step": 0.03,
        "params": {"total": 123456},
        "accuracy": acc,
        "accuracy_by_passes": {"1": acc, "4": acc},
        "table": {"used_frac": 0.7} if lookup else None,
        "overlap": (
            {s: {"same": matrix, "control": [[0.05, 0.05], [0.1, 0.1]]} for s in splits[1:]}
            if lookup else None
        ),
    }


def test_report_from_partial_results(tmp_path):
    assert "No finished runs" in build_report(tmp_path)
    for variant, seed, held, lookup in [("A", 0, 0.01, False), ("C", 0, 0.5, True), ("C", 1, 0.7, True)]:
        run_dir = tmp_path / f"{variant}_s{seed}"
        run_dir.mkdir()
        (run_dir / "final.json").write_text(json.dumps(fake_final(variant, seed, held, lookup)))
    text = build_report(tmp_path)
    assert "0.600 (0.500-0.700)" in text  # C, mean and range over two seeds
    # pass 2 of the chain vs pass 1 of the direct question, then its control
    assert "| C | held_second | 0.800 (0.800-0.800) | 0.100 (0.100-0.100) |" in text
    assert "lookup every pass (C): 0.600" in text
    assert "(B)" not in text  # B has not run, so no line about it
