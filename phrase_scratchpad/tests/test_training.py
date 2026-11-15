"""End-to-end checks: configurations, training, resuming, the sweep, the report."""

import json

import numpy as np
import pytest

import phrase_scratchpad.clutrr as clutrr
from phrase_scratchpad.configs import VARIANTS, build_config, build_run, load_run_config
from phrase_scratchpad.report import build_report
from phrase_scratchpad.sweep import BASELINES, EXTRAS, READING, RUNS, run_name
from phrase_scratchpad.sweep import main as sweep_main
from phrase_scratchpad.tests.test_tasks import stories
from phrase_scratchpad.train import config_from_args, parse_args, run

TINY = {"d_model": 16, "n_heads": 2, "n_subkeys": 4, "d_key": 8, "top_k": 2, "lookup_heads": 2}


@pytest.fixture
def fake_stories(monkeypatch):
    """Stand-in family stories, so no test needs the downloaded data."""
    many = [dict(s) for _ in range(6) for s in stories()[:3]]
    for i, s in enumerate(many):
        s["story"] = s["story"] + f" [N{i}] ."  # make every story distinct
    monkeypatch.setattr(clutrr, "read_stories", lambda data_dir=None: [dict(s) for s in many])


def tiny(variant, **train):
    settings = dict(steps=6, batch_size=8, warmup=2, eval_every=3, log_every=3, eval_size=12)
    settings.update(train)
    cfg = build_config(variant, "smoke", model=TINY, **settings)
    cfg.task_overrides = {"maths": {"max_operand": 6}, "facts": {"n_people": 40}}
    return cfg


def test_variant_settings():
    assert set(RUNS + EXTRAS + BASELINES + READING) == set(VARIANTS)
    assert RUNS[:3] == ["maths", "clutrr", "facts"]  # each task alone comes first
    for name in ("maths", "clutrr", "facts"):
        cfg = build_config(name)
        assert cfg.tasks == (name,) and cfg.supervised == (name,)
        assert not cfg.table_off and not cfg.one_hop_every_step and not cfg.model.get("core_ff")
    split = build_config("mix_split")
    assert split.tasks == ("maths", "clutrr", "facts")
    assert split.supervised == split.table_off == ("maths", "clutrr")  # never the facts
    mix = build_config("mix")
    assert mix.supervised == ("maths", "clutrr") and mix.table_off == ()
    control = build_config("facts_unsupervised")
    assert control.tasks == ("facts",) and control.supervised == ()
    for cfg in (split, mix, control):  # one architecture and loss for stage 3
        assert cfg.model["core_ff"] == 512 and cfg.one_hop_every_step
    assert build_config("facts_plain").model["scratchpad"] is False
    assert all(build_config(v).hide_subject for v in VARIANTS)
    # A reading-test run is its control with one setting changed.
    for name, base in (("facts_reader", "facts"), ("facts_unsupervised_reader", "facts_unsupervised")):
        reader, plain = build_config(name), build_config(base)
        assert reader.reader_frac == 0.5 and plain.reader_frac == 0.0
        reader.reader_frac, reader.variant = 0.0, base
        assert reader == plain
    assert all(build_config(v).reader_frac == 0.0 for v in RUNS + EXTRAS + BASELINES)
    assert READING == ["facts_reader", "facts", "facts_unsupervised_reader", "facts_unsupervised"]


def test_block_applications_per_step(fake_stories):
    suite, model_cfg = build_run(tiny("mix_split"))
    assert model_cfg.n_passes == 1  # arithmetic and facts: one lookup per step
    assert [t.n_passes for t in suite.tasks] == [None, 2, None]  # stories read with two
    # Only fact questions have their subject hidden, for half the chains.
    assert [t.hide_prob for t in suite.tasks] == [0.0, 0.0, 0.5]
    assert [t.reader_frac for t in suite.tasks] == [0.0, 0.0, 0.0]
    mixed = tiny("mix_split", reader_frac=0.5)  # only fact questions go through the note
    assert [t.reader_frac for t in build_run(mixed)[0].tasks] == [0.0, 0.0, 0.5]
    off = tiny("facts")
    off.hide_subject = False
    assert build_run(off)[0].tasks[0].hide_prob == 0.0
    plain_suite, plain_cfg = build_run(tiny("clutrr_plain"))
    assert plain_cfg.n_passes == 4 and plain_suite.tasks[0].n_passes is None
    with pytest.raises(ValueError):
        build_config("maths", table_off=("maths",))  # needs a dense core


def test_command_line_overrides():
    cfg = config_from_args(parse_args(
        ["--variant", "mix_split", "--preset", "pilot", "--out", "x", "--n-people", "50",
         "--max-operand", "7", "--max-stories", "9", "--d-model", "64", "--steps", "5"]
    ))
    assert cfg.task_overrides == {
        "maths": {"max_operand": 7}, "facts": {"n_people": 50}, "clutrr": {"max_stories": 9}
    }
    assert cfg.model["d_model"] == 64 and cfg.model["core_ff"] == 512 and cfg.steps == 5


def test_train_resume_and_report(tmp_path, fake_stories):
    root = tmp_path / "runs"
    cfg = tiny("mix_split")
    first = run(cfg, root / "mix_split")
    assert set(first["tasks"]) == {"maths", "clutrr", "facts"}
    assert first["table_off"] == ["maths", "clutrr"]
    maths, story, facts = (first["tasks"][t] for t in ("maths", "clutrr", "facts"))
    assert maths["trained_with_table"] is False and facts["trained_with_table"] is True
    assert maths["accuracy"] == maths["accuracy_table_off"]  # scored as it was trained
    assert facts["accuracy"] == facts["accuracy_table_on"]
    assert set(facts["scratchpad"]["chain_held_second"]["interventions"]) == {
        "normal", "cleared", "shuffled", "true_bridge"
    }
    assert set(facts["two_call"]["chain_held_second"]) == {"1", "2"}
    assert story["two_call"] == {} and "chain_unseen" in story["scratchpad"]
    assert set(facts["accuracy_by_steps"]) == {"1", "2", "3", "4"}
    assert 0 <= facts["majority_baseline"]["chain_unseen"] <= 1

    saved = load_run_config(root / "mix_split")
    assert tuple(saved.table_off) == ("maths", "clutrr") and saved.model["core_ff"] == 512
    suite, model_cfg = build_run(saved)
    assert model_cfg.question_len == suite.question_len and model_cfg.answer_len == 4

    # Resuming continues from the checkpoint and keeps earlier evaluations.
    longer = tiny("mix_split", steps=9)
    second = run(longer, root / "mix_split", resume=True)
    assert second["steps"] == 9
    steps = [json.loads(line)["step"] for line in (root / "mix_split" / "metrics.jsonl").read_text().splitlines()]
    assert steps == [3, 6, 9]

    run(tiny("facts_plain"), root / "facts_plain")
    report = build_report(root)
    for heading in ("## Accuracy", "## Is the scratchpad used?", "## Two-call ceiling",
                    "## What depends on the table", "## Readout"):
        assert heading in report
    assert "| mix_split | maths | off |" in report and "| mix_split | facts | on |" in report
    assert "| facts_plain | facts |" in report


def test_training_is_repeatable(tmp_path):
    a = run(tiny("maths"), tmp_path / "a")
    b = run(tiny("maths"), tmp_path / "b")
    assert a["tasks"]["maths"]["accuracy"] == b["tasks"]["maths"]["accuracy"]
    assert a["params"] == b["params"]
    timing = run(tiny("maths"), tmp_path / "t", time_only=2)
    assert timing["sec_per_step"] > 0 and (tmp_path / "t" / "timing.json").exists()


def test_reading_test_run_and_its_report(tmp_path):
    root = tmp_path / "runs"
    for variant, seed in (("facts_reader", 0), ("facts_reader", 1), ("facts", 0)):
        cfg = tiny(variant)
        cfg.seed = seed
        final = run(cfg, root / run_name(variant, seed, [0, 1]))
        facts = final["tasks"]["facts"]
        held = facts["scratchpad"]["chain_held_second"]
        assert set(held["interventions"]) == {"normal", "cleared", "shuffled", "true_bridge"}
        assert 0 <= held["subject_hidden"] <= 1 and 0 <= held["fresh_question"] <= 1
        for split in ("one_hop", "one_hop_held"):
            entry = facts["reader"][split]
            assert entry["n"] > 0
            assert all(0 <= entry[k] <= 1 for k in entry if k != "n")
    assert load_run_config(root / "facts_reader_s1").reader_frac == 0.5
    assert load_run_config(root / "facts_s0").reader_frac == 0.0

    report = build_report(root)
    for heading in ("## Is the scratchpad used?", "## Reading a known fact from the note",
                    "## The reading test", "## Readout"):
        assert heading in report
    assert "| facts_reader, seed 1 | facts |" in report and "| facts, seed 0 | facts |" in report
    assert "| facts_reader | 2 |" in report and "| facts | 1 |" in report  # seeds per run
    assert "own note in a fresh question" in report
    assert "These lines give the mean over seeds." in report


def test_the_control_run_is_unchanged_by_the_reader_code(tmp_path):
    """A run that asks nothing through the note draws the same questions as
    before the option existed: the same batches for the same seed."""
    cfg = tiny("facts")
    suite, _ = build_run(cfg)
    task = suite.tasks[0]
    a = task.sample(np.random.default_rng([0, 0]), 4, 4)
    rng = np.random.default_rng([0, 0])
    ids = rng.integers(0, len(task.subject), 4)
    picked = task.train_chains[rng.integers(0, len(task.train_chains), 4)]
    assert (a.question[:4] == task.fact_items(ids).question).all()
    assert (a.question[4:] == task.chain_items(picked).question).all()


def test_sweep_runs_every_seed(tmp_path):
    out = tmp_path / "sweep"
    small = ["--n-people", "40", "--d-model", "16", "--n-heads", "2", "--n-subkeys", "4",
             "--d-key", "8", "--top-k", "2", "--lookup-heads", "2", "--batch-size", "8",
             "--warmup", "2", "--eval-size", "12"]
    sweep_main(["--preset", "smoke", "--out", str(out), "--only", "facts_reader", "facts",
                "--seeds", "0", "1", "--steps", "4", "--pilot-steps", "2", *small])
    names = sorted(p.parent.name for p in out.glob("*/final.json"))
    assert names == ["facts_reader_s0", "facts_reader_s1", "facts_s0", "facts_s1"]
    assert json.loads((out / "facts_reader_s1" / "final.json").read_text())["seed"] == 1
    assert json.loads((out / "plan.json").read_text())["signature"]["seeds"] == [0, 1]
    assert "## The reading test" in (out / "report.md").read_text()
    assert run_name("facts", 0, [0]) == "facts"  # a single seed 0 keeps the old layout
