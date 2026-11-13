"""End-to-end checks: configurations, training, resuming, the sweep, the report."""

import json

import numpy as np
import pytest

import phrase_scratchpad.clutrr as clutrr
from phrase_scratchpad.configs import VARIANTS, build_config, build_run, load_run_config
from phrase_scratchpad.report import build_report
from phrase_scratchpad.sweep import BASELINES, EXTRAS, RUNS
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
    assert set(RUNS + EXTRAS + BASELINES) == set(VARIANTS)
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
    # Every run hides the subject of half its two-step training questions.
    assert all(build_config(v).hide_subject_prob == 0.5 for v in VARIANTS)


def test_block_applications_per_step(fake_stories):
    suite, model_cfg = build_run(tiny("mix_split"))
    assert model_cfg.n_passes == 1  # arithmetic and facts: one lookup per step
    assert [t.n_passes for t in suite.tasks] == [None, 2, None]  # stories read with two
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
