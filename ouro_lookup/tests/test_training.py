import json

import mlx.core as mx
import numpy as np
import pytest

from ouro_lookup import evaluate
from ouro_lookup.configs import VARIANTS, build_config, load_run_config
from ouro_lookup.data import World, WordTokenizer
from ouro_lookup.report import build_report
from ouro_lookup.train import config_from_args, parse_args, run

SMALL = dict(batch_size=24, warmup=20, lr=3e-3, eval_size=24, periodic_size=24, log_every=1000)


def small_config(tiny_dir, variant, steps, **overrides):
    return build_config(
        variant, "smoke", weights=str(tiny_dir), steps=steps,
        world=dict(n_people=12, n_relations=2), addons=dict(n_subkeys=16),
        **{**SMALL, **overrides},
    )


def test_table_learns_the_facts_and_switching_it_off_loses_them(tiny_dir, tmp_path):
    final = run(small_config(tiny_dir, "table", 150), tmp_path / "table")
    assert final["accuracy"]["one_hop"] >= 0.9
    assert final["accuracy_original"]["one_hop"] <= 0.1
    assert final["chain_mode"] == "note"
    assert final["params"]["adapters"] == 0 and final["params"]["table"] > 0
    detail = final["chains"]["chain_train"]
    assert {"note_right", "shuffled_notes", "correct_notes", "two_call", "by_loop"} <= set(detail)
    assert len(detail["by_loop"]["bridge"]) == 4
    # The original model's loss on ordinary text is measured with everything off.
    assert final["ordinary_text_loss"]["original"] > 0
    saved = json.loads((tmp_path / "table" / "final.json").read_text())
    assert saved["accuracy"] == final["accuracy"]
    assert "table" in build_report(tmp_path)


def test_resume_picks_up_where_it_stopped(tiny_dir, tmp_path, capsys):
    out = tmp_path / "run"
    run(small_config(tiny_dir, "table_adapters", 6, eval_every=3), out)
    first = [json.loads(line)["step"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert first == [3, 6]
    run(small_config(tiny_dir, "table_adapters", 12, eval_every=3), out, resume=True)
    assert "resumed from" in capsys.readouterr().out
    steps = [json.loads(line)["step"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert steps == [3, 6, 9, 12]
    assert load_run_config(out).steps == 12


def test_resume_restores_the_table_and_its_optimizer(tiny_dir, tmp_path):
    out = tmp_path / "run"
    run(small_config(tiny_dir, "table", 4, eval_every=4), out)
    before = mx.load(str(out / "checkpoint.safetensors"))
    assert int(np.array(before["optim.tables.step"])) == 4
    assert not any(name.startswith("model.base.") for name in before)
    final = run(small_config(tiny_dir, "table", 8, eval_every=4), out, resume=True)
    after = mx.load(str(out / "checkpoint.safetensors"))
    assert final["steps"] == 8 and int(np.array(after["optim.tables.step"])) == 8
    # Training went on from the saved table, not from a fresh one.
    moved = np.abs(np.array(after["model.tables.0.write"]) - np.array(before["model.tables.0.write"]))
    assert 0 < moved.max() < np.abs(np.array(after["model.tables.0.write"])).max()


@pytest.mark.parametrize("variant", ["latent", "adapters"])
def test_other_runs_train_and_report(tiny_dir, tmp_path, variant):
    final = run(small_config(tiny_dir, variant, 6, eval_every=3), tmp_path / variant)
    assert np.isfinite(final["sec_per_step"])
    if variant == "latent":
        assert final["chain_mode"] == "latent"
        assert "note_right" not in final["chains"]["chain_train"]
        assert {"accuracy_table_off", "accuracy_adapters_off"} <= set(final)
    else:
        assert final["params"]["table"] == 0 and "table_used" not in final
    assert variant in build_report(tmp_path)


def test_timing_mode_writes_a_timing_file(tiny_dir, tmp_path):
    timing = run(small_config(tiny_dir, "table", 100), tmp_path / "t", time_only=2)
    assert timing["sec_per_step"] > 0 and timing["tokens_per_step"] == 12 * 6 + 12 * 9
    assert (tmp_path / "t" / "timing.json").exists() and not (tmp_path / "t" / "final.json").exists()


def test_flags_reach_the_configuration(tiny_dir):
    cfg = config_from_args(parse_args([
        "--variant", "latent", "--preset", "pilot", "--out", "x", "--weights", str(tiny_dir),
        "--n-people", "20", "--n-subkeys", "32", "--table-layers", "0", "2", "--teach-loop", "3",
        "--adapter-rank", "2", "--lr", "0.01",
    ]))
    assert cfg.world == {"n_people": 20, "n_relations": 4}
    assert cfg.addon_config().table_layers == (0, 2) and cfg.addon_config().adapter_rank == 2
    assert cfg.teach_loop == 3 and cfg.lr == 0.01 and cfg.chain_mode == "latent"


def test_every_variant_builds():
    for variant in VARIANTS:
        cfg = build_config(variant, "smoke")
        assert cfg.chain_mode in ("note", "latent")
    with pytest.raises(ValueError):
        build_config("table", "smoke", teach_loop=2)  # teaching a loop is for one-pass runs
    with pytest.raises(ValueError):
        build_config("table", "smoke", addons=dict(table_layers=()))  # nothing to train


class Oracle:
    """Stands in for a model: answers one-step questions from the world and,
    for a two-step question, gives the note or answer it is told to."""

    tables, adapters, training = [], [], False

    def __init__(self, world, chain_reply):
        self.world, self.chain_reply = world, chain_reply
        w = world
        self.fact = {(int(t[2]), int(t[4])): int(a) for t, a in
                     zip(w.fact_tokens, w.name_token[w.fact_answer])}
        ids = np.arange(len(w.chain_first))
        chains = w.chains(ids)
        reply = {"bridge": chains.bridge, "answer": chains.answer}[chain_reply]
        self.chain = {tuple(t.tolist()): int(r) for t, r in zip(w.chain_tokens, reply)}

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def states(self, tokens, **switches):
        rows = np.array(tokens)
        if rows.shape[1] == self.world.chain_tokens.shape[1]:
            tops = [self.chain[tuple(row.tolist())] for row in rows]
        else:  # unknown people get an answer that is nobody's name
            tops = [self.fact.get((int(row[2]), int(row[4])), 1) for row in rows]
        state = np.zeros((len(rows), rows.shape[1], 400), dtype=np.float32)
        state[np.arange(len(rows)), -1, tops] = 1.0
        return [mx.array(state)] * 4

    def read(self, state):
        return state


def oracle_world():
    return World(WordTokenizer().encode, n_people=30, n_relations=3, seed=2, eval_size=60)


def test_a_right_note_is_carried_into_a_fresh_question():
    world = oracle_world()
    model = Oracle(world, "bridge")
    sets = evaluate.eval_ids(world, 60)
    accuracy = evaluate.accuracy_by_split(model, world, sets, "note")
    assert all(value == 1.0 for value in accuracy.values())
    detail = evaluate.chain_detail(model, world, sets[1], "note")["chain_unseen"]
    assert detail["note_right"] == 1.0 and detail["correct_notes"] == 1.0
    assert detail["two_call"] == 1.0 and detail["both_facts_known"] == 1.0
    assert detail["shuffled_notes"] < 0.3  # someone else's note gives the wrong answer


def test_a_wrong_note_gives_a_wrong_answer_even_when_both_facts_are_known():
    world = oracle_world()
    model = Oracle(world, "answer")  # writes the final answer where the note should be
    sets = evaluate.eval_ids(world, 60)
    detail = evaluate.chain_detail(model, world, sets[1], "note")["chain_unseen"]
    assert detail["note_right"] < 0.2 and detail["both_facts_known"] == 1.0
    assert detail["accuracy_when_both_known"] < 0.2 and detail["two_call"] == 1.0
    # Read in one pass instead, the same replies are the right answers.
    latent = evaluate.accuracy_by_split(model, world, sets, "latent")
    assert latent["chain_unseen"] == 1.0
