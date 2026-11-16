"""Checks on the two-day run: removed layers and what stands in for them,
the ordinary text, the known facts, and the flexibility runs."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ouro_lookup import flex, takeover, text
from ouro_lookup.data import ORDINARY
from ouro_lookup.download import load_tokenizer
from ouro_lookup.model import AddonConfig, LookupOuro
from ouro_lookup.ouro import load_ouro
from ouro_lookup.probes import Probes, all_probes
from ouro_lookup.twoday import Run, run_list
from ouro_lookup.twoday import main as twoday_main
from ouro_lookup.twoday_report import build_report

SMALL_TAKEOVER = dict(
    remove_layers=(1,), seq_len=12, batch_size=4, warmup=5, eval_windows=8, periodic_windows=4,
    log_every=1000,
)
SMALL_FLEX = dict(
    batch_size=16, lr=3e-3, adapter_lr=3e-3, warmup=5, check_every=20, log_every=1000,
)
FAKE_PROBES = [("names", "The mentor of Daniel is", "Alice"), ("names", "The coach of Alice is", "Greg")]


@pytest.fixture(scope="session")
def tiny_text(tiny_dir, tmp_path_factory):
    """A directory of prepared text in the tiny model's vocabulary."""
    root = tmp_path_factory.mktemp("text")
    source = root / "source.txt"
    source.write_text("\n".join(ORDINARY * 6) + "\n")
    text.prepare(tiny_dir, root / "prepared", text_file=source)
    return root / "prepared"


def build(tiny_dir, **addons):
    mx.random.seed(0)
    model = LookupOuro(load_ouro(tiny_dir, dtype=mx.float32), AddonConfig(n_subkeys=8, **addons))
    model.eval()
    return model


TOKENS = mx.array([[0, 2, 6, 4, 20, 5], [0, 2, 7, 4, 21, 5]])


def same(a, b, tol=1e-5):
    return np.allclose(np.array(a), np.array(b), atol=tol)


# -- removed layers and their stand-ins ----------------------------------


def test_original_ignores_everything_added_and_removed(tiny_dir):
    base = load_ouro(tiny_dir, dtype=mx.float32)
    want = base.loop_states(TOKENS)
    for addons in (
        dict(table_layers=(1,), remove_layers=(1,)),
        dict(table_layers=(), remove_layers=(0, 2), dense_width=8),
        dict(table_layers=(), remove_layers=(1,), adapter_rank=4),
    ):
        model = build(tiny_dir, **addons)
        got = model.states(TOKENS, original=True)
        assert all(same(g, w) for g, w in zip(got, want))
        # Removing a layer's MLP changes the output.
        assert not same(model.states(TOKENS)[-1], want[-1], 1e-4)


def test_stand_ins_start_as_nothing_and_can_be_switched_off(tiny_dir):
    for addons in (
        dict(table_layers=(1,), remove_layers=(1,)),
        dict(table_layers=(), remove_layers=(1,), dense_width=8),
    ):
        model = build(tiny_dir, **addons)
        nothing = model.states(TOKENS, table=False)[-1]
        assert same(model.states(TOKENS)[-1], nothing)  # zero until trained
        # Give the stand-in something to say: the output moves, and the
        # switch still returns the model with nothing put back.
        if model.dense:
            model.dense[0].down = mx.ones_like(model.dense[0].down) * 0.05
        else:
            model.tables[0].write = mx.ones_like(model.tables[0].write) * 0.05
        assert not same(model.states(TOKENS)[-1], nothing, 1e-4)
        assert same(model.states(TOKENS, table=False)[-1], nothing)


def test_what_the_mlp_half_added_is_reported_for_removed_layers(tiny_dir):
    model = build(tiny_dir, table_layers=(2,), remove_layers=(0, 2), dense_width=8)
    loops = model.base.cfg.total_ut_steps
    states, adds = model.states(TOKENS, original=True, return_adds=True)
    assert len(adds) == 2 * loops and all(a.shape == states[0].shape for a in adds)
    # The first is exactly the original MLP's contribution at layer 0, loop 1.
    body = model.base.model
    cos, sin = model.base.positions(TOKENS.shape[1])
    h = body.layers[0].attend(body.embed_tokens(TOKENS), cos, sin)
    assert same(adds[0], body.layers[0].feed_forward(h))
    # With stand-ins, their outputs are reported instead, in the same order.
    _, stand = model.states(TOKENS, return_adds=True)
    assert len(stand) == 2 * loops and all(s is not None for s in stand)
    # Adapters put nothing at a removed layer.
    bare = build(tiny_dir, table_layers=(), remove_layers=(1,), adapter_rank=4)
    assert bare.states(TOKENS, return_adds=True)[1] == [None] * loops


def test_sizes_and_costs(tiny_dir):
    model = build(tiny_dir, table_layers=(1,), remove_layers=(1,))
    assert model.removed_parameters() == 3 * 64 * 128  # one tiny MLP
    cost = model.stand_in_cost()
    assert cost == {"stand_in": 64 * 4 * 64 + 4 * 8 * 64 + 4 * 8 * 2 * 64, "removed_mlp": 3 * 64 * 128}
    dense = build(tiny_dir, table_layers=(), remove_layers=(0, 1), dense_width=8)
    assert dense.count_parameters()["dense"] == 2 * 3 * 64 * 8
    assert dense.stand_in_cost()["stand_in"] == 3 * 64 * 8
    # On the real model a small MLP of this width costs exactly what the
    # default table costs.
    d, heads, d_key, n_subkeys, top_k = 2048, 4, 64, 128, 8
    table_cost = d * heads * d_key + heads * n_subkeys * d_key + heads * top_k * 2 * d
    assert 3 * d * takeover.SAME_COST_WIDTH == table_cost
    with pytest.raises(ValueError):
        build(tiny_dir, table_layers=(), dense_width=8)  # nothing removed to stand in for
    with pytest.raises(ValueError):
        build(tiny_dir, remove_layers=(7,))


# -- text -----------------------------------------------------------------


def test_wikitext_spacing_is_undone():
    line = ' He said " it is 1 @,@ 000 km ( 620 mi ) long " , and the well @-@ known river \'s mouth is n\'t wide . \n'
    assert text.clean(line) == (
        'He said "it is 1,000 km (620 mi) long", and the well-known river\'s mouth isn\'t wide.'
    )
    long = "A sentence that is long enough to count as a paragraph . " * 5
    lines = [" = Heading = \n", " \n", long + "\n", " short line . \n", long + "\n"]
    assert len(list(text.paragraphs(lines))) == 2
    assert len(list(text.paragraphs(lines, max_chars=len(lines[0]) + len(long) + 5))) == 1


def test_text_windows(tiny_dir, tiny_text):
    _, start = load_tokenizer(tiny_dir)
    data = text.Text.load(tiny_text, start, 10)
    meta = json.loads((tiny_text / "meta.json").read_text())
    assert len(data.train) == meta["train_tokens"] and len(data.valid) == meta["valid_tokens"]
    batch = data.batch(np.random.default_rng(0), 5)
    assert batch.shape == (5, 10) and batch.dtype == np.int32 and (batch[:, 0] == start).all()
    held = data.held_out(1000)
    assert (held == data.held_out(1000)).all() and (held[:, 0] == start).all()
    assert len(held) == (len(data.valid) - 9) // 9 + 1 and len(data.held_out(2)) == 2
    assert (held[0, 1:] == data.valid[:9]).all() and (held[1, 1:] == data.valid[9:18]).all()
    # Preparing again with the same settings reuses what is there.
    assert text.prepare(tiny_dir, tiny_text, text_file=meta["source"]) == meta
    with pytest.raises(SystemExit):
        text.Text.load(tiny_text / "missing", start, 10)


def test_text_from_the_downloaded_files(tiny_dir, tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    long = "The train to the coast leaves every morning at nine and takes about two hours . " * 4
    lines = {
        text.TRAIN: [" = Title = \n", " \n", f" {long}\n", f" {long}\n", " caption .\n"],
        text.VALID: [f" {long}\n"],
    }
    assert text.TRAIN.startswith("train") and text.VALID.startswith("validation")
    for name, rows in lines.items():
        pq.write_table(pa.table({"text": rows}), tmp_path / name)
    monkeypatch.setattr(text, "fetch", lambda name, directory: tmp_path / name)
    meta = text.prepare(tiny_dir, tmp_path / "out")
    assert meta["train_tokens"] == 2 * meta["valid_tokens"] > 0
    assert "leaves every morning at nine and takes about two hours." in meta["example"]


# -- known facts ------------------------------------------------------------


def test_probe_list_is_well_formed():
    probes = all_probes()
    assert len(probes) > 250 and len(set(probes)) == len(probes)
    assert {c for c, _, _ in probes} == {
        "capitals", "currencies", "authors", "elements", "opposites", "places", "sequences",
        "general",
    }
    assert all(start.strip() == start and start and word.strip() == word and word
               for _, start, word in probes)
    assert ("capitals", "The capital of France is", "Paris") in probes
    assert ("authors", "Pride and Prejudice was written by", "Jane") in probes
    assert ("sequences", "Monday, Tuesday, Wednesday,", "Thursday") in probes


def test_probes_are_scored_against_what_the_original_knew(tiny_dir):
    encode, start = load_tokenizer(tiny_dir)
    probes = Probes(encode, start, FAKE_PROBES)
    model = build(tiny_dir, table_layers=(1,), remove_layers=(1,))
    right = probes.right(model, original=True)
    assert right.shape == (2,) and right.dtype == bool
    # Make the first one right by construction: ask for whatever it says.
    said = int(mx.argmax(model.answer_logits(mx.array([probes.tokens[0]]), original=True)).item())
    probes.answer[0] = said
    assert probes.right(model, original=True)[0]
    score = probes.score(np.array([True, True]), known=np.array([True, False]))
    assert score == {"known": 1, "right": 1, "by_category": {"names": {"known": 1, "right": 1}}}


# -- the take-over run ------------------------------------------------------


def small_takeover(variant, tiny_dir, tiny_text, **overrides):
    return takeover.build_config(
        variant, weights=str(tiny_dir), text=str(tiny_text), dtype="float32",
        addons=dict(n_subkeys=8) if "table" in variant else None,
        **{**SMALL_TAKEOVER, "steps": 12, "eval_every": 6, **overrides},
    )


def test_takeover_variants():
    for name in takeover.VARIANTS:
        cfg = takeover.build_config(name)
        addon = cfg.addon_config()
        assert addon.remove_layers == takeover.REMOVE == tuple(range(8, 16))
        assert (addon.table_layers == takeover.REMOVE) == name.startswith("table")
    assert takeover.build_config("dense").addon_config().dense_width == 112
    assert takeover.build_config("dense_wide").addon_config().dense_width == 448
    assert takeover.build_config("adapters").addon_config().adapter_rank == 8
    args = takeover.parse_args(["--variant", "table", "--out", "x", "--n-subkeys", "64",
                                "--remove-layers", "3", "4", "--seq-len", "32", "--steps", "7"])
    cfg = takeover.config_from_args(args)
    assert cfg.addon_config().n_subkeys == 64 and cfg.remove_layers == (3, 4)
    assert cfg.seq_len == 32 and cfg.steps == 7


def test_imitation_is_perfect_when_nothing_changed(tiny_dir, tiny_text):
    model = build(tiny_dir, table_layers=(1,))  # a table that still says nothing
    kl, match, local = takeover.imitation_losses(model, TOKENS)
    assert abs(kl.item()) < 1e-5 and abs(match.item()) < 1e-6 and local.item() == 0.0
    cut = build(tiny_dir, table_layers=(1,), remove_layers=(1,))
    kl, match, local = takeover.imitation_losses(cut, TOKENS)
    assert kl.item() > 1e-4 and match.item() > 1e-4
    assert abs(local.item() - 1.0) < 1e-4  # a silent stand-in misses all of its target


def test_takeover_run_resume_and_timing(tiny_dir, tiny_text, tmp_path):
    cfg = small_takeover("table", tiny_dir, tiny_text)
    final = takeover.run(cfg, tmp_path / "table", probes=FAKE_PROBES)
    assert final["kind"] == "takeover" and final["steps"] == 12
    assert final["tokens_trained"] == 12 * 4 * 12
    assert final["removed"] == {"layers": [1], "numbers": 3 * 64 * 128,
                                "share_of_model": final["removed"]["share_of_model"]}
    held = final["held_out"]
    assert held["nothing"] != held["original"] and np.isfinite(held["trained"])
    assert held["kl_nothing"] > 0 and held["kl_trained"] > 0
    assert final["probes"]["listed"] == 2 and len(final["table_used"]) == 1
    assert set(final["sentences"]) == {"original", "nothing", "trained"}
    steps = [json.loads(line)["step"]
             for line in (tmp_path / "table" / "metrics.jsonl").read_text().splitlines()]
    assert steps == [6, 12]
    assert takeover.load_run_config(tmp_path / "table") == cfg

    # An interrupted run picks up from its checkpoint.
    part = small_takeover("dense", tiny_dir, tiny_text, steps=6)
    takeover.run(part, tmp_path / "dense", probes=FAKE_PROBES)
    arrays, meta = mx.load(str(tmp_path / "dense" / "checkpoint.safetensors"), return_metadata=True)
    assert meta["step"] == "6" and any(k.startswith("model.dense.") for k in arrays)
    longer = small_takeover("dense", tiny_dir, tiny_text, steps=12)
    assert takeover.run(longer, tmp_path / "dense", resume=True, probes=FAKE_PROBES)["steps"] == 12

    timing = takeover.run(cfg, tmp_path / "t", time_only=2, probes=FAKE_PROBES)
    assert timing["sec_per_step"] > 0 and timing["tokens_per_step"] == 48
    assert (tmp_path / "t" / "timing.json").exists()


@pytest.mark.parametrize("variant", ["dense", "adapters"])
def test_a_stand_in_learns_to_imitate(variant, tiny_dir, tiny_text, tmp_path):
    cfg = small_takeover(variant, tiny_dir, tiny_text, steps=150, eval_every=150, lr=3e-3,
                         adapter_lr=3e-3)
    held = takeover.run(cfg, tmp_path / variant, probes=FAKE_PROBES)["held_out"]
    # Closer to the original's predictions than with nothing put back, on
    # text it was not trained on.
    assert held["kl_trained"] < 0.7 * held["kl_nothing"]


# -- flexibility ------------------------------------------------------------


def small_flex(mode, variant, tiny_dir, **overrides):
    return flex.build_config(
        mode, variant, preset="smoke", weights=str(tiny_dir), dtype="float32",
        addons=dict(n_subkeys=8) if "table" in variant else None, **{**SMALL_FLEX, **overrides},
    )


def test_flex_variants():
    table = flex.build_config("retain", "table").addon_config()
    assert table.table_layers is None and table.adapter_rank == 0  # the middle layer, as in stage 1
    matched = flex.build_config("retain", "adapters_matched").addon_config()
    assert matched.table_layers == () and matched.adapter_rank == 344
    # As many trained numbers as the table, on the real model's sizes.
    adapters = 24 * 2 * 2 * 2048 * 344
    table_numbers = 16384 * 2 * 2048 + 2048 * 256 + 256 + 4 * 2 * 128 * 32
    assert abs(adapters - table_numbers) / table_numbers < 0.001
    assert flex.build_config("edit", "table_fixed_keys").freeze_keys
    assert not flex.build_config("edit", "table").freeze_keys
    with pytest.raises(ValueError):
        flex.build_config("forget", "table")
    args = flex.parse_args(["--mode", "edit", "--variant", "adapters", "--out", "x",
                            "--n-people", "30", "--n-edits", "7", "--joint-steps", "9"])
    cfg = flex.config_from_args(args)
    assert cfg.mode == "edit" and cfg.world["n_people"] == 30
    assert cfg.n_edits == 7 and cfg.joint_steps == 9


def test_facts_arrive_in_batches_of_whole_people(tiny_dir, tmp_path):
    cfg = small_flex("retain", "table", tiny_dir, batches=3, phase_steps=300, target=0.8)
    final = flex.run(cfg, tmp_path / "retain")
    assert final["kind"] == "flex" and final["mode"] == "retain"
    assert final["facts"] == {"people": 24, "relations": 3, "facts": 72}
    phases = final["phases"]
    assert [p["facts"] for p in phases] == [24, 24, 24]
    assert all(len(p["recall"]) == 3 and 0 < p["steps"] <= 300 for p in phases)
    summary = final["summary"]
    # The table learns the first batch, and stops as soon as it has.
    assert phases[0]["recall"][0] >= 0.8 and phases[0]["reached_target"]
    assert phases[0]["steps"] < 300 and phases[0]["steps"] % cfg.check_every == 0
    matrix = np.array([p["recall"] for p in phases])
    assert summary["learned"] == pytest.approx(np.diag(matrix).mean())
    assert summary["kept_at_end"] == pytest.approx(matrix[-1, :-1].mean())
    assert summary["first_batch_at_end"] == matrix[-1, 0]
    progress = json.loads((tmp_path / "retain" / "progress.json").read_text())
    assert len(progress["phases"]) == 3

    # Resuming after the second batch trains only the third.
    progress["phases"] = progress["phases"][:2]
    (tmp_path / "retain" / "progress.json").write_text(json.dumps(progress))
    again = flex.run(cfg, tmp_path / "retain", resume=True)
    assert len(again["phases"]) == 3 and again["phases"][:2] == phases[:2]


def test_fixed_keys_stop_moving_after_the_first_batch(tiny_dir, tmp_path):
    cfg = small_flex("retain", "table_fixed_keys", tiny_dir, batches=2, phase_steps=40)
    world, model = flex.build_run(cfg)
    tokens, answers = world.fact_tokens, world.name_token[world.fact_answer]

    def keys():
        table = model.tables[0]
        parts = [table.query.weight, table.query.bias, table.subkeys,
                 table.norm.running_mean, table.norm.running_var]
        return [np.array(p) for p in parts]

    def train(seed):
        flex.train_phase(model, cfg, tokens, answers, 20, np.random.default_rng(seed),
                         check=dict, check_every=1000, compile_step=False)

    before = keys()
    train(0)
    moved = keys()
    assert all(not np.array_equal(a, b) for a, b in zip(before, moved))
    flex.freeze_keys(model)
    names = [name for name, _ in tree_flatten(model.trainable_parameters())]
    assert names == ["tables.0.fire", "tables.0.write"]
    write = np.array(model.tables[0].write)
    train(1)
    assert all(np.array_equal(a, b) for a, b in zip(moved, keys()))
    assert not np.array_equal(write, np.array(model.tables[0].write))  # entries still learn


def test_changing_facts(tiny_dir, tmp_path):
    cfg = small_flex("edit", "table", tiny_dir, joint_steps=600, n_edits=6, edit_steps=60,
                     edit_check=20)
    final = flex.run(cfg, tmp_path / "edit")
    assert final["mode"] == "edit" and final["joint"]["recall"] >= 0.7
    edit = final["edit"]
    assert edit["changed"] == 6 and edit["untouched"] == 66
    # Before the change the old answers are given and the new ones are not.
    assert edit["before"]["old_answer"] >= 0.5 and edit["before"]["recall"] <= 0.34
    assert [c["step"] for c in edit["checks"]] == [20, 40, 60]
    assert edit["at_end"] == edit["checks"][-1]
    assert edit["at_end"]["recall"] >= 0.8  # the new answers took
    assert set(final["ordinary_text_loss"]) == {"original", "after_joint", "at_end"}
    timing = flex.run(cfg, tmp_path / "t", time_only=2)
    assert timing["sec_per_step"] > 0 and timing["eval_sec"] > 0


# -- the sweep --------------------------------------------------------------


def test_run_order():
    runs = run_list([0, 1])
    names = [run.name for run in runs]
    assert len(names) == len(set(names)) == 21
    # The main comparison of each part comes first.
    assert names[:3] == ["takeover_table", "takeover_dense", "takeover_adapters"]
    assert names[3:7] == ["retain_table_s0", "retain_adapters_s0", "edit_table_s0", "edit_adapters_s0"]
    assert names[11:13] == ["takeover_table_small", "takeover_dense_wide"]
    assert all(name.endswith("_s1") for name in names[13:])
    assert Run("retain", "table_fixed_keys", 1).timing_key == Run("edit", "table", 0).timing_key
    assert len(run_list([0])) == 13


def test_sweep_and_report(tiny_dir, tiny_text, tmp_path):
    out = tmp_path / "sweep"
    twoday_main([
        "--out", str(out), "--steps", "8", "--pilot-steps", "2", "--skip-check",
        "--weights", str(tiny_dir), "--text", str(tiny_text), "--dtype", "float32",
        "--only", "takeover_dense", "retain_table_s0", "edit_adapters_s0", "retain_table_s1",
        "--takeover-extra",
        "--remove-layers 1 --seq-len 12 --batch-size 4 --warmup 2 --eval-windows 8 --periodic-windows 4",
        "--flex-extra",
        "--preset smoke --n-subkeys 8 --batches 2 --phase-steps 20 --check-every 10 "
        "--joint-steps 20 --n-edits 4 --edit-steps 10 --edit-check 5 --batch-size 8 --warmup 2",
    ])
    done = sorted(p.parent.name for p in out.glob("*/final.json"))
    assert done == ["edit_adapters_s0", "retain_table_s0", "retain_table_s1", "takeover_dense"]
    plan = json.loads((out / "plan.json").read_text())
    assert plan["steps"] == 8 and set(plan["estimated_minutes"]) == set(done)
    assert set(plan["timings"]) == {"takeover_dense", "flex_table", "flex_adapters"}
    report = (out / "report.md").read_text()
    assert report == build_report(out)
    for heading in ("## Taking over the removed layers", "### Known facts by kind",
                    "### During training", "## Adding facts batch by batch",
                    "### Recall of each batch after each phase", "## Changing facts"):
        assert heading in report
    assert "| dense | held-out loss |" in report and "after 50% of the steps" in report
    assert "| nothing put back | 0 | 0% |" in report and "| dense |" in report
    assert "| table | 2 |" in report  # two seeds of the retention run
    # A second start with other settings is refused; the same settings resume.
    with pytest.raises(SystemExit):
        twoday_main(["--out", str(out), "--steps", "9", "--skip-check"])
