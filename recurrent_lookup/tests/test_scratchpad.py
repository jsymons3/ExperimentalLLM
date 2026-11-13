"""Checks on the scratchpad model and the code around it."""

import json
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from recurrent_lookup.configs import build_config, load_run_config, model_config
from recurrent_lookup.data import GraphConfig, MultiHopData
from recurrent_lookup.diagnose import diagnose_run, to_markdown
from recurrent_lookup.evaluate import (
    decode_by_depth,
    eval_mode,
    forward,
    retrieval_overlap,
    slot_contents,
)
from recurrent_lookup.loading import load_run
from recurrent_lookup.model import ModelConfig, RecurrentModel
from recurrent_lookup.report import build_report
from recurrent_lookup.sweep import SLOT_RUNS, baseline_differences, planned_config
from recurrent_lookup.train import build_optimizer, load_checkpoint, run, save_checkpoint
from recurrent_lookup.two_call import diagnose as two_call_diagnose

TOKENS = mx.array([[3, 41, 45], [7, 42, 44]])
TARGETS = mx.array([5, 9])
ORIGINAL_PARAMETERS = {
    "embed.weight", "pos", "norm_attn.weight", "norm_attn.bias",
    "attn.qkv.weight", "attn.out.weight", "norm_ff.weight", "norm_ff.bias",
    "ff.query.weight", "ff.query.bias", "ff.norm.running_mean",
    "ff.norm.running_var", "ff.subkeys", "ff.fire", "ff.write",
    "norm_out.weight", "norm_out.bias",
}


def small(**kw):
    base = dict(vocab_size=50, n_entities=40, d_model=32, n_subkeys=8, d_key=16, top_k=4)
    base.update(kw)
    return ModelConfig(**base)


def slot_model(seed=0, **kw):
    mx.random.seed(seed)
    base = dict(num_slots=1, n_passes=2, outer_steps_train=(2, 3, 4))
    base.update(kw)
    model = RecurrentModel(small(**base))
    model.eval()
    return model


def gradients(model, steps):
    def loss_fn(model):
        return nn.losses.cross_entropy(model(TOKENS, steps=steps), TARGETS, reduction="mean")

    return dict(tree_flatten(nn.value_and_grad(model, loss_fn)(model)[1]))


def test_no_slots_is_the_original_model():
    mx.random.seed(0)
    model = RecurrentModel(small(num_slots=0, n_passes=4))
    model.eval()
    assert {k for k, _ in tree_flatten(model.parameters())} == ORIGINAL_PARAMETERS

    # The original forward pass, written out independently.
    h = model.embed(TOKENS) + model.pos[:3]
    for _ in range(4):
        h = h + model.attn(model.norm_attn(h))
        h = h + model.ff(model.norm_ff(h))[0]
    expected = model.norm_out(h[:, -1]) @ model.embed.weight[:40].T

    assert np.array_equal(np.array(model(TOKENS)), np.array(expected))
    assert np.array_equal(np.array(model(TOKENS, n_passes=4, steps=1)), np.array(expected))
    with pytest.raises(ValueError):
        model(TOKENS, steps=2)


def test_one_hot_through_the_soft_blend_returns_the_embedding_exactly():
    model = slot_model()
    ids = [0, 7, 39]
    one_hot = np.zeros((3, 40), dtype=np.float32)
    one_hot[np.arange(3), ids] = 1
    blended = np.array(model.token_form(mx.array(one_hot)))
    assert np.array_equal(blended, np.array(model.embed.weight)[ids])


def test_final_loss_reaches_parameters_used_only_at_step_one():
    model = slot_model()
    grads = gradients(model, steps=3)
    # Row 0 of the step embedding is only read at step 1, so its gradient can
    # only arrive through what step 1 wrote into the slot.
    step_grad = np.abs(np.array(grads["step_emb"]))
    assert step_grad[0].max() > 0 and step_grad[2].max() > 0
    assert step_grad[3].max() == 0  # step 4 never ran
    assert float(mx.abs(grads["write_head.weight"]).max()) > 0
    assert float(mx.abs(grads["slot_empty"]).max()) > 0

    # With a single step nothing written is ever read.
    assert float(mx.abs(gradients(model, steps=1)["write_head.weight"]).max()) == 0


def test_raw_and_hard_contents_also_carry_gradient():
    for content, extra in (("raw", "raw_proj.weight"), ("hard", "write_head.weight")):
        grads = gradients(slot_model(slot_content=content), steps=2)
        assert np.abs(np.array(grads["step_emb"]))[0].max() > 0, content
        assert float(mx.abs(grads[extra]).max()) > 0, content


def test_hard_content_is_the_top_entity_embedding():
    model = slot_model(slot_content="hard")
    logits, trace = model(TOKENS, steps=1, return_trace=True)
    content = model.slot_content(logits, None)
    top = np.array(mx.argmax(logits, axis=-1))
    np.testing.assert_allclose(
        np.array(content), np.array(model.embed.weight)[top], atol=1e-5
    )


def test_slot_update_is_a_soft_overwrite_with_token_form_content():
    model = slot_model(num_slots=3)
    _, trace = model(TOKENS, steps=2, return_trace=True)
    write = np.array(trace["write"][0])  # (B, K+1)
    assert write.shape == (2, 4)
    np.testing.assert_allclose(write.sum(axis=1), 1, atol=1e-5)
    probs = np.array(mx.softmax(trace["step_logits"][0], axis=-1))
    content = probs @ np.array(model.embed.weight)[:40]
    empty = np.array(model.slot_empty)
    w = write[:, :3, None]
    expected = (1 - w) * empty + w * content[:, None, :]
    np.testing.assert_allclose(np.array(trace["slots"][0]), expected, atol=1e-5)


def never_write(model):
    """Force the write head to always choose "write nothing"."""
    K = model.cfg.num_slots
    model.write_head.weight = mx.zeros_like(model.write_head.weight)
    model.write_head.bias = mx.array([-50.0] * K + [50.0])


def test_state_is_fresh_each_step_unless_carried():
    fresh = slot_model(step_embedding=False)
    never_write(fresh)
    _, trace = fresh(TOKENS, steps=3, return_trace=True)
    first = np.array(trace["step_logits"][0])
    # Nothing was written and there is no step signal, so every step repeats.
    assert np.array_equal(first, np.array(trace["step_logits"][1]))
    assert np.array_equal(first, np.array(trace["step_logits"][2]))

    carried = slot_model(step_embedding=False, carry_hidden=True)
    never_write(carried)
    _, trace = carried(TOKENS, steps=2, return_trace=True)
    assert not np.array_equal(
        np.array(trace["step_logits"][0]), np.array(trace["step_logits"][1])
    )


def test_question_positions_compute_what_the_original_model_computes():
    mx.random.seed(0)
    original = RecurrentModel(small(n_passes=2))
    original.eval()
    model = slot_model(seed=1)
    model.update(original.parameters())  # share every original parameter
    _, expected = original(TOKENS, return_retrieval=True)
    _, trace = model(TOKENS, steps=1, return_trace=True)
    assert trace["retrieved"][0].shape[1] == 3 + 1 + 1  # question, slot, work
    for want, got in zip(expected, trace["retrieved"]):
        assert np.array_equal(np.array(want), np.array(got[:, :3]))


def test_slot_position_makes_a_slot_look_like_a_typed_entity():
    model = slot_model(slot_position=0)
    entity = mx.array([[7], [12]])
    typed = model.embed(entity) + model.pos[0]  # entity 7 typed at position 0
    held = model.slot_inputs(model.embed.weight[entity])  # entity 7 held in the slot
    assert np.array_equal(np.array(typed), np.array(held))
    plain = slot_model().slot_inputs(model.embed.weight[entity])
    assert np.array_equal(np.array(plain), np.array(model.embed.weight[entity]))


def test_more_steps_than_trained_reuse_the_last_step_embedding():
    model = slot_model(outer_steps_train=(2,))
    assert model.step_emb.shape == (2, 32)
    assert model(TOKENS, steps=6).shape == (2, 40)
    assert model(TOKENS).shape == (2, 40)  # default: the fewest trained steps


def test_variant_configs():
    c = model_config(build_config("full", "C"))
    soft = model_config(build_config("full", "slot_soft"))
    assert (soft.num_slots, soft.slot_content, soft.n_passes) == (1, "soft", 2)
    assert tuple(soft.outer_steps_train) == (2, 3, 4)
    assert soft.n_entries == c.n_entries and c.num_slots == 0 and c.n_passes == 4
    assert model_config(build_config("full", "slot_raw")).slot_content == "raw"
    assert model_config(build_config("full", "slot_soft_k3")).num_slots == 3
    assert build_config("full", "slot_soft_prefix").prefix_loss_weight == 1.0
    assert build_config("full", "slot_soft").prefix_loss_weight == 0.0

    # Switching slots on by flag gets the same defaults as the variants.
    flagged = model_config(build_config("smoke", "C", model={"num_slots": 1}))
    assert (flagged.n_passes, tuple(flagged.outer_steps_train)) == (2, (2, 3, 4))
    with pytest.raises(ValueError):
        build_config("smoke", "C", model={"outer_steps_train": (2, 3)})
    with pytest.raises(ValueError):
        build_config("smoke", "C", prefix_loss_weight=1.0)


def test_saved_configs_load_old_and_new(tmp_path):
    new = build_config("smoke", "slot_soft", steps=10)
    saved = asdict(new)
    saved["some_future_key"] = 1
    (tmp_path / "config.json").write_text(json.dumps(saved))
    loaded = load_run_config(tmp_path)
    assert model_config(loaded).num_slots == 1
    assert list(model_config(loaded).outer_steps_train) == [2, 3, 4]

    # A config written before the scratchpad existed.
    old = {
        "preset": "full", "variant": "C", "seed": 0,
        "graph": asdict(GraphConfig(n_entities=5000)), "model": {"n_subkeys": 128},
        "steps": 128200, "batch_size": 512, "two_hop_frac": 0.5, "lr": 0.001,
        "warmup": 500, "weight_decay": 0.1, "grad_clip": 1.0, "eval_every": 3205,
        "log_every": 100, "eval_passes": [1, 2, 4, 8],
    }
    (tmp_path / "config.json").write_text(json.dumps(old))
    loaded = load_run_config(tmp_path)
    assert model_config(loaded).num_slots == 0 and loaded.table_weight_decay is None
    assert "warmup" in loaded.lr_schedule


def test_table_weight_decay_is_separate_only_when_asked():
    zero = lambda model: {k: mx.zeros_like(v) for k, v in tree_flatten(model.trainable_parameters())}
    from mlx.utils import tree_unflatten

    for table_decay, table_shrinks in ((None, True), (0.0, False)):
        mx.random.seed(0)
        model = RecurrentModel(small())
        cfg = build_config("smoke", "C", lr=0.1, warmup=1, table_weight_decay=table_decay)
        before = dict(tree_flatten(model.trainable_parameters()))
        before = {k: np.array(v) for k, v in before.items()}
        optimizer = build_optimizer(cfg)
        optimizer.update(model, tree_unflatten(list(zero(model).items())))
        after = {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}
        # With zero gradients the only change is weight decay.
        assert not np.array_equal(before["attn.qkv.weight"], after["attn.qkv.weight"])
        for name in ("ff.subkeys", "ff.fire", "ff.write"):
            assert np.array_equal(before[name], after[name]) != table_shrinks, name


def test_checkpoint_round_trip_with_separate_table_decay(tmp_path):
    cfg = build_config("smoke", "slot_soft", lr=0.01, warmup=5, table_weight_decay=0.0)
    model = slot_model()
    model.train()
    optimizer = build_optimizer(cfg)
    for _ in range(3):
        optimizer.update(model, nn.value_and_grad(model, lambda m: m(TOKENS).sum())(model)[1])
    path = tmp_path / "checkpoint.safetensors"
    save_checkpoint(path, model, optimizer, step=3)

    restored, restored_opt = slot_model(seed=5), build_optimizer(cfg)
    assert load_checkpoint(path, restored, restored_opt) == 3
    old = dict(tree_flatten(optimizer.state))
    new = dict(tree_flatten(restored_opt.state))
    assert old.keys() == new.keys()
    assert all(np.array_equal(np.array(old[k]), np.array(new[k])) for k in old)
    assert int(new["states.0.step"]) == int(new["states.1.step"]) == 3  # schedule position
    model.eval()
    assert np.array_equal(np.array(model(TOKENS)), np.array(restored(TOKENS)))


def tiny_data():
    return MultiHopData(GraphConfig(n_entities=40, n_relations=10, eval_size=30))


def test_scratchpad_evaluations_run_and_have_the_right_shape():
    data = tiny_data()
    model = slot_model()
    chains = data.eval_sets["two_hop_held_second"]
    with eval_mode(model):
        decode = decode_by_depth(model, data, chains, 4)
        assert decode["depth"] == [1, 2, 3, 4] and len(decode["bridge"]) == 4
        assert "first_hop_position_bridge" not in decode
        contents = slot_contents(model, data, chains)
        assert set(contents) == {"bridge_in_slot", "bridge_in_readout", "write_weight"}
        assert 0 <= contents["write_weight"] <= 1
        overlap = retrieval_overlap(model, data, chains)
        assert np.array(overlap["same"]).shape == (2, 2)  # steps, not passes
        assert forward(model, mx.array(data.encode(chains)[0]), 3).shape == (len(chains), 40)
    calls = two_call_diagnose(model, data, (1, 2), (2,))
    assert set(calls["two_hop_held_second"]) == {"1", "2"}

    mx.random.seed(0)
    original = RecurrentModel(small())
    with eval_mode(original):
        decode = decode_by_depth(original, data, chains, 4)
    assert len(decode["first_hop_position_bridge"]) == 4


def test_a_token_form_slot_decodes_to_the_entity_it_holds():
    data = tiny_data()
    model = slot_model()
    chains = data.eval_sets["two_hop_id"]
    bridge = data.tail[chains[:, 0], chains[:, 1]]
    # Make step 1 write the true bridge: an oracle for the measurement itself.
    original = model.__class__._scratchpad_forward

    def oracle(self, tokens, n_passes, steps, return_retrieval, return_trace):
        logits, trace = original(self, tokens, n_passes, steps, False, True)
        slots = self.embed.weight[mx.array(bridge[: tokens.shape[0]])][:, None, :]
        trace["slots"][0] = slots
        return logits, trace

    model.__class__._scratchpad_forward = oracle
    try:
        with eval_mode(model):
            assert slot_contents(model, data, chains, batch_size=len(chains))["bridge_in_slot"] == 1.0
    finally:
        model.__class__._scratchpad_forward = original


def test_slot_run_order_and_baseline_check():
    assert SLOT_RUNS == [
        ("slot_soft", 0), ("slot_raw", 0), ("slot_soft_prefix", 0),
        ("slot_soft", 1), ("slot_soft", 2), ("slot_soft_k3", 0),
    ]
    base = build_config("full", "C", steps=128200)
    same = planned_config("full", "slot_soft", [], 128200)
    assert baseline_differences(base, same) == []
    assert same.steps == 128200 and model_config(same).num_slots == 1

    changed = planned_config("full", "slot_raw", ["--lr", "0.0003", "--top-k", "16"], 1000)
    problems = " ".join(baseline_differences(base, changed))
    assert "lr:" in problems and "steps:" in problems and "model top_k" in problems
    smaller_graph = planned_config("pilot", "slot_soft", [], 128200)
    assert any(d.startswith("graph") for d in baseline_differences(base, smaller_graph))


def test_train_evaluate_diagnose_and_report_a_slot_run(tmp_path):
    cfg = build_config(
        "smoke", "slot_soft_prefix",
        graph={"n_entities": 30, "n_relations": 8, "eval_size": 20},
        model={"d_model": 16, "n_heads": 2, "n_subkeys": 4, "d_key": 8, "top_k": 2, "lookup_heads": 2},
        steps=8, batch_size=8, warmup=2, eval_every=4, log_every=4,
    )
    run_dir = tmp_path / "slot_soft_prefix_s0"
    final = run(cfg, run_dir)
    assert final["depth_unit"] == "steps" and final["eval_depth"] == 2
    assert set(final["accuracy_by_steps"]) == {"1", "2", "3", "4", "6", "8"}
    assert final["trained_steps"] == [2, 3, 4]
    assert set(final["slots"]["two_hop_held_second"]) >= {"bridge_in_slot"}
    assert set(final["two_call"]["two_hop_held_second"]) == {"1", "2"}
    assert "warmup" in json.loads((run_dir / "config.json").read_text())["lr_schedule"]

    cfg2, data, model, step = load_run(run_dir)
    assert step == 8 and model.cfg.num_slots == 1 and cfg2.prefix_loss_weight == 1.0

    result = diagnose_run(run_dir)
    assert result["reproduces_final"]["status"] == "identical"
    text = to_markdown(result)
    assert "Slot contents after step 1" in text and "two calls, 1 steps each" in text

    report = build_report([tmp_path])
    assert "accuracy by number of outer steps" in report
    assert "what is in the slot after step 1" in report
    assert "Two-call ceiling" in report
    assert "| slot_soft_prefix | 1 |" in report
