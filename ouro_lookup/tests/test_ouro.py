import json

import mlx.core as mx
import numpy as np
import pytest

from ouro_lookup.ouro import OuroConfig, load_ouro, rotary_tables, rotate


def test_loads_and_runs(tiny_dir):
    model = load_ouro(tiny_dir)
    tokens = mx.array([[0, 5, 9, 3, 7]])
    states = model.loop_states(tokens)
    assert len(states) == model.cfg.total_ut_steps == 4
    assert states[0].shape == (1, 5, model.cfg.hidden_size)
    assert model(tokens).shape == (1, 5, model.cfg.vocab_size)
    assert len(model.loop_states(tokens, loops=2)) == 2
    assert len(load_ouro(tiny_dir, loops=3).loop_states(tokens)) == 3


def test_later_tokens_do_not_change_earlier_predictions(tiny_dir):
    model = load_ouro(tiny_dir)
    a = np.array(model(mx.array([[0, 5, 9, 3, 7]])))
    b = np.array(model(mx.array([[0, 5, 9, 8, 2]])))
    assert np.allclose(a[0, :3], b[0, :3], atol=1e-5)
    assert not np.allclose(a[0, 3:], b[0, 3:], atol=1e-3)


def test_word_order_matters(tiny_dir):
    model = load_ouro(tiny_dir)
    a = np.array(model(mx.array([[0, 5, 9, 3]]))[0, -1])
    b = np.array(model(mx.array([[0, 9, 5, 3]]))[0, -1])
    assert not np.allclose(a, b, atol=1e-3)


def test_rotation_keeps_length_and_depends_on_distance_only():
    cos, sin = rotary_tables(6, 8, 10000.0, mx.float32)
    q = mx.random.normal((1, 1, 6, 8))
    r = rotate(q, cos, sin)
    assert np.allclose(np.linalg.norm(np.array(r), axis=-1), np.linalg.norm(np.array(q), axis=-1), atol=1e-5)
    # The same vector at positions (1, 3) and (2, 4): equal dot products.
    v = mx.broadcast_to(mx.random.normal((1, 1, 1, 8)), (1, 1, 6, 8))
    rv = np.array(rotate(v, cos, sin))[0, 0]
    assert np.isclose(rv[1] @ rv[3], rv[2] @ rv[4], atol=1e-5)


def test_conversion_changes_precision(tiny_dir):
    model = load_ouro(tiny_dir, dtype=mx.float16)
    assert model.dtype == mx.float16
    assert model.loop_states(mx.array([[0, 5, 9]]))[-1].dtype == mx.float16


def test_rejects_what_this_port_does_not_implement(tmp_path, tiny_dir):
    raw = json.loads((tiny_dir / "config.json").read_text())
    for change in ({"num_key_value_heads": 2}, {"hidden_act": "gelu"}, {"layer_types": ["sliding_attention"]}):
        (tmp_path / "config.json").write_text(json.dumps({**raw, **change}))
        with pytest.raises(ValueError):
            OuroConfig.from_json(tmp_path / "config.json")


def test_checkpoint_must_match_the_model(tmp_path, tiny_dir):
    raw = json.loads((tiny_dir / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**raw, "num_hidden_layers": 2}))
    (tmp_path / "model.safetensors").write_bytes((tiny_dir / "model.safetensors").read_bytes())
    with pytest.raises(ValueError, match="does not match"):
        load_ouro(tmp_path)
