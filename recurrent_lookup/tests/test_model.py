"""Checks on the model: retrieval, causality, parameter matching, resume."""

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from recurrent_lookup.configs import build_config, model_config
from recurrent_lookup.evaluate import eval_mode
from recurrent_lookup.model import (
    CausalSelfAttention,
    ModelConfig,
    ProductKeyLookup,
    RecurrentModel,
    count_parameters,
    lookup_param_count,
)
from recurrent_lookup.train import load_checkpoint, save_checkpoint


def small(**kw):
    base = dict(vocab_size=50, n_entities=40, d_model=32, n_subkeys=8, d_key=16, top_k=4)
    base.update(kw)
    return ModelConfig(**base)


def test_product_key_retrieval_matches_brute_force():
    mx.random.seed(0)
    cfg = small(query_norm="none", lookup_heads=3)
    lookup = ProductKeyLookup(cfg)
    x = mx.random.normal((5, 3, cfg.d_model))
    index, score = lookup.retrieve(x)

    n, half = cfg.n_subkeys, cfg.d_key // 2
    q = np.array(lookup.query(x)).reshape(5, 3, 3, 2, half)
    keys = np.array(lookup.subkeys)  # (H, 2, n, half)
    first = np.einsum("bthd,hnd->bthn", q[..., 0, :], keys[:, 0])
    second = np.einsum("bthd,hnd->bthn", q[..., 1, :], keys[:, 1])
    full = (first[..., :, None] + second[..., None, :]).reshape(5, 3, 3, n * n)

    index, score = np.array(index), np.array(score)
    np.testing.assert_allclose(
        np.take_along_axis(full, index.astype(np.int64), axis=-1), score, atol=1e-4
    )
    best = -np.sort(-full, axis=-1)[..., : cfg.top_k]
    np.testing.assert_allclose(-np.sort(-score, axis=-1), best, atol=1e-4)


def test_attention_is_causal():
    mx.random.seed(0)
    attn = CausalSelfAttention(32, 4)
    x = mx.random.normal((2, 3, 32))
    changed = mx.concatenate([x[:, :2], mx.random.normal((2, 1, 32))], axis=1)
    np.testing.assert_allclose(
        np.array(attn(x)[:, :2]), np.array(attn(changed)[:, :2]), atol=1e-5
    )


def test_two_hop_prefix_retrieves_what_the_one_hop_question_retrieves():
    mx.random.seed(0)
    model = RecurrentModel(small())
    model.eval()
    two = mx.array([[3, 41, 45], [7, 42, 44]])
    _, r_two = model(two, return_retrieval=True)
    _, r_one = model(two[:, :2], return_retrieval=True)
    for a, b in zip(r_two, r_one):
        assert np.array_equal(np.sort(np.array(a[:, :2]), -1), np.sort(np.array(b), -1))


def test_lookup_on_first_pass_only():
    model = RecurrentModel(small(lookup_passes="first"))
    _, retrieved = model(mx.array([[3, 41, 45]]), return_retrieval=True)
    assert retrieved[0] is not None and all(r is None for r in retrieved[1:])


def test_parameter_counts():
    cfg = small()
    counts = count_parameters(RecurrentModel(cfg))
    assert counts["feed_forward"] == lookup_param_count(cfg)

    c = RecurrentModel(model_config(build_config("smoke", "C")))
    b = RecurrentModel(model_config(build_config("smoke", "B")))
    c_total, b_total = count_parameters(c)["total"], count_parameters(b)["total"]
    assert abs(c_total - b_total) / c_total < 0.01


def test_table_sizes_of_the_e_runs():
    sizes = {v: model_config(build_config("full", v)).n_entries for v in ("E_small", "C", "E_large")}
    assert sizes == {"E_small": 1024, "C": 16384, "E_large": 262144}


def test_query_norm_statistics_only_move_in_training():
    mx.random.seed(0)
    model = RecurrentModel(small())
    tokens = mx.array([[3, 41, 45], [7, 42, 44]])
    before = np.array(model.ff.norm.running_mean)
    with eval_mode(model):
        model(tokens)
    assert np.array_equal(before, np.array(model.ff.norm.running_mean))
    model(tokens)
    assert not np.array_equal(before, np.array(model.ff.norm.running_mean))
    assert "running_mean" not in model.ff.norm.trainable_parameters()


def test_checkpoint_round_trip(tmp_path):
    mx.random.seed(0)
    tokens = mx.array([[3, 41, 45], [7, 42, 44]])
    model = RecurrentModel(small())
    optimizer = optim.AdamW(learning_rate=1e-3)
    optimizer.init(model.trainable_parameters())
    model(tokens)  # move the running statistics off their initial values
    path = tmp_path / "checkpoint.safetensors"
    save_checkpoint(path, model, optimizer, step=17)

    mx.random.seed(1)
    restored = RecurrentModel(small())
    restored_opt = optim.AdamW(learning_rate=1e-3)
    assert load_checkpoint(path, restored, restored_opt) == 17
    model.eval(), restored.eval()
    np.testing.assert_allclose(
        np.array(model(tokens)), np.array(restored(tokens)), atol=1e-6
    )


def test_every_option_trains_one_step():
    import mlx.nn as nn

    tokens, targets = mx.array([[3, 41, 45], [7, 42, 44]]), mx.array([5, 9])
    options = [
        dict(query_norm="ema"),
        dict(query_norm="layer"),
        dict(query_norm="none"),
        dict(lookup_passes="first"),
        dict(tie_embeddings=False),
        dict(ff="dense", d_ff=64),
    ]
    for option in options:
        mx.random.seed(0)
        model = RecurrentModel(small(**option))
        optimizer = optim.AdamW(learning_rate=1e-2)

        def loss_fn(model):
            return nn.losses.cross_entropy(model(tokens), targets, reduction="mean")

        before = loss_fn(model).item()
        for _ in range(5):
            _, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
        assert loss_fn(model).item() < before, option
        assert model(tokens).shape == (2, 40)
