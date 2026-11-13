import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from core_memory.data import Corpus, articles, prepare
from core_memory.model import Config, CoreMemoryLM, is_memory
from core_memory.train import load_checkpoint, make_optimizer, save_checkpoint


def small(**kwargs):
    fields = {
        "width": 16,
        "heads": 2,
        "passes": 3,
        "context": 12,
        "subkeys": 4,
        "lookup_heads": 1,
        "top_k": 2,
        "key_width": 8,
        "lookup_budget": 2,
    }
    fields.update(kwargs)
    return CoreMemoryLM(Config(**fields))


def test_memory_off_is_a_real_bypass():
    class Bomb(nn.Module):
        def __call__(self, x):
            raise AssertionError("memory was accessed")

    m = small()
    m.memory = Bomb()
    m.eval()
    logits, calls = m(mx.array([[1, 2, 3]]), "off")
    assert logits.shape == (1, 3, 257)
    assert np.array(calls).sum() == 0


def test_budget_and_causal_gate():
    mx.random.seed(0)
    m = small()
    m.eval()
    m.gate.bias = mx.array([10.0])
    x = mx.array([[1, 2, 3, 4]])
    logits, calls = m(x)
    np.testing.assert_array_equal(np.array(calls), 2)
    changed, _other = m(mx.array([[1, 2, 9, 8]]))
    np.testing.assert_allclose(
        np.array(logits[:, :2]), np.array(changed[:, :2]), atol=1e-5
    )
    _, unlimited = m(x, "on")
    np.testing.assert_array_equal(np.array(unlimited), 3)
    m.train()
    for _ in range(3):
        assert np.array(m(x)[1]).max() <= 2
    m = small(lookup_budget=0)
    m.eval()
    np.testing.assert_allclose(np.array(m(x)[0]), np.array(m(x, "off")[0]), atol=1e-6)


def test_routing_and_memory_receive_gradients():
    mx.random.seed(3)
    m = small()
    m.eval()
    m.gate.bias = mx.array([1.0])
    x = mx.array([[1, 2, 3]])
    y = mx.array([[2, 3, 4]])

    def loss(m):
        logits, calls = m(x)
        return (
            nn.losses.cross_entropy(logits, y, reduction="mean") + 0.01 * calls.mean()
        )

    _, grads = nn.value_and_grad(m, loss)(m)
    flat = dict(tree_flatten(grads))
    for key in (
        "gate.weight",
        "memory.query.weight",
        "memory.subkeys",
        "memory.fire",
        "memory.write",
        "ff.up.weight",
    ):
        assert mx.abs(flat[key]).max().item() > 0, key


def test_differential_decay_includes_embeddings_but_not_table():
    m = small()
    before = {k: np.array(v) for k, v in tree_flatten(m.parameters())}
    opt = make_optimizer(0.1, 1, 0.5, 0)
    opt.update(m, tree_map(mx.zeros_like, m.trainable_parameters()))
    after = dict(tree_flatten(m.parameters()))
    for k in ("memory.fire", "memory.write", "memory.subkeys"):
        assert is_memory(k)
        np.testing.assert_array_equal(before[k], np.array(after[k]))
    np.testing.assert_allclose(
        np.array(after["embed.weight"]), before["embed.weight"] * 0.95, atol=1e-6
    )
    np.testing.assert_array_equal(
        before["out_norm.weight"], np.array(after["out_norm.weight"])
    )


def test_document_boundaries_and_duplicates(tmp_path):
    prepare(
        {"train": ["a" * 20, "b" * 20], "valid": ["c" * 20]}, tmp_path, {"test": True}
    )
    c = Corpus(tmp_path, "train", 8)
    x, y = c.sample(np.random.default_rng(0), 100)
    for row in np.concatenate([x, y], axis=1):
        assert not (97 in row and 98 in row)
    with pytest.raises(ValueError, match="duplicate"):
        prepare({"train": ["same"], "valid": ["same"]}, tmp_path / "bad", {})
    with pytest.raises(ValueError, match="long enough"):
        Corpus(tmp_path, "valid", 30)
    assert (
        len(
            articles(
                [
                    " = First = ",
                    "text",
                    " == Section == ",
                    "more",
                    " = Second = ",
                    "other",
                ]
            )
        )
        == 2
    )


def test_checkpoint_restores_optimizer_sampling_and_routing(tmp_path):
    mx.random.seed(9)
    m = small()
    opt = make_optimizer(0.001, 3, 0.1, 0)
    rng = np.random.default_rng(7)
    x = mx.array([[1, 2, 3]])
    y = mx.array([[2, 3, 4]])

    def update(m, opt):
        key = mx.random.key(int(rng.integers(0, 2**32)))

        def loss(m):
            return nn.losses.cross_entropy(m(x, key=key)[0], y, reduction="mean")

        value, grads = nn.value_and_grad(m, loss)(m)
        opt.update(m, grads)
        mx.eval(m.parameters(), opt.state, value)
        return value.item()

    update(m, opt)
    path = tmp_path / "checkpoint.safetensors"
    save_checkpoint(path, m, opt, 4, rng)
    expected_sample = rng.integers(100, size=10)
    expected = update(m, opt)
    restored = small()
    ropt = make_optimizer(0.001, 3, 0.1, 0)
    assert load_checkpoint(path, restored, ropt, rng) == 4
    np.testing.assert_array_equal(rng.integers(100, size=10), expected_sample)
    assert update(restored, ropt) == pytest.approx(expected, abs=1e-6)
    for (k, a), (_, b) in zip(
        tree_flatten(m.parameters()), tree_flatten(restored.parameters())
    ):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6, err_msg=k)


def test_training_resume_and_configuration_guard(tmp_path):
    from core_memory.train import parse_args, run

    corpus = tmp_path / "data"
    prepare(
        {"train": ["training text " * 30], "valid": ["validation text " * 30]},
        corpus,
        {},
    )
    common = [
        "--data",
        str(corpus),
        "--width",
        "16",
        "--heads",
        "2",
        "--context",
        "8",
        "--subkeys",
        "4",
        "--key-width",
        "8",
        "--passes",
        "2",
        "--lookup-budget",
        "1",
        "--batch-size",
        "2",
        "--warmup",
        "1",
        "--gate-warmup",
        "0",
        "--memory-dropout",
        "0",
        "--eval-every",
        "2",
        "--eval-batches",
        "1",
    ]
    run(parse_args(common + ["--out", str(tmp_path / "full"), "--steps", "4"]))
    run(parse_args(common + ["--out", str(tmp_path / "resume"), "--steps", "2"]))
    run(
        parse_args(
            common + ["--out", str(tmp_path / "resume"), "--steps", "4", "--resume"]
        )
    )
    a = mx.load(str(tmp_path / "full/checkpoint.safetensors"))
    b = mx.load(str(tmp_path / "resume/checkpoint.safetensors"))
    for k in a:
        np.testing.assert_allclose(np.array(a[k]), np.array(b[k]), atol=1e-6, err_msg=k)
    with pytest.raises(ValueError, match="change training"):
        run(
            parse_args(
                common
                + [
                    "--out",
                    str(tmp_path / "resume"),
                    "--steps",
                    "6",
                    "--resume",
                    "--lr",
                    ".02",
                ]
            )
        )
