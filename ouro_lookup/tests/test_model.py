import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ouro_lookup.model import AddonConfig, LookupOuro
from ouro_lookup.ouro import load_ouro

TOKENS = mx.array([[0, 5, 9, 3, 7], [0, 6, 2, 8, 4]])


def build(tiny_dir, **kw):
    kw = {"n_subkeys": 16, "adapter_rank": 4, **kw}
    mx.random.seed(0)
    return LookupOuro(load_ouro(tiny_dir), AddonConfig(**kw))


def trained_a_little(model):
    """Give the additions some non-zero content, as training would."""
    for table in model.tables:
        table.write = mx.random.normal(table.write.shape) * 0.1
    for adapter in model.adapters:
        adapter.q_up = mx.random.normal(adapter.q_up.shape) * 0.1
        adapter.v_up = mx.random.normal(adapter.v_up.shape) * 0.1
    return model


def logits(model, **switches):
    return np.array(model.answer_logits(TOKENS, **switches))


def test_starts_as_the_unmodified_model(tiny_dir):
    model = build(tiny_dir).eval()
    base = np.array(model.base(TOKENS)[:, -1])
    assert np.allclose(logits(model), base, atol=1e-6)


def test_each_addition_can_be_switched_off(tiny_dir):
    model = trained_a_little(build(tiny_dir)).eval()
    base = np.array(model.base(TOKENS)[:, -1])
    everything = logits(model)
    assert np.allclose(logits(model, table=False, adapters=False), base, atol=1e-6)
    assert not np.allclose(everything, base, atol=1e-3)
    assert not np.allclose(logits(model, table=False), everything, atol=1e-3)
    assert not np.allclose(logits(model, adapters=False), everything, atol=1e-3)
    assert not np.allclose(logits(model, table=False), base, atol=1e-3)


def test_table_sits_at_the_middle_layer_by_default(tiny_dir):
    assert build(tiny_dir).table_layers == (1,)  # 3 layers
    assert build(tiny_dir, table_layers=(2, 0)).table_layers == (0, 2)
    no_table = build(tiny_dir, table_layers=())
    assert no_table.tables == [] and len(no_table.adapters) == 3
    assert build(tiny_dir, adapter_rank=0).adapters == []


def test_the_table_is_consulted_once_per_loop(tiny_dir):
    model = build(tiny_dir).eval()
    states, retrieved = model.states(TOKENS, return_retrieved=True)
    assert len(states) == 4 and len(retrieved) == 4
    assert retrieved[0].shape == (2, 5, 4, 8)  # questions, positions, heads, top-k
    assert len(model.states(TOKENS, loops=2)) == 2


def test_only_the_additions_are_trained(tiny_dir):
    model = build(tiny_dir)
    names = [name for name, _ in tree_flatten(model.trainable_parameters())]
    assert names and all(name.startswith(("tables.", "adapters.")) for name in names)
    assert not any("running" in name for name in names)
    saved = model.addon_parameters()
    assert not any(name.startswith("base.") for name in saved)
    assert any(name.endswith("running_mean") for name in saved)
    counts = model.count_parameters()
    assert counts["trained"] == sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    assert counts["frozen"] == sum(v.size for _, v in tree_flatten(model.base.parameters()))


def answer_loss(model, **switches):
    return nn.losses.cross_entropy(model.answer_logits(TOKENS, **switches), mx.array([3, 4])).mean()


def test_gradients_reach_the_table_and_the_adapters(tiny_dir):
    model = trained_a_little(build(tiny_dir)).eval()
    _, grads = nn.value_and_grad(model, answer_loss)(model)
    flat = dict(tree_flatten(grads))
    assert not any(name.startswith("base.") for name in flat)
    for name in ("tables.0.write", "tables.0.fire", "tables.0.subkeys", "tables.0.query.weight",
                 "adapters.0.q_up", "adapters.2.v_down"):
        assert np.abs(np.array(flat[name])).max() > 0, name


def test_adapters_can_be_applied_without_being_trained(tiny_dir):
    model = trained_a_little(build(tiny_dir)).eval()
    frozen = lambda m: answer_loss(m, adapters_learn=False)
    _, grads = nn.value_and_grad(model, frozen)(model)
    flat = dict(tree_flatten(grads))
    assert all(np.abs(np.array(v)).max() == 0 for k, v in flat.items() if k.startswith("adapters."))
    assert np.abs(np.array(flat["tables.0.write"])).max() > 0
    # The adapters are still applied: the answer is the same as with them learning.
    assert np.allclose(logits(model, adapters_learn=False), logits(model), atol=1e-6)


def test_runs_in_reduced_precision(tiny_dir):
    mx.random.seed(0)
    model = LookupOuro(load_ouro(tiny_dir, dtype=mx.float16), AddonConfig(n_subkeys=16, adapter_rank=4))
    model = trained_a_little(model).eval()
    states = model.states(TOKENS)
    assert states[-1].dtype == mx.float16
    out = model.answer_logits(TOKENS)
    assert out.dtype == mx.float32 and np.isfinite(np.array(out)).all()
    assert model.tables[0].write.dtype == mx.float32
