"""Ouro with a lookup table beside one of its layers, and small adapters.

Ouro's own weights are frozen. Two kinds of trainable additions exist:

* A table (the product-key lookup from ``recurrent_lookup``) that reads the
  same state as the MLP of one layer and adds its result to the state
  alongside the MLP's. Because Ouro applies its whole layer stack four times,
  the table is consulted four times per token, once per loop, each time with
  a query formed from the current state. Its stored vectors start at zero, so
  before training the model is exactly Ouro.

* Low-rank adapters on the query and value projections of every attention
  layer: small trained corrections added beside the frozen weights. They also
  start as no change.

Either can be switched off when the model is called, which is how the
evaluation tells what each one contributes.

For the take-over experiment the MLP of some layers can also be removed. A
table at such a layer then stands in for the MLP instead of sitting beside
it, and so can a small dense MLP, for comparison.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from phrase_scratchpad.model import MaskedLookup

from .ouro import Ouro


@dataclass
class AddonConfig:
    # Layers that get a table beside their MLP. None: the middle layer
    # (layer 12 of Ouro's 24); () for no table.
    table_layers: tuple = None
    n_subkeys: int = 128  # each table has n_subkeys ** 2 entries
    lookup_heads: int = 4  # independent queries per token
    top_k: int = 8  # entries retrieved per query
    d_key: int = 64  # query width per head
    adapter_rank: int = 0  # 0 for no adapters
    adapter_scale: float = 2.0
    # Layers whose own MLP is removed. A table at one of these layers stands
    # in for the MLP; ``dense_width`` puts a small MLP of that hidden width at
    # each of them instead (0 for none).
    remove_layers: tuple = ()
    dense_width: int = 0


@dataclass
class TableConfig:
    """What ``ProductKeyLookup`` reads from its configuration."""

    d_model: int
    n_subkeys: int
    lookup_heads: int
    top_k: int
    d_key: int
    query_norm: str = "ema"

    @property
    def n_entries(self):
        return self.n_subkeys**2


class Table(MaskedLookup):
    """The lookup, working in 32-bit whatever precision the model runs in."""

    def __init__(self, cfg: TableConfig):
        super().__init__(cfg)
        self.write = mx.zeros_like(self.write)  # adds nothing until trained

    def __call__(self, x):
        y = x.astype(mx.float32)
        # Ouro's normed states are small; bring them to unit size so entries
        # respond as they do in the small models this table was built for.
        y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + 1e-6)
        out, index = super().__call__(y)
        return out.astype(x.dtype), index


class SmallMLP(nn.Module):
    """A small MLP of Ouro's own kind, to stand in for a removed one.

    It reads what the removed MLP read, brought to unit size as for the
    table, and works in 32-bit. Its output layer starts at zero.
    """

    def __init__(self, d, width):
        super().__init__()
        self.gate = mx.random.normal((d, width)) * (d**-0.5)
        self.up = mx.random.normal((d, width)) * (d**-0.5)
        self.down = mx.zeros((width, d))

    def __call__(self, x):
        y = x.astype(mx.float32)
        y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + 1e-6)
        return ((nn.silu(y @ self.gate) * (y @ self.up)) @ self.down).astype(x.dtype)


class Adapter(nn.Module):
    """Low-rank corrections to one attention layer's query and value projections."""

    def __init__(self, d_in, d_out, rank, scale):
        super().__init__()
        self.scale = scale
        self.q_down = mx.random.normal((d_in, rank)) * (d_in**-0.5)
        self.q_up = mx.zeros((rank, d_out))
        self.v_down = mx.random.normal((d_in, rank)) * (d_in**-0.5)
        self.v_up = mx.zeros((rank, d_out))

    def __call__(self, x, learns=True):
        """(change to queries, change to values). With ``learns`` false the
        adapter is applied but receives no gradient from this call."""
        keep = (lambda p: p) if learns else mx.stop_gradient
        y = x.astype(mx.float32)
        dq = ((y @ keep(self.q_down)) @ keep(self.q_up)) * self.scale
        dv = ((y @ keep(self.v_down)) @ keep(self.v_up)) * self.scale
        return dq.astype(x.dtype), dv.astype(x.dtype)


class LookupOuro(nn.Module):
    def __init__(self, base: Ouro, cfg: AddonConfig):
        super().__init__()
        self.cfg = cfg
        self.base = base
        self.base.freeze()
        d = base.cfg.hidden_size
        inner = base.cfg.num_attention_heads * base.cfg.head_dim
        n_layers = base.cfg.num_hidden_layers
        layers = (n_layers // 2,) if cfg.table_layers is None else cfg.table_layers
        if any(not 0 <= i < n_layers for i in layers):
            raise ValueError(f"table layers {layers} are outside 0..{n_layers - 1}")
        self.table_layers = tuple(sorted(set(layers)))
        self.remove_layers = tuple(sorted(set(cfg.remove_layers)))
        if any(not 0 <= i < n_layers for i in self.remove_layers):
            raise ValueError(f"removed layers {self.remove_layers} are outside 0..{n_layers - 1}")
        if cfg.dense_width and not self.remove_layers:
            raise ValueError("a small MLP stands in for a removed one: name the layers to remove")
        table_cfg = TableConfig(d, cfg.n_subkeys, cfg.lookup_heads, cfg.top_k, cfg.d_key)
        self.tables = [Table(table_cfg) for _ in self.table_layers]
        self.dense = [
            SmallMLP(d, cfg.dense_width) for _ in (self.remove_layers if cfg.dense_width else ())
        ]
        self.adapters = [
            Adapter(d, inner, cfg.adapter_rank, cfg.adapter_scale)
            for _ in range(n_layers if cfg.adapter_rank else 0)
        ]

    def states(
        self, tokens, table=True, adapters=True, adapters_learn=True, loops=None,
        return_retrieved=False, original=False, return_adds=False,
    ):
        """The normed state after each loop: a list of (B, T, d) arrays.

        ``table`` and ``adapters`` switch the additions off for this call.
        ``table`` false also switches off a small MLP standing in for a
        removed one; the removed MLPs stay removed. ``original`` runs Ouro
        untouched: nothing added and nothing removed.
        ``adapters_learn`` false applies the adapters without training them.
        With ``return_retrieved`` also returns the entries each lookup chose.
        With ``return_adds`` also returns what the MLP half added to the state
        at each layer in ``remove_layers``, loop by loop and layer by layer:
        the original MLP's output when ``original``, otherwise what stands in
        for it (None where nothing does).
        """
        body = self.base.model
        h = body.embed_tokens(tokens)
        cos, sin = self.base.positions(tokens.shape[1])
        table = table and not original
        table_at = {layer: k for k, layer in enumerate(self.table_layers)} if table else {}
        dense_at = {layer: k for k, layer in enumerate(self.remove_layers)} if table else {}
        dense_at = dense_at if self.dense else {}
        removed = () if original else self.remove_layers
        use_adapters = adapters and not original and len(self.adapters) > 0
        states, retrieved, adds = [], [], []
        for _ in range(loops or self.base.cfg.total_ut_steps):
            for i, layer in enumerate(body.layers):
                h = layer.attend(h, cos, sin, self.adapters[i] if use_adapters else None, adapters_learn)
                add = None if i in removed else layer.feed_forward(h)
                if i in table_at or i in dense_at:
                    x = layer.post_attention_layernorm(h)
                    if i in table_at:
                        out, index = self.tables[table_at[i]](x)
                        retrieved.append(index)
                        add = out if add is None else add + out
                    if i in dense_at:
                        out = self.dense[dense_at[i]](x)
                        add = out if add is None else add + out
                if i in self.remove_layers:
                    adds.append(add)
                if add is not None:
                    h = h + add
            h = body.norm(h)
            states.append(h)
        out = [states]
        if return_retrieved:
            out.append(retrieved)
        if return_adds:
            out.append(adds)
        return out[0] if len(out) == 1 else tuple(out)

    def read(self, state):
        """Next-token logits for a state, in 32-bit."""
        return self.base.lm_head(state).astype(mx.float32)

    def answer_logits(self, tokens, **switches):
        """Logits for the token after the question, from the last loop: (B, vocab)."""
        return self.read(self.states(tokens, **switches)[-1][:, -1])

    # -- the trainable additions, for checkpoints -------------------------

    def addon_parameters(self):
        """Everything except Ouro's own weights, including the running
        statistics the tables keep for their queries."""
        return {
            name: value
            for name, value in tree_flatten(self.parameters())
            if not name.startswith("base.")
        }

    def count_parameters(self):
        groups = {"table": 0, "adapters": 0}
        kinds = {"tables": "table", "adapters": "adapters", "dense": "dense"}
        for name, value in tree_flatten(self.trainable_parameters()):
            kind = kinds[name.split(".")[0]]
            groups[kind] = groups.get(kind, 0) + value.size
        groups["trained"] = sum(groups.values())
        groups["frozen"] = sum(v.size for _, v in tree_flatten(self.base.parameters()))
        return groups

    def removed_parameters(self):
        """How many of Ouro's own numbers sit in the removed MLPs."""
        layers = self.base.model.layers
        return sum(
            v.size for i in self.remove_layers for _, v in tree_flatten(layers[i].mlp.parameters())
        )

    def stand_in_cost(self):
        """Multiplications per token, each time a layer is applied, for what
        stands in for one removed MLP, next to the removed MLP's own."""
        cfg, d = self.cfg, self.base.cfg.hidden_size
        cost = 0
        if any(i in self.table_layers for i in self.remove_layers):
            cost += d * cfg.lookup_heads * cfg.d_key  # forming the queries
            cost += cfg.lookup_heads * cfg.n_subkeys * cfg.d_key  # scoring the sub-keys
            cost += cfg.lookup_heads * min(cfg.top_k, cfg.n_subkeys) * 2 * d  # the entries read
        if self.dense:
            cost += 3 * d * cfg.dense_width
        return {"stand_in": cost, "removed_mlp": 3 * d * self.base.cfg.intermediate_size}
