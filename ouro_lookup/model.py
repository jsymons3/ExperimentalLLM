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
        table_cfg = TableConfig(d, cfg.n_subkeys, cfg.lookup_heads, cfg.top_k, cfg.d_key)
        self.tables = [Table(table_cfg) for _ in self.table_layers]
        self.adapters = [
            Adapter(d, inner, cfg.adapter_rank, cfg.adapter_scale)
            for _ in range(n_layers if cfg.adapter_rank else 0)
        ]

    def states(
        self, tokens, table=True, adapters=True, adapters_learn=True, loops=None,
        return_retrieved=False,
    ):
        """The normed state after each loop: a list of (B, T, d) arrays.

        ``table`` and ``adapters`` switch the additions off for this call.
        ``adapters_learn`` false applies the adapters without training them.
        With ``return_retrieved`` also returns the entries each lookup chose.
        """
        body = self.base.model
        h = body.embed_tokens(tokens)
        cos, sin = self.base.positions(tokens.shape[1])
        table_at = {layer: k for k, layer in enumerate(self.table_layers)} if table else {}
        use_adapters = adapters and len(self.adapters) > 0
        states, retrieved = [], []
        for _ in range(loops or self.base.cfg.total_ut_steps):
            for i, layer in enumerate(body.layers):
                h = layer.attend(h, cos, sin, self.adapters[i] if use_adapters else None, adapters_learn)
                add = layer.feed_forward(h)
                if i in table_at:
                    out, index = self.tables[table_at[i]](layer.post_attention_layernorm(h))
                    add = add + out
                    retrieved.append(index)
                h = h + add
            h = body.norm(h)
            states.append(h)
        return (states, retrieved) if return_retrieved else states

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
        for name, value in tree_flatten(self.trainable_parameters()):
            groups["table" if name.startswith("tables.") else "adapters"] += value.size
        groups["trained"] = groups["table"] + groups["adapters"]
        groups["frozen"] = sum(v.size for _, v in tree_flatten(self.base.parameters()))
        return groups
