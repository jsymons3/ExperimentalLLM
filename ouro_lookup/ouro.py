"""Ouro, ByteDance's looped language model, in MLX.

Ouro-1.4B is an ordinary decoder-only transformer (24 layers, width 2048) whose
whole layer stack is applied ``total_ut_steps`` times (4 by default) with the
same weights. After each loop the state goes through the final norm; that
normed state is both what the output head reads and what the next loop starts
from. Paper: https://arxiv.org/abs/2510.25741. Weights (Apache-2.0):
https://huggingface.co/ByteDance/Ouro-1.4B.

This file follows the published PyTorch code (``modeling_ouro.py``) and uses
the checkpoint's parameter names, so the weights load as they are. It has no
generation cache: this project only runs whole short sequences.

The pieces of a layer are separate methods so that ``model.py`` can add a
lookup table and adapters between them.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class OuroConfig:
    vocab_size: int = 49152
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    total_ut_steps: int = 4  # how many times the layer stack is applied

    @classmethod
    def from_json(cls, path):
        raw = json.loads(Path(path).read_text())
        if raw.get("num_key_value_heads", raw["num_attention_heads"]) != raw["num_attention_heads"]:
            raise ValueError("this port assumes one key/value head per attention head")
        if raw.get("hidden_act", "silu") != "silu":
            raise ValueError(f"unsupported activation {raw['hidden_act']!r}")
        if "sliding_attention" in raw.get("layer_types", []):
            raise ValueError("this port has no sliding-window attention")
        return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})


class RMSNorm(nn.Module):
    """As in the original: the statistics are computed in 32-bit."""

    def __init__(self, dims, eps):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        y = x.astype(mx.float32)
        y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + self.eps)
        return self.weight * y.astype(x.dtype)


def rotary_tables(length, head_dim, theta, dtype):
    """cos and sin of every position's rotation angles: (length, head_dim / 2)."""
    inv_freq = theta ** (-mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
    angles = mx.arange(length).astype(mx.float32)[:, None] * inv_freq[None, :]
    return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def rotate(x, cos, sin):
    """Rotary position encoding of (B, heads, T, head_dim) queries or keys."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


class Attention(nn.Module):
    def __init__(self, cfg: OuroConfig):
        super().__init__()
        d, inner = cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(d, inner, bias=False)
        self.k_proj = nn.Linear(d, inner, bias=False)
        self.v_proj = nn.Linear(d, inner, bias=False)
        self.o_proj = nn.Linear(inner, d, bias=False)

    def __call__(self, x, cos, sin, adapter=None, adapter_learns=True):
        B, T, _ = x.shape
        q, v = self.q_proj(x), self.v_proj(x)
        if adapter is not None:
            # Small trained corrections to the query and value projections.
            dq, dv = adapter(x, adapter_learns)
            q, v = q + dq, v + dv

        def heads(y):
            return y.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(self.k_proj(x)), heads(v)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask="causal" if T > 1 else None
        )
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: OuroConfig):
        super().__init__()
        d, inner = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(d, inner, bias=False)
        self.up_proj = nn.Linear(d, inner, bias=False)
        self.down_proj = nn.Linear(inner, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    """Attention then MLP, each normed before and after ("sandwich" norm)."""

    def __init__(self, cfg: OuroConfig):
        super().__init__()
        d, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(d, eps)
        self.input_layernorm_2 = RMSNorm(d, eps)
        self.post_attention_layernorm = RMSNorm(d, eps)
        self.post_attention_layernorm_2 = RMSNorm(d, eps)

    def attend(self, h, cos, sin, adapter=None, adapter_learns=True):
        """The state after the attention half of the layer."""
        y = self.self_attn(self.input_layernorm(h), cos, sin, adapter, adapter_learns)
        return h + self.input_layernorm_2(y)

    def feed_forward(self, h):
        """What the MLP half adds to the state."""
        return self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(h)))

    def __call__(self, h, cos, sin):
        h = self.attend(h, cos, sin)
        return h + self.feed_forward(h)


class Body(nn.Module):
    def __init__(self, cfg: OuroConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Layer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Trained to decide when to stop looping. Loaded but not used here:
        # every question runs all the loops.
        self.early_exit_gate = nn.Linear(cfg.hidden_size, 1)


class Ouro(nn.Module):
    def __init__(self, cfg: OuroConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Body(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype

    def positions(self, length):
        return rotary_tables(length, self.cfg.head_dim, self.cfg.rope_theta, self.dtype)

    def loop_states(self, tokens, loops=None):
        """The normed state after each loop: a list of (B, T, d) arrays."""
        h = self.model.embed_tokens(tokens)
        cos, sin = self.positions(tokens.shape[1])
        states = []
        for _ in range(loops or self.cfg.total_ut_steps):
            for layer in self.model.layers:
                h = layer(h, cos, sin)
            h = self.model.norm(h)
            states.append(h)
        return states

    def __call__(self, tokens, loops=None):
        """Next-token logits from the last loop: (B, T, vocab)."""
        return self.lm_head(self.loop_states(tokens, loops)[-1])


def load_ouro(directory, dtype=None, loops=None):
    """Build Ouro from a directory holding config.json and model.safetensors.

    ``dtype`` converts the weights (they are stored in bfloat16); ``loops``
    overrides how many times the layer stack is applied.
    """
    directory = Path(directory)
    cfg = OuroConfig.from_json(directory / "config.json")
    if loops:
        cfg.total_ut_steps = loops
    model = Ouro(cfg)
    weights = mx.load(str(directory / "model.safetensors"))
    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    if missing or extra:
        raise ValueError(f"checkpoint does not match the model: missing {missing}, extra {extra}")
    for name, value in weights.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"{name}: checkpoint {value.shape}, model {expected[name].shape}")
    if dtype is not None:
        weights = {name: value.astype(dtype) for name, value in weights.items()}
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model
