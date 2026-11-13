"""A dense recurrent language-model core with optional learned expert memory."""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from recurrent_lookup.model import (
    CausalSelfAttention,
    DenseFF,
    ModelConfig,
    ProductKeyLookup,
)


@dataclass
class Config:
    vocab_size: int = 257  # UTF-8 bytes plus a document boundary
    context: int = 128
    width: int = 128
    heads: int = 4
    passes: int = 4
    subkeys: int = 64
    lookup_heads: int = 2
    top_k: int = 4
    key_width: int = 32
    lookup_budget: int = 2  # maximum memory contributions per position

    def __post_init__(self):
        if self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("width must be positive and divisible by heads")
        if self.passes < 1 or not 0 <= self.lookup_budget <= self.passes:
            raise ValueError("lookup_budget must be between zero and passes")
        if (
            min(
                self.context,
                self.subkeys,
                self.lookup_heads,
                self.top_k,
                self.key_width,
            )
            < 1
            or self.key_width % 2
        ):
            raise ValueError("positive sizes and an even key_width are required")


def is_memory(path, weight=None):
    # Router/query weights remain in the core; only keys and expert vectors
    # receive the memory decay. Embeddings are part of the penalized core.
    return path in ("memory.subkeys", "memory.fire", "memory.write")


class CoreMemoryLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = mx.random.normal((cfg.context, cfg.width)) * cfg.width**-0.5
        self.attn_norm = nn.LayerNorm(cfg.width)
        self.attn = CausalSelfAttention(cfg.width, cfg.heads)
        self.ff_norm = nn.LayerNorm(cfg.width)
        self.ff = DenseFF(cfg.width, 4 * cfg.width)
        self.memory_norm = nn.LayerNorm(cfg.width)
        self.memory = ProductKeyLookup(
            ModelConfig(
                vocab_size=cfg.vocab_size,
                n_entities=cfg.vocab_size,
                d_model=cfg.width,
                n_subkeys=cfg.subkeys,
                lookup_heads=cfg.lookup_heads,
                top_k=cfg.top_k,
                d_key=cfg.key_width,
                query_norm="layer",
            )
        )
        self.gate = nn.Linear(cfg.width, 1)
        self.gate.weight = mx.zeros_like(self.gate.weight)
        self.gate.bias = mx.zeros_like(self.gate.bias)
        self.out_norm = nn.LayerNorm(cfg.width)

    def __call__(self, tokens, mode="gated", key=None):
        """Return logits at all positions and logical lookup counts per position.

        off: genuinely skip retrieval. on: forced retrieval on every pass,
        bypassing the budget, for an upper-access ablation. gated: stochastic
        binary gates in training (straight-through gradients), deterministic
        threshold >= 0.5 in evaluation, with a hard per-position budget.

        Gated training/evaluation computes candidate retrievals even for gates
        that are zero. Counts measure contributions, NOT hardware savings.
        """
        if mode not in ("off", "on", "gated"):
            raise ValueError(mode)
        if tokens.shape[1] > self.cfg.context:
            raise ValueError("sequence exceeds configured context")
        h = self.embed(tokens) + self.position[: tokens.shape[1]]
        used = mx.zeros(tokens.shape)
        keys = (
            mx.random.split(key, self.cfg.passes)
            if key is not None
            else [None] * self.cfg.passes
        )
        for pass_index in range(self.cfg.passes):
            h = h + self.attn(self.attn_norm(h))
            dense, _ = self.ff(self.ff_norm(h))
            h = h + dense
            if mode == "off" or (mode == "gated" and self.cfg.lookup_budget == 0):
                continue
            x = self.memory_norm(h)
            if mode == "on":
                gate = mx.ones(tokens.shape)
            else:
                prob = mx.sigmoid(self.gate(x).squeeze(-1))
                binary = (
                    (mx.random.uniform(shape=prob.shape, key=keys[pass_index]) < prob)
                    if self.training
                    else (prob >= 0.5)
                )
                gate = prob + mx.stop_gradient(binary.astype(prob.dtype) - prob)
                gate = gate * mx.stop_gradient(
                    (used < self.cfg.lookup_budget).astype(prob.dtype)
                )
            value, _ = self.memory(x)
            h = h + gate[..., None] * value
            used = used + gate
        return self.out_norm(h) @ self.embed.weight.T, used
