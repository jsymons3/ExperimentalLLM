"""One small transformer block applied repeatedly, with a learned lookup table.

The block is: causal self-attention, then a feed-forward part. The feed-forward
part is either an ordinary MLP (``ff="dense"``) or a product-key lookup over a
table of single-neuron experts (``ff="lookup"``), as in "Mixture of A Million
Experts" (https://arxiv.org/abs/2407.04153). The same block, with the same
weights, is applied ``n_passes`` times, so with the lookup every pass issues a
new query from the updated state.

Nothing in the table is written by hand. Keys, query projection and the stored
neurons all start random and are trained by the task loss.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class ModelConfig:
    vocab_size: int
    n_entities: int
    d_model: int = 128
    n_heads: int = 4
    max_len: int = 3
    n_passes: int = 4
    ff: str = "lookup"  # "lookup" or "dense"
    d_ff: int = 512  # hidden width when ff == "dense"
    n_subkeys: int = 128  # the table has n_subkeys ** 2 entries
    lookup_heads: int = 4  # independent queries per token
    top_k: int = 8  # entries retrieved per query
    d_key: int = 64  # query width per head, split into two halves
    query_norm: str = "ema"  # "ema", "layer" or "none"
    lookup_passes: str = "every"  # "every", or "first" (pass 1 only)
    tie_embeddings: bool = True

    @property
    def n_entries(self):
        return self.n_subkeys**2


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x):
        B, T, D = x.shape
        H = self.n_heads
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, D // H).transpose(0, 2, 3, 1)
        v = v.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)
        scores = (q @ k) * (D // H) ** -0.5
        scores = scores + mx.triu(mx.full((T, T), -1e9), k=1)
        y = mx.softmax(scores, axis=-1) @ v
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, D))


class DenseFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def __call__(self, x):
        return self.down(nn.gelu(self.up(x))), None


class QueryNorm(nn.Module):
    """Normalise each query dimension to zero mean and unit variance.

    Product-key memories normalise queries with batch norm so that queries
    spread over the whole key set and more of the table gets used. Ordinary
    batch norm uses the statistics of the current call while training and
    running averages afterwards. Here the same layer is called once per pass,
    so those two would disagree, and the same question would be normalised
    differently depending on the pass it is asked in. To keep a query's
    meaning independent of the pass, this layer always normalises with the
    running averages, and only updates them while training.
    """

    def __init__(self, dims, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.running_mean = mx.zeros((dims,))
        self.running_var = mx.ones((dims,))
        self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def unfreeze(self, *args, **kwargs):
        super().unfreeze(*args, **kwargs)
        self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def __call__(self, q):
        if self.training:
            flat = mx.stop_gradient(q.reshape(-1, q.shape[-1]))
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * flat.mean(axis=0)
            self.running_var = (1 - m) * self.running_var + m * flat.var(axis=0)
        return (q - self.running_mean) * mx.rsqrt(self.running_var + self.eps)


class ProductKeyLookup(nn.Module):
    """Top-k retrieval over n_subkeys**2 single-neuron experts.

    Entry i holds two vectors: ``fire[i]`` decides how strongly the entry
    responds to the current state, and ``write[i]`` is what it adds to the
    state. The output for one token is

        sum over heads and retrieved entries of
            softmax(key score) * gelu(fire[i] . x) * write[i]

    Finding the best of n**2 keys costs about 2n comparisons: the query is cut
    in half, each half is scored against its own n sub-keys, and a full key is
    a pair of sub-keys whose score is the sum of the two half scores.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        assert cfg.d_key % 2 == 0
        self.n_heads = cfg.lookup_heads
        self.n_subkeys = cfg.n_subkeys
        self.top_k = min(cfg.top_k, cfg.n_subkeys)
        self.half = cfg.d_key // 2
        self.query_norm = cfg.query_norm

        self.query = nn.Linear(d, self.n_heads * cfg.d_key)
        if cfg.query_norm == "ema":
            self.norm = QueryNorm(self.n_heads * cfg.d_key)
        elif cfg.query_norm == "layer":
            self.norm = nn.LayerNorm(self.half)
        elif cfg.query_norm != "none":
            raise ValueError(f"unknown query_norm {cfg.query_norm!r}")

        # Each head has its own sub-keys; all heads share the entries.
        self.subkeys = mx.random.normal(
            (self.n_heads, 2, cfg.n_subkeys, self.half)
        ) * (self.half**-0.5)
        self.fire = mx.random.normal((cfg.n_entries, d)) * (d**-0.5)
        self.write = mx.random.normal((cfg.n_entries, d)) * 0.02

    def retrieve(self, x):
        """Return (index, score), each (B, T, heads, top_k)."""
        B, T, _ = x.shape
        H, n, k = self.n_heads, self.n_subkeys, self.top_k

        q = self.query(x)
        if self.query_norm == "ema":
            q = self.norm(q)
        q = q.reshape(B, T, H, 2, self.half)
        if self.query_norm == "layer":
            q = self.norm(q)

        # Score each half of the query against its own sub-keys: (B,T,H,2,n).
        sub = mx.matmul(self.subkeys, q[..., None]).squeeze(-1)
        top = mx.argpartition(-mx.stop_gradient(sub), kth=k - 1, axis=-1)[..., :k]
        top_score = mx.take_along_axis(sub, top, axis=-1)

        # The best k full keys are among the k x k pairs of per-half winners.
        pair_score = top_score[..., 0, :, None] + top_score[..., 1, None, :]
        pair_index = top[..., 0, :, None] * n + top[..., 1, None, :]
        pair_score = pair_score.reshape(B, T, H, k * k)
        pair_index = pair_index.reshape(B, T, H, k * k)
        best = mx.argpartition(
            -mx.stop_gradient(pair_score), kth=k - 1, axis=-1
        )[..., :k]
        index = mx.take_along_axis(pair_index, best, axis=-1)
        score = mx.take_along_axis(pair_score, best, axis=-1)
        return index, score

    def __call__(self, x):
        index, score = self.retrieve(x)
        weight = mx.softmax(score, axis=-1)  # (B,T,H,k)
        fire = self.fire[index]  # (B,T,H,k,d)
        write = self.write[index]
        act = nn.gelu((fire * x[:, :, None, None, :]).sum(axis=-1))
        out = ((weight * act)[..., None] * write).sum(axis=(2, 3))
        return out, index


class RecurrentModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.pos = mx.random.normal((cfg.max_len, d)) * (d**-0.5)
        self.norm_attn = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, cfg.n_heads)
        self.norm_ff = nn.LayerNorm(d)
        if cfg.ff == "dense":
            self.ff = DenseFF(d, cfg.d_ff)
        elif cfg.ff == "lookup":
            self.ff = ProductKeyLookup(cfg)
        else:
            raise ValueError(f"unknown ff {cfg.ff!r}")
        self.norm_out = nn.LayerNorm(d)
        if not cfg.tie_embeddings:
            self.unembed = nn.Linear(d, cfg.n_entities, bias=False)

    def __call__(self, tokens, n_passes=None, return_retrieval=False):
        """tokens: (B, T). Returns logits over entities for the last position.

        With ``return_retrieval`` also returns, for each pass, the table
        indices retrieved, shape (B, T, heads, top_k), or None for a pass
        without a lookup.
        """
        cfg = self.cfg
        n_passes = n_passes or cfg.n_passes
        h = self.embed(tokens) + self.pos[: tokens.shape[1]]
        retrieved = []
        for p in range(n_passes):
            h = h + self.attn(self.norm_attn(h))
            if cfg.ff == "dense" or cfg.lookup_passes == "every" or p == 0:
                out, index = self.ff(self.norm_ff(h))
                h = h + out
                retrieved.append(index)
            else:
                retrieved.append(None)
        h = self.norm_out(h[:, -1])
        if cfg.tie_embeddings:
            logits = h @ self.embed.weight[: cfg.n_entities].T
        else:
            logits = self.unembed(h)
        return (logits, retrieved) if return_retrieval else logits


def lookup_param_count(cfg: ModelConfig):
    """Trainable parameters in a ProductKeyLookup built from cfg."""
    d, H, half = cfg.d_model, cfg.lookup_heads, cfg.d_key // 2
    n = (d + 1) * H * cfg.d_key  # query projection
    n += H * 2 * cfg.n_subkeys * half  # sub-keys
    n += 2 * cfg.n_entries * d  # fire and write
    if cfg.query_norm == "layer":
        n += 2 * half
    return n


def matched_dense_width(cfg: ModelConfig):
    """Hidden width that gives a DenseFF the parameter count of the lookup."""
    d = cfg.d_model
    return max(1, round((lookup_param_count(cfg) - d) / (2 * d + 1)))


def count_parameters(model: RecurrentModel):
    groups = {"embedding": 0, "attention": 0, "feed_forward": 0, "other": 0}
    for name, value in tree_flatten(model.trainable_parameters()):
        if name.startswith(("embed", "unembed", "pos")):
            groups["embedding"] += value.size
        elif name.startswith("attn"):
            groups["attention"] += value.size
        elif name.startswith("ff"):
            groups["feed_forward"] += value.size
        else:
            groups["other"] += value.size
    groups["total"] = sum(groups.values())
    return groups
