"""One small transformer block applied repeatedly, with a learned lookup table.

The block is: causal self-attention, then a feed-forward part. The feed-forward
part is either an ordinary MLP (``ff="dense"``) or a product-key lookup over a
table of single-neuron experts (``ff="lookup"``), as in "Mixture of A Million
Experts" (https://arxiv.org/abs/2407.04153). The same block, with the same
weights, is applied ``n_passes`` times, so with the lookup every pass issues a
new query from the updated state.

Nothing in the table is written by hand. Keys, query projection and the stored
neurons all start random and are trained by the task loss.

With ``num_slots > 0`` the model also has a scratchpad. The sequence becomes
[question tokens, slots, work position] and the model runs in outer steps. A
step applies the block ``n_passes`` times, reads the work position out through
the tied embedding, and writes that result into the slots as a blend of token
embeddings. The next step starts from the question and the slots only, so the
earlier result is read the way a typed token is read.
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
    n_passes: int = 4  # block applications (per outer step, with slots)
    ff: str = "lookup"  # "lookup" or "dense"
    d_ff: int = 512  # hidden width when ff == "dense"
    n_subkeys: int = 128  # the table has n_subkeys ** 2 entries
    lookup_heads: int = 4  # independent queries per token
    top_k: int = 8  # entries retrieved per query
    d_key: int = 64  # query width per head, split into two halves
    query_norm: str = "ema"  # "ema", "layer" or "none"
    lookup_passes: str = "every"  # "every", or "first" (pass 1 only)
    tie_embeddings: bool = True
    # Scratchpad. The defaults switch it off, which is the original model.
    num_slots: int = 0  # slot positions; 0 means no scratchpad
    slot_content: str = "soft"  # what is written: "soft", "hard" or "raw"
    outer_steps_train: tuple = (1,)  # outer step counts sampled in training
    carry_hidden: bool = False  # keep hidden state between outer steps
    step_embedding: bool = True  # tell the work position which step it is on

    @property
    def n_entries(self):
        return self.n_subkeys**2

    @property
    def default_steps(self):
        """Outer steps used when none are asked for."""
        return min(self.outer_steps_train)


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

        # Scratchpad parameters exist only when it is on, so checkpoints and
        # seeded initialisation of the original model are unchanged.
        if cfg.num_slots:
            if cfg.slot_content not in ("soft", "hard", "raw"):
                raise ValueError(f"unknown slot_content {cfg.slot_content!r}")
            self.slot_empty = mx.random.normal((d,)) * (d**-0.5)
            self.slot_type = mx.zeros((cfg.num_slots, d))
            self.work_emb = mx.random.normal((d,)) * (d**-0.5)
            if cfg.step_embedding:
                n_steps = max(cfg.outer_steps_train)
                self.step_emb = mx.random.normal((n_steps, d)) * (d**-0.5)
            self.write_head = nn.Linear(d, cfg.num_slots + 1)
            if cfg.slot_content == "raw":
                self.raw_proj = nn.Linear(d, d)

    def block(self, h, lookup=True):
        """One application of the shared block."""
        h = h + self.attn(self.norm_attn(h))
        if not lookup:
            return h, None
        out, index = self.ff(self.norm_ff(h))
        return h + out, index

    def readout(self, h):
        """Logits over entities from a (B, d) state."""
        h = self.norm_out(h)
        if self.cfg.tie_embeddings:
            return h @ self.embed.weight[: self.cfg.n_entities].T
        return self.unembed(h)

    def token_form(self, probs):
        """Blend of entity embeddings, weighted by probs (B, n_entities).

        A one-hot row returns that entity's input embedding exactly.
        """
        return probs @ self.embed.weight[: self.cfg.n_entities]

    def slot_content(self, logits, h_work):
        """The vector offered to the slots at the end of a step."""
        kind = self.cfg.slot_content
        if kind == "raw":
            return self.raw_proj(h_work)
        soft = self.token_form(mx.softmax(logits, axis=-1))
        if kind == "soft":
            return soft
        # hard: the top entity's embedding, with the soft blend's gradient.
        hard = self.embed.weight[mx.argmax(mx.stop_gradient(logits), axis=-1)]
        return soft + mx.stop_gradient(hard - soft)

    def __call__(
        self,
        tokens,
        n_passes=None,
        steps=None,
        return_retrieval=False,
        return_trace=False,
    ):
        """tokens: (B, T). Returns logits over entities.

        ``n_passes`` is the number of block applications: in total for the
        original model, per outer step for a scratchpad model. ``steps`` is
        the number of outer steps and only applies to a scratchpad model.

        With ``return_retrieval`` also returns, for each block application in
        order, the table indices retrieved, shape (B, positions, heads,
        top_k), or None where no lookup ran. With ``return_trace`` (scratchpad
        only) returns a dict instead, with per-step ``step_logits``, ``slots``
        (after the step's write), ``write`` weights and ``retrieved``.
        """
        cfg = self.cfg
        n_passes = n_passes or cfg.n_passes
        if cfg.num_slots:
            return self._scratchpad_forward(
                tokens, n_passes, steps or cfg.default_steps, return_retrieval, return_trace
            )
        if steps not in (None, 1) or return_trace:
            raise ValueError("outer steps and traces need num_slots > 0")

        h = self.embed(tokens) + self.pos[: tokens.shape[1]]
        retrieved = []
        for p in range(n_passes):
            h, index = self.block(
                h, cfg.ff == "dense" or cfg.lookup_passes == "every" or p == 0
            )
            retrieved.append(index)
        logits = self.readout(h[:, -1])
        return (logits, retrieved) if return_retrieval else logits

    def _scratchpad_forward(self, tokens, n_passes, steps, return_retrieval, return_trace):
        cfg = self.cfg
        B, L = tokens.shape
        K, d = cfg.num_slots, cfg.d_model
        question = self.embed(tokens) + self.pos[:L]
        slots = mx.broadcast_to(self.slot_empty, (B, K, d))
        trace = {"step_logits": [], "slots": [], "write": [], "retrieved": []}
        h = None
        for t in range(steps):
            work = self.work_emb
            if cfg.step_embedding:
                # Steps beyond those trained reuse the last step's embedding.
                step = self.step_emb[min(t, self.step_emb.shape[0] - 1)]
                work = work + step
            work = mx.broadcast_to(work, (B, 1, d))
            if cfg.carry_hidden and h is not None:
                # Keep the state at question and work positions; only the slot
                # positions are refreshed with what was just written.
                kept = h[:, -1:]
                if cfg.step_embedding:
                    kept = kept + step
                h = mx.concatenate([h[:, :L], slots + self.slot_type, kept], axis=1)
            else:
                # Fresh state: only the slots carry anything between steps.
                h = mx.concatenate([question, slots + self.slot_type, work], axis=1)
            for i in range(n_passes):
                h, index = self.block(
                    h, cfg.ff == "dense" or cfg.lookup_passes == "every" or i == 0
                )
                trace["retrieved"].append(index)
            logits = self.readout(h[:, -1])
            content = self.slot_content(logits, h[:, -1])
            write = mx.softmax(self.write_head(h[:, -1]), axis=-1)  # (B, K+1)
            w = write[:, :K, None]  # the last weight means "write nothing"
            slots = (1 - w) * slots + w * content[:, None, :]
            trace["step_logits"].append(logits)
            trace["slots"].append(slots)
            trace["write"].append(write)
        if return_trace:
            return logits, trace
        return (logits, trace["retrieved"]) if return_retrieval else logits


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
    scratchpad = ("slot_", "work_emb", "step_emb", "write_head", "raw_proj")
    for name, value in tree_flatten(model.trainable_parameters()):
        if name.startswith(("embed", "unembed", "pos")):
            groups["embedding"] += value.size
        elif name.startswith("attn"):
            groups["attention"] += value.size
        elif name.startswith("ff"):
            groups["feed_forward"] += value.size
        elif name.startswith(scratchpad):
            groups["scratchpad"] = groups.get("scratchpad", 0) + value.size
        else:
            groups["other"] += value.size
    groups["total"] = sum(groups.values())
    return groups
