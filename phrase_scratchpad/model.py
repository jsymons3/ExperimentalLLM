"""The shared recurrent block with a scratchpad of several tokens.

This extends the scratchpad that worked on the graph task (one entity, fully
overwritten, written as the single predicted token) from one token to a short
phrase. The sequence is

    [question tokens] [K scratchpad slots] [K work positions]

and the model runs in outer steps. A step applies the shared block
``n_passes`` times. Work position i then predicts token i of an answer, and
the K predicted tokens' own embeddings replace the scratchpad. The next step
starts fresh from the question and the scratchpad. The last step's K
predictions are the answer.

The block is the one from ``recurrent_lookup``: causal attention, then a
product-key lookup (or a dense MLP for a control).

With ``core_ff > 0`` the block also has a small dense MLP that is always on,
and the lookup can be switched off per question. Questions that need no
stored facts can then be trained with the table off, which leaves the
attention and the small MLP (the core) as the only place their procedure can
be learned.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from recurrent_lookup.model import DenseFF, ModelConfig, ProductKeyLookup

PAD = 0


@dataclass
class PhraseConfig:
    vocab_size: int
    n_output: int  # answers and the scratchpad only use token ids below this
    question_len: int
    answer_len: int  # K: tokens in an answer, and slots in the scratchpad
    d_model: int = 128
    n_heads: int = 4
    n_passes: int = 2  # block applications per outer step
    ff: str = "lookup"  # "lookup" or "dense"
    d_ff: int = 512
    core_ff: int = 0  # hidden width of an always-on dense MLP; 0 means none
    n_subkeys: int = 128  # the table has n_subkeys ** 2 entries
    lookup_heads: int = 4
    top_k: int = 8
    d_key: int = 64
    scratchpad: bool = True  # False: no slots and a single step
    slot_content: str = "hard"  # "hard": the predicted tokens; "soft": a blend
    outer_steps_train: tuple = (2, 3, 4)

    @property
    def default_steps(self):
        return min(self.outer_steps_train) if self.scratchpad else 1


class MaskedAttention(nn.Module):
    """Multi-head self-attention with an additive mask."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, mask):
        B, S, D = x.shape
        H = self.n_heads
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, S, H, D // H).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, H, D // H).transpose(0, 2, 3, 1)
        v = v.reshape(B, S, H, D // H).transpose(0, 2, 1, 3)
        scores = (q @ k) * (D // H) ** -0.5 + mask
        y = mx.softmax(scores, axis=-1) @ v
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, S, D))


class MaskedLookup(ProductKeyLookup):
    """The product-key lookup, with padding kept out of its query statistics.

    Padded question positions are never attended to, so what they retrieve
    does not matter, but they would distort the running mean and variance
    used to normalise queries.
    """

    def retrieve(self, x, valid=None):
        B, T, _ = x.shape
        H, n, k = self.n_heads, self.n_subkeys, self.top_k
        norm = self.norm
        q = self.query(x)
        if self.training:
            flat = mx.stop_gradient(q)
            weight = mx.ones((B, T, 1)) if valid is None else valid[..., None]
            count = mx.maximum(weight.sum(), 1.0)
            mean = (flat * weight).sum(axis=(0, 1)) / count
            var = (((flat - mean) ** 2) * weight).sum(axis=(0, 1)) / count
            m = norm.momentum
            norm.running_mean = (1 - m) * norm.running_mean + m * mean
            norm.running_var = (1 - m) * norm.running_var + m * var
        q = (q - norm.running_mean) * mx.rsqrt(norm.running_var + norm.eps)
        q = q.reshape(B, T, H, 2, self.half)

        # As in ProductKeyLookup: score each half query against its sub-keys,
        # then pick the best full keys among the pairs of per-half winners.
        sub = mx.matmul(self.subkeys, q[..., None]).squeeze(-1)
        top = mx.argpartition(-mx.stop_gradient(sub), kth=k - 1, axis=-1)[..., :k]
        top_score = mx.take_along_axis(sub, top, axis=-1)
        pair_score = top_score[..., 0, :, None] + top_score[..., 1, None, :]
        pair_index = top[..., 0, :, None] * n + top[..., 1, None, :]
        pair_score = pair_score.reshape(B, T, H, k * k)
        pair_index = pair_index.reshape(B, T, H, k * k)
        best = mx.argpartition(-mx.stop_gradient(pair_score), kth=k - 1, axis=-1)[..., :k]
        return (
            mx.take_along_axis(pair_index, best, axis=-1),
            mx.take_along_axis(pair_score, best, axis=-1),
        )

    def __call__(self, x, valid=None):
        index, score = self.retrieve(x, valid)
        weight = mx.softmax(score, axis=-1)
        fire = self.fire[index]
        write = self.write[index]
        act = nn.gelu((fire * x[:, :, None, None, :]).sum(axis=-1))
        return ((weight * act)[..., None] * write).sum(axis=(2, 3)), index


class PhraseModel(nn.Module):
    def __init__(self, cfg: PhraseConfig):
        super().__init__()
        self.cfg = cfg
        d, K = cfg.d_model, cfg.answer_len
        if cfg.slot_content not in ("hard", "soft"):
            raise ValueError(f"unknown slot_content {cfg.slot_content!r}")
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.pos = mx.random.normal((cfg.question_len, d)) * (d**-0.5)
        self.norm_attn = nn.LayerNorm(d)
        self.attn = MaskedAttention(d, cfg.n_heads)
        self.norm_ff = nn.LayerNorm(d)
        if cfg.ff == "dense":
            self.ff = DenseFF(d, cfg.d_ff)
        elif cfg.ff == "lookup":
            self.ff = MaskedLookup(
                ModelConfig(
                    vocab_size=cfg.vocab_size,
                    n_entities=cfg.n_output,
                    d_model=d,
                    n_subkeys=cfg.n_subkeys,
                    lookup_heads=cfg.lookup_heads,
                    top_k=cfg.top_k,
                    d_key=cfg.d_key,
                )
            )
        else:
            raise ValueError(f"unknown ff {cfg.ff!r}")
        if cfg.core_ff:
            self.norm_core = nn.LayerNorm(d)
            self.core = DenseFF(d, cfg.core_ff)
        self.norm_out = nn.LayerNorm(d)
        self.work_emb = mx.random.normal((K, d)) * (d**-0.5)
        if cfg.scratchpad:
            self.slot_empty = mx.random.normal((d,)) * (d**-0.5)
            # Zero at the start, so a slot looks like a bare token until the
            # model learns otherwise.
            self.slot_type = mx.zeros((K, d))
            self.step_emb = mx.random.normal((max(cfg.outer_steps_train), d)) * (d**-0.5)

    def block(self, h, mask, valid, table_on=None):
        """One application of the shared block.

        ``table_on`` (B,) of 0/1 switches the lookup off for some questions.
        """
        h = h + self.attn(self.norm_attn(h), mask)
        if self.cfg.core_ff:
            h = h + self.core(self.norm_core(h))[0]
        if self.cfg.ff == "dense":
            out, index = self.ff(self.norm_ff(h))
        else:
            # Questions with the table off are also left out of the lookup's
            # query statistics.
            if table_on is not None:
                valid = valid * table_on[:, None]
            out, index = self.ff(self.norm_ff(h), valid)
        if table_on is not None:
            out = out * table_on[:, None, None]
        return h + out, index

    def readout(self, h):
        """Logits over answer tokens from work-position states (B, K, d)."""
        return self.norm_out(h) @ self.embed.weight[: self.cfg.n_output].T

    def token_form(self, logits):
        """What is written to the scratchpad for these predictions: (B, K, d).

        Hard content is exactly the predicted tokens' embeddings. The soft
        blend supplies its gradient, since choosing a token has none.
        """
        answers = self.embed.weight[: self.cfg.n_output]
        soft = mx.softmax(logits, axis=-1) @ answers
        if self.cfg.slot_content == "soft":
            return soft
        hard = answers[mx.argmax(mx.stop_gradient(logits), axis=-1)]
        return soft + mx.stop_gradient(hard - soft)

    def __call__(
        self, question, steps=None, return_trace=False, slot_override=None, table_on=None
    ):
        """question: (B, L) token ids, padded at the end. Returns logits
        (B, K, n_output) for the answer's K tokens.

        With ``return_trace`` also returns a dict with every step's logits,
        the scratchpad after every step's write, and the table indices
        retrieved by every block application. ``slot_override`` (B, K, d)
        replaces the scratchpad that step 2 reads; it is for evaluation only.
        ``table_on`` (B,) of 0/1 switches the lookup off for some questions.
        """
        cfg = self.cfg
        B, L = question.shape
        K, d = cfg.answer_len, cfg.d_model
        if slot_override is not None and (self.training or not cfg.scratchpad):
            raise ValueError("slot overrides are evaluation-only and need a scratchpad")
        steps = (steps or cfg.default_steps) if cfg.scratchpad else 1
        n_extra = 2 * K if cfg.scratchpad else K

        asked = self.embed(question) + self.pos[:L]
        valid = mx.concatenate(
            [(question != PAD).astype(mx.float32), mx.ones((B, n_extra))], axis=1
        )
        size = L + n_extra
        # Causal order, and nothing attends to padded question positions.
        mask = mx.triu(mx.full((size, size), -1e9), k=1) + (1 - valid)[:, None, None, :] * -1e9

        trace = {"step_logits": [], "slots": [], "retrieved": []}
        slots = mx.broadcast_to(self.slot_empty, (B, K, d)) if cfg.scratchpad else None
        for t in range(steps):
            if cfg.scratchpad:
                work = self.work_emb + self.step_emb[min(t, self.step_emb.shape[0] - 1)]
                parts = [asked, slots + self.slot_type, mx.broadcast_to(work, (B, K, d))]
            else:
                parts = [asked, mx.broadcast_to(self.work_emb, (B, K, d))]
            h = mx.concatenate(parts, axis=1)
            for _ in range(cfg.n_passes):
                h, index = self.block(h, mask, valid, table_on)
                trace["retrieved"].append(index)
            logits = self.readout(h[:, -K:])
            trace["step_logits"].append(logits)
            if cfg.scratchpad:
                slots = self.token_form(logits)  # full overwrite
                if t == 0 and slot_override is not None:
                    slots = slot_override
                trace["slots"].append(slots)
        return (logits, trace) if return_trace else logits


def count_parameters(model):
    groups = {"embedding": 0, "attention": 0, "feed_forward": 0, "scratchpad": 0, "other": 0}
    for name, value in tree_flatten(model.trainable_parameters()):
        if name.startswith(("embed", "pos")):
            groups["embedding"] += value.size
        elif name.startswith("attn"):
            groups["attention"] += value.size
        elif name.startswith("ff"):
            groups["feed_forward"] += value.size
        elif name.startswith(("core", "norm_core")):
            groups["core_mlp"] = groups.get("core_mlp", 0) + value.size
        elif name.startswith(("slot_", "step_emb", "work_emb")):
            groups["scratchpad"] += value.size
        else:
            groups["other"] += value.size
    groups["total"] = sum(groups.values())
    return groups
