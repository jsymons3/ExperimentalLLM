"""Paired ablations measure memory dependence, not fact/abstraction labels."""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_unflatten

from .data import Corpus
from .model import Config, CoreMemoryLM


def token_losses(model, x, y, mode):
    logits, calls = model(mx.array(x), mode=mode)
    return np.array(
        nn.losses.cross_entropy(logits, mx.array(y), reduction="none")
    ), np.array(calls)


def evaluate(model, corpus, batches=8, batch_size=8, seed=912):
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    losses = {m: [] for m in ("off", "on", "gated")}
    calls, examples = [], []
    try:
        for _ in range(batches):
            x, y = corpus.sample(rng, batch_size)
            for mode, mode_losses in losses.items():
                nll, count = token_losses(model, x, y, mode)
                mode_losses.append(nll)
                if mode == "gated":
                    calls.append(count)
            if len(examples) < 12:
                delta = losses["off"][-1] - losses["on"][-1]
                for row in range(min(2, len(x))):
                    pos = int(delta[row].argmax())
                    context = bytes(
                        int(v) for v in x[row, max(0, pos - 48) : pos + 1] if v < 256
                    ).decode("utf-8", errors="replace")
                    examples.append(
                        {
                            "context": context,
                            "target_byte": int(y[row, pos]),
                            "off_minus_on_nll": float(delta[row, pos]),
                        }
                    )
    finally:
        model.train(was_training)
    flat = {k: np.concatenate(v).ravel() for k, v in losses.items()}
    gap = flat["off"] - flat["on"]
    count = np.concatenate(calls).ravel()
    correlation = (
        float(np.corrcoef(gap, count)[0, 1])
        if np.std(gap) > 0 and np.std(count) > 0
        else None
    )
    return {
        "sampled_tokens": len(gap),
        "modes": {
            m: {"nll": float(v.mean()), "bits_per_token": float(v.mean() / math.log(2))}
            for m, v in flat.items()
        },
        "off_minus_on_nll": float(gap.mean()),
        "memory_helped_fraction": float((gap > 0).mean()),
        "gated_calls_per_token": float(count.mean()),
        "gap_call_correlation": correlation,
        "examples": examples,
        "interpretation": "Positive off-minus-on loss means memory helped prediction. It does not label a token as factual. Bits/token includes EOS.",
    }


def probe_completions(model, path):
    """Optional user-labelled JSONL {prompt,target,kind}; never used for training."""
    model.eval()
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = list(item["prompt"].encode("utf-8"))
        target = list(item["target"].encode("utf-8"))
        if (
            not prompt
            or not target
            or len(prompt) + len(target) - 1 > model.cfg.context
        ):
            raise ValueError(
                "each probe needs a nonempty prompt/target fitting the context"
            )
        all_tokens = np.array([prompt + target], dtype=np.int32)
        result = {k: item[k] for k in ("prompt", "target", "kind")}
        for mode in ("off", "on", "gated"):
            nll, _ = token_losses(model, all_tokens[:, :-1], all_tokens[:, 1:], mode)
            result[mode + "_nll"] = float(nll[:, len(prompt) - 1 :].mean())
        result["off_minus_on_nll"] = result["off_nll"] - result["on_nll"]
        records.append(result)
    groups = {
        k: float(np.mean([r["off_minus_on_nll"] for r in records if r["kind"] == k]))
        for k in sorted({r["kind"] for r in records})
    }
    return {
        "mean_gap_by_user_label": groups,
        "records": records,
        "note": "Labels are supplied by the user, not inferred from loss.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--probes", type=Path)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    cfg = json.loads((a.run / "config.json").read_text())
    model = CoreMemoryLM(Config(**cfg["model"]))
    arrays, meta = mx.load(str(a.run / "checkpoint.safetensors"), return_metadata=True)
    model.update(
        tree_unflatten(
            [(k[6:], v) for k, v in arrays.items() if k.startswith("model.")]
        )
    )
    result = {
        "step": int(meta["step"]),
        "split": a.split,
        "evaluation": evaluate(
            model,
            Corpus(cfg["data"], a.split, model.cfg.context),
            a.batches,
            a.batch_size,
        ),
    }
    if a.probes:
        result["probes"] = probe_completions(model, a.probes)
    dest = a.run / f"evaluation-{a.split}.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
