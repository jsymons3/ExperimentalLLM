"""Compare native chain answers with an explicit predicted-entity handoff.

The true intermediate entity is used only for scoring and for the separate
oracle-second-hop probe. The two-call path receives the model's prediction.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from .configs import RunConfig, graph_config, model_config
from .data import MultiHopData
from .evaluate import eval_mode
from .model import RecurrentModel


def predict(model, tokens, passes, batch_size):
    return np.concatenate(
        [
            np.array(
                mx.argmax(
                    model(mx.array(tokens[i : i + batch_size]), n_passes=passes),
                    axis=-1,
                )
            )
            for i in range(0, len(tokens), batch_size)
        ]
    )


def summarize(rows, bridge, target, first, oracle_second, second, native):
    known_first = first == bridge
    known_second = oracle_second == target
    both = known_first & known_second
    composed = second == target
    native_correct = {p: pred == target for p, pred in native.items()}

    def rate(correct, mask=None):
        selected = correct if mask is None else correct[mask]
        return {
            "correct": int(selected.sum()),
            "n": len(selected),
            "accuracy": float(selected.mean()) if len(selected) else None,
        }

    result = {
        "n": len(rows),
        "first_fact": rate(known_first),
        "second_fact_with_true_bridge": rate(known_second),
        "both_facts_known": rate(both),
        "two_call": rate(composed),
        "native": {str(p): rate(c) for p, c in native_correct.items()},
        "on_both_known": {
            "two_call": rate(composed, both),
            "native": {str(p): rate(c, both) for p, c in native_correct.items()},
        },
        "two_call_failures_on_both_known": int((both & ~composed).sum()),
        "native_wrong_but_two_call_right": {
            str(p): int((~c & composed).sum()) for p, c in native_correct.items()
        },
    }
    first_native = next(iter(native))
    examples = np.flatnonzero(both & ~native_correct[first_native] & composed)[:5]
    result["examples_known_facts_native_wrong"] = [
        {
            "head": int(rows[i, 0]),
            "r1": int(rows[i, 1]),
            "r2": int(rows[i, 2]),
            "true_bridge": int(bridge[i]),
            "predicted_bridge": int(first[i]),
            "target": int(target[i]),
            "two_call_answer": int(second[i]),
            "native_answers": {str(p): int(pred[i]) for p, pred in native.items()},
        }
        for i in examples
    ]
    return result


def diagnose(model, data, one_hop_passes=(2, 4), direct_passes=(4, 8), batch_size=512):
    result = {}
    with eval_mode(model):
        for split, rows in data.eval_sets.items():
            if not split.startswith("two_hop") or not len(rows):
                continue
            h, r1, r2 = rows.T
            bridge = data.tail[h, r1]
            tokens, target = data.encode_two_hop(rows)
            native = {p: predict(model, tokens, p, batch_size) for p in direct_passes}
            by_passes = {}
            for passes in one_hop_passes:
                first_tokens, _ = data.encode_one_hop(np.stack([h, r1], axis=1))
                first = predict(model, first_tokens, passes, batch_size)
                oracle_tokens, _ = data.encode_one_hop(np.stack([bridge, r2], axis=1))
                oracle_second = predict(model, oracle_tokens, passes, batch_size)
                # Crucial: feed predicted first-hop answers, never true bridges.
                second_tokens, _ = data.encode_one_hop(np.stack([first, r2], axis=1))
                second = predict(model, second_tokens, passes, batch_size)
                by_passes[str(passes)] = summarize(
                    rows, bridge, target, first, oracle_second, second, native
                )
            result[split] = by_passes
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--one-hop-passes", type=int, nargs="+", default=[2, 4])
    p.add_argument("--direct-passes", type=int, nargs="+", default=[4, 8])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if min(a.one_hop_passes + a.direct_passes + [a.batch_size]) < 1:
        p.error("pass counts and batch size must be positive")
    if a.cpu:
        mx.set_default_device(mx.cpu)
    cfg = RunConfig(**json.loads((a.run / "config.json").read_text()))
    checkpoint = a.run / "checkpoint.safetensors"
    with checkpoint.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    arrays, meta = mx.load(str(checkpoint), return_metadata=True)
    model = RecurrentModel(model_config(cfg))
    model.update(
        tree_unflatten(
            [(k[6:], v) for k, v in arrays.items() if k.startswith("model.")]
        )
    )
    mx.eval(model.parameters())
    del arrays
    results = diagnose(
        model,
        MultiHopData(graph_config(cfg)),
        a.one_hop_passes,
        a.direct_passes,
        a.batch_size,
    )
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checksum,
        "step": int(meta["step"]),
        "device": str(mx.default_device()),
        "graph": cfg.graph,
        "trained_passes": model.cfg.n_passes,
        "one_hop_passes_per_call": a.one_hop_passes,
        "direct_passes": a.direct_passes,
        "method": "Same fixed evaluation chains as the training report. Two-call feeds the predicted entity into a fresh one-hop prompt. True bridges are used only for oracle scoring and selecting the both-known subset. No updates to model or checkpoint.",
        "caveats": [
            "100% two-call accuracy on both-known is expected by construction, not independent evidence of learned composition.",
            "The diagnostic supplies decomposition and a discrete entity bottleneck externally; it does not identify which internal mechanism fails.",
            "Two calls with two passes each match four recurrent block applications, not exact FLOPs or latency; two four-pass calls use eight applications.",
            "Both-known subsets differ with one-hop pass count. Compare methods within each subset.",
        ],
        "splits": results,
    }
    out = a.out or a.run / "two-call-diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for split, variants in results.items():
        for passes, x in variants.items():
            print(
                json.dumps(
                    {
                        "split": split,
                        "passes_per_call": passes,
                        "both_known": x["both_facts_known"],
                        "native": x["native"],
                        "two_call": x["two_call"],
                        "on_both_known": x["on_both_known"],
                    }
                ),
                flush=True,
            )
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
