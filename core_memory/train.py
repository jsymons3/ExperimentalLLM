"""Train controlled natural-text experiments; preserve optimizer and RNG on resume."""

import argparse
import hashlib
import json
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from .data import Corpus
from .evaluate import evaluate
from .model import Config, CoreMemoryLM, is_memory


def make_optimizer(lr, warmup, core_decay, memory_decay):
    schedule = lambda step: lr * mx.minimum(1.0, (step + 1) / warmup)
    return optim.MultiOptimizer(
        [
            optim.AdamW(schedule, weight_decay=memory_decay),
            optim.AdamW(schedule, weight_decay=core_decay),
            optim.AdamW(schedule, weight_decay=0),
        ],
        filters=[is_memory, lambda path, w: w.ndim >= 2],
    )


def save_checkpoint(path, model, optimizer, step, rng):
    arrays = {f"model.{k}": v for k, v in tree_flatten(model.parameters())}
    arrays.update({f"optim.{k}": v for k, v in tree_flatten(optimizer.state)})
    tmp = str(path) + ".tmp.safetensors"
    mx.save_safetensors(
        tmp,
        arrays,
        metadata={"step": str(step), "numpy_rng": json.dumps(rng.bit_generator.state)},
    )
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, rng):
    arrays, meta = mx.load(str(path), return_metadata=True)
    model.update(
        tree_unflatten(
            [(k[6:], v) for k, v in arrays.items() if k.startswith("model.")]
        )
    )
    optimizer.init(model.trainable_parameters())
    optimizer.state = tree_unflatten(
        [(k[6:], v) for k, v in arrays.items() if k.startswith("optim.")]
    )
    rng.bit_generator.state = json.loads(meta["numpy_rng"])
    return int(meta["step"])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--variant", choices=["core", "unpressured", "selective"], default="selective"
    )
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--core-decay", type=float)
    p.add_argument("--memory-decay", type=float)
    p.add_argument("--memory-dropout", type=float)
    p.add_argument("--lookup-cost", type=float)
    p.add_argument("--gate-warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-seconds", type=float, default=120)
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cpu", action="store_true")
    for name, field in Config.__dataclass_fields__.items():
        if name != "vocab_size":
            p.add_argument(
                "--" + name.replace("_", "-"), type=int, default=field.default
            )
    a = p.parse_args(argv)
    defaults = {
        "core": (0.1, 0, 1, 0),
        "unpressured": (0.01, 0.01, 0, 0),
        "selective": (0.1, 0, 0.25, 0.01),
    }[a.variant]
    for key, default in zip(
        ("core_decay", "memory_decay", "memory_dropout", "lookup_cost"), defaults
    ):
        if getattr(a, key) is None:
            setattr(a, key, default)
    if (
        min(a.steps, a.batch_size, a.warmup, a.eval_every, a.eval_batches, a.log_every)
        < 1
    ):
        p.error("steps, batch sizes, warmup and intervals must be positive")
    if (
        not 0 <= a.memory_dropout <= 1
        or min(
            a.lr,
            a.core_decay,
            a.memory_decay,
            a.lookup_cost,
            a.gate_warmup,
            a.max_seconds,
        )
        < 0
        or a.checkpoint_seconds <= 0
    ):
        p.error("invalid training pressure or time setting")
    return a


def run(a):
    if a.cpu:
        mx.set_default_device(mx.cpu)
    cfg = Config(
        **{k: getattr(a, k) for k in Config.__dataclass_fields__ if k != "vocab_size"}
    )
    config = {
        k: (str(v.resolve()) if isinstance(v, Path) else v)
        for k, v in vars(a).items()
        if k not in Config.__dataclass_fields__ and k not in ("out", "resume")
    }
    config["model"] = asdict(cfg)
    config["corpus_sha256"] = hashlib.sha256(
        (a.data / "manifest.json").read_bytes()
    ).hexdigest()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.safetensors"
    config_path = out / "config.json"
    if config_path.exists():
        if not a.resume or not checkpoint.exists():
            raise ValueError("existing run requires --resume and a checkpoint")
        old = json.loads(config_path.read_text())
        operational = {
            "steps",
            "eval_every",
            "eval_batches",
            "log_every",
            "checkpoint_seconds",
            "max_seconds",
            "cpu",
        }
        if {k: v for k, v in old.items() if k not in operational} != {
            k: v for k, v in config.items() if k not in operational
        }:
            raise ValueError(
                "resume would change training settings or corpus; use a fresh run"
            )
    elif a.resume:
        raise ValueError("cannot resume a missing run")
    train = Corpus(a.data, "train", cfg.context)
    valid = Corpus(a.data, "valid", cfg.context)
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = CoreMemoryLM(cfg)
    optimizer = make_optimizer(a.lr, a.warmup, a.core_decay, a.memory_decay)
    optimizer.init(model.trainable_parameters())
    step = load_checkpoint(checkpoint, model, optimizer, rng) if a.resume else 0
    if a.steps <= step:
        raise ValueError("target steps must exceed checkpoint step")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if (out / "final.json").exists():
        (out / "final.json").unlink()
    metrics = out / "metrics.jsonl"
    if metrics.exists():
        records = [json.loads(l) for l in metrics.read_text().splitlines()]
        metrics.write_text(
            "".join(json.dumps(x) + "\n" for x in records if x["step"] <= step)
        )
    mx.eval(model.parameters(), optimizer.state)
    counts = {"core": 0, "memory": 0}
    for path, w in tree_flatten(model.trainable_parameters()):
        counts["memory" if is_memory(path) else "core"] += w.size
    print(
        json.dumps(
            {
                "variant": a.variant,
                "start_step": step,
                "target_step": a.steps,
                "parameters": counts,
                "device": str(mx.default_device()),
            }
        ),
        flush=True,
    )

    state = [model.state, optimizer.state]

    def make_step(mode):
        def loss_fn(m, x, y, key):
            logits, calls = m(x, mode=mode, key=key)
            nll = nn.losses.cross_entropy(logits, y, reduction="mean")
            charge = calls.mean() if mode == "gated" else mx.array(0.0)
            return nll + a.lookup_cost * charge, (nll, calls.mean())

        vg = nn.value_and_grad(model, loss_fn)

        def update(x, y, key):
            (loss, aux), grads = vg(model, x, y, key)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            return loss, aux

        # MLX 0.32.4 compilation fails on this stateful multi-output step.
        # Ordinary lazy execution works on both CPU and Metal; do not silently
        # retry a failed compiled trace, which can leave traced arrays in state.
        return update

    updates = {mode: make_step(mode) for mode in ("off", "on", "gated")}
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    old_handlers = {
        s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)
    }
    started = last_checkpoint = time.monotonic()
    window = []
    reason = "step target"
    try:
        while step < a.steps:
            if stop["requested"] or (
                a.max_seconds and time.monotonic() - started >= a.max_seconds
            ):
                reason = "signal" if stop["requested"] else "time budget"
                break
            x, y = train.sample(rng, a.batch_size)
            # Consume the same RNG draws in every control so a common seed
            # really gives the same sequence of text windows across variants.
            drop_memory = rng.random() < a.memory_dropout
            if a.variant == "core" or drop_memory:
                mode = "off"
            elif a.variant == "unpressured" or step < a.gate_warmup:
                mode = "on"
            else:
                mode = "gated"
            # One saved NumPy RNG drives sampling, memory dropout and explicit
            # MLX routing keys, avoiding reliance on a mutable global MLX RNG.
            key = mx.random.key(int(rng.integers(0, 2**32)))
            loss, (nll, calls) = updates[mode](mx.array(x), mx.array(y), key)
            mx.eval(loss, nll, calls, state)
            if not np.isfinite(loss.item()):
                raise FloatingPointError("non-finite training loss")
            step += 1
            window.append([nll.item(), calls.item()])
            if step % a.log_every == 0:
                avg = np.mean(window, axis=0)
                print(
                    f"step {step:7d} | nll {avg[0]:.4f} | calls/token {avg[1]:.3f} | last mode {mode}",
                    flush=True,
                )
            if step % a.eval_every == 0 or step == a.steps:
                result = evaluate(model, valid, a.eval_batches, a.batch_size)
                record = {
                    "step": step,
                    "tokens_seen": step * a.batch_size * cfg.context,
                    "elapsed_seconds": time.monotonic() - started,
                    "train_window_nll": float(np.mean(window, axis=0)[0]),
                    "valid": result,
                }
                with metrics.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"eval {step} | off {result['modes']['off']['nll']:.4f} | on {result['modes']['on']['nll']:.4f} | gated {result['modes']['gated']['nll']:.4f} | calls {result['gated_calls_per_token']:.3f}",
                    flush=True,
                )
                window = []
            if time.monotonic() - last_checkpoint >= a.checkpoint_seconds:
                save_checkpoint(checkpoint, model, optimizer, step, rng)
                last_checkpoint = time.monotonic()
        save_checkpoint(checkpoint, model, optimizer, step, rng)
        final = {
            "step": step,
            "tokens_seen": step * a.batch_size * cfg.context,
            "equivalent_corpus_passes": step
            * a.batch_size
            * cfg.context
            / len(train.tokens),
            "stop_reason": reason,
            "parameters": counts,
            "device": str(mx.default_device()),
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "train": evaluate(model, train, a.eval_batches, a.batch_size),
            "valid": evaluate(model, valid, a.eval_batches, a.batch_size),
        }
        (out / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        print(
            f"Finished at step {step}: {reason}; checkpoint and final.json saved.",
            flush=True,
        )
        return final
    finally:
        for s, handler in old_handlers.items():
            signal.signal(s, handler)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
