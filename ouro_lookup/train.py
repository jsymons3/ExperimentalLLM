"""Train and evaluate one run.

    python -m ouro_lookup.train --variant table --preset pilot --steps 1000 \
        --out runs/ouro/pilot/table

Ouro's own weights stay frozen; only the table and the adapters are trained.
Writes to the output directory:
    config.json            the resolved configuration
    metrics.jsonl          one line per periodic evaluation
    final.json             end-of-run evaluation: accuracy by split with the
                           additions on and off, what limits two-step accuracy,
                           the note comparisons, loss on ordinary text
    checkpoint.safetensors the table, the adapters and the optimizer state,
                           for --resume (Ouro's weights are not copied)
"""

import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .configs import VARIANTS, build_config, save_run_config
from .data import PRESETS, World
from .download import load_tokenizer
from .evaluate import full_evaluation, periodic_evaluation
from .model import LookupOuro
from .ouro import load_ouro

DTYPES = {"stored": None, "float16": mx.float16, "float32": mx.float32}
ADDON_FLAGS = ("n_subkeys", "lookup_heads", "top_k", "d_key", "adapter_rank", "adapter_scale")
TRAIN_FLAGS = (
    "weights", "dtype", "loops", "teach_loop", "teach_weight", "adapters_learn", "steps",
    "batch_size", "two_hop_frac", "lr", "adapter_lr", "warmup", "weight_decay", "grad_clip",
    "eval_every", "log_every", "eval_size", "periodic_size",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", choices=list(VARIANTS), default="table")
    p.add_argument("--preset", choices=list(PRESETS), default="pilot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory for this run")
    p.add_argument("--resume", action="store_true", help="continue from checkpoint")
    p.add_argument(
        "--time-only", type=int, default=0, metavar="N",
        help="time N training steps and the evaluations, write timing.json, exit",
    )
    p.add_argument("--no-compile", action="store_true", help="skip mx.compile")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0)

    m = p.add_argument_group("model")
    m.add_argument("--weights", help="directory with Ouro's config, tokenizer and weights")
    m.add_argument("--dtype", choices=list(DTYPES), help="precision Ouro runs in")
    m.add_argument("--loops", type=int, help="times the layer stack is applied (Ouro's own: 4)")
    m.add_argument("--table-layers", type=int, nargs="*", metavar="L", help="layers with a table")
    m.add_argument("--n-subkeys", type=int, help="the table has n_subkeys**2 entries")
    m.add_argument("--lookup-heads", type=int)
    m.add_argument("--top-k", type=int)
    m.add_argument("--d-key", type=int)
    m.add_argument("--adapter-rank", type=int)
    m.add_argument("--adapter-scale", type=float)

    d = p.add_argument_group("facts")
    d.add_argument("--n-people", type=int)
    d.add_argument("--n-relations", type=int)

    t = p.add_argument_group("training")
    t.add_argument("--teach-loop", type=int, help="latent runs: loop graded on the intermediate answer")
    t.add_argument("--teach-weight", type=float)
    t.add_argument("--adapters-learn", choices=["chains", "all"])
    t.add_argument("--steps", type=int)
    t.add_argument("--batch-size", type=int, help="questions per step")
    t.add_argument("--two-hop-frac", type=float)
    t.add_argument("--lr", type=float, help="learning rate of the table")
    t.add_argument("--adapter-lr", type=float)
    t.add_argument("--warmup", type=int)
    t.add_argument("--weight-decay", type=float, help="of the table")
    t.add_argument("--grad-clip", type=float)
    t.add_argument("--eval-every", type=int)
    t.add_argument("--log-every", type=int)
    t.add_argument("--eval-size", type=int)
    t.add_argument("--periodic-size", type=int)
    return p.parse_args(argv)


def config_from_args(args):
    def given(names):
        return {k: getattr(args, k) for k in names if getattr(args, k) is not None}

    addons = given(ADDON_FLAGS)
    if args.table_layers is not None:
        addons["table_layers"] = list(args.table_layers)
    return build_config(
        args.variant, args.preset, args.seed,
        world=given(("n_people", "n_relations")), addons=addons, **given(TRAIN_FLAGS),
    )


def build_run(cfg):
    """(world, model) for a configuration."""
    encode, start = load_tokenizer(cfg.weights)
    world = World(encode, start=start, eval_size=cfg.eval_size, seed=cfg.seed, **cfg.world)
    mx.random.seed(cfg.seed)
    base = load_ouro(cfg.weights, dtype=DTYPES[cfg.dtype], loops=cfg.loops or None)
    model = LookupOuro(base, cfg.addon_config())
    loops = base.cfg.total_ut_steps
    if cfg.teach_loop and not 1 <= cfg.teach_loop < loops:
        raise ValueError(f"teach_loop must be between 1 and {loops - 1}")
    return world, model


def build_optimizers(cfg):
    """One optimizer for the table and one for the adapters."""

    def schedule(peak):
        return lambda step: peak * mx.minimum(1.0, (step + 1) / cfg.warmup)

    return {
        "tables": optim.AdamW(learning_rate=schedule(cfg.lr), weight_decay=cfg.weight_decay),
        "adapters": optim.AdamW(learning_rate=schedule(cfg.adapter_lr), weight_decay=0.0),
    }


def save_checkpoint(path, model, optimizers, step):
    arrays = {f"model.{k}": v for k, v in model.addon_parameters().items()}
    for name, optimizer in optimizers.items():
        arrays.update({f"optim.{name}.{k}": v for k, v in tree_flatten(optimizer.state)})
    tmp = str(path) + ".tmp.safetensors"
    mx.save_safetensors(tmp, arrays, metadata={"step": str(step)})
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizers):
    arrays, metadata = mx.load(str(path), return_metadata=True)
    model.update(tree_unflatten([(k[6:], v) for k, v in arrays.items() if k.startswith("model.")]))
    trainable = model.trainable_parameters()
    for name, optimizer in optimizers.items():
        prefix = f"optim.{name}."
        items = [(k[len(prefix):], v) for k, v in arrays.items() if k.startswith(prefix)]
        if name in trainable and items:
            optimizer.init({name: trainable[name]})
            optimizer.state = tree_unflatten(items)
    return int(metadata["step"])


def format_accuracy(acc, chains=True):
    def a(split):
        return f"{acc[split]:.3f}" if split in acc else "  -  "

    line = f"one-step {a('one_hop')}"
    if "one_hop_table_off" in acc:
        line += f" (table off {a('one_hop_table_off')})"
    if not chains:  # two-step questions were not trained in this run
        return line
    return line + (
        f" | two-step: trained {a('chain_train')} unseen {a('chain_unseen')} "
        f"held-out 1st {a('chain_held_first')} 2nd {a('chain_held_second')} "
        f"both {a('chain_held_both')}"
    )


def run(cfg, out_dir, resume=False, time_only=0, compile_step=True, ckpt_minutes=10.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    world, model = build_run(cfg)
    optimizers = build_optimizers(cfg)
    ckpt_path = out / "checkpoint.safetensors"
    step0 = 0
    if resume and ckpt_path.exists():
        step0 = load_checkpoint(ckpt_path, model, optimizers)
        print(f"resumed from {ckpt_path} at step {step0}")
    else:
        trainable = model.trainable_parameters()
        for name, optimizer in optimizers.items():
            if trainable.get(name):
                optimizer.init({name: trainable[name]})
    mx.eval(model.parameters(), [o.state for o in optimizers.values()])

    params = model.count_parameters()
    eval_every = cfg.eval_every or max(50, cfg.steps // 40)
    save_run_config(cfg, out)
    print(f"run {cfg.variant} seed {cfg.seed} ({VARIANTS[cfg.variant]['about']})")
    print(f"facts  {json.dumps(world.summary())}")
    print(
        f"model  {asdict(model.cfg)}, table at layers {list(model.table_layers)}, "
        f"{model.base.dtype}, {model.base.cfg.total_ut_steps} loops"
    )
    print(f"params {params}")

    n_chain = int(round(cfg.batch_size * cfg.two_hop_frac)) if len(world.train_chains) else 0
    n_one = cfg.batch_size - n_chain
    learn_all = cfg.adapters_learn == "all"
    teach = cfg.teach_loop - 1 if cfg.teach_loop else None
    state = [model.state] + [o.state for o in optimizers.values()]

    def cross_entropy(logits, target):
        return nn.losses.cross_entropy(logits, target, reduction="sum")

    def loss_fn(model, one_tokens, one_answer, chain_tokens, chain_answer, chain_bridge):
        """Mean loss per question.

        One-step questions are graded on their answer. Two-step questions are
        graded on the intermediate answer in "note" mode and on the final
        answer in "latent" mode. With ``teach_loop`` the state after that loop
        is graded too: the answer of a one-step question, the intermediate
        answer of a two-step one.
        """
        states = model.states(one_tokens, adapters_learn=learn_all)
        total = cross_entropy(model.read(states[-1][:, -1]), one_answer)
        if teach is not None:
            early = cross_entropy(model.read(states[teach][:, -1]), one_answer)
            total = total + cfg.teach_weight * early
        if n_chain:
            states = model.states(chain_tokens)
            target = chain_bridge if cfg.chain_mode == "note" else chain_answer
            total = total + cross_entropy(model.read(states[-1][:, -1]), target)
            if teach is not None:
                early = cross_entropy(model.read(states[teach][:, -1]), chain_bridge)
                total = total + cfg.teach_weight * early
        return total / cfg.batch_size

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(*batch):
        loss, grads = loss_and_grad(model, *batch)
        grads, _ = optim.clip_grad_norm(grads, cfg.grad_clip)
        for name, optimizer in optimizers.items():
            if grads.get(name):
                optimizer.update(model, {name: grads[name]})
        return loss

    if compile_step:
        train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)
    rng = np.random.default_rng([cfg.seed, step0])

    def one_step():
        if n_chain:
            one, chain = world.sample(rng, n_one, n_chain)
        else:  # a world too small to have training chains: one unused row
            one, chain = world.facts(rng.integers(0, len(world.fact_answer), n_one)), world.chains([0])
        arrays = [one.tokens, one.answer, chain.tokens, chain.answer, chain.bridge]
        loss = train_step(*[mx.array(a) for a in arrays])
        mx.eval(loss, state)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError("non-finite training loss; stopping this run")
        return value

    if time_only:
        for _ in range(3):  # compile and warm up
            one_step()
        t0 = time.perf_counter()
        for _ in range(time_only):
            one_step()
        sec_per_step = (time.perf_counter() - t0) / time_only
        t0 = time.perf_counter()
        periodic_evaluation(model, world, cfg.chain_mode, cfg.periodic_size)
        eval_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        full_evaluation(model, world, cfg.chain_mode, cfg.eval_size)
        final_sec = time.perf_counter() - t0
        tokens = n_one * world.fact_tokens.shape[1] + n_chain * world.chain_tokens.shape[1]
        timing = {
            "sec_per_step": sec_per_step,
            "eval_sec": eval_sec,
            "final_eval_sec": final_sec,
            "tokens_per_step": tokens,
            "tokens_per_sec": tokens / sec_per_step,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "params": params,
        }
        (out / "timing.json").write_text(json.dumps(timing, indent=2))
        print(
            f"timing: {sec_per_step * 1000:.0f} ms/step ({timing['tokens_per_sec']:.0f} tokens/s), "
            f"periodic eval {eval_sec:.1f}s, final eval {final_sec:.1f}s, "
            f"peak memory {timing['peak_memory_gb']:.1f} GB"
        )
        return timing

    # Keep only the evaluations at or before the step we are starting from.
    metrics_path = out / "metrics.jsonl"
    if metrics_path.exists():
        kept = [
            line
            for line in metrics_path.read_text().splitlines()
            if json.loads(line)["step"] <= step0
        ]
        metrics_path.write_text("".join(line + "\n" for line in kept))
    train_sec, window_loss, window_n = 0.0, 0.0, 0
    started = last_ckpt = time.perf_counter()
    step = step0
    try:
        while step < cfg.steps:
            t0 = time.perf_counter()
            window_loss += one_step()
            train_sec += time.perf_counter() - t0
            window_n += 1
            step += 1

            evaluate_now = step % eval_every == 0 or step == cfg.steps
            if step % cfg.log_every == 0 and not evaluate_now:
                print(
                    f"step {step:7d} | loss {window_loss / window_n:.4f} | "
                    f"{train_sec / (step - step0) * 1000:.0f} ms/step",
                    flush=True,
                )
            if evaluate_now:
                accuracy = periodic_evaluation(model, world, cfg.chain_mode, cfg.periodic_size)
                record = {
                    "step": step,
                    "loss": window_loss / window_n,
                    "sec_per_step": train_sec / (step - step0),
                    "elapsed_min": (time.perf_counter() - started) / 60,
                    "accuracy": accuracy,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"step {step:7d} | loss {record['loss']:.4f} | "
                    f"{format_accuracy(accuracy, n_chain > 0)}",
                    flush=True,
                )
                window_loss, window_n = 0.0, 0
            if time.perf_counter() - last_ckpt > ckpt_minutes * 60:
                save_checkpoint(ckpt_path, model, optimizers, step)
                last_ckpt = time.perf_counter()
    except KeyboardInterrupt:
        save_checkpoint(ckpt_path, model, optimizers, step)
        print(f"\ninterrupted at step {step}; checkpoint saved. Add --resume to continue.")
        raise SystemExit(130)

    save_checkpoint(ckpt_path, model, optimizers, step)
    final = {
        "variant": cfg.variant,
        "about": VARIANTS[cfg.variant]["about"],
        "seed": cfg.seed,
        "preset": cfg.preset,
        "steps": step,
        "facts": world.summary(),
        "params": params,
        "sec_per_step": train_sec / max(1, step - step0),
        "wall_minutes": (time.perf_counter() - started) / 60,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "platform": platform.platform(),
    }
    final.update(full_evaluation(model, world, cfg.chain_mode, cfg.eval_size))
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print("final  " + format_accuracy(final["accuracy"], n_chain > 0))
    return final


def main(argv=None):
    args = parse_args(argv)
    run(
        config_from_args(args),
        args.out,
        resume=args.resume,
        time_only=args.time_only,
        compile_step=not args.no_compile,
        ckpt_minutes=args.checkpoint_minutes,
    )


if __name__ == "__main__":
    main()
