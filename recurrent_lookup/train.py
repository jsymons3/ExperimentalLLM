"""Train and evaluate one run.

    python -m recurrent_lookup.train --preset smoke --variant C --steps 3000 \
        --out runs/smoke/C

Writes to the output directory:
    config.json            the resolved configuration, including the learning
                           rate schedule
    metrics.jsonl          one line per periodic evaluation
    final.json             end-of-run evaluation (accuracy by depth, table
                           usage, retrieval overlap, two-call ceiling, what
                           the answer position holds at each depth, slot
                           contents, timing, parameters)
    checkpoint.safetensors model and optimizer state (with the schedule
                           position), for --resume
"""

import argparse
import json
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

from .configs import PRESETS, VARIANTS, build_config, graph_config, model_config
from .data import MultiHopData
from .evaluate import accuracy_by_split, eval_mode, full_evaluation, table_usage
from .model import RecurrentModel, count_parameters

# The parameters that make up the lookup table: its keys and its entries.
TABLE_PARAMETERS = ("ff.subkeys", "ff.fire", "ff.write")

GRAPH_FLAGS = (
    "n_entities",
    "n_relations",
    "held_out_frac",
    "two_hop_train_frac",
    "eval_size",
)
MODEL_FLAGS = (
    "d_model",
    "n_heads",
    "n_passes",
    "n_subkeys",
    "lookup_heads",
    "top_k",
    "d_key",
    "query_norm",
    "num_slots",
    "slot_content",
    "outer_steps_train",
)
TRAIN_FLAGS = (
    "steps",
    "batch_size",
    "two_hop_frac",
    "lr",
    "warmup",
    "weight_decay",
    "grad_clip",
    "eval_every",
    "log_every",
    "prefix_loss_weight",
    "table_weight_decay",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", choices=list(PRESETS), default="smoke")
    p.add_argument("--variant", choices=list(VARIANTS), default="C")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory for this run")
    p.add_argument("--resume", action="store_true", help="continue from checkpoint")
    p.add_argument(
        "--time-only",
        type=int,
        default=0,
        metavar="N",
        help="time N training steps and the evaluations, write timing.json, exit",
    )
    p.add_argument("--no-compile", action="store_true", help="skip mx.compile")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0)

    g = p.add_argument_group("graph overrides")
    g.add_argument("--n-entities", type=int)
    g.add_argument("--n-relations", type=int)
    g.add_argument("--held-out-frac", type=float)
    g.add_argument("--two-hop-train-frac", type=float)
    g.add_argument("--eval-size", type=int)

    m = p.add_argument_group("model overrides")
    m.add_argument("--d-model", type=int)
    m.add_argument("--n-heads", type=int)
    m.add_argument(
        "--n-passes",
        "--inner-passes",
        dest="n_passes",
        type=int,
        help="block applications (per outer step, with a scratchpad)",
    )
    m.add_argument("--n-subkeys", type=int, help="run C table has n_subkeys**2 entries")
    m.add_argument("--lookup-heads", type=int)
    m.add_argument("--top-k", type=int)
    m.add_argument("--d-key", type=int)
    m.add_argument("--query-norm", choices=["ema", "layer", "none"])
    m.add_argument("--no-tie-embeddings", action="store_true")

    s = p.add_argument_group("scratchpad (off unless num_slots > 0)")
    s.add_argument("--num-slots", type=int, help="slot positions; 0 is the original model")
    s.add_argument("--slot-content", choices=["soft", "hard", "raw"])
    s.add_argument(
        "--outer-steps-train",
        type=int,
        nargs="+",
        metavar="T",
        help="outer step counts to sample from per batch (default 2 3 4)",
    )
    s.add_argument("--carry-hidden", action="store_true", help="keep hidden state between steps")
    s.add_argument("--no-step-embedding", action="store_true")
    s.add_argument(
        "--prefix-loss-weight",
        type=float,
        help="train the step-1 readout of two-hop training chains toward the bridge",
    )

    t = p.add_argument_group("training overrides")
    t.add_argument("--steps", type=int)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--two-hop-frac", type=float)
    t.add_argument("--lr", type=float)
    t.add_argument("--warmup", type=int)
    t.add_argument("--weight-decay", type=float)
    t.add_argument(
        "--table-weight-decay",
        type=float,
        help="separate weight decay for the table (default: same as --weight-decay)",
    )
    t.add_argument("--grad-clip", type=float)
    t.add_argument("--eval-every", type=int)
    t.add_argument("--log-every", type=int)
    return p.parse_args(argv)


def config_from_args(args):
    def given(names):
        return {k: getattr(args, k) for k in names if getattr(args, k) is not None}

    model = given(MODEL_FLAGS)
    if args.no_tie_embeddings:
        model["tie_embeddings"] = False
    if args.carry_hidden:
        model["carry_hidden"] = True
    if args.no_step_embedding:
        model["step_embedding"] = False
    return build_config(
        preset=args.preset,
        variant=args.variant,
        seed=args.seed,
        graph=given(GRAPH_FLAGS),
        model=model,
        **given(TRAIN_FLAGS),
    )


def build_optimizer(cfg):
    """AdamW with the run's schedule, and a separate table decay if asked for."""

    def schedule(step):
        return cfg.lr * mx.minimum(1.0, (step + 1) / cfg.warmup)

    core = optim.AdamW(learning_rate=schedule, weight_decay=cfg.weight_decay)
    if cfg.table_weight_decay is None:
        return core
    table = optim.AdamW(learning_rate=schedule, weight_decay=cfg.table_weight_decay)
    return optim.MultiOptimizer([table, core], [lambda name, _: name in TABLE_PARAMETERS])


def save_checkpoint(path, model, optimizer, step):
    arrays = {f"model.{k}": v for k, v in tree_flatten(model.parameters())}
    arrays.update({f"optim.{k}": v for k, v in tree_flatten(optimizer.state)})
    tmp = str(path) + ".tmp.safetensors"
    mx.save_safetensors(tmp, arrays, metadata={"step": str(step)})
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer):
    arrays, metadata = mx.load(str(path), return_metadata=True)
    model_items = [(k[6:], v) for k, v in arrays.items() if k.startswith("model.")]
    optim_items = [(k[6:], v) for k, v in arrays.items() if k.startswith("optim.")]
    model.update(tree_unflatten(model_items))
    optimizer.init(model.trainable_parameters())
    optimizer.state = tree_unflatten(optim_items)
    return int(metadata["step"])


def format_eval(acc, table):
    def a(name):
        return f"{acc[name]:.3f}" if name in acc else "  -  "

    line = (
        f"1hop {a('one_hop')} held {a('one_hop_held')} | "
        f"2hop train {a('two_hop_train')} id {a('two_hop_id')} | "
        f"held-out 1st {a('two_hop_held_first')} 2nd {a('two_hop_held_second')} "
        f"both {a('two_hop_held_both')}"
    )
    if table:
        line += f" | table used {table['used_frac']:.2f}"
    return line


def run(cfg, out_dir, resume=False, time_only=0, compile_step=True, ckpt_minutes=10.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = MultiHopData(graph_config(cfg))
    mx.random.seed(cfg.seed)
    model = RecurrentModel(model_config(cfg))
    optimizer = build_optimizer(cfg)
    slots = bool(model.cfg.num_slots)

    ckpt_path = out / "checkpoint.safetensors"
    step0 = 0
    if resume and ckpt_path.exists():
        step0 = load_checkpoint(ckpt_path, model, optimizer)
        print(f"resumed from {ckpt_path} at step {step0}")
    mx.eval(model.parameters(), optimizer.state)

    params = count_parameters(model)
    eval_depths = cfg.eval_steps if slots else cfg.eval_passes
    eval_every = cfg.eval_every or max(100, cfg.steps // 40)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"run {cfg.variant} seed {cfg.seed} ({VARIANTS[cfg.variant]})")
    print(f"graph  {data.summary()}")
    print(f"model  {cfg.model}")
    print(f"params {params}")

    n_two = int(round(cfg.batch_size * cfg.two_hop_frac))
    n_one = cfg.batch_size - n_two

    state = [model.state, optimizer.state]

    def make_train_step(steps):
        """A training step at a fixed number of outer steps (None: no slots).

        The loss is cross-entropy on the readout at the final step. With a
        prefix loss, the step-1 readout of two-hop chains is also trained
        toward the bridge entity.
        """
        kwargs = {"steps": steps} if slots else {}

        def loss_fn(model, tok1, y1, tok2, y2, bridge):
            total = mx.array(0.0)
            if n_one:
                logits = model(tok1, **kwargs)
                total = total + nn.losses.cross_entropy(logits, y1, reduction="sum")
            if n_two and cfg.prefix_loss_weight:
                logits, trace = model(tok2, return_trace=True, **kwargs)
                first = nn.losses.cross_entropy(
                    trace["step_logits"][0], bridge, reduction="sum"
                )
                total = total + cfg.prefix_loss_weight * first
                total = total + nn.losses.cross_entropy(logits, y2, reduction="sum")
            elif n_two:
                logits = model(tok2, **kwargs)
                total = total + nn.losses.cross_entropy(logits, y2, reduction="sum")
            return total / cfg.batch_size

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def train_step(tok1, y1, tok2, y2, bridge):
            loss, grads = loss_and_grad(model, tok1, y1, tok2, y2, bridge)
            grads, _ = optim.clip_grad_norm(grads, cfg.grad_clip)
            optimizer.update(model, grads)
            return loss

        if compile_step:
            return partial(mx.compile, inputs=state, outputs=state)(train_step)
        return train_step

    step_options = list(model.cfg.outer_steps_train) if slots else [None]
    train_steps = {steps: make_train_step(steps) for steps in step_options}

    rng = np.random.default_rng([cfg.seed, step0])

    def one_step():
        (tok1, y1), (tok2, y2) = data.sample_batch(rng, n_one, n_two)
        bridge = data.tail[tok2[:, 0], tok2[:, 1] - data.n_entities]
        # The number of outer steps is drawn once per batch.
        steps = int(rng.choice(step_options)) if slots else None
        loss = train_steps[steps](
            mx.array(tok1), mx.array(y1), mx.array(tok2), mx.array(y2), mx.array(bridge)
        )
        mx.eval(loss, state)
        return loss.item()

    def periodic_eval():
        with eval_mode(model):
            return accuracy_by_split(model, data), table_usage(model, data, sample=4000)

    if time_only:
        for _ in range(5):  # compile and warm up
            one_step()
        t0 = time.perf_counter()
        for _ in range(time_only):
            one_step()
        sec_per_step = (time.perf_counter() - t0) / time_only
        t0 = time.perf_counter()
        periodic_eval()
        eval_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        full_evaluation(model, data, eval_depths)
        final_sec = time.perf_counter() - t0
        timing = {
            "sec_per_step": sec_per_step,
            "eval_sec": eval_sec,
            "final_eval_sec": final_sec,
            "params": params,
        }
        (out / "timing.json").write_text(json.dumps(timing, indent=2))
        print(
            f"timing: {sec_per_step * 1000:.1f} ms/step, periodic eval "
            f"{eval_sec:.1f}s, final eval {final_sec:.1f}s"
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

            if step % cfg.log_every == 0 and step % eval_every != 0:
                print(
                    f"step {step:7d} | loss {window_loss / window_n:.4f} | "
                    f"{train_sec / (step - step0) * 1000:.1f} ms/step",
                    flush=True,
                )
            if step % eval_every == 0 or step == cfg.steps:
                acc, table = periodic_eval()
                record = {
                    "step": step,
                    "loss": window_loss / window_n,
                    "sec_per_step": train_sec / (step - step0),
                    "elapsed_min": (time.perf_counter() - started) / 60,
                    "accuracy": acc,
                    "table": table,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"step {step:7d} | loss {record['loss']:.4f} | "
                    f"{format_eval(acc, table)}",
                    flush=True,
                )
                window_loss, window_n = 0.0, 0
            if time.perf_counter() - last_ckpt > ckpt_minutes * 60:
                save_checkpoint(ckpt_path, model, optimizer, step)
                last_ckpt = time.perf_counter()
    except KeyboardInterrupt:
        save_checkpoint(ckpt_path, model, optimizer, step)
        print(f"\ninterrupted at step {step}; checkpoint saved. Add --resume to continue.")
        raise SystemExit(130)

    save_checkpoint(ckpt_path, model, optimizer, step)
    final = {
        "variant": cfg.variant,
        "seed": cfg.seed,
        "preset": cfg.preset,
        "steps": step,
        "trained_passes": model.cfg.n_passes,
        "trained_steps": list(model.cfg.outer_steps_train) if slots else None,
        "params": params,
        "sec_per_step": train_sec / max(1, step - step0),
        "wall_minutes": (time.perf_counter() - started) / 60,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "platform": platform.platform(),
    }
    final.update(full_evaluation(model, data, eval_depths))
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print("final  " + format_eval(final["accuracy"], final["table"]))
    unit = final["depth_unit"]
    for depth, acc in final[f"accuracy_by_{unit}"].items():
        print(f"  {unit}={depth}: {format_eval(acc, None)}")
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
