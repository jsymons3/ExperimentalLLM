"""Train and evaluate one run.

    python -m phrase_scratchpad.train --variant maths --preset smoke --steps 2000 \
        --out runs/phrase/smoke/maths

Writes to the output directory:
    config.json            the resolved configuration
    metrics.jsonl          one line per periodic evaluation
    final.json             end-of-run evaluation per task: accuracy by split and
                           by number of steps, guessing baseline, what the
                           scratchpad holds, slot interventions, two-call ceiling
    checkpoint.safetensors model and optimizer state, for --resume
"""

import argparse
import json
import math
import platform
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from recurrent_lookup.train import build_optimizer, load_checkpoint, save_checkpoint

from .configs import TASK_BATCH, VARIANTS, build_config, build_run
from .evaluate import full_evaluation, periodic_evaluation
from .model import PhraseModel, count_parameters
from .tasks import PRESETS

MODEL_FLAGS = (
    "d_model", "n_heads", "n_passes", "ff", "core_ff", "n_subkeys", "lookup_heads",
    "top_k", "d_key", "slot_content", "outer_steps_train",
)
TRAIN_FLAGS = (
    "steps", "batch_size", "two_hop_frac", "lr", "warmup", "weight_decay",
    "table_weight_decay", "grad_clip", "bridge_loss_weight", "hide_subject_prob", "table_off",
    "eval_every", "log_every", "eval_size",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", choices=list(VARIANTS), default="maths")
    p.add_argument("--preset", choices=list(PRESETS), default="smoke")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory for this run")
    p.add_argument("--resume", action="store_true", help="continue from checkpoint")
    p.add_argument(
        "--time-only", type=int, default=0, metavar="N",
        help="time N training steps and the evaluations, write timing.json, exit",
    )
    p.add_argument("--no-compile", action="store_true", help="skip mx.compile")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0)

    d = p.add_argument_group("task size overrides")
    d.add_argument("--max-operand", type=int, help="arithmetic: largest operand")
    d.add_argument("--n-people", type=int, help="facts: how many people to include")
    d.add_argument("--max-stories", type=int, help="family stories: how many to train on")

    m = p.add_argument_group("model overrides")
    m.add_argument("--d-model", type=int)
    m.add_argument("--n-heads", type=int)
    m.add_argument("--n-passes", type=int, help="block applications per outer step")
    m.add_argument("--ff", choices=["lookup", "dense"])
    m.add_argument("--core-ff", type=int, help="width of an always-on dense MLP; 0 for none")
    m.add_argument("--n-subkeys", type=int, help="the table has n_subkeys**2 entries")
    m.add_argument("--lookup-heads", type=int)
    m.add_argument("--top-k", type=int)
    m.add_argument("--d-key", type=int)
    m.add_argument("--slot-content", choices=["hard", "soft"])
    m.add_argument("--outer-steps-train", type=int, nargs="+", metavar="T")

    t = p.add_argument_group("training overrides")
    t.add_argument("--steps", type=int)
    t.add_argument("--batch-size", type=int, help="questions per task in each batch")
    t.add_argument("--two-hop-frac", type=float)
    t.add_argument("--lr", type=float)
    t.add_argument("--warmup", type=int)
    t.add_argument("--weight-decay", type=float)
    t.add_argument("--table-weight-decay", type=float)
    t.add_argument("--grad-clip", type=float)
    t.add_argument("--bridge-loss-weight", type=float)
    t.add_argument(
        "--table-off", nargs="*", metavar="TASK",
        help="tasks trained with the lookup table switched off (needs --core-ff)",
    )
    t.add_argument(
        "--hide-subject-prob", type=float,
        help="share of two-step training questions whose subject is hidden after step one",
    )
    t.add_argument("--eval-every", type=int)
    t.add_argument("--log-every", type=int)
    t.add_argument("--eval-size", type=int)
    return p.parse_args(argv)


def config_from_args(args):
    def given(names):
        return {k: getattr(args, k) for k in names if getattr(args, k) is not None}

    overrides = {}
    if args.max_operand is not None:
        overrides["maths"] = {"max_operand": args.max_operand}
    if args.n_people is not None:
        overrides["facts"] = {"n_people": args.n_people}
    if args.max_stories is not None:
        overrides["clutrr"] = {"max_stories": args.max_stories}
    return build_config(
        args.variant, args.preset, args.seed,
        model=given(MODEL_FLAGS), task_overrides=overrides, **given(TRAIN_FLAGS),
    )


def format_accuracy(name, acc):
    def a(split):
        return f"{acc[split]:.3f}" if split in acc else "  -  "

    line = (
        f"{name}: 1hop {a('one_hop')} held {a('one_hop_held')} | "
        f"chains train {a('chain_train')} unseen {a('chain_unseen')}"
    )
    if "chain_held_second" in acc:
        line += (
            f" | held-out 1st {a('chain_held_first')} 2nd {a('chain_held_second')} "
            f"both {a('chain_held_both')}"
        )
    return line


def run(cfg, out_dir, resume=False, time_only=0, compile_step=True, ckpt_minutes=10.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    suite, model_cfg = build_run(cfg)
    mx.random.seed(cfg.seed)
    model = PhraseModel(model_cfg)
    optimizer = build_optimizer(cfg)

    ckpt_path = out / "checkpoint.safetensors"
    step0 = 0
    if resume and ckpt_path.exists():
        step0 = load_checkpoint(ckpt_path, model, optimizer)
        print(f"resumed from {ckpt_path} at step {step0}")
    mx.eval(model.parameters(), optimizer.state)

    params = count_parameters(model)
    eval_every = cfg.eval_every or max(100, cfg.steps // 40)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"run {cfg.variant} seed {cfg.seed} ({VARIANTS[cfg.variant]})")
    print(f"tasks  {json.dumps(suite.summary())}")
    print(f"model  {asdict(model_cfg)}")
    print(f"params {params}")

    # Questions per task in a batch, split into one-step and two-step.
    sizes = [
        max(2, int(round(cfg.batch_size * TASK_BATCH.get(task.name, 1.0)))) for task in suite.tasks
    ]
    counts = [(size - int(round(size * cfg.two_hop_frac)), int(round(size * cfg.two_hop_frac))) for size in sizes]
    state = [model.state, optimizer.state]

    def per_question_loss(logits, target):
        """Cross-entropy averaged over the K answer positions: (B,)."""
        return nn.losses.cross_entropy(logits, target, reduction="none").mean(axis=-1)

    def make_train_step(steps):
        """A training step at a fixed number of outer steps.

        Every question is graded on its final answer. Chains of a supervised
        task are also taught their intermediate answer at step 1. With
        ``one_hop_every_step`` a one-step question is graded at every step.
        Questions of a ``table_off`` task run with the lookup switched off.
        With ``hide_subject_prob`` some two-step questions lose their subject
        after the first step.
        """

        def task_loss(task, model, question, answer, bridge, teach_bridge, every_step, table_on, hide):
            logits, trace = model(
                question,
                steps=steps,
                return_trace=True,
                table_on=table_on if cfg.table_off else None,
                hide_prefix=hide if cfg.hide_subject_prob else None,
                n_passes=task.n_passes,
            )
            earlier = trace["step_logits"][:-1]
            share = (1 - every_step) + every_step / (len(earlier) + 1)
            total = (per_question_loss(logits, answer) * share).sum()
            if cfg.one_hop_every_step:
                for step_logits in earlier:
                    total = total + (per_question_loss(step_logits, answer) * every_step).sum() / (
                        len(earlier) + 1
                    )
            if earlier and cfg.bridge_loss_weight:
                first = per_question_loss(trace["step_logits"][0], bridge)
                total = total + cfg.bridge_loss_weight * (first * teach_bridge).sum()
            return total

        def loss_fn(model, *batch):
            # Each task's questions have their own length, so each task is a
            # separate forward pass: seven arrays per task. Tasks count
            # equally, whatever their batch sizes.
            total = mx.array(0.0)
            for i, (task, size) in enumerate(zip(suite.tasks, sizes)):
                total = total + task_loss(task, model, *batch[7 * i : 7 * i + 7]) / size
            return total / len(sizes)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def train_step(*batch):
            loss, grads = loss_and_grad(model, *batch)
            grads, _ = optim.clip_grad_norm(grads, cfg.grad_clip)
            optimizer.update(model, grads)
            return loss

        if compile_step:
            return partial(mx.compile, inputs=state, outputs=state)(train_step)
        return train_step

    step_options = list(model_cfg.outer_steps_train) if model_cfg.scratchpad else [1]
    train_steps = {steps: make_train_step(steps) for steps in step_options}
    rng = np.random.default_rng([cfg.seed, step0])

    def make_batch():
        arrays = []
        for task, (n_one, n_chain) in zip(suite.tasks, counts):
            items = task.sample(rng, n_one, n_chain)
            teach = items.chain & (task.name in cfg.supervised)
            every = ~items.chain & cfg.one_hop_every_step
            table_on = np.full(len(items), task.name not in cfg.table_off)
            hidden = rng.random(len(items)) < cfg.hide_subject_prob
            arrays += [
                items.question,
                items.answer,
                items.bridge,
                teach.astype(np.float32),
                every.astype(np.float32),
                table_on.astype(np.float32),
                (items.subject_len * hidden).astype(np.int32),
            ]
        return [mx.array(a) for a in arrays]

    def one_step():
        batch = make_batch()
        # The number of outer steps is drawn once per batch.
        steps = int(rng.choice(step_options))
        loss = train_steps[steps](*batch)
        mx.eval(loss, state)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError("non-finite training loss; stopping this run")
        return value

    if time_only:
        for _ in range(2 * len(step_options) + 2):  # compile and warm up
            one_step()
        t0 = time.perf_counter()
        for _ in range(time_only):
            one_step()
        sec_per_step = (time.perf_counter() - t0) / time_only
        t0 = time.perf_counter()
        periodic_evaluation(model, suite, cfg.table_off)
        eval_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        full_evaluation(model, suite, cfg.eval_steps, cfg.table_off)
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
                accuracy = periodic_evaluation(model, suite, cfg.table_off)
                record = {
                    "step": step,
                    "loss": window_loss / window_n,
                    "sec_per_step": train_sec / (step - step0),
                    "elapsed_min": (time.perf_counter() - started) / 60,
                    "accuracy": accuracy,
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"step {step:7d} | loss {record['loss']:.4f}", flush=True)
                for name, acc in accuracy.items():
                    print("    " + format_accuracy(name, acc), flush=True)
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
        "supervised": list(cfg.supervised),
        "table_off": list(cfg.table_off),
        "params": params,
        "sec_per_step": train_sec / max(1, step - step0),
        "wall_minutes": (time.perf_counter() - started) / 60,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "platform": platform.platform(),
    }
    final.update(full_evaluation(model, suite, cfg.eval_steps, cfg.table_off))
    (out / "final.json").write_text(json.dumps(final, indent=2))
    for name, entry in final["tasks"].items():
        print("final  " + format_accuracy(name, entry["accuracy"]))
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
