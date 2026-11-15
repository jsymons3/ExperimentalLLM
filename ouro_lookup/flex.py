"""Flexibility: add facts batch by batch, or change some, with no revision.

    python -m ouro_lookup.flex --mode retain --variant table --out runs/flex/retain_table
    python -m ouro_lookup.flex --mode edit --variant adapters --out runs/flex/edit_adapters

Ouro stays frozen; a table or adapters hold the made-up facts, as in stage 1.
Only one-step questions are used.

``retain``: the people are split into batches. Each batch's facts are trained
alone until they are recalled, then every batch is tested. Earlier batches
are never shown again, so their recall shows what later learning destroys.

``edit``: all the facts are trained together. Then a hundred of them get a
new answer and only those are trained. The test is whether the new answers
are given, and how many of the untouched facts survive.

Writes config.json, progress.json (one record per finished phase),
final.json, and checkpoint.safetensors (the trained additions, saved after
each phase, so ``--resume`` restarts at the next one).
"""

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .configs import DEFAULT_WEIGHTS
from .data import PRESETS, World
from .download import load_tokenizer
from .evaluate import ordinary_text_loss
from .model import AddonConfig, LookupOuro
from .ouro import load_ouro
from .train import DTYPES, learning_rate, load_checkpoint, save_checkpoint

# Rank 344 gives adapters as many trained numbers as the table (67.6 million
# against 67.7). Their scale is lowered with the square root of the rank so
# that a step changes the model about as much as at rank 8.
MATCHED_RANK = 344
VARIANTS = {
    "table": dict(about="the table alone", addons=dict(adapter_rank=0)),
    "adapters": dict(
        about="adapters alone, rank 8", addons=dict(adapter_rank=8, table_layers=())
    ),
    "adapters_matched": dict(
        about="adapters alone, with as many trained numbers as the table (rank 344)",
        addons=dict(
            adapter_rank=MATCHED_RANK, adapter_scale=2.0 * (8 / MATCHED_RANK) ** 0.5,
            table_layers=(),
        ),
    ),
    "table_fixed_keys": dict(
        about="the table, with the part that forms and matches queries frozen after "
        "the first phase",
        addons=dict(adapter_rank=0),
        freeze_keys=True,
    ),
}


@dataclass
class FlexConfig:
    mode: str = "retain"  # "retain" or "edit"
    variant: str = "table"
    preset: str = "full"
    seed: int = 0
    weights: str = DEFAULT_WEIGHTS
    dtype: str = "stored"
    world: dict = field(default_factory=dict)
    addons: dict = field(default_factory=dict)
    freeze_keys: bool = False
    batch_size: int = 64  # questions per step
    lr: float = 5e-4  # table
    adapter_lr: float = 2e-4
    warmup: int = 100  # at the start of every phase
    lr_final: float = 0.3  # share of the peak rate left at a phase's last allowed step
    weight_decay: float = 0.1  # table only
    grad_clip: float = 1.0
    target: float = 0.98  # a phase stops once its own facts are recalled this well
    check_every: int = 100
    log_every: int = 50
    # retain
    batches: int = 6
    phase_steps: int = 2000  # most steps a batch may take
    # edit
    joint_steps: int = 4000  # most steps the first, joint phase may take
    n_edits: int = 100
    edit_steps: int = 200
    edit_check: int = 20

    def addon_config(self):
        values = dict(self.addons)
        if values.get("table_layers") is not None:
            values["table_layers"] = tuple(values["table_layers"])
        return AddonConfig(**values)


def build_config(mode, variant, preset="full", seed=0, world=None, addons=None, **overrides):
    if mode not in ("retain", "edit"):
        raise ValueError("mode is 'retain' or 'edit'")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    spec = {k: v for k, v in VARIANTS[variant].items() if k != "about"}
    return FlexConfig(
        mode=mode, variant=variant, preset=preset, seed=seed,
        world={**PRESETS[preset], **(world or {})},
        addons={**spec.pop("addons"), **(addons or {})},
        **{**spec, **overrides},
    )


def load_run_config(run_dir):
    saved = json.loads((Path(run_dir) / "config.json").read_text())
    known = {f.name for f in fields(FlexConfig)}
    return FlexConfig(**{k: v for k, v in saved.items() if k in known})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["retain", "edit"], default="retain")
    p.add_argument("--variant", choices=list(VARIANTS), default="table")
    p.add_argument("--preset", choices=list(PRESETS), default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory for this run")
    p.add_argument("--resume", action="store_true", help="continue after the last finished phase")
    p.add_argument(
        "--time-only", type=int, default=0, metavar="N",
        help="time N training steps and one test of every fact, write timing.json, exit",
    )
    p.add_argument("--no-compile", action="store_true", help="skip mx.compile")
    p.add_argument("--weights", help="directory with Ouro's config, tokenizer and weights")
    p.add_argument("--dtype", choices=list(DTYPES), help="precision Ouro runs in")
    p.add_argument("--n-people", type=int)
    p.add_argument("--n-relations", type=int)
    p.add_argument("--n-subkeys", type=int)
    p.add_argument("--adapter-rank", type=int)
    for name, kind in (
        ("batch_size", int), ("lr", float), ("adapter_lr", float), ("warmup", int),
        ("lr_final", float), ("weight_decay", float), ("grad_clip", float), ("target", float),
        ("check_every", int), ("log_every", int), ("batches", int), ("phase_steps", int),
        ("joint_steps", int), ("n_edits", int), ("edit_steps", int), ("edit_check", int),
    ):
        p.add_argument("--" + name.replace("_", "-"), type=kind)
    return p.parse_args(argv)


def config_from_args(args):
    skip = {"mode", "variant", "preset", "seed", "out", "resume", "time_only", "no_compile"}
    given = {k: v for k, v in vars(args).items() if v is not None and k not in skip}
    world = {k: given.pop(k) for k in ("n_people", "n_relations") if k in given}
    addons = {k: given.pop(k) for k in ("n_subkeys", "adapter_rank") if k in given}
    return build_config(
        args.mode, args.variant, args.preset, args.seed, world=world, addons=addons, **given
    )


def build_run(cfg):
    """(world, model) for a configuration."""
    encode, start = load_tokenizer(cfg.weights)
    world = World(encode, start=start, seed=cfg.seed, **cfg.world)
    mx.random.seed(cfg.seed)
    model = LookupOuro(load_ouro(cfg.weights, dtype=DTYPES[cfg.dtype]), cfg.addon_config())
    return world, model


def recall(model, tokens, answers, batch=128, **switches):
    """Share of questions whose most likely next token is the answer."""
    if not len(tokens):
        return None
    was_training = model.training
    model.eval()
    try:
        right = 0
        for at in range(0, len(tokens), batch):
            logits = model.answer_logits(mx.array(tokens[at : at + batch]), **switches)
            right += int((np.array(mx.argmax(logits, axis=-1)) == answers[at : at + batch]).sum())
        return right / len(tokens)
    finally:
        model.train(was_training)


def freeze_keys(model):
    """Stop training the part of each table that forms and matches queries.
    Entries can still be written; where a question lands no longer moves."""
    for table in model.tables:
        table.query.freeze()
        table.freeze(keys=["subkeys"], recurse=False)
        table.norm.momentum = 0.0  # the running query statistics stay as they are


def train_phase(model, cfg, tokens, answers, max_steps, rng, check, check_every, stop=None,
                compile_step=True, label=""):
    """Train on these questions and no others, with fresh optimizers.

    ``check()`` is called every ``check_every`` steps and returns a record
    whose "recall" ends the phase early once it reaches ``stop``. Returns
    (steps taken, records).
    """
    schedule = SimpleNamespace(steps=max_steps, warmup=cfg.warmup, lr_final=cfg.lr_final)
    optimizers = {
        "tables": optim.AdamW(learning_rate(schedule, cfg.lr), weight_decay=cfg.weight_decay),
        "adapters": optim.AdamW(learning_rate(schedule, cfg.adapter_lr), weight_decay=0.0),
    }
    trainable = model.trainable_parameters()
    for name, optimizer in optimizers.items():
        if trainable.get(name):
            optimizer.init({name: trainable[name]})
    state = [model.state] + [o.state for o in optimizers.values()]

    def loss_fn(model, batch, target):
        logits = model.read(model.states(batch)[-1][:, -1])
        return nn.losses.cross_entropy(logits, target, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(batch, target):
        loss, grads = loss_and_grad(model, batch, target)
        grads, _ = optim.clip_grad_norm(grads, cfg.grad_clip)
        for name, optimizer in optimizers.items():
            if grads.get(name):
                optimizer.update(model, {name: grads[name]})
        return loss

    if compile_step:
        train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)

    records, window, window_n, started = [], 0.0, 0, time.perf_counter()
    for step in range(1, max_steps + 1):
        rows = rng.integers(0, len(tokens), cfg.batch_size)
        loss = train_step(mx.array(tokens[rows]), mx.array(answers[rows]))
        mx.eval(loss, state)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError("non-finite training loss; stopping this run")
        window, window_n = window + value, window_n + 1
        checking = step % check_every == 0 or step == max_steps
        if checking:
            record = {"step": step, "loss": window / window_n, **check()}
            records.append(record)
            shown = ", ".join(f"{k} {v:.3f}" for k, v in record.items() if k not in ("step", "loss"))
            print(f"{label} step {step:5d} | loss {record['loss']:.4f} | {shown}", flush=True)
            window, window_n = 0.0, 0
            if stop is not None and record["recall"] >= stop:
                break
        elif step % cfg.log_every == 0:
            print(f"{label} step {step:5d} | loss {window / window_n:.4f}", flush=True)
    sec = (time.perf_counter() - started) / step
    return step, records, sec


def run(cfg, out_dir, resume=False, time_only=0, compile_step=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world, model = build_run(cfg)
    params = model.count_parameters()
    P, R = world.n_people, world.n_relations
    tokens, answers = world.fact_tokens, world.name_token[world.fact_answer]
    sentences = SimpleNamespace(encode=world.encode, start=world.start)

    def ordinary(**switches):
        was_training = model.training
        model.eval()
        try:
            return ordinary_text_loss(model, sentences, **switches)
        finally:
            model.train(was_training)

    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"run {cfg.mode} {cfg.variant} seed {cfg.seed} ({VARIANTS[cfg.variant]['about']})")
    print(f"facts  {P} people, {R} relations, {P * R} one-step facts")
    print(f"model  {asdict(model.cfg)}, table at layers {list(model.table_layers)}, {model.base.dtype}")
    print(f"params {params}")

    if time_only:
        rng = np.random.default_rng(cfg.seed)
        train_phase(model, cfg, tokens, answers, 3, rng, dict, 10**9, compile_step=compile_step)
        _, _, sec = train_phase(
            model, cfg, tokens, answers, time_only, rng, dict, 10**9, compile_step=compile_step
        )
        t0 = time.perf_counter()
        recall(model, tokens, answers)
        eval_sec = time.perf_counter() - t0
        tokens_per_step = cfg.batch_size * tokens.shape[1]
        timing = {
            "sec_per_step": sec, "eval_sec": eval_sec, "tokens_per_step": tokens_per_step,
            "tokens_per_sec": tokens_per_step / sec,
            "peak_memory_gb": mx.get_peak_memory() / 1e9, "params": params,
        }
        (out / "timing.json").write_text(json.dumps(timing, indent=2))
        print(
            f"timing: {sec * 1000:.0f} ms/step ({timing['tokens_per_sec']:.0f} tokens/s), "
            f"testing every fact {eval_sec:.1f}s, peak memory {timing['peak_memory_gb']:.1f} GB"
        )
        return timing

    # Phases already finished, if resuming.
    progress_path, ckpt_path = out / "progress.json", out / "checkpoint.safetensors"
    phases = []
    if resume and progress_path.exists() and ckpt_path.exists():
        phases = json.loads(progress_path.read_text())["phases"]
        load_checkpoint(ckpt_path, model, {})
        print(f"resumed after phase {len(phases)}")
    started = time.perf_counter()
    original_loss = ordinary(table=False, adapters=False)

    def finish_phase(record):
        phases.append(record)
        save_checkpoint(ckpt_path, model, {}, len(phases))
        progress_path.write_text(json.dumps({"phases": phases}, indent=2))

    if cfg.mode == "retain":
        # Whole people per batch: all the facts about a person arrive together.
        order = np.random.default_rng([cfg.seed, 1000]).permutation(P)
        groups = [
            np.sort((people[:, None] * R + np.arange(R)).reshape(-1))
            for people in np.array_split(order, cfg.batches)
        ]
        for b, ids in enumerate(groups):
            if cfg.freeze_keys and b >= 1:
                freeze_keys(model)
            if b < len(phases):
                continue
            steps, records, sec = train_phase(
                model, cfg, tokens[ids], answers[ids], cfg.phase_steps,
                np.random.default_rng([cfg.seed, b]),
                check=lambda ids=ids: {"recall": recall(model, tokens[ids], answers[ids])},
                check_every=cfg.check_every, stop=cfg.target, compile_step=compile_step,
                label=f"batch {b + 1}/{len(groups)}",
            )
            row = [recall(model, tokens[g], answers[g]) for g in groups]
            finish_phase({
                "phase": b, "facts": len(ids), "steps": steps, "sec_per_step": sec,
                "reached_target": records[-1]["recall"] >= cfg.target,
                "recall": row, "ordinary_text_loss": ordinary(),
            })
            print("    recall of each batch now: " + " ".join(f"{x:.3f}" for x in row), flush=True)
        matrix = np.array([p["recall"] for p in phases])
        n = len(matrix)
        result = {
            "phases": phases,
            "summary": {
                "learned": float(np.mean(np.diag(matrix))),
                "kept_at_end": float(matrix[-1, :-1].mean()) if n > 1 else None,
                "first_batch_at_end": float(matrix[-1, 0]),
                "all_at_end": float(matrix[-1].mean()),
                "steps": [p["steps"] for p in phases],
                "reached_target": [p["reached_target"] for p in phases],
            },
            "ordinary_text_loss": {
                "original": original_loss, "at_end": phases[-1]["ordinary_text_loss"]
            },
        }
    else:
        every = np.arange(P * R)
        if not phases:
            steps, records, sec = train_phase(
                model, cfg, tokens, answers, cfg.joint_steps, np.random.default_rng([cfg.seed, 0]),
                check=lambda: {"recall": recall(model, tokens, answers)},
                check_every=2 * cfg.check_every, stop=cfg.target, compile_step=compile_step,
                label="all facts",
            )
            finish_phase({
                "phase": 0, "facts": len(every), "steps": steps, "sec_per_step": sec,
                "reached_target": records[-1]["recall"] >= cfg.target,
                "recall": records[-1]["recall"], "ordinary_text_loss": ordinary(),
            })
        if cfg.freeze_keys:
            freeze_keys(model)
        # The facts to change, each to a person who is neither the old answer
        # nor the person asked about.
        rng = np.random.default_rng([cfg.seed, 2000])
        edited = np.sort(rng.choice(P * R, size=min(cfg.n_edits, P * R), replace=False))
        old = world.fact_answer[edited]
        new = np.array([
            rng.choice(np.setdiff1d(np.arange(P), [o, s]))
            for o, s in zip(old, world.fact_person[edited])
        ])
        new_answers = world.name_token[new]
        kept = np.setdiff1d(every, edited)

        def look():
            return {
                "recall": recall(model, tokens[edited], new_answers),  # new answers given
                "old_answer": recall(model, tokens[edited], answers[edited]),
                "untouched": recall(model, tokens[kept], answers[kept]),
            }

        before = look()
        steps, records, sec = train_phase(
            model, cfg, tokens[edited], new_answers, cfg.edit_steps,
            np.random.default_rng([cfg.seed, 1]), check=look, check_every=cfg.edit_check,
            compile_step=compile_step, label="changed facts",
        )
        finish_phase({"phase": 1, "facts": len(edited), "steps": steps, "sec_per_step": sec})
        took = next((r for r in records if r["recall"] >= 0.95), None)
        result = {
            "joint": phases[0],
            "edit": {
                "changed": len(edited), "untouched": len(kept), "before": before,
                "checks": records, "when_changed": took, "at_end": records[-1],
            },
            "ordinary_text_loss": {
                "original": original_loss, "after_joint": phases[0]["ordinary_text_loss"],
                "at_end": ordinary(),
            },
        }

    final = {
        "kind": "flex", "mode": cfg.mode, "variant": cfg.variant,
        "about": VARIANTS[cfg.variant]["about"], "seed": cfg.seed,
        "facts": {"people": P, "relations": R, "facts": P * R},
        "params": params,
        "wall_minutes": (time.perf_counter() - started) / 60,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "mlx_version": mx.__version__, "device": str(mx.default_device()),
        "platform": platform.platform(),
        **result,
    }
    (out / "final.json").write_text(json.dumps(final, indent=2))
    if cfg.mode == "retain":
        s = final["summary"]
        kept = "-" if s["kept_at_end"] is None else f"{s['kept_at_end']:.3f}"
        print(
            f"final  learned {s['learned']:.3f} | earlier batches kept at the end {kept} | "
            f"first batch {s['first_batch_at_end']:.3f}"
        )
    else:
        e = final["edit"]["at_end"]
        print(
            f"final  new answers given {e['recall']:.3f} | old answers still given "
            f"{e['old_answer']:.3f} | untouched facts kept {e['untouched']:.3f}"
        )
    return final


def main(argv=None):
    args = parse_args(argv)
    run(
        config_from_args(args), args.out, resume=args.resume, time_only=args.time_only,
        compile_step=not args.no_compile,
    )


if __name__ == "__main__":
    main()
