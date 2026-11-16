"""Take over part of Ouro: remove some MLPs and train something to stand in.

    python -m ouro_lookup.takeover --variant table --steps 2000 --out runs/takeover/table

The MLP of layers 8 to 15 is removed (277 million of Ouro's numbers). What
stands in for them is trained to imitate the original model on ordinary text;
Ouro's remaining weights stay frozen. The original model is the teacher, so
no labels are needed. Three things are matched at every position:

    the next-token distribution after the last loop,
    the state after each loop,
    and, where something stands in for a removed MLP, what that MLP added.

Writes to the output directory:
    config.json            the resolved configuration
    metrics.jsonl          one line per periodic evaluation
    final.json             held-out text loss and well-known facts for the
                           original model, with nothing put back, and with the
                           trained stand-in; sizes and costs
    checkpoint.safetensors the trained additions and optimizer state
"""

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from . import text as text_data
from .configs import DEFAULT_WEIGHTS
from .download import load_tokenizer
from .evaluate import ordinary_text_loss
from .model import AddonConfig, LookupOuro
from .ouro import load_ouro
from .probes import Probes
from .train import DTYPES, learning_rate, load_checkpoint, save_checkpoint

REMOVE = tuple(range(8, 16))
# A small MLP of this width costs what the default table costs per token
# (688,128 multiplications, 2% of the MLP it stands in for).
SAME_COST_WIDTH = 112

VARIANTS = {
    "table": dict(
        about="a table of 16,384 entries stands in for each removed MLP",
        addons=dict(table=True, n_subkeys=128),
    ),
    "dense": dict(
        about="a small MLP with the table's computation stands in for each removed MLP",
        addons=dict(dense_width=SAME_COST_WIDTH),
    ),
    "adapters": dict(
        about="nothing stands in; adapters on every attention layer",
        addons=dict(adapter_rank=8),
    ),
    "table_small": dict(
        about="a table of 4,096 entries stands in for each removed MLP",
        addons=dict(table=True, n_subkeys=64),
    ),
    "dense_wide": dict(
        about="a small MLP with four times the table's computation stands in",
        addons=dict(dense_width=4 * SAME_COST_WIDTH),
    ),
    # Extras, not in the default two-day run.
    "table_adapters": dict(
        about="a table of 16,384 entries per removed MLP, and adapters",
        addons=dict(table=True, n_subkeys=128, adapter_rank=8),
    ),
    "table_large": dict(
        about="a table of 36,864 entries stands in for each removed MLP",
        addons=dict(table=True, n_subkeys=192),
    ),
}


@dataclass
class TakeoverConfig:
    variant: str = "table"
    seed: int = 0
    weights: str = DEFAULT_WEIGHTS
    dtype: str = "stored"
    text: str = text_data.DEFAULT_DIR  # directory written by ouro_lookup.text
    remove_layers: tuple = REMOVE
    addons: dict = field(default_factory=dict)
    seq_len: int = 96  # tokens per window of text, start token included
    batch_size: int = 8  # windows per step
    steps: int = 10000
    lr: float = 5e-4  # tables and small MLPs, at their peak
    adapter_lr: float = 2e-4
    warmup: int = 200
    lr_final: float = 0.1
    schedule_initial_steps: int = 0  # optional original budget before an extension
    schedule_transition_step: int = 0  # preserve that schedule through this step
    schedule_steps: int = 0  # keep an existing schedule when stopping earlier
    weight_decay: float = 0.1  # tables and small MLPs
    grad_clip: float = 1.0
    kl_weight: float = 1.0  # next-token distribution after the last loop
    match_weight: float = 1.0  # state after each loop
    local_weight: float = 1.0  # what each removed MLP added
    eval_every: int = 0  # 0: forty evaluations per run
    log_every: int = 50
    eval_windows: int = 1024  # held-out windows in the final evaluation
    periodic_windows: int = 64

    def addon_config(self):
        values = dict(self.addons)
        layers = tuple(self.remove_layers)
        values["table_layers"] = layers if values.pop("table", False) else ()
        return AddonConfig(remove_layers=layers, **values)


def takeover_learning_rate(cfg, peak):
    """Extend a run without changing the learning rates it already used.

    Give every control the same transition, including runs starting fresh.
    The original ramp/decay is followed until the transition; the remaining
    decay is stretched to the new end, without a learning-rate jump.
    """
    if cfg.schedule_steps:
        if cfg.schedule_steps < cfg.steps:
            raise ValueError("schedule_steps must cover all requested training steps")
        cfg = replace(cfg, steps=cfg.schedule_steps)
    if not cfg.schedule_initial_steps and not cfg.schedule_transition_step:
        return learning_rate(cfg, peak)
    initial, transition = cfg.schedule_initial_steps, cfg.schedule_transition_step
    if not cfg.warmup <= transition < initial <= cfg.steps:
        raise ValueError("schedule extension needs warmup <= transition < initial <= steps")
    original = learning_rate(replace(cfg, steps=initial), peak)
    at_transition = original(mx.array(transition, dtype=mx.uint64))

    def rate(step):
        progress = mx.clip((step - transition) / (cfg.steps - transition), 0.0, 1.0)
        extended = at_transition + (peak * cfg.lr_final - at_transition) * progress
        return mx.where(step < transition, original(step), extended)

    return rate


def completed_checkpoint_step(model, optimizers, saved_step):
    """SIGINT may arrive after an update but before the Python step counter.

    Checkpoint serialization evaluates the pending state. Use the optimizer
    counter to avoid replaying that completed update's training window.
    Finished, weights-only checkpoints have no optimizer counter to check.
    """
    trainable = model.trainable_parameters()
    counts = {int(optimizer.state["step"].item())
              for name, optimizer in optimizers.items() if trainable.get(name)}
    if counts == {0}:
        return saved_step
    if len(counts) != 1 or next(iter(counts)) not in (saved_step, saved_step + 1):
        raise ValueError(f"checkpoint step {saved_step} disagrees with optimizer steps {counts}")
    completed = counts.pop()
    if completed != saved_step:
        print(f"checkpoint includes completed update {completed}; advancing past saved counter {saved_step}")
    return completed


def build_config(variant, seed=0, addons=None, **overrides):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    cfg = TakeoverConfig(
        variant=variant, seed=seed, addons={**VARIANTS[variant]["addons"], **(addons or {})},
        **overrides,
    )
    cfg.remove_layers = tuple(cfg.remove_layers)
    return cfg


def load_run_config(run_dir):
    saved = json.loads((Path(run_dir) / "config.json").read_text())
    known = {f.name for f in fields(TakeoverConfig)}
    cfg = TakeoverConfig(**{k: v for k, v in saved.items() if k in known})
    cfg.remove_layers = tuple(cfg.remove_layers)
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", choices=list(VARIANTS), default="table")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory for this run")
    p.add_argument("--resume", action="store_true", help="continue from checkpoint")
    p.add_argument(
        "--time-only", type=int, default=0, metavar="N",
        help="time N training steps and the evaluations, write timing.json, exit",
    )
    p.add_argument("--no-compile", action="store_true", help="skip mx.compile")
    p.add_argument("--checkpoint-minutes", type=float, default=10.0)
    p.add_argument("--weights", help="directory with Ouro's config, tokenizer and weights")
    p.add_argument("--dtype", choices=list(DTYPES), help="precision Ouro runs in")
    p.add_argument("--text", help="directory with the prepared text")
    p.add_argument("--remove-layers", type=int, nargs="+", metavar="L")
    p.add_argument("--n-subkeys", type=int, help="each table has n_subkeys**2 entries")
    p.add_argument("--dense-width", type=int)
    p.add_argument("--adapter-rank", type=int)
    for name, kind in (
        ("seq_len", int), ("batch_size", int), ("steps", int), ("lr", float),
        ("adapter_lr", float), ("warmup", int), ("lr_final", float), ("weight_decay", float),
        ("schedule_initial_steps", int), ("schedule_transition_step", int),
        ("schedule_steps", int),
        ("grad_clip", float), ("kl_weight", float), ("match_weight", float),
        ("local_weight", float), ("eval_every", int), ("log_every", int),
        ("eval_windows", int), ("periodic_windows", int),
    ):
        p.add_argument("--" + name.replace("_", "-"), type=kind)
    return p.parse_args(argv)


def config_from_args(args):
    skip = {"variant", "seed", "out", "resume", "time_only", "no_compile", "checkpoint_minutes"}
    addon_flags = ("n_subkeys", "dense_width", "adapter_rank")
    given = {k: v for k, v in vars(args).items() if v is not None and k not in skip}
    addons = {k: given.pop(k) for k in addon_flags if k in given}
    return build_config(args.variant, args.seed, addons=addons, **given)


def build_run(cfg):
    """(model, text, encode, start) for a configuration."""
    encode, start = load_tokenizer(cfg.weights)
    mx.random.seed(cfg.seed)
    base = load_ouro(cfg.weights, dtype=DTYPES[cfg.dtype])
    model = LookupOuro(base, cfg.addon_config())
    text = text_data.Text.load(cfg.text, start, cfg.seq_len)
    return model, text, encode, start


def relative_error(got, want):
    """Mean squared difference, as a share of the target's mean square."""
    got, want = got.astype(mx.float32), want.astype(mx.float32)
    return mx.mean((got - want) ** 2) / (mx.mean(want**2) + 1e-8)


def imitation_losses(model, tokens):
    """(next-token divergence, state error, stand-in error) against the
    original model, each averaged over every position."""
    want_states, want_adds = model.states(tokens, original=True, return_adds=True)
    want_states = [mx.stop_gradient(s) for s in want_states]
    states, adds = model.states(tokens, return_adds=True)
    want = nn.log_softmax(model.read(want_states[-1]), axis=-1)
    got = nn.log_softmax(model.read(states[-1]), axis=-1)
    kl = mx.mean(mx.sum(mx.exp(want) * (want - got), axis=-1))
    match = sum(relative_error(s, w) for s, w in zip(states, want_states)) / len(states)
    pairs = [(a, mx.stop_gradient(w)) for a, w in zip(adds, want_adds) if a is not None]
    local = sum(relative_error(a, w) for a, w in pairs) / len(pairs) if pairs else mx.array(0.0)
    return kl, match, local


def text_losses(model, windows, batch=16):
    """Next-token loss on windows of text for the original model, with
    nothing standing in, and as trained; and how far the last two are from
    the original's next-token distribution."""
    total = dict(original=0.0, nothing=0.0, trained=0.0, kl_nothing=0.0, kl_trained=0.0)
    count = 0
    for at in range(0, len(windows), batch):
        tokens = mx.array(windows[at : at + batch])
        target = tokens[:, 1:]
        logp = {
            "original": nn.log_softmax(model.read(model.states(tokens, original=True)[-1]), axis=-1),
            "nothing": nn.log_softmax(
                model.read(model.states(tokens, table=False, adapters=False)[-1]), axis=-1
            ),
            "trained": nn.log_softmax(model.read(model.states(tokens)[-1]), axis=-1),
        }
        for name, value in logp.items():
            picked = mx.take_along_axis(value[:, :-1], target[..., None], axis=-1)
            total[name] -= mx.sum(picked).item()
        p = mx.exp(logp["original"])
        for name in ("nothing", "trained"):
            total["kl_" + name] += mx.sum(p * (logp["original"] - logp[name])).item()
        count += target.size
    out = {name: value / count for name, value in total.items() if not name.startswith("kl")}
    positions = count + len(windows)  # the divergence also covers the last position
    out.update({name: value / positions for name, value in total.items() if name.startswith("kl")})
    gap = out["nothing"] - out["original"]
    out["gap_recovered"] = (out["nothing"] - out["trained"]) / gap if gap > 0 else None
    return out


def trained_loss(model, windows, batch=16):
    """Next-token loss of the trained model alone, for the periodic look."""
    total, count = 0.0, 0
    for at in range(0, len(windows), batch):
        tokens = mx.array(windows[at : at + batch])
        logits = model.read(model.states(tokens)[-1][:, :-1])
        total += nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="sum").item()
        count += tokens[:, 1:].size
    return total / count


def table_usage(model, windows, batch=16):
    """Share of each table's entries retrieved at least once on this text."""
    if not model.tables:
        return None
    seen = [set() for _ in model.tables]
    for at in range(0, len(windows), batch):
        _, retrieved = model.states(mx.array(windows[at : at + batch]), return_retrieved=True)
        for k, index in enumerate(retrieved):
            seen[k % len(model.tables)].update(np.unique(np.array(index)).tolist())
    return [len(s) / table.fire.shape[0] for s, table in zip(seen, model.tables)]


def final_evaluation(model, text, probes, encode, start, cfg):
    was_training = model.training
    model.eval()
    try:
        windows = text.held_out(cfg.eval_windows)
        known = probes.right(model, original=True)
        sentences = SimpleNamespace(encode=encode, start=start)
        cost = model.stand_in_cost()
        removed = model.removed_parameters()
        frozen = model.count_parameters()["frozen"]
        return {
            "removed": {
                "layers": list(model.remove_layers),
                "numbers": removed,
                "share_of_model": removed / frozen,
            },
            "cost": {**cost, "share": cost["stand_in"] / cost["removed_mlp"]},
            "held_out": {
                "windows": len(windows),
                "tokens": int(windows[:, 1:].size),
                **text_losses(model, windows),
            },
            "probes": {
                "listed": len(probes),
                "nothing": probes.score(probes.right(model, table=False, adapters=False), known),
                "trained": probes.score(probes.right(model), known),
            },
            "sentences": {
                "original": ordinary_text_loss(model, sentences, original=True),
                "nothing": ordinary_text_loss(model, sentences, table=False, adapters=False),
                "trained": ordinary_text_loss(model, sentences),
            },
            "table_used": table_usage(model, windows[:128]),
        }
    finally:
        model.train(was_training)


def run(cfg, out_dir, resume=False, time_only=0, compile_step=True, ckpt_minutes=10.0, probes=None):
    """Train and evaluate. ``probes`` replaces the built-in list of
    well-known facts (for tests on a stand-in model)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, text, encode, start = build_run(cfg)
    probes = Probes(encode, start, probes)
    optimizers = {
        "tables": optim.AdamW(takeover_learning_rate(cfg, cfg.lr), weight_decay=cfg.weight_decay),
        "dense": optim.AdamW(takeover_learning_rate(cfg, cfg.lr), weight_decay=cfg.weight_decay),
        "adapters": optim.AdamW(takeover_learning_rate(cfg, cfg.adapter_lr), weight_decay=0.0),
    }
    ckpt_path = out / "checkpoint.safetensors"
    step0 = 0
    if resume and ckpt_path.exists():
        step0 = load_checkpoint(ckpt_path, model, optimizers)
        step0 = completed_checkpoint_step(model, optimizers, step0)
        print(f"resumed from {ckpt_path} at step {step0}")
    else:
        trainable = model.trainable_parameters()
        for name, optimizer in optimizers.items():
            if trainable.get(name):
                optimizer.init({name: trainable[name]})
    mx.eval(model.parameters(), [o.state for o in optimizers.values()])

    params = model.count_parameters()
    if not params["trained"]:
        raise ValueError("nothing to train: a run needs a table, a small MLP or adapters")
    eval_every = cfg.eval_every or max(50, cfg.steps // 40)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    tokens_per_step = cfg.batch_size * cfg.seq_len
    print(f"run {cfg.variant} seed {cfg.seed} ({VARIANTS[cfg.variant]['about']})")
    print(
        f"model  MLP removed from layers {list(model.remove_layers)} "
        f"({model.removed_parameters():,} numbers); {asdict(model.cfg)}; {model.base.dtype}"
    )
    print(f"params {params}")
    print(f"text   {len(text.train):,} training tokens, {len(text.valid):,} held out")

    state = [model.state] + [o.state for o in optimizers.values()]

    def loss_fn(model, tokens):
        kl, match, local = imitation_losses(model, tokens)
        total = cfg.kl_weight * kl + cfg.match_weight * match + cfg.local_weight * local
        return total, kl, match, local

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(tokens):
        values, grads = loss_and_grad(model, tokens)
        grads, _ = optim.clip_grad_norm(grads, cfg.grad_clip)
        for name, optimizer in optimizers.items():
            if grads.get(name):
                optimizer.update(model, {name: grads[name]})
        return mx.stack(values)

    if compile_step:
        train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)
    rng = np.random.default_rng([cfg.seed, 0])
    # Resume at the same training windows as an uninterrupted run, so the
    # matched-data comparison does not change after a restart.
    for _ in range(step0):
        rng.integers(0, len(text.train) - cfg.seq_len + 2, cfg.batch_size)

    def one_step():
        values = train_step(mx.array(text.batch(rng, cfg.batch_size)))
        mx.eval(values, state)
        values = np.array(values)
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite training loss; stopping this run")
        return values

    def periodic():
        was_training = model.training
        model.eval()
        try:
            return {
                "held_out_loss": trained_loss(model, text.held_out(cfg.periodic_windows)),
                "probes_right": int((probes.right(model) & known).sum()),
                "probes_known": int(known.sum()),
            }
        finally:
            model.train(was_training)

    model.eval()
    known = probes.right(model, original=True)
    model.train()

    if time_only:
        for _ in range(3):  # compile and warm up
            one_step()
        t0 = time.perf_counter()
        for _ in range(time_only):
            one_step()
        sec_per_step = (time.perf_counter() - t0) / time_only
        t0 = time.perf_counter()
        periodic()
        eval_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        final_evaluation(model, text, probes, encode, start, cfg)
        final_sec = time.perf_counter() - t0
        timing = {
            "sec_per_step": sec_per_step,
            "eval_sec": eval_sec,
            "final_eval_sec": final_sec,
            "tokens_per_step": tokens_per_step,
            "tokens_per_sec": tokens_per_step / sec_per_step,
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

    metrics_path = out / "metrics.jsonl"
    if metrics_path.exists():  # keep only the evaluations up to where we resume
        kept = [
            line for line in metrics_path.read_text().splitlines()
            if json.loads(line)["step"] <= step0
        ]
        metrics_path.write_text("".join(line + "\n" for line in kept))
    train_sec, window, window_n = 0.0, np.zeros(4), 0
    started = last_ckpt = time.perf_counter()
    step = step0
    try:
        while step < cfg.steps:
            t0 = time.perf_counter()
            window += one_step()
            train_sec += time.perf_counter() - t0
            window_n += 1
            step += 1

            evaluate_now = step % eval_every == 0 or step == cfg.steps
            if step % cfg.log_every == 0 or evaluate_now:
                total, kl, match, local = (window / window_n).tolist()
                line = (
                    f"step {step:7d} | loss {total:.4f} (next token {kl:.3f}, states {match:.3f}, "
                    f"stand-in {local:.3f}) | {train_sec / (step - step0) * 1000:.0f} ms/step"
                )
                if evaluate_now:
                    record = {
                        "step": step, "loss": total, "kl": kl, "match": match, "local": local,
                        "sec_per_step": train_sec / (step - step0),
                        "elapsed_min": (time.perf_counter() - started) / 60,
                        **periodic(),
                    }
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    line += (
                        f" | held-out loss {record['held_out_loss']:.3f}, known facts "
                        f"{record['probes_right']} of {record['probes_known']}"
                    )
                    window, window_n = np.zeros(4), 0
                print(line, flush=True)
            if time.perf_counter() - last_ckpt > ckpt_minutes * 60:
                save_checkpoint(ckpt_path, model, optimizers, step)
                last_ckpt = time.perf_counter()
    except KeyboardInterrupt:
        save_checkpoint(ckpt_path, model, optimizers, step)
        print(f"\ninterrupted at step {step}; checkpoint saved. Add --resume to continue.")
        raise SystemExit(130)

    # The finished run keeps the trained additions only: the optimizer state
    # of a large table is twice its size again.
    save_checkpoint(ckpt_path, model, {}, step)
    final = {
        "kind": "takeover",
        "variant": cfg.variant,
        "about": VARIANTS[cfg.variant]["about"],
        "seed": cfg.seed,
        "steps": step,
        "tokens_trained": step * tokens_per_step,
        "params": params,
        "sec_per_step": train_sec / max(1, step - step0),
        "wall_minutes": (time.perf_counter() - started) / 60,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "platform": platform.platform(),
    }
    final.update(final_evaluation(model, text, probes, encode, start, cfg))
    (out / "final.json").write_text(json.dumps(final, indent=2))
    held, facts = final["held_out"], final["probes"]
    print(
        f"final  held-out loss: original {held['original']:.3f}, nothing put back "
        f"{held['nothing']:.3f}, trained {held['trained']:.3f} | known facts: "
        f"{facts['nothing']['right']} with nothing, {facts['trained']['right']} trained, "
        f"of {facts['trained']['known']}"
    )
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
