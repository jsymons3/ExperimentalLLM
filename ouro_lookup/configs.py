"""The runs: what is added to Ouro and how two-step questions are handled."""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .data import PRESETS
from .model import AddonConfig

DEFAULT_WEIGHTS = "data/ouro-1.4b"

# Two ways to answer a two-step question such as
# "The doctor of the mentor of Daniel is":
#
#   chain_mode "note": a discrete handoff, as in the scratchpad that worked on
#       the graph task. The question is taught to produce the intermediate
#       answer (the mentor). That one token is written into a fresh one-step
#       question ("The doctor of Alice is") and the model runs again from
#       scratch. Same model and same table on both steps; nothing else is
#       carried over.
#   chain_mode "latent": one pass. The answer must come out of Ouro's own
#       loops, which hand on a continuous state. With teach_loop = k the state
#       after loop k is also graded: it must read as the intermediate answer
#       on a two-step question, and as the answer on a one-step question.
#
# adapters_learn "chains" trains the adapters on two-step questions only, so
# the facts themselves can only be stored in the table; "all" trains them on
# every question.
VARIANTS = {
    "table": dict(
        about="frozen Ouro plus the table; two-step questions go through a written note",
        addons=dict(adapter_rank=0),
        chain_mode="note",
    ),
    "table_adapters": dict(
        about="table plus adapters; two-step questions go through a written note",
        addons=dict(adapter_rank=8),
        chain_mode="note",
        adapters_learn="chains",
    ),
    "latent": dict(
        about="table plus adapters; two-step questions answered in one pass through "
        "Ouro's loops, with the intermediate answer taught at loop 2",
        addons=dict(adapter_rank=8),
        chain_mode="latent",
        teach_loop=2,
        adapters_learn="chains",
    ),
    "adapters": dict(
        about="control: adapters only, no table, written note",
        addons=dict(adapter_rank=8, table_layers=()),
        chain_mode="note",
        adapters_learn="all",
    ),
    # Extras, not in the default sweep.
    "table_latent": dict(
        about="frozen Ouro plus the table, one pass, intermediate answer taught at loop 2",
        addons=dict(adapter_rank=0),
        chain_mode="latent",
        teach_loop=2,
    ),
    "latent_untaught": dict(
        about="table plus adapters, one pass, graded on the final answer only",
        addons=dict(adapter_rank=8),
        chain_mode="latent",
        teach_loop=0,
        adapters_learn="chains",
    ),
    "table_adapters_all": dict(
        about="written note, with adapters that also learn from one-step questions",
        addons=dict(adapter_rank=8),
        chain_mode="note",
        adapters_learn="all",
    ),
}


@dataclass
class RunConfig:
    variant: str = "table"
    preset: str = "full"
    seed: int = 0
    weights: str = DEFAULT_WEIGHTS  # directory with Ouro's files
    dtype: str = "stored"  # "stored" (bfloat16), "float16" or "float32"
    loops: int = 0  # times the layer stack is applied; 0 for the model's own 4
    world: dict = field(default_factory=dict)  # n_people, n_relations, ...
    addons: dict = field(default_factory=dict)  # AddonConfig fields
    chain_mode: str = "note"
    teach_loop: int = 0
    teach_weight: float = 1.0
    adapters_learn: str = "chains"
    # Training.
    steps: int = 8000
    batch_size: int = 64  # questions per step
    two_hop_frac: float = 0.5
    lr: float = 1e-3  # table
    adapter_lr: float = 2e-4
    warmup: int = 200
    weight_decay: float = 0.1  # table only
    grad_clip: float = 1.0
    eval_every: int = 0  # 0: forty evaluations per run
    log_every: int = 50
    eval_size: int = 500  # questions per group in the final evaluation
    periodic_size: int = 128  # questions per group in the periodic ones

    def addon_config(self):
        values = dict(self.addons)
        if values.get("table_layers") is not None:
            values["table_layers"] = tuple(values["table_layers"])
        return AddonConfig(**values)


def build_config(variant, preset, seed=0, world=None, addons=None, **overrides):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    spec = {k: v for k, v in VARIANTS[variant].items() if k != "about"}
    merged_addons = {**spec.pop("addons", {}), **(addons or {})}
    cfg = RunConfig(
        variant=variant,
        preset=preset,
        seed=seed,
        world={**PRESETS[preset], **(world or {})},
        addons=merged_addons,
        **{**spec, **overrides},
    )
    if cfg.chain_mode not in ("note", "latent"):
        raise ValueError("chain_mode is 'note' or 'latent'")
    if cfg.adapters_learn not in ("chains", "all"):
        raise ValueError("adapters_learn is 'chains' or 'all'")
    if cfg.teach_loop and cfg.chain_mode == "note":
        raise ValueError("teach_loop applies to chain_mode 'latent' only")
    addon = cfg.addon_config()
    if addon.table_layers == () and not addon.adapter_rank:
        raise ValueError("a run needs a table, adapters, or both")
    return cfg


def load_run_config(run_dir):
    raw = json.loads((Path(run_dir) / "config.json").read_text())
    known = {f.name for f in fields(RunConfig)}
    return RunConfig(**{k: v for k, v in raw.items() if k in known})


def save_run_config(cfg, run_dir):
    (Path(run_dir) / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
