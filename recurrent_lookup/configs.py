"""Graph-size presets and the run variants."""

import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path

from .data import GraphConfig
from .model import ModelConfig, matched_dense_width

# n_subkeys is the sub-key count for run C; its table has n_subkeys**2 entries.
PRESETS = {
    "smoke": {"n_entities": 200, "n_subkeys": 32},  # 4k facts, 1k entries
    "pilot": {"n_entities": 1000, "n_subkeys": 64},  # 20k facts, 4k entries
    "full": {"n_entities": 5000, "n_subkeys": 128},  # 100k facts, 16k entries
}

VARIANTS = {
    "A": "recurrent block, ordinary feed-forward, no lookup",
    "B": "A with the feed-forward widened to C's parameter count",
    "C": "recurrent block with a lookup on every pass",
    "D": "C with the lookup on the first pass only",
    "E_small": "C with a table 16x smaller",
    "E_large": "C with a table 16x larger",
    "slot_soft": "C with one scratchpad slot written in token form",
    "slot_raw": "slot_soft, but the slot holds a raw vector",
    "slot_soft_prefix": "slot_soft, plus a loss that puts the bridge in step 1",
    "slot_soft_overwrite": "bridge loss, full overwrite with soft entity content",
    "slot_hard_overwrite": "bridge loss, full overwrite with one predicted entity",
    "slot_soft_k3": "slot_soft with three slots",
}

SLOT_KEYS = (
    "num_slots",
    "slot_content",
    "slot_write",
    "outer_steps_train",
    "carry_hidden",
    "step_embedding",
    "slot_position",
)
# Settings a scratchpad model gets unless they are given explicitly.
SLOT_DEFAULTS = {"n_passes": 2, "outer_steps_train": (2, 3, 4)}
SLOT_VARIANTS = {
    "slot_soft": {"num_slots": 1, "slot_content": "soft"},
    "slot_raw": {"num_slots": 1, "slot_content": "raw"},
    "slot_soft_prefix": {"num_slots": 1, "slot_content": "soft"},
    "slot_soft_overwrite": {"num_slots": 1, "slot_content": "soft", "slot_write": "overwrite"},
    "slot_hard_overwrite": {"num_slots": 1, "slot_content": "hard", "slot_write": "overwrite"},
    "slot_soft_k3": {"num_slots": 3, "slot_content": "soft"},
}


@dataclass
class RunConfig:
    preset: str = "smoke"
    variant: str = "C"
    seed: int = 0
    graph: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    steps: int = 20000
    batch_size: int = 512
    two_hop_frac: float = 0.5  # share of each batch that is two-hop
    lr: float = 1e-3
    warmup: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 0  # 0 means steps // 40
    log_every: int = 100
    eval_passes: tuple = (1, 2, 4, 8)
    # Outer step counts evaluated at the end of a scratchpad run.
    eval_steps: tuple = (1, 2, 3, 4, 6, 8)
    # Weight of the extra loss that trains the step-1 readout of two-hop
    # training chains toward the bridge entity. 0 switches it off.
    prefix_loss_weight: float = 0.0
    # Weight decay for the table (sub-keys and entries). None means the same
    # decay as everything else, which is what the first sweep used.
    table_weight_decay: float = None
    lr_schedule: str = "linear warmup from 0 to lr over `warmup` steps, then constant"


def build_config(preset="smoke", variant="C", seed=0, graph=None, model=None, **train):
    """Assemble a RunConfig. ``graph`` and ``model`` override preset fields."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    base = PRESETS[preset]
    graph_cfg = GraphConfig(n_entities=base["n_entities"])
    for key, value in (graph or {}).items():
        setattr(graph_cfg, key, value)

    # Model fields are filled in against run C first, so that B can be matched
    # to C's parameter count whatever overrides are in play.
    fields = {"n_subkeys": base["n_subkeys"]}
    fields.update(model or {})
    if variant in SLOT_VARIANTS:
        fields = {**SLOT_VARIANTS[variant], **fields}
        if variant in ("slot_soft_prefix", "slot_soft_overwrite", "slot_hard_overwrite"):
            train.setdefault("prefix_loss_weight", 1.0)
    if fields.get("num_slots"):
        fields = {**SLOT_DEFAULTS, **fields}
    elif "outer_steps_train" in fields or train.get("prefix_loss_weight"):
        raise ValueError("outer steps and the prefix loss need num_slots > 0")
    if "outer_steps_train" in fields:
        fields["outer_steps_train"] = tuple(fields["outer_steps_train"])
    reference = ModelConfig(
        vocab_size=graph_cfg.n_entities + graph_cfg.n_relations,
        n_entities=graph_cfg.n_entities,
        **{k: v for k, v in fields.items() if k not in SLOT_KEYS},
    )
    if variant == "A":
        fields.update(ff="dense", d_ff=4 * reference.d_model)
    elif variant == "B":
        fields.update(ff="dense", d_ff=matched_dense_width(reference))
    elif variant == "D":
        fields.update(lookup_passes="first")
    elif variant == "E_small":
        fields.update(n_subkeys=max(2, reference.n_subkeys // 4))
    elif variant == "E_large":
        fields.update(n_subkeys=reference.n_subkeys * 4)

    return RunConfig(
        preset=preset,
        variant=variant,
        seed=seed,
        graph=asdict(graph_cfg),
        model=fields,
        **train,
    )


def load_run_config(run_dir):
    """The RunConfig a run directory was trained with."""
    saved = json.loads((Path(run_dir) / "config.json").read_text())
    known = {f.name for f in dataclass_fields(RunConfig)}
    return RunConfig(**{k: v for k, v in saved.items() if k in known})


def graph_config(cfg: RunConfig):
    return GraphConfig(**cfg.graph)


def model_config(cfg: RunConfig):
    graph = graph_config(cfg)
    return ModelConfig(
        vocab_size=graph.n_entities + graph.n_relations,
        n_entities=graph.n_entities,
        **cfg.model,
    )
