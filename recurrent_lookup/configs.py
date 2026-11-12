"""Graph-size presets and the run variants A-E."""

from dataclasses import asdict, dataclass, field

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
    reference = ModelConfig(
        vocab_size=graph_cfg.n_entities + graph_cfg.n_relations,
        n_entities=graph_cfg.n_entities,
        **fields,
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


def graph_config(cfg: RunConfig):
    return GraphConfig(**cfg.graph)


def model_config(cfg: RunConfig):
    graph = graph_config(cfg)
    return ModelConfig(
        vocab_size=graph.n_entities + graph.n_relations,
        n_entities=graph.n_entities,
        **cfg.model,
    )
