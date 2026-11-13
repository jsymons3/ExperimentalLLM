"""Load a finished or in-progress run from its directory."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .configs import graph_config, load_run_config, model_config
from .data import MultiHopData
from .model import RecurrentModel


def load_run(run_dir):
    """Return (config, data, model, step) for a run directory. Read-only."""
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    model = RecurrentModel(model_config(cfg))
    arrays, metadata = mx.load(
        str(run_dir / "checkpoint.safetensors"), return_metadata=True
    )
    weights = {k[6:]: v for k, v in arrays.items() if k.startswith("model.")}
    missing = {k for k, _ in tree_flatten(model.parameters())} - set(weights)
    if missing:
        raise ValueError(
            f"{run_dir}: checkpoint has no values for {sorted(missing)}; "
            "it was saved by a different model configuration"
        )
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return cfg, MultiHopData(graph_config(cfg)), model, int(metadata["step"])
