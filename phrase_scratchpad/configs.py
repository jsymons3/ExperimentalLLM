"""Run variants for the three stages and their controls."""

import json
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path

from .model import PhraseConfig
from .tasks import build_suite

VARIANTS = {
    "maths": "stage 1: arithmetic chains, intermediate result taught",
    "clutrr": "stage 1: family-story questions, intermediate link taught",
    "facts": "stage 2: real-name fact chains, intermediate answer taught",
    "mix": "stage 3: all three; intermediate taught for arithmetic and stories only",
    "mix_split": "stage 3 with the table switched off on arithmetic and stories",
    "facts_unsupervised": "control for stage 3: facts alone, intermediate never taught",
    "maths_plain": "baseline: arithmetic, no scratchpad, same block applications",
    "clutrr_plain": "baseline: family stories, no scratchpad, same block applications",
    "facts_plain": "baseline: facts, no scratchpad, same block applications",
}
# Tasks whose questions can be answered from the question alone. In stage 3
# they are the ones taught the intermediate step, and in mix_split they run
# with the table off.
REASONING = ("maths", "clutrr")
ALL = ("maths", "clutrr", "facts")
# Share of ``batch_size`` each task gets. Family stories are several times
# longer than the other questions and there are far fewer of them, so a
# quarter-size batch costs about the same and still shows each story often.
TASK_BATCH = {"clutrr": 0.25}
_PLAIN = {"scratchpad": False, "n_passes": 4, "outer_steps_train": (1,)}
# The stage 3 runs share one architecture: the lookup plus a small dense core.
_CORE = {"core_ff": 512}
_STAGE3 = {"one_hop_every_step": True, "model": _CORE}
_SETTINGS = {
    "maths": {"tasks": ("maths",), "supervised": ("maths",)},
    "clutrr": {"tasks": ("clutrr",), "supervised": ("clutrr",)},
    "facts": {"tasks": ("facts",), "supervised": ("facts",)},
    "mix": {"tasks": ALL, "supervised": REASONING, **_STAGE3},
    "mix_split": {"tasks": ALL, "supervised": REASONING, "table_off": REASONING, **_STAGE3},
    "facts_unsupervised": {"tasks": ("facts",), "supervised": (), **_STAGE3},
    "maths_plain": {"tasks": ("maths",), "supervised": (), "model": _PLAIN},
    "clutrr_plain": {"tasks": ("clutrr",), "supervised": (), "model": _PLAIN},
    "facts_plain": {"tasks": ("facts",), "supervised": (), "model": _PLAIN},
}


@dataclass
class RunConfig:
    variant: str = "maths"
    preset: str = "smoke"
    seed: int = 0
    tasks: tuple = ("maths",)
    # Tasks whose training chains are also taught their intermediate answer:
    # the first step's output is trained toward the bridge.
    supervised: tuple = ()
    # Train one-step questions to be answered at every outer step, not only
    # the last. Used where no intermediate answer is taught, so that the
    # first step still learns to answer one-step questions.
    one_hop_every_step: bool = False
    # Share of two-step training questions whose subject is hidden after the
    # first step, so that they can only be answered through the scratchpad.
    hide_subject_prob: float = 0.0
    # Tasks trained with the lookup table switched off. Only sensible for
    # questions that need no stored facts, and with a dense core (core_ff).
    table_off: tuple = ()
    model: dict = field(default_factory=dict)
    task_overrides: dict = field(default_factory=dict)
    steps: int = 20000
    batch_size: int = 256  # questions per task in each batch (see TASK_BATCH)
    two_hop_frac: float = 0.5
    lr: float = 1e-3
    warmup: int = 500
    weight_decay: float = 0.1
    table_weight_decay: float = None  # None: the same as weight_decay
    grad_clip: float = 1.0
    bridge_loss_weight: float = 1.0
    eval_every: int = 0  # 0 means steps // 40
    log_every: int = 100
    eval_size: int = 2000
    eval_steps: tuple = (1, 2, 3, 4)
    lr_schedule: str = "linear warmup from 0 to lr over `warmup` steps, then constant"


def build_config(variant="maths", preset="smoke", seed=0, model=None, **train):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    settings = dict(_SETTINGS[variant])
    fields = {**settings.pop("model", {}), **(model or {})}
    if "outer_steps_train" in fields:
        fields["outer_steps_train"] = tuple(fields["outer_steps_train"])
    if {**settings, **train}.get("table_off") and not fields.get("core_ff"):
        raise ValueError("switching the table off for a task needs a dense core (core_ff)")
    return RunConfig(
        variant=variant, preset=preset, seed=seed, model=fields, **{**settings, **train}
    )


def load_run_config(run_dir):
    saved = json.loads((Path(run_dir) / "config.json").read_text())
    known = {f.name for f in dataclass_fields(RunConfig)}
    return RunConfig(**{k: v for k, v in saved.items() if k in known})


def build_run(cfg: RunConfig):
    """The task suite and the model configuration for a run."""
    suite = build_suite(
        cfg.tasks, cfg.preset, eval_size=cfg.eval_size, overrides=cfg.task_overrides
    )
    model_cfg = PhraseConfig(
        vocab_size=len(suite.vocab),
        n_output=suite.n_output,
        question_len=suite.question_len,
        answer_len=suite.answer_len,
        **cfg.model,
    )
    return suite, model_cfg
