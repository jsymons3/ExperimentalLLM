# Learned core / memory separation

A separate, byte-level language-model experiment on natural text. The original
`recurrent_lookup` graph experiment and its checkpoints are unchanged. This is
a new model trained from scratch; graph-model checkpoints are not compatible.

The hypothesis: a constrained dense core, a learned expert memory, occasional
memory-off training, and a price for memory access might encourage different
roles. These pressures **do not guarantee** abstractions in the core or facts
in the memory. Memory may store syntax or phrase completions; the core may
retain common facts. Stronger weight decay is a regularizer, not a direct
measurement or charge for stored information.

## Architecture and objective

Each recurrent pass has causal attention and an always-active dense MLP,
followed by the existing PEER-style product-key expert memory as an optional
residual. The table is learned end to end. Facts are not inserted into it.
Query normalization is per token (LayerNorm), avoiding batch statistics that
could leak future-token information during language-model training.

The gate reads the current state before retrieval. During training it samples
a binary choice with a straight-through gradient. At evaluation it uses a
probability threshold of 0.5. A hard per-position budget limits the number of
passes allowed to contribute memory output. The objective on gated batches is
next-token cross entropy plus `lookup_cost * mean_calls_per_token`. A call
means access through all lookup heads on one recurrent pass, not one expert.

Memory-off batches genuinely skip the table, and still predict the original
text. There is no artificial "need lookup" answer. They preserve pressure for
the core to predict useful language, but can also encourage fact memorization
inside the core. The first `gate_warmup` steps use forced memory on batches
where memory is available, to give it a chance to learn before gating.

Embeddings (including the tied output projection), core matrices and routing
matrices use the core weight decay. Expert `fire`/`write` vectors and subkeys
use memory decay. Biases and normalization vectors get no decay.

**Compute limitation:** gated mode computes candidate memory output even when
the gate rejects it, so its budget limits contributions rather than GPU time.
The `off` mode skips lookup computation. True sparse dispatch is future work;
do not claim speed savings from gate counts. A lookup budget does not prevent
the table from participating in reasoning.

Defaults (untuned): width 128, 4 passes, 4,096 entries, 2 heads x top-4 experts,
128-byte context, and at most 2 memory contributions per token position.
Byte contexts are short; this is a mechanism pilot, not a useful pretrained LM.
Training uses ordinary MLX lazy execution on CPU/Metal. JIT compilation is
disabled because MLX 0.32.4 rejected this stateful multi-output training step.

## Setup and data

From the repository root, using the existing Apple-silicon Python environment:

```bash
env -u PYTHONPATH .venv/bin/python -m pip install -r core_memory/requirements.txt
env -u PYTHONPATH .venv/bin/python -m core_memory.data --wikitext-2 --out data/wikitext-2-bytes
```

[WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext) is the small
Wikipedia-derived starting corpus. Preparation records the exact downloaded
revision, source licenses, document hashes and packed-file hashes. It preserves
the official split boundaries, groups rows by top-level article headings,
removes exact duplicates within a split, and rejects exact duplicates across
splits. WikiText contains some title-like table fragments, so grouping is
conservative, not a perfect reconstruction of Wikipedia articles. Related
articles and near duplicates are not detected. All files remain local under
ignored `data/`; the source is CC-BY-SA-3.0/GFDL, as recorded in its dataset card.

For a different corpus, including exported Simple English Wikipedia documents:

```bash
env -u PYTHONPATH .venv/bin/python -m core_memory.data --out data/my-corpus \
  --train corpus/train/*.txt --valid corpus/valid/*.txt --test corpus/test/*.txt
```

Each supplied UTF-8 file is a document. Supply document-level splits yourself;
keep paraphrases of a fact in the intended evaluation set when testing transfer.
The tokenizer uses bytes 0–255 plus document boundary token 256, with no learned
vocabulary or unknown-token issue. Sampled windows never cross documents.

## Run the comparison

Run these sequentially, using the same seed and exposure for each:

```bash
env -u PYTHONPATH .venv/bin/python -m core_memory.train --data data/wikitext-2-bytes --variant core --out runs/core-memory/core --steps 10000
env -u PYTHONPATH .venv/bin/python -m core_memory.train --data data/wikitext-2-bytes --variant unpressured --out runs/core-memory/unpressured --steps 10000
env -u PYTHONPATH .venv/bin/python -m core_memory.train --data data/wikitext-2-bytes --variant selective --out runs/core-memory/selective --steps 10000
env -u PYTHONPATH .venv/bin/python -m core_memory.report runs/core-memory
```

| Variant | Training access | Core decay | Table decay | Off batches | Lookup cost |
|---|---|---:|---:|---:|---:|
| core | Off | 0.1 | 0 | All | 0 |
| unpressured | Always on | 0.01 | 0.01 | None | 0 |
| selective | Learned gate and budget | 0.1 | 0 | 25% | 0.01 |

The core control allocates an unused table but does not train or access it.
Its `off` score is the relevant result; its on/gated scores use untrained
weights. It has fewer active parameters, so it is not a parameter-matched
control. The unpressured comparison tests a bundle of changes. To identify the
effect of a specific pressure, repeat selective runs changing one knob at a
time: `--lookup-cost 0`, `--memory-dropout 0`, or
`--core-decay 0.01 --memory-decay 0.01`. Repeat across seeds before conclusions.

`--max-seconds 600` bounds a pilot's training time; checkpoint and final
evaluation can take additional time. `--cpu` avoids using the GPU. For an
unattended run prefix the command with `caffeinate -is`, keep power connected,
and keep the lid open. This command does not automatically detach a run.

Checkpoints are atomic and include the model, all optimizer states and the
NumPy RNG that drives both data sampling and explicit MLX routing keys.
SIGINT/SIGTERM finishes the current update, checkpoints and evaluates. Repeat
the same arguments with a larger `--steps` and `--resume` to continue. Resume
rejects changed training settings or corpus manifests. Changing reporting
intervals, runtime limits or device is allowed, though switching device need
not reproduce floating-point results exactly.

## Measure the proposed separation

Periodic validation and the final train/validation comparison use **identical
sampled windows** for three modes: forced off, forced on at every pass (ignoring
the budget), and deterministic gated access. NLL is in nats; bits/token includes
the boundary symbol. Evaluation reports the loss gap, gate usage, correlation
between usage and the gap, and a few contexts with large gaps. These contexts
are examples, not semantic labels. Results use a fixed reproducible sample,
not the complete corpus; increase `--eval-batches` for less sampling noise.

```bash
env -u PYTHONPATH .venv/bin/python -m core_memory.evaluate runs/core-memory/selective --split valid --batches 64
```

The test split is reserved for final evaluation, not periodic tuning. Optional
JSONL probes allow manually annotated completion tests, with each line having
`prompt`, `target`, and `kind` fields. For example, `kind` can distinguish
context-supplied reasoning from facts taught in training and tested with a new
wording. `--probes probes.jsonl` scores just the target bytes and summarizes
memory benefit by the supplied labels. Arbitrary labels do not establish a
valid benchmark; make sure context length, training exposure and controls fit
the intended question.

A large off/on gap means **memory dependence**, not "this is a fact."
Names/numbers are imperfect proxies, and unpredictable wording also creates
gaps. A separate model trained on other documents does not solve that semantic
identification problem. Useful evidence would include retained core-only
competence on context-supplied rules, memory benefits on independently labelled
facts across paraphrases, and successful interventions or edits to those facts.
Keep the existing synthetic chain tests as complementary evidence; ordinary
language-model loss cannot establish systematic composition.

## Verification

```bash
env -u PYTHONPATH .venv/bin/python -m pytest --rootdir=. --confcutdir=. core_memory/tests recurrent_lookup/tests -q
```

Tests cover true memory bypass, causal predictions, strict routing budgets,
gradient flow into core/router/table, correct decay groups, document boundaries,
cross-split duplicate rejection, and optimizer/sampling/routing restoration.

Initial local verification (2026-11-12, MLX 0.32.4): all 26 tests across both
experiments passed. Three CPU pilots used width 64, context 64, 1,024 memory
entries, batch 4, 1,000 steps, seed 0, warmup 50 and gate warmup 200. Each saw
256,000 byte targets, only about 0.023 equivalent passes over the training
corpus. All three consumed identical sampling/routing RNG streams.

Validation NLL was 2.2548 for core-only; 2.2561 with always-on memory versus
2.2782 when that model's memory was disabled; and 2.2517 for the selective
model, whose deterministic gate made effectively no calls. These tiny pilots
verify execution and expose early gate collapse. They do not show semantic
separation, nor a reliable advantage between models. Try a longer memory
warmup and a smaller lookup cost as controlled ablations before a long sweep.
