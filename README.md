# ExperimentalLLM

Small language-model experiments.

| Experiment | Where | What it is |
|---|---|---|
| Multi-token scratchpad | [`phrase_scratchpad/`](phrase_scratchpad/README.md) | The scratchpad that worked on the graph task, extended to short phrases and tested in stages: arithmetic and family stories, real-name facts, then all three with reasoning kept in a small core and facts in the table. Built, not yet run at full size. |
| Learned core / memory separation | [`core_memory/`](core_memory/README.md) | Natural-text recurrent LM with a dense core, learned expert memory, memory-off training and budgeted gating. Paired evaluations measure memory dependence. |
| Recurrent block + learned lookup | [`recurrent_lookup/`](recurrent_lookup/README.md) | A tiny block applied repeatedly, with a product-key lookup table on every pass, tested on multi-hop facts. Includes a scratchpad variant that writes intermediate results in token form. MLX, runs on an Apple silicon laptop. |
| Diffusion LM with a global workspace | `hiPhiLLM.ipynb` | Described below. JAX/Flax notebook. |

Latest results (November 13, 2026): the two 64,000-step synthetic scratchpad
controls are complete. Removing bridge supervision leaves held-out second-hop
accuracy at 0.25%. The parameter-matched dense control recalls 98.2% of individual
facts and writes the correct bridge on 98.0% of held-out second-hop questions,
but answers 0% internally versus 90.0% through two separate calls. It also fails
on trained chains, so this is a training failure under the tested recipe, not
proof of an inherent dense-model limitation. See the
[completed controls and diagnostics](recurrent_lookup/README.md#completed-follow-up-results-november-13)
and [machine-readable results](recurrent_lookup/results/followup-2026-11-13.json).
The Wikipedia scratchpad adaptation has not started.

## hiPhiLLM.ipynb

This is a **single-notebook research scaffold** that:

- Trains a diffusion LM in **embedding space** (predicting ε).
- Uses a **Global Workspace (GWS)** + per-layer **state buffers** + optional **reverse write-back**.
- Adds **true self-conditioning** (2-pass x₀ feedback) with strict `stop_gradient`.
- Runs on **single device (`jit`)** or **multi-device (`pmap`)** automatically.
- Adds an **instrumental GWS auxiliary objective**: predict the **final token** of the sequence from GWS.
- Includes **learned positional embeddings**, so the model actually knows token order.
- Saves and restores the **full optimizer/training state**, not just parameters.

## Files expected
- `data.txt` — plain text training corpus
- `sp.model` — SentencePiece model for tokenization
