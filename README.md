# ExperimentalLLM

Small language-model experiments.

| Experiment | Where | What it is |
|---|---|---|
| Learned core / memory separation | [`core_memory/`](core_memory/README.md) | Natural-text recurrent LM with a dense core, learned expert memory, memory-off training and budgeted gating. Paired evaluations measure memory dependence. |
| Recurrent block + learned lookup | [`recurrent_lookup/`](recurrent_lookup/README.md) | A tiny block applied repeatedly, with a product-key lookup table on every pass, tested on multi-hop facts. MLX, runs on an Apple silicon laptop. |
| Diffusion LM with a global workspace | `hiPhiLLM.ipynb` | Described below. JAX/Flax notebook. |

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
