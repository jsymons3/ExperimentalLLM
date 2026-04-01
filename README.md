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
