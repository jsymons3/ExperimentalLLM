# Recurrent block + learned lookup: multi-hop fact test

A toy test of one idea: a tiny network that runs the same block several times
and, on every pass, looks something up in a large learned table.

The network is taught a made-up set of facts ("A's boss is B", "B's city is
C") and chained questions ("what is A's boss's city?"). Some facts are only
ever shown on their own, never inside a chain. The test is whether the model
answers chained questions that need those facts. If the lookup model does and
the same block without the table (or with a single lookup) does not, repeated
passes plus lookup is doing something neither does alone.

Nothing in the table is hand-written. Keys, queries and stored values start
random and are learned from the task loss.

## Setup

Apple silicon Mac, Python 3.10 or newer.

```bash
pip install -r recurrent_lookup/requirements.txt
```

Run everything from the repository root.

## Run it

1. Check the code works on your machine (seconds):

```bash
python -m pytest recurrent_lookup -q
```

2. Smoke run on a 200-entity graph, to see loss fall and the numbers print:

```bash
python -m recurrent_lookup.train --preset smoke --variant C --steps 3000 --out runs/smoke/C
```

3. Pilot on the full graph before committing a night. Look at 1-hop recall and
   whether loss is still falling:

```bash
python -m recurrent_lookup.train --preset full --variant C --steps 5000 --out runs/pilot/C
```

```bash
python -m recurrent_lookup.train --preset full --variant A --steps 5000 --out runs/pilot/A
```

   If learning is slow, use `--preset pilot` (1,000 entities, 20,000 facts) for
   the sweep instead of `full`.

4. The overnight sweep. `caffeinate` stops the Mac sleeping:

```bash
caffeinate -is python -m recurrent_lookup.sweep --preset full --hours 7.5 --out runs/overnight
```

   It times each variant, picks one step count for all ten runs so the total
   fits the budget, runs them most-important-first, and writes
   `runs/overnight/report.md`. If it is interrupted, the same command resumes:
   finished runs are skipped and a half-finished run continues from its
   checkpoint. Extra flags are passed to every run, for example `--lr 3e-4`.

5. Rebuild the report at any time, including while the sweep is running:

```bash
python -m recurrent_lookup.report runs/overnight
```

To train one run for longer, repeat its command with a larger `--steps` and
`--resume`.

## The task

- A random graph: 5,000 entities, 20 relations, one tail per (entity,
  relation), so 100,000 facts (`full` preset).
- One-hop question `[h, r]`, two-hop question `[h, r1, r2]`. The answer is read
  at the last position.
- Every fact is trained as a one-hop question.
- 5% of facts are held out of two-hop training: no training chain uses them.
- 40% of the remaining chains are two-hop training data. The rest are test
  chains, in four groups:

| Group | What the chain needs |
|---|---|
| 2-hop unseen | Only facts that do appear in training chains |
| held-out 1st | A held-out fact for the first hop |
| held-out 2nd | A held-out fact for the second hop |
| held-out both | Held-out facts for both hops |

Held-out 2nd and both are the main result. With causal attention the prefix
`[h, r1]` is processed identically in a one-hop and a two-hop question, so the
first hop comes for free. The second hop is right only if the model fetches a
fact it was never trained to use in second position.

## The model

One pre-norm transformer block (width 128, causal attention), applied 4 times
with the same weights. Input and output embeddings are tied. The feed-forward
part is either an MLP or a product-key lookup over single-neuron experts, as
in [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153): each
retrieved entry `i` adds `softmax(key score) * gelu(fire[i] . x) * write[i]`
to the state. Defaults are 4 query heads and top-8 entries per head.

## The runs

| Run | What it is | What it tests |
|---|---|---|
| A | Recurrent block, ordinary feed-forward, no lookup | Baseline at about equal compute |
| B | A with the feed-forward widened to C's parameter count | Whether "just make it bigger" does as well |
| C | Lookup on every pass, 16,384 entries | The idea |
| D | C with the lookup on the first pass only | Whether per-pass retrieval matters |
| E_small, E_large | C with 1,024 and 262,144 entries | Whether capacity grows with table size |

A and C run with three seeds, the rest once: ten runs. All runs share the same
graph and splits; the seed changes initialisation and batch order.

## What is measured

- One-hop recall: does the model store the facts at all.
- Two-hop accuracy per test group.
- Accuracy at 1, 2, 4 and 8 passes at test time.
- Table usage: the share of entries a sample of training questions retrieves.
  Entries that are never retrieved never train.
- Retrieval overlap: for a chain a -> b -> c, do the entries retrieved while
  answering the chain match those retrieved by the direct question about b?
  The control asks about a different entity with the same relation.

Success would be C clearly above A and B on the held-out groups, D well below
C, and overlap well above its control. C scoring high on trained chains and
near chance on held-out ones means it memorised chains (a case table).

## Choices this code makes that the chat left open

- Equal steps, not equal minutes. The chat suggested about 45 minutes per run.
  The sweep gives every run the same number of updates and sizes that number
  to the total budget, so slower variants take more of the night.
- Held-out chains are reported by hop, for the reason given above.
- In run D the later passes have attention only: no lookup and no feed-forward.
- Query normalisation uses running statistics in both training and
  evaluation (see `QueryNorm` in `model.py`). Ordinary batch norm would
  normalise the same question differently on different passes.
  `--query-norm layer` and `--query-norm none` are available.
- One weight decay (0.1) applies to everything, including the table.

## Status

- Tested only on MLX's Linux CPU build, at reduced size, because that build is
  too slow for more. The tests pass, and training, resume, the sweep and the
  report all run end to end.
- Not yet run on an Apple GPU or at full size. Speed, memory and whether the
  defaults learn the 100,000-fact graph in one night are unknown.
- The longest test was run C on 100 entities (2,000 facts) with width 64 and a
  256-entry table, for 4,000 steps. It reached 97% one-hop recall, 39% on
  trained chains (still rising), 15% on unseen chains and 2% on held-out 2nd
  hop, where chance is 1%. So the code learns, but nothing here shows the idea
  working yet.
- In that test one-hop recall was 36% with 1 pass, 85% with 2 and 97% with 4.
  The model spread a single lookup over all four passes, which leaves none for
  a second hop. If the pilot shows the same pattern in `final.json`, train
  with more passes (`--n-passes 8`).
- Learning rate, weight decay, batch size, heads and top-k are untuned
  starting points.
- In the study this task comes from
  ([Grokked Transformers are Implicit Reasoners](https://arxiv.org/abs/2405.15071)),
  composition appeared only after training far past the point of fitting the
  training set. One night may be too short. A flat held-out column on a first
  night is not yet a negative result.

## Files

| File | Contents |
|---|---|
| `data.py` | Graph generation and splits |
| `model.py` | Recurrent block, product-key lookup, dense baseline |
| `configs.py` | Presets and the A-E variants |
| `train.py` | One run: training, periodic evaluation, checkpoints |
| `evaluate.py` | Accuracy, table usage, retrieval overlap |
| `sweep.py` | The ten-run sweep inside a time budget |
| `report.py` | Markdown summary of finished runs |
| `tests/` | Split leakage, causality, retrieval and resume checks |

Each run directory holds `config.json`, `metrics.jsonl` (one line per periodic
evaluation), `final.json` and `checkpoint.safetensors`. The checkpoint for
`E_large` is roughly 800 MB.
