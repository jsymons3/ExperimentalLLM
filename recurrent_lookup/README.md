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

### Two-call diagnostic

```bash
python -m recurrent_lookup.two_call runs/continuation-2026-11-12/E_large_s0
```

This read-only checkpoint evaluation compares native chains at 4/8 passes
with two explicit one-hop calls at 2/4 passes each (for a scratchpad run:
2/4 outer steps against calls of 1/2 steps). The second call receives
the predicted intermediate entity. It also measures native accuracy restricted
to chains whose two component facts are independently recalled correctly.
Two-call accuracy on that subset should be 100% by construction; the useful
measure is how often native chaining fails despite correct component recall.
Decomposition and an entity-token handoff are supplied externally, so this
does not establish that the network learned either operation internally.
Outputs include exact counts, example failures and the checkpoint SHA-256.

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

| Run | What it is | Parameters | What it tests |
|---|---|---|---|
| A | Recurrent block, ordinary feed-forward, no lookup | 0.84M | Baseline at about equal compute |
| B | A with the feed-forward widened to C's parameter count | 4.97M | Whether "just make it bigger" does as well |
| C | Lookup on every pass, 16,384 entries | 4.97M | The idea |
| D | C with the lookup on the first pass only | 4.97M | Whether per-pass retrieval matters |
| E_small | C with 1,024 entries | 1.01M | Whether capacity grows with table size |
| E_large | C with 262,144 entries | 68.0M | Same |

Parameter counts are for the `full` preset. A and C run with three seeds, the
rest once: ten runs. All runs share the same
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
- One weight decay (0.1) applies to everything, including the table, unless
  `--table-weight-decay` is given.

## Status

- Tested only on MLX's Linux CPU build, at reduced size, because that build is
  too slow for more. The tests pass, and training, resume, the sweep and the
  report all run end to end.
- At full size, all six variants were built and took one training step on a
  batch of 16. Nothing more was run at full size, and nothing on an Apple GPU.
  Speed, memory and whether the defaults learn the 100,000-fact graph in one
  night are unknown.
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

## Scratchpad variant

The first sweep showed the table stores facts but the model cannot hand the
result of one lookup to the next. This variant gives the model a place to
write an intermediate result in the same form as a typed token, and tests
whether it can then be read like one, including for held-out facts. It is a
hypothesis under test, not a result.

### How it works

- The sequence is `[question tokens, K slots, work position]`. Attention stays
  causal, so the work position, at the end, reads everything.
- The model runs in outer steps. A step applies the block `n_passes` times
  (2 by default), reads the work position out through the tied embedding, and
  writes the result into the slots.
- What is written is `softmax(logits) @ E`: a blend of entity embeddings, so a
  slot holding entity `b` looks like `b` typed into the question. A write head
  gives one weight per slot plus one for "write nothing".
- The next step starts fresh from the question and the slots. Nothing else is
  carried over.
- The loss is cross-entropy on the last step's readout. The number of steps is
  drawn per batch from 2, 3 or 4. No hop labels are used.

### Flags

All default to the original model. The scratchpad variants, and any run with
`--num-slots` above 0, use the values in the last column.

| Flag | Values | With slots |
|---|---|---|
| `--num-slots` | 0 (original model), 1, 3 | 1 |
| `--slot-content` | `soft` (token form), `hard` (top entity, straight-through), `raw` (linear layer on the work state) | `soft` |
| `--inner-passes` (same as `--n-passes`) | block applications per step | 2 |
| `--outer-steps-train` | step counts to sample from per batch | 2 3 4 |
| `--carry-hidden` | keep hidden state between steps | off |
| `--no-step-embedding` | drop the per-step embedding on the work position | embedding on |
| `--prefix-loss-weight` | extra loss: step-1 readout of training chains toward the bridge | 0 |
| `--slot-position` | diagnostic: give slots the position embedding of a question position (0 is the entity position) | none |
| `--table-weight-decay` | separate decay for sub-keys and entries | same as `--weight-decay` |

### Step 1: checks on the existing checkpoints (no training)

```bash
python -m recurrent_lookup.diagnose runs/overnight/B_s0 runs/overnight/C_s0 runs/continuation-2026-11-12/E_large_s0
```

For each run this prints three things and saves them as `diagnostics.md` and
`diagnostics.json` in the run directory:

- Whether the checkpoint reproduces the accuracies in `final.json`. For
  `C_s0` this is the check that the new code has not changed the old model.
- The two-call table, for all five chain groups.
- What the answer position holds after each pass: how often its top entity is
  the bridge, how often the answer, and how often the first-hop position holds
  the bridge.

### Step 2: the scratchpad runs

See the time estimate first. This times 200 steps of each variant and stops:

```bash
python -m recurrent_lookup.sweep --runs slots --preset full --steps 128200 --baseline runs/overnight/C_s0 --out runs/slots --estimate-only
```

Then run them. The same command resumes after an interruption:

```bash
caffeinate -is python -m recurrent_lookup.sweep --runs slots --preset full --steps 128200 --baseline runs/overnight/C_s0 --out runs/slots
```

| Run | Slots | Slot content | Extra loss | Seeds |
|---|---|---|---|---|
| `slot_soft` | 1 | token form | none | 3 |
| `slot_raw` | 1 | raw vector | none | 1 |
| `slot_soft_prefix` | 1 | token form | step-1 readout toward the bridge, training chains only | 1 |
| `slot_soft_k3` | 3 | token form | none | 1 |

They run in this order: `slot_soft` seed 0, `slot_raw`, `slot_soft_prefix`,
`slot_soft` seeds 1 and 2, `slot_soft_k3`. `--baseline` makes the sweep refuse
to start if the graph, optimizer, schedule, batch size or step count differ
from run C's, and puts the first sweep's runs in the same report. To rebuild
the report by hand:

```bash
python -m recurrent_lookup.report runs/slots runs/overnight
```

### What the report adds for these runs

- Accuracy at 1, 2, 3, 4, 6 and 8 outer steps.
- What is in the slot after step 1: how often it decodes to the bridge entity.
- Retrieval overlap between step 2 of a chain and the direct question about
  the bridge.
- The two-call ceiling on the same checkpoint.

### How to read the result

- `slot_soft` far above C on held-out 2nd hop, near its own two-call ceiling,
  with the bridge usually in the slot: the mechanism works.
- `slot_soft` fails and `slot_soft_prefix` works: the mechanism is sound and
  the difficulty is getting it to emerge. That points to a curriculum.
- `slot_soft` and `slot_raw` work equally: token form is not what matters. The
  fresh state per step is.
- Bridge in the slot but held-out accuracy low: the model writes the bridge
  and cannot read it from a slot the way it reads it from the question.
- Everything fails: report it with the diagnostics. Do not tune blindly.

### Where this differs from the design brief

- **Work position.** The brief describes the work position as the existing
  answer position. The original model has none: it reads the answer at the
  last question token. The scratchpad model follows the brief's forward pass
  and adds a learned work token after the slots. With `num_slots = 0` the
  original layout is used, so old checkpoints load and reproduce.
- **Weight decay.** The brief asks both to change only the model variant
  relative to run C and to give the table no weight decay. Run C decayed the
  table at 0.1. These runs keep run C's setting so the comparison has one
  difference. `--table-weight-decay 0` is available.
- **Evaluation depth.** The headline numbers for scratchpad runs use 2 steps
  of 2 passes, the same 4 block applications as run C.
- **Cost.** A scratchpad run processes two more positions per question and
  about 6 block applications instead of 4, so expect well over the 1.5 times
  run C's step time that the block count alone suggests. At tiny size on CPU
  it measured about 2 times.
- **Bridge loss weight.** The brief gives no value for `slot_soft_prefix`. It
  uses 1.0.
- **Readout.** The model's readout covers entities only, so the slot content
  is a blend of entity embeddings.
- **Carried state.** With `--carry-hidden`, question and work positions keep
  their state and only the slot positions are refreshed. No planned run uses
  it.
- **Not built.** The Simple English Wikipedia version. The brief recommends
  waiting for these results, and it overlaps with `core_memory/`, which
  already has a natural-text pipeline with memory-off batches.

### Status of the scratchpad code

- Initial verification used MLX's Linux CPU build at small size. Old checkpoints
  reproduce their stored accuracies exactly, retraining the original variants
  gives identical numbers, and the six-run sweep, the diagnostics and the
  report run end to end.
- November 13 Mac verification: 54 tests pass on CPU and Apple GPU, including
  checkpoint/resume for all three overwrite-comparison variants. Six full-size
  runs completed on Apple GPU; the results are recorded below.

### A small-scale test, and what it suggests

Five runs on a 60-entity graph (1,200 facts), width 32, a 256-entry table,
8,000 steps, one seed each. Chance is 1.7%. A scratchpad run took about 2.4
times run C's time per step.

| Run | Trained chains | Unseen chains | Held-out 2nd | Bridge in slot (held-out 2nd) | Write weight |
|---|---|---|---|---|---|
| C | 19.0% | 13.7% | 4.3% | - | - |
| `slot_soft` | 12.3% | 5.0% | 2.0% | 1.7% | 0.05 |
| `slot_soft_prefix` | 60.7% | 49.0% | 1.0% | 93.0% | 0.28 |
| `slot_soft_prefix --slot-position 0` | 6.3% | 7.0% | 1.7% | 1.7% | 0.02 |
| `slot_soft_prefix --slot-position 0 --no-step-embedding` | 36.0% | 30.3% | 3.0% | 94.3% | 0.31 |

- `slot_soft` did not use its slot. The loss grades only the last step and
  training never uses a single step, so nothing trains the first step
  directly: one-hop accuracy was 10% at 1 step and 96% at 2. The model
  answered from scratch in a later step and wrote almost nothing.
- With the bridge loss the slot is used and chaining works for facts that
  appear in training chains: 49% on unseen chains, against 14% for run C.
- Held-out 2nd hop stayed at chance in every run. With the bridge loss the
  bridge was in the slot for 93% of those chains and two separate calls
  answered 72% of them, but the model chained 1%. Retrieval overlap with the
  direct question was at its control. So at this scale the model writes the
  bridge in token form and still does not look up the second fact the way a
  direct question does.
- Giving the slot the entity position's embedding, the follow-up the brief
  names for this outcome, did not change that.

This is one seed on a very small model. It shows what to look for in the full
runs, not what they will find.

### Full-overwrite comparison (Mac overnight run)

This comparison isolates two mixtures: retaining the old slot, and blending
several predicted entity embeddings. All three arms use the same bridge loss
(weight 1.0), graph, initial weights for a given seed, batch size, optimizer,
and sampled outer depths. The graph is the full 5,000-entity / 100,000-fact
problem. Each arm runs with seeds 0 and 1, in this order within each seed:

| Variant | Slot write | Content |
|---|---|---|
| `slot_soft_prefix` | Learned partial write | Probability-weighted entity embeddings |
| `slot_soft_overwrite` | Always replaces the whole slot | Probability-weighted entity embeddings |
| `slot_hard_overwrite` | Always replaces the whole slot | Predicted top entity, soft straight-through gradient |

The full-overwrite variants are trained that way from initialization. They
never receive the true intermediate entity as input during training. The
write-head parameters remain present to preserve matched initialization but
are bypassed in overwrite mode. Full overwrite currently requires one slot.

On this checkout, start or resume from Terminal with:

```bash
bash scripts/run_overwrite_overnight.sh
```

The launcher uses `.venv`, keeps the Mac awake with `caffeinate`, and writes
to `runs/overwrite-2026-11-13/overnight.log`. Keep the Mac plugged in, its lid
open, and that Terminal session running. Ctrl-C saves the current checkpoint;
the same command resumes. Checkpoints also save every five minutes. A lock
prevents accidentally launching the same sweep twice.

The timing pilot measures 500 steps and full diagnostics for each arm. The
planner chooses equal steps across six runs for an estimated seven-hour
budget, capped at 128,200 steps per run. Allow extra time for sustained-load
slowdowns. `plan.json` records the exact step count and timings; rerunning
reuses it. Do not change training settings while resuming; use a fresh output
directory for a different experiment. This comparison matches its three arms,
not necessarily the training duration of the historical run C.

The prepared November 13 plan is **64,000 steps per run**, six runs, about
seven hours total on this MacBook. The timing pilots completed 500 measured
training steps per arm at full size, with full end-of-run diagnostics.

In a second Terminal:

```bash
tail -f runs/overwrite-2026-11-13/overnight.log
```

`report.md` updates after every completed run. In addition to the usual
accuracies, depth curves, and two-call comparisons (including chains whose
component facts are both known), `final.json` records slot interventions:
normal, reset to the learned empty vector, shuffled between questions, true
bridge written with the original write weight, and true bridge as a clean
embedding. They replace only the handoff after step 1. Oracle conditions use
answers at evaluation time and must not be reported as model accuracy.

Judge the comparison primarily on held-out second-hop and both-held-out
accuracy, with independent fact recall and the two-call diagnostic alongside
them. A falling training loss or a decodable bridge alone is insufficient.
Both seeds use the same graph; they measure initialization/training variation,
not generalization across independently generated graphs.

### Completed full-size results, November 13

All six runs completed 64,000 steps, using training code `670dd16`, MLX 0.32.4,
and Apple GPU. The measured training/evaluation sweep finished in approximately
6 hours 43 minutes. The machine-readable [results](results/overwrite-2026-11-13.json)
include resolved configurations, both seeds, depth curves, exact diagnostic
counts, and hashes of the original local final-result files.

Means across seeds 0 and 1 (same graph):

| Variant | Ordinary fact recall | Recall of facts excluded from chains | Unseen chains | Held-out second | Both held out |
|---|---|---|---|---|---|
| Gated soft (`slot_soft_prefix`) | 93.8% | 64.9% | 81.6% | 49.9% | 26.2% |
| Full soft (`slot_soft_overwrite`) | 89.8% | 40.6% | 69.3% | 24.9% | 8.3% |
| Full hard (`slot_hard_overwrite`) | 95.4% | 83.5% | 88.6% | 69.6% | 61.5% |

The hard version wins in both seeds: held-out second-hop accuracy is 72.6%
and 66.5%; both-held-out accuracy is 62.9% and 60.05%. Shuffling its scratchpad
reduces held-out second-hop accuracy to 0% in each seed. On that split, among
chains whose two facts are correctly recalled in separate one-step calls,
native accuracy is 86.5% and 81.4%. Those subsets differ between models and
should not be mistaken for a fixed-cohort comparison.

The gated model's final write weight is about 0.995 in both seeds: its gate did
not remain collapsed at full scale. Forcing full overwrite with soft content
performs worse. The benefit is therefore not explained by removing the gate
alone. Hard content improves individual fact recall as well as composition;
the aggregate improvement cannot all be attributed to a better handoff.

This supports discrete entity handoffs for supervised synthetic composition.
Every arm receives bridge supervision, and these are two seeds on one graph.
The experiment does not establish unsupervised emergence, natural-language
transfer, separation of abstractions from facts, or an advantage of the lookup
table over a dense model with the same scratchpad training.

## Files

| File | Contents |
|---|---|
| `data.py` | Graph generation and splits |
| `model.py` | Recurrent block, product-key lookup, dense baseline, scratchpad |
| `configs.py` | Presets and the run variants |
| `train.py` | One run: training, periodic evaluation, checkpoints |
| `evaluate.py` | Accuracy, table usage, retrieval overlap, slot contents |
| `sweep.py` | A set of runs, one after another: the first comparison or the scratchpad runs |
| `report.py` | Markdown summary of finished runs |
| `two_call.py` | Two-call diagnostic on one checkpoint, with exact counts |
| `diagnose.py` | Read-only checks on checkpoints: reproduction, two-call, per-pass decoding |
| `loading.py` | Load a run directory |
| `tests/` | Split leakage, causality, retrieval, resume and scratchpad checks |

Each run directory holds `config.json`, `metrics.jsonl` (one line per periodic
evaluation), `final.json` and `checkpoint.safetensors`. The checkpoint for
`E_large` is roughly 800 MB.
