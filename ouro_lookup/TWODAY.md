# Ouro, the two-day run

Stage 1 showed that a frozen Ouro can use a learned table, and that small
adapters do the same job with 43 times fewer trained numbers. The job was
2,160 made-up facts, which either can hold with room to spare. This run asks
two questions where a table should differ from adapters, if it differs at
all:

1. **Take-over.** Remove part of Ouro and train something to do its work. Can
   a table take over what the removed part knew, at a small share of its
   computation? What was removed is far more than adapters or a small dense
   layer have room for.
2. **Flexibility.** Add facts batch by batch, and change facts, without
   showing the old ones again. Does a table lose less than adapters?

It is built and checked at small size. It has not been run on an Apple GPU.
See "Status".

## Run it

Apple silicon Mac, from the repository root. Have about 15 GB of disk free.

1. Check the code (several minutes; no download needed):

```bash
.venv/bin/python -m pytest --rootdir=. --confcutdir=. ouro_lookup/tests -q
```

2. While you are at the keyboard, run the quick check. It downloads the text
   on first use (160 MB), trains a short take-over run and a small retention
   run on the real model, and should end with "passed". It should take about
   ten minutes; that is an estimate.

```bash
bash scripts/run_ouro_twoday.sh --check-only
```

3. Time every kind of run and print the plan, without starting:

```bash
bash scripts/run_ouro_twoday.sh --estimate-only
```

4. Start the 21 runs:

```bash
bash scripts/run_ouro_twoday.sh
```

The default budget is 44 hours and the plan never exceeds it. The same
command resumes after an interruption. Results are in
`runs/ouro-twoday/report.md`, rewritten after every finished run, and the log
is `runs/ouro-twoday/sweep.log`. Keep the Mac plugged in with the lid open.

Useful variations (a different budget or run list needs a fresh `--out`):

```bash
bash scripts/run_ouro_twoday.sh --hours 30 --out runs/ouro-twoday-30h
bash scripts/run_ouro_twoday.sh --only takeover_table takeover_dense --out runs/takeover-two
bash scripts/run_ouro_twoday.sh --dtype float32      # if a loss turns non-finite
```

To keep the results, copy `report.md`, `plan.json` and each run's
`config.json`, `final.json` and its `metrics.jsonl` or `progress.json` into
`ouro_lookup/results/`, as for stage 1. Checkpoints stay out of the
repository.

## Part 1: taking over eight layers

### What is removed

The MLP of layers 8 to 15: 277 million numbers, 19% of the model, and about a
fifth of its computation. Measured on the original model before building
this (PyTorch on CPU, 16 everyday sentences and 269 sentence starts with a
well-known next word):

| MLPs removed | Loss on the sentences | Known facts right |
|---|---|---|
| None | 2.97 | 238 |
| Layer 12 only | 3.01 | not measured |
| Layers 10 to 13 | 3.22 | not measured |
| Layers 8 to 15 | 4.03 | 39 |

The model still writes grammatical text without them but loses most of what
it knew. It named 99 capital cities before and none after ("The capital of
Kenya is the"). All eight sequence patterns survive ("Monday, Tuesday,
Wednesday," is still followed by "Thursday"). One middle MLP alone barely
matters, so eight are removed to leave something to measure.

### What stands in

| Run | What stands in for each removed MLP | Trained numbers | Computation |
|---|---|---|---|
| `takeover_table` | A table of 16,384 entries | 541 million | 2% of the MLP |
| `takeover_dense` | A small MLP of width 112 | 5.5 million | 2%, the same as the table |
| `takeover_adapters` | Nothing; adapters on every attention layer | 1.6 million | None once folded in |
| `takeover_table_small` | A table of 4,096 entries | 138 million | 2% |
| `takeover_dense_wide` | A small MLP of width 448 | 22 million | 8% |

Computation counts multiplications per token. A table reads 32 entries
whatever its size, so its cost does not grow with what it holds. A small MLP
of width 112 costs exactly what the default table costs.

Ouro's remaining weights stay frozen. The stand-in is trained to imitate the
original model on ordinary text, so no labels are needed. At every position
three things are matched: the next-token distribution after the last loop,
the state after each loop, and what each removed MLP added to the state.
Every run sees the same windows of text in the same order and takes the same
number of steps.

The text is the first 120 million characters of WikiText-103, 25.7 million
tokens of Wikipedia articles, with its spaced-out punctuation undone.
It is downloaded from Hugging Face and checked on first use, and is not
committed. Reading it needs `pyarrow`, which the launcher installs.

### What is measured

- **Held-out loss**: next-token loss on articles never trained on, for the
  original model, with nothing put back, and with the trained stand-in. The
  report gives the share of the lost ground recovered.
- **Known facts**: the sentence starts the original model completes
  correctly, by kind. Nothing is trained on them, though Wikipedia states
  many of the same facts in other words.
- Distance from the original's predictions, loss on the sixteen sentences,
  the share of table entries used, speed and memory.

### How to read it

- **The table takes over** if `takeover_table` recovers clearly more of the
  lost ground and of the known facts than `takeover_dense` and
  `takeover_adapters`. That would be the first advantage of a table over the
  cheap alternatives in this repository.
- **Size is what matters** if `takeover_table` beats `takeover_table_small`
  and `takeover_dense_wide` beats `takeover_dense`. Compare the table with
  the wide MLP to see what four times the computation buys instead.
- **Adapters are enough again** if `takeover_adapters` restores the facts.
  That would be a second result against the table.
- **Nothing takes over** if every run stays near the "nothing put back" row
  on known facts. The text may then be too little: by my estimate each run
  sees roughly ten million tokens, and the published work on such tables
  used a trillion.

## Part 2: flexibility

Made-up facts on the frozen model, as in stage 1: 360 people, six relations,
2,160 one-step facts. No two-step questions.

**Adding facts batch by batch** (`retain_*`). The people are split into six
batches of 60. Each batch's 360 facts are trained alone, with fresh
optimizers, until 98% are recalled or 2,000 steps have passed. Every batch is
then tested. Earlier batches are never shown again.

**Changing facts** (`edit_*`). All the facts are trained together until 98%
are recalled. Then 100 of them get a new answer and only those are trained,
for 200 steps, with a test every 20. The report gives how many untouched
facts survive at the first test where 95% of the new answers are given, and
at the end.

| Variant | What holds the facts |
|---|---|
| `table` | The table alone, 67.7 million numbers |
| `adapters` | Adapters alone, rank 8, 1.6 million numbers |
| `adapters_matched` | Adapters alone, rank 344: as many trained numbers as the table |
| `table_fixed_keys` | The table, with the part that forms and matches queries frozen after the first phase, so later learning can only write entries |

Each runs with seeds 0 and 1. The seed changes the facts, the starting
weights and the order of questions.

### How to read it

- **The table is more flexible** if its earlier batches survive later ones,
  or its untouched facts survive a change, clearly better than both adapter
  runs.
- **Shared keys are the weak point** if `table_fixed_keys` keeps what `table`
  loses.
- **No advantage** if all four forget alike. With the stage 1 result that
  would leave no measured reason to prefer the table at this size.

## Order and time

The runs go in this order, so that an interruption keeps the main comparison
of each part:

1. `takeover_table`, `takeover_dense`, `takeover_adapters`
2. Retention and editing for `table` and `adapters`, seed 0
3. The same for `adapters_matched` and `table_fixed_keys`, seed 0
4. `takeover_table_small`, `takeover_dense_wide`
5. Everything in 2 and 3 again with seed 1

Each take-over run gets up to 4.5 hours (`--takeover-hours`) and all five get
the same step count. The flexibility runs stop when their facts are learned,
so the plan counts their longest possible time and the real total should be
shorter.

## Choices worth knowing about

- **Table sizes.** The larger table is 0.54 billion numbers. Training keeps
  three more arrays of the table's size, so a 2 billion number table does not
  fit in 48 GB. `--only takeover_table_large` runs 1.2 billion (36,864
  entries per layer) and needs about 25 GB for the table alone.
- **Adapters have no local target.** A table or small MLP is also taught what
  the removed MLP added. Adapters put nothing in that place, so they learn
  from the other two targets only.
- **One seed for the take-over runs.** A second would double their 22 hours.
- **Rank-344 adapters** use a scale lowered with the square root of the rank
  (0.305 against 2.0), so that a step changes the model about as much as at
  rank 8. This is untuned.
- **Batches are whole people.** All six facts about a person arrive together,
  and later batches bring people the model has not been asked about.
- **Not in this run**: deleting a single fact from the table, and teaching
  the table to stay silent on ordinary text. In Part 1 the table is trained
  on ordinary text, so that question does not arise there.
- **Attention.** Sequences up to 256 tokens use the explicit calculation that
  fixed stage 1 on the Apple GPU. Whether the fused kernel can be trained
  through at these lengths was never tested there.
- **A known fact** counts when the model's most likely next token is the
  first token of the expected word.

## Status

Checked while building, on a Linux machine with no GPU:

- **The code runs end to end** on the tiny random stand-in for Ouro: the
  quick check, the timing, all 21 runs, resume and the report. 20 tests cover
  the new parts, and the 38 earlier Ouro tests still pass.
- **The removal does what the table above says** on the real weights.
- **A table can learn to stand in, on the real weights.** With a PyTorch copy
  of the recipe, a table of 4,096 entries per layer was trained for 100 steps
  on 12,800 tokens. On 16 held-out windows the loss was 3.22 for the
  original, 4.71 with nothing put back, 4.14 after 50 steps and 4.07 after
  100. That is a sign the recipe learns, not a forecast: it says nothing yet
  about facts.
- **The training code runs on the real weights at full size**, in MLX, in
  the stored 16-bit format. One take-over step with the full
  541-million-number table on a 12-token window gave finite gradients, and
  one update lowered the next-token mismatch on that window from 1.29 to
  0.51. Two steps of the rank-344 adapters on two questions were finite, with
  the loss going from 11.1 to 7.7.

Not checked:

- **Any of it on an Apple GPU.** Speed, memory and the time per run are
  unknown until the plan is printed. Run steps 2 and 3 above first.
- **Whether facts come back.** The small check measured loss only.
- **Sequences of 96 tokens in training.** Stage 1 trained on nine at most.
  The quick check trains on 48.
- **The step limits of the flexibility runs.** In stage 1 the table needed
  about 60 presentations of each fact. The limits allow about 350. If a
  batch is not learned in time, the report says so in "every batch learned".

## Files

| File | What it is |
|---|---|
| `takeover.py` | One take-over run: training, evaluation, checkpoints |
| `flex.py` | One retention or editing run |
| `text.py` | Fetches, checks and tokenises the ordinary text |
| `probes.py` | The well-known facts and how they are scored |
| `twoday.py` | The quick check, the timing and the runs in order |
| `twoday_report.py` | `report.md` from finished runs |
| `tests/test_twoday.py` | 20 tests |
