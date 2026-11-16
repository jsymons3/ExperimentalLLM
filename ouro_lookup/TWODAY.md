# Ouro, the two-day run

Stage 1 showed that a frozen Ouro can use a learned table, and that small
adapters do the same job with 43 times fewer trained numbers. The job was
2,160 made-up facts, which either can hold with room to spare. This run asks
two questions where a table should differ from adapters, if it differs at
all:

1. **Take-over.** Remove part of Ouro and train something to do its work. Can
   a table take over what the removed part knew, at a small share of its
   computation? Whether adapters or a small dense layer can recover the
   same behaviour is an empirical question, not something their parameter
   counts settle in advance.
2. **Flexibility.** Add facts batch by batch, and change facts, without
   showing the old ones again. Does a table lose less than adapters?

The Mac sweep started November 15, 2026 and stopped November 16 after 12 of
21 runs completed. The Metal compiler service became unavailable; the saved
results are intact and training remains stopped. See the
[partial results and interpretation](results/twoday-2026-11-15/README.md).

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

The default planning budget is 44 hours; timings are estimates, not a
guarantee of wall-clock duration. The same
command resumes after an interruption. Results are in
`runs/ouro-twoday/report.md`, rewritten after every finished run, and the log
is `runs/ouro-twoday/sweep.log`. Keep the Mac plugged in with the lid open.
Take-over runs save weights and optimizer state every five minutes by
default (`--checkpoint-minutes N` changes this); flexibility runs save at
phase boundaries. Finished runs are skipped on restart. An abrupt crash
can lose updates since the last checkpoint, including the start of a run
if it had not yet saved one.

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

When extending an interrupted take-over run, preserve its original
learning-rate history with `--schedule-initial-steps OLD_TOTAL` and
`--schedule-transition-step COMPLETED_UPDATES`, passed through
`--takeover-extra`. Apply these settings to every take-over control, even
those not started yet. The original schedule is used up to the transition;
its remaining decay is then stretched smoothly to the new total. Keep a
copy of the original plan before revising the saved step count and plan
signature. Use an interrupted checkpoint with optimizer state for this;
finished checkpoints retain only the trained additions.

The optimizer counter resolves an interruption arriving just after an
update but before the logged step counter advances. The resumed run skips
exactly that many training windows. Sparse routing can amplify small
floating-point differences after recompilation, so this preserves the
training protocol without promising bitwise identical weights.

To stop earlier while preserving the already-used learning-rate schedule,
set the smaller `--steps` budget and pass `--schedule-steps OLD_TOTAL` to
every take-over run as well. This truncates the existing schedule rather
than restarting or speeding up its decay.

### Short mechanism diagnostics

`python -m ouro_lookup.mechanism` compares answer-token ranks in intact,
damaged and repaired models, then trains two fresh tables for 1,500 updates.
It expects a saved reference `config.json` and `checkpoint.safetensors` in
`runs/ouro-mechanism/reference`; `--reference` and `--out` override these.
Do not run this concurrently with training: it uses the same GPU.

The primary comparison is the 99 country–capital probes. One arm removes
all training paragraphs containing a canonical country name or an alias
listed in `mechanism.py`; the other removes the same number of paragraphs
randomly within each character-length bin. The validation text stays the
same, both arms see the same number of token positions, and both start from
the original frozen model with fresh tables and the same learning-rate
history. The reference checkpoint is a longer-run diagnostic, not the
matched short-run control. Other probe categories are secondary outcomes.

`--prepare-only` builds and audits the datasets without training. The saved
manifest records the aliases, paragraph counts and text hashes; each dataset
also saves its original paragraph indices. `--rank-run DIR --rank-output
FILE` independently measures a saved checkpoint. Results include per-probe
rank intervals (for ties), log probabilities and margins against the best
alternative. This avoids interpreting every top-one change as a whole fact
being gained or lost.

Recovery without explicit subject mentions weakens the direct-reteaching
explanation. It does not exclude unnamed mentions, aliases absent from the
list, facts conveyed by the teacher on other text, or reconstruction of
distributed computations. A high answer rank before repair supports residual
accessibility; a low rank does not establish that the information was erased.
Topic distribution shifts remain a confound even with random-removal and
length matching. These are one-seed diagnostics, not definitive localization
of factual storage. The first-token scoring limitation still applies.

## Part 1: replacing eight feed-forward modules

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

The original small check retained grammatical-looking text but lost many
of these next-token completions. It named 99 capital cities before and none
after ("The capital of Kenya is the"). All eight originally correct sequence
patterns survived. This does not establish that the removed modules held
facts exclusively, that factual information was erased, or that reasoning
survived: damaging a computation can make existing information inaccessible.
One middle MLP alone barely changed the measured loss, so eight are removed
to leave a measurable recovery task. Attention in all 24 layers remains.

### What stands in

| Run | What stands in for each removed MLP | Trained numbers | Computation |
|---|---|---|---|
| `takeover_table` | A table of 16,384 entries | 541 million | 2% of the MLP |
| `takeover_dense` | A small MLP of width 112 | 5.5 million | 2%, the same as the table |
| `takeover_adapters` | Nothing; adapters on every attention layer | 1.6 million | None once folded in |
| `takeover_table_small` | A table of 4,096 entries | 138 million | 2% |
| `takeover_dense_wide` | A small MLP of width 448 | 22 million | 8% |

Computation counts multiplications per token, not measured latency. A table
reads 32 entries at these sizes; scoring its subkeys still grows with table
size. A small MLP of width 112 matches the default table's counted
multiplications. Sparse memory traffic and routing overhead may make the
table slower in practice. The 541-million-parameter replacement is larger
than the 277 million removed parameters, so this is not parameter compression.

This is a frozen-model distillation/recovery experiment inspired by
[Memory Layers at Scale](https://arxiv.org/html/2412.09764v2). That work
pretrained memory-augmented models on up to a trillion tokens; it did not
establish this particular frozen-core repair recipe. Success here would
support replacing dense computation under the tested conditions, not prove
that facts have been cleanly separated from general reasoning.

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
- **Attention.** Short questions retain the explicit calculation that fixed
  stage 1. Compiled bfloat16 gradients now pass on the Mac at lengths 48 and
  96 with Ouro's real head width, so text windows use fused attention.
- **A known fact** counts when the model's most likely next token is the
  first token of the expected word.

## Status

### Partial Mac results, November 16, 2026

Four takeover variants completed 14,800 updates each on the same 5,683,200
input token positions. The small dense replacement has the lowest held-out
loss (3.231) and the most originally known probes correct (193/238), ahead of
the large table (3.277 and 174/238). This supplies no repair-efficiency benefit
for the table under this recipe.

All eight seed-0 retention and editing runs completed. Earlier-batch recall
at the end is 30.9% for the table, 14.2% for parameter-matched adapters and
7.7% for small adapters. The fixed-key variant reaches 53.5%, but its first
batch required 500 rather than 300 updates before freezing; the comparison
does not isolate the effect of freezing. Editing does not establish a table
advantage over parameter-matched adapters.

The wider dense takeover and all eight seed-1 flexibility runs are unfinished.
The [archive](results/twoday-2026-11-15/README.md) includes the unmodified
report, configurations, metrics, final diagnostics and provenance. Checkpoints
remain local. No run was restarted for publication.

### Mac review and preflight, November 15, 2026

- All 150 tests across the four packages pass, including 61 Ouro tests.
  An existing tiny recall test needed 300 rather than 150 steps to converge
  reliably on the Mac; its accuracy and table-off requirements are unchanged.
- The real 541-million-parameter table trained for 40 steps on 7,680 text
  tokens with fused attention at length 48. On 1,504 held-out target tokens,
  loss was 2.994 for the original model, 4.828 after removing the MLPs and
  3.943 with the trained replacement: 48.3% of that gap recovered. First-token
  probes rose from 38 to 78 of 238 originally correct. This is a preflight,
  not a comparison against the dense or adapter controls.
- The small retention preflight learned both 72-fact batches to 100%, but
  retained only 59.7% of the first batch after learning the second. No
  comparative retention benefit is established by this check.
- A failed preflight now stops the sweep. Resuming an editing run after its
  last checkpoint reuses its saved diagnostics instead of editing twice.
  Takeover resumption advances the original data generator to the saved
  step, preserving the matched window sequence.
- Compiled attention gradients pass at lengths 48 and 96 on Ouro's real
  bfloat16 head width. Text-window training uses fused attention; short fact
  questions retain the stage-1 workaround.
- Text timings use a batch of four 96-token windows to leave memory headroom
  on the Mac. The planned full comparison will use the same batch and number
  of steps for each takeover variant. Full results are not yet established.

### Checks before the Mac review

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

Not checked at that earlier stage:

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
| `tests/test_twoday.py` | 21 tests |
