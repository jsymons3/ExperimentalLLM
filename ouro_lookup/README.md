# Ouro with a lookup table: stage 1

The earlier experiments trained tiny models from scratch. This one attaches
the same lookup table to a pretrained looped language model,
[Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B), keeps the model's
own weights frozen, and asks three questions:

1. **Can a frozen pretrained model use a learned table at all?** It is taught
   made-up facts. It should answer them with the table on and not with it
   off.
2. **Can it chain two facts it has never seen chained?** Two routes are
   compared: a written note that is handed to a fresh computation (the
   discrete handoff that worked in `recurrent_lookup/`), and one pass through
   Ouro's own loops, which hand on a continuous state.
3. **What do small adapters add, and could they do the job without a
   table?**

All four full-size runs completed on an Apple GPU on November 15, 2026.
The table works with a frozen core and a written handoff, but adapters alone
match its accuracy with about 43 times fewer trainable parameters. The latent
route fails on new chains despite recalling their facts. See the
[results and interpretation](results/mac-2026-11-14/README.md) and
[full diagnostic report](results/mac-2026-11-14/report.md).

The follow-up [two-day run](TWODAY.md) completed 12 of 21 runs before a Metal
compiler service failure on November 16. A small dense replacement repairs
the removed feed-forward modules better than the large table. In the separate
sequential-learning test, the table retains more earlier facts than both
adapter controls, including the parameter-matched one. Editing shows no clear
advantage over the larger adapters. These are preliminary results from one
seed; the second seed and wider dense control remain unfinished. See the
[follow-up results and limitations](results/twoday-2026-11-15/README.md) and
[saved report](results/twoday-2026-11-15/report.md).

## Setup

Apple silicon Mac, from the repository root, in the existing environment:

```bash
.venv/bin/python -m pip install -r ouro_lookup/requirements.txt
```

The model is downloaded on first use (2.9 GB, into `data/ouro-1.4b`,
verified against known checksums):

```bash
.venv/bin/python -m ouro_lookup.download
```

That command also asks the model to complete "The capital of France is" and
fails if the answer is not "Paris", which would mean the weights or this port
are wrong.

## Run it

1. Check the code on your machine (under two minutes, no download needed):

```bash
.venv/bin/python -m pytest --rootdir=. --confcutdir=. ouro_lookup/tests -q
```

2. While you are at the keyboard, run the quick check. It teaches the frozen
   model 64 facts through the table in 300 steps and should end with
   "passed". Expect a few minutes.

```bash
bash scripts/run_ouro_lookup.sh --check-only
```

3. See how long the runs will take, without starting them:

```bash
bash scripts/run_ouro_lookup.sh --estimate-only
```

4. Start the four runs (eight hours by default):

```bash
bash scripts/run_ouro_lookup.sh
```

The same command resumes after an interruption. Results are written to
`runs/ouro-lookup/report.md` after each finished run, and the log is
`runs/ouro-lookup/sweep.log`. Expect about 7 GB of disk use: the model, and
under 1 GB of checkpoint per run.

Useful variations:

```bash
bash scripts/run_ouro_lookup.sh --hours 10
bash scripts/run_ouro_lookup.sh --only table table_adapters --out runs/ouro-two
bash scripts/run_ouro_lookup.sh --dtype float32      # if a loss turns non-finite
```

A different budget or run list needs a fresh `--out` directory.

## What is attached to Ouro

Ouro-1.4B is an ordinary transformer of 24 layers whose whole stack is
applied four times with the same weights. Its weights are loaded as
published and never change.

- **The table.** The product-key lookup from `recurrent_lookup/`, with 16,384
  entries. It sits beside the MLP of layer 12: it reads the state that MLP
  reads and adds its result to the state alongside the MLP's. Because the
  stack is applied four times, the table is consulted four times per token,
  each time with a query formed from the current state. Its stored vectors
  start at zero, so before training the model is exactly Ouro.
- **Adapters** (in some runs). Low-rank corrections, rank 8, to the query and
  value projections of every attention layer. They also start as no change.
  About 1.6 million numbers, against 68 million in the table and 1.4 billion
  frozen.

Either can be switched off when the model is called. That is how the report
tells what each contributes.

## The facts

People are known by one first name, and each has a mentor, a manager, a
doctor, a lawyer, a landlord and a coach, drawn at random from the other
people:

    The mentor of Daniel is Alice.

The full size is 360 people and 2,160 facts. The names are real first names
but the facts are random, so Ouro cannot know them: with nothing attached it
should score zero, and the report shows that number. Every name and relation
is a single token, so an answer is one token.

A two-step question chains two facts:

    The doctor of the mentor of Daniel is ...

Every fact is taught as a one-step question. A tenth of the facts never
appear in any two-step training question. Two-step test questions are grouped
by which of their facts were kept out: neither ("unseen"), the first, the
second, or both. This is the split used in the rest of the repository.

## Two ways to answer a two-step question

**Written note.** The two-step question is taught to produce the
intermediate answer, here the mentor's name. That one token is written into
a fresh one-step question, "The doctor of Alice is", and the model runs again
from scratch. It is the same model and the same table on both steps, and
nothing but the note is carried over. This costs nothing extra in training,
because the second step is an ordinary one-step question.

The procedure is supplied, not discovered: the code builds the second
question. Success shows that a fact stored in the table can be reused through
the note, including a fact never seen in a two-step question. It does not
show that the model worked out the two steps for itself.

**One pass.** The question is answered directly, after Ouro's fourth loop.
The intermediate answer is taught too: after loop 2 the state must read as
the mentor's name. Ouro's state can be read out as a next token after every
loop, which is how the original model was trained, so this needs no new
parts. What is handed from loop to loop is a continuous state, the form that
did worse than discrete tokens in our earlier runs.

## The runs

| Run | Additions | Two-step questions | Purpose |
|---|---|---|---|
| `table` | Table only | Written note | Can a fully frozen model use the table, and chain through a note? |
| `table_adapters` | Table and adapters | Written note | Do adapters help it write the right note? |
| `latent` | Table and adapters | One pass | Does Ouro's own looping chain the facts? |
| `adapters` | Adapters only | Written note | Control: what adapters manage with no table |

They run in that order with the same number of steps, so the most important
comparison finishes first.

In `table_adapters` and `latent` the adapters are trained on two-step
questions only. A one-step question still passes through them, but sends them
no training signal. This removes direct one-step supervision of the adapters;
chain labels can still teach them factual information. Use the table-off
interventions to measure dependence rather than assuming every fact is stored
only in the table.

Extra runs, left out because each one shortens the others: `table_latent`
(one pass with no adapters), `latent_untaught` (one pass, graded on the final
answer only) and `table_adapters_all` (adapters that also learn from one-step
questions). Add them with `--only`.

## What the report shows

- **Accuracy** by group of questions, with a guessing rate to compare
  against.
- **What each addition contributes**: the same questions with the table off,
  with the adapters off, and on the original model.
- **What limits two-step accuracy**: how often both facts are known as
  one-step questions, accuracy among those, and what asking two separate
  questions would reach.
- **The written note**: how often the note is right, and accuracy with the
  model's own notes, with notes shuffled between questions, and with the
  correct note supplied.
- **What the state reads as after each loop**: for the one-pass run, whether
  the intermediate answer shows up at loop 2 and the final one at loop 4.
- **Side effects and cost**: loss on sixteen ordinary sentences before and
  after, the share of the table in use, speed and memory.

### How to read it

- `table`, one-step, high with the table and zero without: a frozen
  pretrained model can use a learned table. This is the first thing to look
  at.
- A written-note run with "held-out 1st" well above the shuffled-notes
  figure: a fact taught only as a one-step question was found again inside a
  two-step question, and reused. That is the result the scratchpad gave on
  the small graph.
- "Held-out 2nd" in a written-note run asks two things. Its second step is
  the held-out fact's own one-step question, so it needs that fact recalled.
  Its first step needs the note not to depend on the outer relation: the
  right note has never been written next to that outer relation in training.
- `latent` near zero on held-out groups while its "two separate questions"
  figure is high: the facts are there, and the continuous loops do not chain
  them. That would repeat the earlier finding at this scale.
- `adapters` close to the table runs on one-step questions: adapters alone
  can hold this many facts, and the table is not needed at this size. The
  comparison then moves to larger fact counts.

## Choices worth knowing about

- **One-token names.** A note is one token. Names of several tokens would
  need the note written a token at a time and are not in this version.
- **A start token.** Every sequence begins with Ouro's end-of-text token. The
  original model predicts ordinary text better with it (loss 2.33 against
  2.57 on six sentences).
- **Question wording.** "The mentor of Daniel is" is short, and a name is not
  the continuation Ouro expects: it prefers "a" or "the". The table has to
  overcome that. Longer wordings that invite a name were not clearly better
  and cost twice the tokens.
- **Switching the adapters off is not a clean test.** The table's contents
  are learned with the adapters in place. Without them the states shift and
  the table's queries can miss. Read "adapters off" as "the table does not
  work alone after being trained with adapters", not as "the adapters hold
  the facts".
- **Precision.** Ouro runs in the 16-bit format its weights are stored in.
  The table and adapters are kept and updated in 32-bit. `--dtype float32`
  runs everything in 32-bit, at roughly twice the memory.
- **Layer 12** is the middle of the stack. `--table-layers` moves the table
  or adds more.
- **Step counts** are set by the time budget, never above 20,000.
- **Learning rate.** The table's rate peaks at 0.0005, half of what the small
  models used, and both rates fall in a straight line to a tenth of their
  peak by the last step. In the checks below accuracy swung between
  evaluations at a constant 0.001, and was steady with the lower, falling
  rate, which learned just as fast.

## Status

### Completed Mac sweep, November 15, 2026

All four variants completed 8,400 steps, seed 0, on 2,160 invented facts.
Ouro's original weights stayed frozen in every run. The sequential sweep ran
from 01:21 to 08:45 Central, after the eight reading tests finished.

| Run | One-step recall | Unseen chains | Held-out 2nd | Held-out both | Trainable parameters |
|---|---:|---:|---:|---:|---:|
| `table` | 100.0% | 100.0% | 99.8% | 97.8% | 67,666,176 |
| `table_adapters` | 100.0% | 100.0% | 100.0% | 100.0% | 69,239,040 |
| `latent` | 100.0% | 3.6% | 0.2% | 0.7% | 69,239,040 |
| `adapters` | 100.0% | 100.0% | 100.0% | 100.0% | 1,572,864 |

Every constituent fact is taught individually; held-out facts are excluded
from chain training. Final evaluations use up to 500 questions per group.
Shuffling the notes reduces held-out second-hop accuracy to 1.0% in all
three written-note runs. Disabling the table in the table-only model reduces
one-step recall to 0%; disabling adapters in the adapter-only model does too.

The memory is usable, but has no demonstrated accuracy or capacity advantage
here. Adapters use less peak memory (7.0 versus 9.2 GB) and cause a smaller rise
in the small ordinary-text loss check (2.970 to 3.311, versus 4.037 for the
table). The table is slightly faster per training step (693 versus 726 ms),
but these are not inference latency or matched-capacity measurements.

The latent model writes a readable correct bridge at loop 2 on 100% of unseen
chains, yet answers only 3.6% after loop 4; two separate questions reach 100%.
This diagnoses a failure to reuse known facts under this recipe. Written
handoffs use two full calls and external question construction, so this does
not isolate discreteness from context reset, extra compute or the supplied
procedure, or show that continuous reasoning is inherently incapable.

The [archive](results/mac-2026-11-14/README.md) contains the original report,
plan, configurations, learning curves, final diagnostics and
[provenance manifest](results/mac-2026-11-14/manifest.json). Checkpoints are
hashed in the manifest but remain local. One seed and a modest fact set do
not establish scaling, continual-update, gating or knowledge-shedding benefits.

### Mac preflight, November 14, 2026

- The first real-model training attempt on MLX 0.32.4 hit a Metal attention
  error: its short-sequence vector kernel did not provide the logsumexp
  required for backpropagation. For short sequences with wide heads, the port
  now uses the equivalent explicit causal attention calculation. Longer
  sequences and the tiny model's narrow heads retain the previous path.
- All 38 Ouro tests pass, including compiled gradient checks at the real
  model's head width. Four comparison prompts retain the original next-token
  prediction. The 16-bit logits are not bit-identical: mean absolute differences
  on those prompts are 0.021 to 0.049, and maximum differences 0.219 to 0.719.
- The real frozen model, in its stored bfloat16 format, learned all 64 facts
  in the 300-step check: 100% recall with the table and 0% without, including
  the facts excluded from chain training. Peak memory was 8.33 GB and training
  averaged 452 ms per step. This validates the small check, not the 2,160-fact
  sweep or long-run stability.

### Earlier checks before the Mac preflight

Checked while building, on a Linux machine with no GPU:

- **The port matches the original.** On the real weights, in 32-bit, this
  MLX model and ByteDance's PyTorch code agree on four test sentences: the
  largest difference in any output score is 0.00007, the state after every
  loop agrees to the same precision, and the top token agrees everywhere.
- **The code runs end to end** on a tiny random stand-in for Ouro
  (`python -m ouro_lookup.tiny`): the quick check, the timing, training with
  resume, evaluation and the report. 35 tests cover it.
- **The training code runs on the real weights.** Three training steps of
  the `table_adapters` run on the real model, in its stored 16-bit format, on
  a batch of four questions: every gradient was finite and the loss went
  from 10.1 to 7.7 to 2.6.
- **The real frozen model learns through the table.** MLX is too slow to
  train the real model further on a CPU, so this was checked with a PyTorch
  copy of the table attached to the original model. 16 made-up facts were
  recalled 100% with the table and 0% without after about 16 steps, at the
  old learning rate and at the new one.
- **The written note works on the real frozen model, at toy size.** With the
  same PyTorch copy: Ouro fully frozen, no adapters, 32 facts about 16 people
  with two relations, 25 two-step training questions, 300 steps at the
  current learning rate. One run:

  | Questions | Count | Note is right | Answer right | With shuffled notes |
  |---|---|---|---|---|
  | One-step facts | 32 | | 100% (0% with the table off) | |
  | Two-step, trained | 25 | 100% | 100% | 16% |
  | Two-step, unseen | 14 | 93% | 93% | 0% |
  | Held-out 1st | 11 | 100% | 100% | 27% |
  | Held-out 2nd | 13 | 100% | 100% | 46% |

  One-step recall was 100% at every evaluation from step 120 on, and the
  held-out groups stayed between 82% and 100%. How to weigh it:

  - It is small: one run, 11 to 14 questions per group, two relations.
  - The note for a two-step question is the answer to its last words ("the
    mentor of Daniel is"), so writing it mostly asks the frozen model to
    ignore the start of the question. That is the easy half of chaining, and
    it is why the note route can work with no adapters at all.
  - Shuffled notes score high on held-out 2nd because several of its
    questions share one held-out fact, and so the same note and answer. A
    swap often hands back an identical note.
  - An earlier run at the old learning rate (a constant 0.001, 150 steps) was
    unsettled: one-step recall read 59%, 59%, 84%, 62% and 97% at successive
    evaluations, and it ended with held-out 2nd at 23%. That low figure did
    not hold.

The earlier checks left Apple GPU training, the 2,160-fact result and long
16-bit training untested. The completed Mac sweep above now supplies all
three: four full-size runs finished in stored bfloat16 with final diagnostics.

## Files

| File | What it is |
|---|---|
| `ouro.py` | Ouro in MLX, loading the published weights |
| `model.py` | The table and adapters attached to it; removed layers and what stands in for them |
| `data.py` | The made-up facts, the two-step questions and the split |
| `configs.py` | The runs and their settings |
| `train.py` | One training run, with checkpoints and resume |
| `evaluate.py` | Accuracy, the note comparisons and the other diagnostics |
| `sweep.py` | The quick check, the timing and the runs in order |
| `report.py` | `report.md` from finished runs |
| `download.py` | Fetches and checks the model |
| `tiny.py` | A tiny random stand-in for tests and dry runs |
| `takeover.py`, `flex.py`, `text.py`, `probes.py`, `twoday.py`, `twoday_report.py` | The two-day run: see [TWODAY.md](TWODAY.md) |
| `tests/` | 61 tests |
