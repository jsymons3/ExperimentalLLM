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

None of this has been trained on the real model with MLX. See "Status".

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
no training signal. Adapters could otherwise memorise the facts themselves,
and the run could not tell where a fact was stored.

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
  evaluations at a constant 0.001, and the lower rate learned just as fast.

## Status

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
- **A first look at the written note**, with the same PyTorch copy: Ouro
  fully frozen, 32 facts about 16 people, 25 two-step training questions, 150
  steps at the old learning rate. One run, with 11 to 14 questions per group:

  | Questions | Note is right | Answer right | With shuffled notes |
  |---|---|---|---|
  | One-step facts | | 97% (0% with the table off) | |
  | Two-step, trained | 96% | 92% | 12% |
  | Two-step, unseen | 57% | 57% | 0% |
  | Held-out 1st | 82% | 82% | 18% |
  | Held-out 2nd | 23% | 23% | 8% |

  This is a sign that the note route can work on a frozen model, not a
  measurement. Training was unsettled: one-step recall read 59%, 59%, 84%,
  62% and 97% at successive evaluations. The low held-out 2nd figure is three
  right out of thirteen and may or may not hold up.

Not checked:

- **Training the real model with MLX.** No step of the real runs has been
  taken on an Apple GPU, so the speed, the memory use and the time per run
  are unknown until the sweep times them. Run steps 2 and 3 above first.
- **The full-size result.** Whether the note or the loops chain held-out
  facts at 2,160 facts is what the runs are for.
- **Long 16-bit training.** Three steps in 16-bit behaved; thousands have
  not been tried. If a loss turns non-finite, rerun with `--dtype float32`.

## Files

| File | What it is |
|---|---|
| `ouro.py` | Ouro in MLX, loading the published weights |
| `model.py` | The table and adapters attached to it |
| `data.py` | The made-up facts, the two-step questions and the split |
| `configs.py` | The runs and their settings |
| `train.py` | One training run, with checkpoints and resume |
| `evaluate.py` | Accuracy, the note comparisons and the other diagnostics |
| `sweep.py` | The quick check, the timing and the runs in order |
| `report.py` | `report.md` from finished runs |
| `download.py` | Fetches and checks the model |
| `tiny.py` | A tiny random stand-in for tests and dry runs |
| `tests/` | 35 tests |
