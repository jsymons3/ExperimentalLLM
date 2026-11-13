# Multi-token scratchpad: arithmetic, family stories and real-name facts

The graph experiment in `recurrent_lookup/` found one thing that works: a
scratchpad that is fully overwritten with the single entity the model
predicts, when the model is taught what the intermediate answer should be.
That result used one-token answers on a made-up graph.

This package takes the next steps, one change at a time:

1. **A scratchpad of several tokens**, on tasks that need no stored facts:
   arithmetic, and questions about short family stories.
2. **Real names and real facts**, with the held-out design of the graph
   experiment.
3. **All three together**, where the intermediate step is taught only on
   arithmetic and stories, to see whether the habit carries over to fact
   chains where it is never taught, and whether reasoning can be kept in a
   small core while facts go to the table.

Nothing here has been run at full size. See "Status".

## Setup

Apple silicon Mac, from the repository root, in the existing environment:

```bash
.venv/bin/python -m pip install -r phrase_scratchpad/requirements.txt
```

The family stories are downloaded on first use (about 20 MB, see "Data"):

```bash
.venv/bin/python -m phrase_scratchpad.clutrr
```

## Run it

1. Check the code on your machine:

```bash
.venv/bin/python -m pytest --rootdir=. --confcutdir=. phrase_scratchpad/tests -q
```

2. See how long the runs would take. This times 200 steps of each and stops:

```bash
bash scripts/run_phrase_scratchpad.sh --estimate-only
```

3. Run them. The default budget is eight hours for five runs:

```bash
bash scripts/run_phrase_scratchpad.sh
```

   Keep the Mac plugged in with the lid open. The same command resumes after
   an interruption. `--hours 10` changes the budget, but a different budget or
   run list needs a fresh `--out` directory. Progress is in
   `runs/phrase-scratchpad/sweep.log`, and `report.md` is rewritten after every
   finished run.

4. Rebuild the report by hand:

```bash
.venv/bin/python -m phrase_scratchpad.report runs/phrase-scratchpad
```

A single run, for example a quick look at arithmetic:

```bash
.venv/bin/python -m phrase_scratchpad.train --variant maths --preset smoke --steps 3000 --out runs/phrase-smoke/maths
```

## The tasks

Every task has one-step questions and two-step questions. A two-step question
has an intermediate answer, the "bridge", which is what the scratchpad should
hold after the first step.

| Task | One-step question | Two-step question | Bridge | Needs stored facts |
|---|---|---|---|---|
| Arithmetic | `47 + 85` | `47 + 85 - 20` | `132` | No |
| Family stories | story, then `Ashley ? Lillian` | story, then `Ashley ? Nicholas` | `daughter Lillian` | No |
| Facts | `Jaws [director]` | `Jaws [director] [date of birth]` | `Steven Spielberg` | Yes |

- **Arithmetic.** Addition and subtraction of whole numbers up to 99, with no
  negative results. Numbers are written with three digit tokens, lowest digit
  first, so the position working out the tens can see the position working
  out the ones.
- **Family stories** (CLUTRR). Two sentences written by crowd workers, such as
  "Ashley's daughter, Lillian, asked her mom to read her a story. Nicholas's
  sister Lillian asked him for some help planting her garden." The answer to
  `Ashley ? Nicholas` is `son`. The two relations have to be read from the
  story and combined: a daughter's brother is a son.
- **Facts.** Real names from Wikipedia-derived data: 20,325 facts about 1,500
  people at full size. A name is one token per word, up to four words, and
  each relation is one token.

### Test splits

Arithmetic and facts use the graph experiment's design. Every one-step
question is trained. Some are never used inside a training chain. Test chains
are grouped by what they need:

| Group | What the chain needs |
|---|---|
| Unseen chains | Only facts that some training chain uses |
| Held-out 1st | A first step that no training chain uses |
| Held-out 2nd | A second step that no training chain uses |
| Held-out both | Both |

Held-out 2nd and both are the main result for facts. For arithmetic they are
weaker evidence, because a sum can be worked out and need not be recalled.

Family stories have nothing stored to hold out. They are trained on the
dataset's training stories and tested on its validation stories, which the
model has never seen.

The report gives a guessing floor for every split: always answering with the
commonest training answer for the question's last relation. It is about 14%
for fact chains (many directors are "American") and 10% for stories.

## The model

The block from `recurrent_lookup/` (causal attention, then the product-key
lookup). The sequence is

    [question tokens] [K scratchpad slots] [K work positions]

Work position `i` predicts token `i` of an answer. After each step the K
predicted tokens' own embeddings replace the scratchpad, and the next step
starts fresh from the question and the scratchpad. The last step's
predictions are the answer. K is 3 for arithmetic, 2 for stories and 4 for
facts or any mix.

### Making the model use the scratchpad

Small checks showed that writing the right intermediate answer is not
enough: the model writes it and then answers without reading it. Two
settings close the ways around, and both are on by default.

- **One block application per step** for arithmetic and facts (the graph
  experiment used two). With two, a step is deep enough to redo both
  arithmetic operations from the question. Stories keep two, because reading
  a relation out of a sentence takes more depth than one lookup.
- **Hiding the subject.** For half of the two-step training questions, the
  subject (the first number, or the film's name) is hidden from the second
  step on. Those questions can only be answered from what step 1 wrote.
  Nothing is hidden at test time.

`--n-passes 2` and `--hide-subject-prob 0` turn them off.

### Two additions used only in stage 3

- **A small dense core**: an always-on MLP (width 512) beside the lookup.
- **A table switch**: the lookup can be turned off per question, leaving
  attention and the small MLP.

## The runs

| Run | Tasks | Intermediate taught on | Table off on | What it tests |
|---|---|---|---|---|
| `maths` | Arithmetic | Arithmetic | - | Stage 1: does a multi-token scratchpad work |
| `clutrr` | Stories | Stories | - | Stage 1: can it read and combine relations in real sentences |
| `facts` | Facts | Facts | - | Stage 2: the graph result with real multi-word names |
| `mix_split` | All three | Arithmetic, stories | Arithmetic, stories | Stage 3: does the habit reach fact chains, with reasoning kept in the core |
| `facts_unsupervised` | Facts | Nothing | - | Control for stage 3: the same facts alone |

They run in that order with the same number of steps. Three more are left out
by default because each extra run shortens all the others:

- `--with-mix` adds `mix`: stage 3 with the table always on, to show whether
  the switch matters.
- `--with-baselines` adds `maths_plain`, `clutrr_plain` and `facts_plain`: the
  same tasks with no scratchpad and the same number of block applications.

`--only maths facts` runs a chosen subset.

### Why the table is switched off on arithmetic and stories

A fact question can only be answered from memory, so a model trained on facts
alone has no reason to develop anything but a table. Arithmetic and story
questions can be answered from the question itself. Training them with the
table off leaves attention and the small MLP as the only place their
procedure can live. Facts are trained with the table on. The report then
shows every task with the table on and off: if the split worked, arithmetic
and stories survive with the table off and facts collapse.

Facts are never trained with the table off. That would only teach the core to
memorise them.

## What the report shows

- **Accuracy** per run, task and split, next to the guessing floor.
- **Is the scratchpad used?** How often step 1 writes exactly the bridge, and
  accuracy when the scratchpad is cleared, shuffled between questions, or
  replaced with the true bridge.
- **Two-call ceiling**: the chain asked as two separate one-step questions,
  the second about whatever the model answered to the first.
- **What depends on the table**: every task with the table on, then off.
- **Accuracy by number of outer steps** at test time.

### How to read it

- **Stage 1 works** if `maths` and `clutrr` are well above their floors on
  two-step questions, the bridge is usually written, and shuffling the
  scratchpad destroys accuracy.
- **Stage 2 works** if `facts` is well above its floor on held-out 2nd chains
  and near its own two-call ceiling.
- **Stage 3 works** if fact chains in `mix_split` beat `facts_unsupervised`
  and the bridge is written for fact chains, where it was never taught.
- **The split works** if, in `mix_split`, arithmetic and stories keep their
  accuracy with the table off and facts lose theirs.

Each run has one seed. Small gaps are noise.

## Choices worth knowing about

- **Words, not subwords.** The fact vocabulary is closed, so each word is a
  token and a name is at most four tokens. A subword tokenizer only matters
  for open text.
- **The facts file is a controlled table, not a copy of Wikipedia.** Different
  entities that share a name are merged, and each subject and relation keeps
  one object. Some rows are therefore not true of any single real entity. The
  test stays valid because answers are defined by the table.
- **The bridge for a story** is the first stated link and the person in the
  middle ("daughter Lillian"). Other choices are possible.
- **Stage 3 grades one-step questions at every step.** The other runs grade
  only the last step, as in the graph experiment. Without an intermediate
  answer for fact chains, nothing else would train the first step on facts.
- **The control is also a test of its own.** `facts_unsupervised` hides the
  subject too, so it shows whether that pressure alone makes the model write
  the intermediate answer without ever being taught it.
- **Stories may not need the scratchpad.** A two-sentence story can be read
  and combined inside one step, and in a small check clearing the scratchpad
  made no difference. Their role is to give the core reasoning that needs no
  stored facts. The report shows whether the scratchpad is used.
- **Stories get a quarter-size batch.** They are several times longer than
  other questions and there are far fewer of them. Tasks count equally in the
  loss whatever their batch size.
- **Stages 1 and 2 use the lookup-only block** that worked on the graph task.
  Only the stage 3 runs have the dense core.
- **The baselines** with no scratchpad use four block applications, the most
  a scratchpad run uses on arithmetic or facts.

## Data

- **Facts**: `facts/people_facts.tsv`, 36,950 rows, committed. Built by
  `prepare_facts.py` from the evidence triples of
  [2WikiMultihopQA](https://huggingface.co/datasets/xanhho/2WikiMultihopQA)
  (Apache-2.0) at a pinned revision. `facts/people_facts.meta.json` records
  the source hashes and filters. Regenerating it needs `pyarrow` and a 375 MB
  download; running the experiment does not.
- **Family stories**: [CLUTRR](https://github.com/facebookresearch/clutrr),
  licensed CC BY-NC 4.0 (non-commercial). It is downloaded to `data/clutrr/`,
  which git ignores, and checked against recorded hashes. 7,717 training
  stories and 1,919 test stories with two relations are used.

## Status

- Written and tested on MLX's Linux CPU build, which is far too slow for a
  full-size run. 25 tests pass, every run trains, resumes, evaluates and
  reports end to end at tiny size, and the existing 58 tests for the other
  experiments still pass.
- Every run was built at full size and took a few training steps and an
  evaluation on a small batch.
- Not run on an Apple GPU or for any length at full size. Speed, memory and
  whether the defaults learn in one night are unknown. `--estimate-only` will
  show the speed.
- If the compiled training step is rejected on the Mac, add `--no-compile`.
- Learning rate, batch size, widths and the stage 3 loss are untuned starting
  points.

### Small checks run while building

These ran on CPU with a much smaller model (width 32 or 48, a 256-entry
table), cut-down tasks, a few thousand steps and one seed each. They show
that the code learns and what the settings do. They do not predict the
full-size results.

Facts, 60 people, intermediate answer taught, 6,000 steps. The guessing floor
on held-out 2nd chains is 31%.

| Setting | Unseen chains | Held-out 2nd | Held-out 2nd with the scratchpad shuffled |
|---|---|---|---|
| Two passes per step, nothing hidden | 92% | 21% | 20% |
| One pass per step, nothing hidden | 98% | 41% | 33% |
| One pass per step, subject hidden half the time (the default) | 99% | 73% | 20% |

Arithmetic, numbers up to 19, intermediate result taught, 6,000 steps.

| Setting | Unseen chains | Held-out 2nd | Unseen chains with the scratchpad shuffled |
|---|---|---|---|
| Two passes per step | 59% | 66% | 58% |
| One pass per step | 76% | 75% | 9% |

- **With two passes the scratchpad was written and then ignored.** Shuffling
  it changed nothing on arithmetic. With one pass it became necessary.
- **Facts found a second way round**: the table learned to send a film
  straight to what it had stored about the film's director, which answers
  chains it has practised with but not held-out second steps. Hiding the
  subject removed that route and held-out 2nd rose from 41% to 73%.
- **Family stories**, 1,500 of them, 4,000 steps: 61% on two-step questions
  about unseen stories with two passes per step, 45% with one, against a 10%
  floor. Clearing the scratchpad made no difference in either case.
- **Facts with no intermediate answer taught** (the stage 3 control): 94% on
  unseen chains and 32% on held-out 2nd, which is the floor. It answered most
  chains in a single step through the shortcut above and never wrote the
  bridge (1%).
- **The full default combination was only smoke-tested.** Arithmetic with
  one pass and hiding together, and the three-task run with the table off,
  both train and evaluate, but neither was run long enough here to show
  whether it learns.

## Files

| File | Contents |
|---|---|
| `tasks.py` | Arithmetic, facts, the shared split design, and several tasks in one vocabulary |
| `clutrr.py` | Family stories: download, parsing, questions |
| `prepare_facts.py` | Builds `facts/people_facts.tsv` |
| `model.py` | Multi-token scratchpad, dense core, table switch |
| `configs.py` | The runs |
| `train.py` | One run: training, periodic evaluation, checkpoints |
| `evaluate.py` | Accuracy, scratchpad interventions, two-call ceiling, table on and off |
| `sweep.py` | The runs one after another inside a time budget |
| `report.py` | Markdown summary of finished runs |
| `tests/` | Labels, splits, masking, scratchpad contents, table switch, resume |
