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

All five full-size runs completed on November 14, 2026. The supervised fact
scratchpad works with multi-word names, but the mixed run does not show transfer
of chaining from arithmetic and stories. See [the completed results](#completed-overnight-results-november-14).

The next run is [the reading test](#the-reading-test-next-run). It is set up
and checked at small size, and not yet run at full size.

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

## The reading test (next run)

### Why

The overnight runs showed where fact chains fail. With the intermediate
answer taught, the note is written correctly on 99.1% of chains whose second
fact was kept out of training chains, yet only 78.3% are answered, and
supplying the correct note does not help (78.6%). Asked as two separate
questions, the same model reaches 93.3%. The second fact is stored. It is not
reliably found when its subject arrives in the scratchpad instead of in the
question, which is the only way that fact was ever asked.

### What changes

Half of the one-step questions about facts that can be a second step are
asked through the note instead:

    ordinary     Alain Jessua [country of citizenship]             answer French
    through      Verdi [director] [country of citizenship]         answer French
    the note     with the note set to "Alain Jessua" after step 1

- The question starts like a two-step question. Its start ("Verdi
  [director]") is the first step of some training chain and has nothing to do
  with the fact.
- After step 1 the scratchpad is set to the fact's subject, whatever the
  model wrote. The answer is the fact's own answer. Step 2 then sees what it
  sees in a real chain: a name in the scratchpad and the relation to apply.
- As with chains, the subject at the start is hidden after step 1 half the
  time. The other half teaches that the note wins over a question still in
  view.
- These questions replace ordinary one-step questions, so every fact is
  presented as often as before. No chain is added, and no chain's answer or
  intermediate answer is shown. A start that happens to lead to the fact's
  subject would make a real chain, so those are never used.

A held-out second fact is therefore still absent from every training chain,
but it is no longer unread through the note. A good score in a `_reader` run
means that one-step labels are enough to make a fact usable in chains. It
does not mean the model reads facts through the note that it never read
that way.

### The runs

| Run | Intermediate taught | Facts asked through the note | Question |
|---|---|---|---|
| `facts_reader` | Yes | Yes | Does teaching the reading close the gap to two separate questions? |
| `facts` | Yes | No | Control. Seed 0 repeats the overnight run (78.3%). |
| `facts_unsupervised_reader` | No | Yes | Once step 2 can read a name, does step 1 start writing one? |
| `facts_unsupervised` | No | No | Control. Seed 0 repeats the overnight run (24.2%). |

Each runs with seeds 0 and 1. The seed changes the starting weights and the
order of questions; the facts and the split are the same. All four runs with
seed 0 come first. A `_reader` run differs from its control in that one
setting (`reader_frac`). The untaught pair has the small dense core and the
loss of the stage 3 runs, as before, so compare within a pair.

### Run it

1. Check the code (under a minute; no download is needed):

```bash
.venv/bin/python -m pytest --rootdir=. --confcutdir=. phrase_scratchpad/tests -q
```

2. Time the runs and print the plan, without starting them:

```bash
bash scripts/run_reading_test.sh --estimate-only
```

3. Start the eight runs:

```bash
bash scripts/run_reading_test.sh
```

At the overnight speed a run takes about 45 minutes at 64,000 steps, so the
eight take about six hours inside the default eight-hour budget. Keep the Mac
plugged in with the lid open. The same command resumes after an interruption.
Progress is in `runs/reading-test/sweep.log`, and
`runs/reading-test/report.md` is rewritten after every finished run. A
different budget, run list or seed list needs a fresh `--out` directory.

To keep the results, copy `report.md`, `plan.json` and each run's
`config.json`, `final.json` and `metrics.jsonl` into
`phrase_scratchpad/results/`, as for the overnight run. Checkpoints stay out
of the repository.

### What the report adds

- **The reading test**: for held-out 2nd chains, the mean and range over
  seeds of accuracy, how often the bridge is written, and accuracy with the
  true bridge supplied, with notes shuffled, with the subject hidden after
  step 1, and with the model's own note written into a fresh question.
- **Reading a known fact from the note**: one-step facts asked as a question
  and asked through the note, separately for facts used in training chains
  and facts never used in one.
- Three more columns under "Is the scratchpad used?", for every run:
  - *Final answer written at step 1*: the chain answered in one step, with
    the scratchpad only carrying the result forward.
  - *Subject hidden after step 1*: accuracy when step 2 can see only the
    relations and the note.
  - *Own note in a fresh question*: the note is written into a new one-step
    question and that is asked on its own. This is the handoff done by
    rewriting the question in code, which is how `ouro_lookup/` chains. It
    needs no reading from the scratchpad.

### How to read it

- **Reading is fixed** if `facts_reader` lifts "true bridge supplied" and
  held-out 2nd from about 78% toward the 93% of two separate questions.
- **The gap was the question in view** if `facts`, the control, already
  scores far higher with the subject hidden than without. The note is then
  read correctly when nothing else can be used, and hiding the subject at
  test time would be the smaller fix.
- **The note appears by itself** if `facts_unsupervised_reader` writes the
  bridge, which no run was taught to do, and beats `facts_unsupervised` on
  held-out 2nd while shuffled notes still destroy the answer. This is the
  matched control a later transfer run would be compared with.
- **Step 2 can read and step 1 does not write** if that run's "true bridge
  supplied" is high while the bridge is not written. Then the missing piece
  is the first step, which is what teaching on another task was meant to
  supply.

### Small checks

On CPU, with 60 people (745 facts), width 48, a 256-entry table, 5,000 steps
and one seed each. There are 182 held-out 2nd chains, and guessing scores
31%. These show that the code does what it should. They do not predict the
full-size result.

| Run | Held-out 2nd | Bridge written | True bridge supplied | Subject hidden | Own note in a fresh question | Held-out fact read through the note |
|---|---|---|---|---|---|---|
| `facts` | 53.3% | 100.0% | 53.3% | 91.8% | 100.0% | 82.1% |
| `facts_reader` | 96.2% | 100.0% | 96.2% | 96.2% | 98.4% | 96.4% |
| `facts_unsupervised` | 29.1% | 6.0% | 33.0% | 32.4% | 37.9% | 64.3% |
| `facts_unsupervised_reader` | 54.9% | 6.0% | 97.8% | 54.4% | 52.2% | 100.0% |

- **With the intermediate taught, the teaching closed the gap**: held-out 2nd
  went from 53% to 96%, and chains with both facts held out from 33% to 88%.
- **The control read its note correctly when the subject was hidden** (92%)
  and not when it was in view (53%). At this size most of the loss came from
  the question still being visible, not from an unreadable note.
- **With nothing taught, step 2 learned to read and step 1 did not learn to
  write.** Supplying the true bridge went from 33% to 98%, but the bridge was
  written on 6% of chains in both runs. Step 1 wrote its guess at the final
  answer instead (38% of held-out 2nd chains, against 22% in the control) and
  step 2 copied it.
- **The untaught gain is not chaining.** Held-out 2nd rose from 29% to 55%,
  but most of it is on country of citizenship (69% to 90%, 62 chains), which
  has a few common answers. Dates of birth rose from 2% to 25% (61 chains).
- Both new runs were also built at full size and took compiled training
  steps and an evaluation on CPU.

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
enough: the model can write it and then answer without reading it. Two
settings were introduced to encourage scratchpad use, and both are on by
default. The completed full-size run confirms strong use on supervised facts,
but not on arithmetic or stories.

- **One block application per step** for arithmetic and facts (the graph
  experiment used two). With two, a step is deep enough to redo both
  arithmetic operations from the question in a small check. One pass did not
  enforce note use in the full-size arithmetic run. Stories keep two, because reading
  a relation out of a sentence takes more depth than one lookup.
- **Hiding the subject, on facts.** For half of the two-step fact questions
  in training, the subject (the film's name) is hidden from the second step
  on. Those questions can only be answered from what step 1 wrote. Nothing is
  hidden at test time. Arithmetic is left alone because hiding slowed it in a
  small check. That choice leaves a shortcut in the completed full-size run.

`--n-passes 2` and `--no-hide-subject` turn them off.

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

`mix_split` jointly updates the core and the table on fact questions. The
table switch tests which capabilities depend on the table; it does not force
all factual information out of the core. Pretraining the core, freezing it,
and then learning facts only in the table is a separate future experiment.

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
- **The control hides the subject too** and uses the same dense-core width.
  It also shows whether that pressure alone
  makes the model write the intermediate answer without being taught it. In
  a small check it did not. This is not a perfectly matched optimization
  control: the mixed model has a larger, task-separated vocabulary, and its
  three task losses are averaged, whereas the control optimizes only facts.
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

### Completed overnight results (November 14)

All five runs finished 64,000 steps, seed 0, between November 13 at 21:33 and
November 14 at 03:55 Central. The complete [report](results/overnight-2026-11-14/report.md),
[interpretation and next experiment](results/overnight-2026-11-14/README.md),
and [manifest](results/overnight-2026-11-14/manifest.json) are committed alongside
every run's configuration, 40 periodic evaluations, and final diagnostics.
Checkpoints remain local and have SHA-256 hashes in the manifest.

| Run / task | Main chain accuracy | Same, notes shuffled | Correct bridge written |
|---|---:|---:|---:|
| `maths` | 83.2% | 81.5% | 99.5% |
| `clutrr` | 99.6% | 99.6% | 99.6% |
| `facts` | 78.3% | 7.4% | 99.1% |
| `mix_split` / maths | 88.5% | 87.9% | 99.5% |
| `mix_split` / clutrr | 97.7% | 95.9% | 93.0% |
| `mix_split` / facts | 24.6% | 17.9% | 4.2% |
| `facts_unsupervised` | 24.2% | 20.2% | 1.6% |

The main split is held-out second step for arithmetic and facts, and unseen
stories for CLUTRR. Scores require every answer token to be correct.

- Supervised multi-word fact chaining uses the scratchpad causally. Unseen
  chains reach 95.9%, held-out second step 78.3%, and both facts held out of
  chain training 68.5%. All constituent one-step facts were trained.
- Mixed training does not demonstrate transfer: held-out second step is
  24.6% versus 24.2% for the control, and unseen chains are lower (62.0%
  versus 67.8%). This is one seed, not evidence of a general impossibility.
- The intended source habit is weak: shuffling arithmetic and story notes
  barely changes mixed-run answers. Correct note writing is not enough.
- Supplying the correct fact bridge only lifts the mixed model to 34.7%,
  whereas two separate calls reach 95.0% (93.0% with one outer step per call,
  matching the native two-step block count). Both writing and using the
  intermediate representation need attention.
- Arithmetic and stories in `mix_split` work with the table off: 88.0% and
  97.7% on unseen chains. Fact unseen chains drop from 62.0% to 14.2% when it
  is disabled. This is useful task-dependent reliance, not a proof that all
  factual knowledge lives in the table: one-step facts still score 51.2%
  with it off.

### Preflight and historical small checks

November 13 Mac preflight: all 83 tests across the three packages pass on Apple
GPU, including compiled training and checkpoint/resume. The review found and
fixed one issue in `mix_split`: a batch with the table off decayed the lookup's
running query mean and variance toward zero. Such batches now leave both
statistics unchanged. The regression check warms the statistics before switching
the table off, so it detects the bug even after facts have been learned.

Initial checks from Claude's environment, before the Mac preflight:

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
- **The small core can do arithmetic alone.** With the table switched off,
  one-step sums reached 100% and unseen chains 38% by step 5,000, still
  rising. That is slower than with the table (76% at 6,000 steps).
- **Hiding the subject slowed arithmetic**: 30% on unseen chains with it
  against 76% without, so it is applied to facts only.
- **The three-task run was only smoke-tested.** It trains and evaluates, but
  was not run long enough here to show whether it learns.

## Files

| File | Contents |
|---|---|
| `tasks.py` | Arithmetic, facts, the shared split design, and several tasks in one vocabulary |
| `clutrr.py` | Family stories: download, parsing, questions |
| `prepare_facts.py` | Builds `facts/people_facts.tsv` |
| `model.py` | Multi-token scratchpad, dense core, table switch |
| `configs.py` | The runs |
| `train.py` | One run: training, periodic evaluation, checkpoints |
| `evaluate.py` | Accuracy, scratchpad interventions, facts read through the note, two-call ceiling, table on and off |
| `sweep.py` | The runs one after another inside a time budget, for one or several seeds |
| `report.py` | Markdown summary of finished runs |
| `tests/` | Labels, splits, masking, scratchpad contents, given notes, table switch, resume, seeds |
