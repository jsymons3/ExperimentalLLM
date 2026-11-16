# Ouro repair, retention and editing: partial Mac results

Twelve of the planned 21 runs completed on an Apple M5 Pro with 48 GB memory,
between November 15 and 16, 2026. The sweep stopped at 08:20 Central on
November 16 after Apple's Metal compiler service became unavailable. The
completed results survived. Training has not been restarted for publication.

The strongest positive result is reduced forgetting when learning successive
fact batches. The table has no demonstrated repair advantage over the small
dense replacement, and no clear editing advantage over parameter-matched
adapters. All completed runs use seed 0; replication remains unfinished.

## Repairing removed feed-forward modules

The MLPs in Ouro layers 8–15 are removed, while attention and the other
original parameters stay frozen. Each addition learns from the original
model on WikiText-103. These four runs each completed 14,800 updates on the
same 5,683,200 input token positions, using batches of four 96-token windows.
Final text loss covers 97,280 held-out target tokens. The probe set contains
238 prompts the intact model answers correctly, scored on the first answer
token only.

| Addition | Trainable parameters | Held-out text loss | Known probes correct |
|---|---:|---:|---:|
| Original Ouro | — | 2.865 | 238/238 |
| Nothing put back | 0 | 4.945 | 38/238 |
| Large table | 541,329,408 | 3.277 | 174/238 |
| Small dense modules | 5,505,024 | 3.231 | 193/238 |
| Adapters | 1,572,864 | 3.302 | 164/238 |
| Smaller table | 138,545,152 | 3.270 | 157/238 |

The small dense modules recover more than the large table while using about
98 times fewer trainable parameters. The large table also exceeds the
276,824,064 parameters removed; this is not parameter compression. Matching
replacement multiplication counts does not match measured inference latency.

Recovery does not locate the facts: the remaining network can retain
information made inaccessible by the removal, and the intact teacher can
transmit information during training. This is a frozen-model repair test,
not a demonstration of moving factual storage out of a reasoning core.
The probes are not directly trained, but their subjects or answers can occur
in the training articles. The parked subject-exclusion experiment did not
finish and supplies no controlled result here.

## Learning successive batches without rehearsal

This is a separate setup with no MLP removal. Frozen Ouro learns 2,160
invented facts in six batches of 360, training only on the newest batch.
Every method reaches the 98% target on every batch. "Earlier facts retained"
is recall of the first five batches after learning the sixth.

| Addition | Trainable parameters | Mean recall when a batch is learned | Earlier facts retained | All facts at the end |
|---|---:|---:|---:|---:|
| Table | 67,666,176 | 99.2% | 30.9% | 42.3% |
| Small adapters | 1,572,864 | 99.3% | 7.7% | 23.0% |
| Parameter-matched adapters | 67,633,152 | 99.8% | 14.2% | 28.5% |
| Table, query/key parameters frozen after batch 1 | 67,666,176 | 99.8% | 53.5% | 61.2% |

The table retains more than either adapter control under the tested recipes.
This is a useful preliminary advantage, although substantial forgetting
remains. The query/key intervention is promising but not cleanly isolated:
the fixed-key run took 500 updates to learn its first batch, versus 300 for
the ordinary table, and their trajectories differed before freezing. They
were not branched from the same first-batch checkpoint. Freeze the same saved
checkpoint in one arm to isolate that effect in a follow-up. Fixed query/key
parameters also do not guarantee fixed routing when their input states change.

The rank-344 adapter control matches parameter count, not optimizer settings,
effective update scale, training time or a tuned performance envelope. One
seed is not enough to establish a general architecture advantage.

## Editing existing facts

After learning all 2,160 facts together, each model receives new answers for
100 facts. Only those changed facts are trained for 200 updates; 2,060 facts
are left unpractised. Every method first reaches at least 95% new-answer
recall at the 40-update check, and reaches 100% by the end.

| Addition | Untouched facts correct at update 40 | Untouched facts correct at update 200 |
|---|---:|---:|
| Table | 93.5% | 79.9% |
| Small adapters | 51.6% | 27.4% |
| Parameter-matched adapters | 91.9% | 84.2% |
| Table with fixed query/key parameters | 93.3% | 78.7% |

The table clearly outperforms the small adapters on this check, but the
parameter-matched adapters are close at the threshold and better at the end.
Freezing keys gives no clear editing benefit. Continuing to train after the
new answers are learned can cause additional collateral forgetting.

## Completion, provenance and limits

- Completed: four takeover runs, four retention variants and four editing
  variants, all seed 0. The wider dense takeover and all eight seed-1
  retention/editing runs are unfinished, not zero-scoring results.
- [Original generated report](report.md), [saved plan](plan.json),
  [launch history](launch.json), [failure status](status.json), and
  [archive manifest](manifest.json). The manifest explicitly has
  `complete: false` and lists all unfinished runs.
- Every completed run includes its original `config.json`, `final.json` and
  `metrics.jsonl` or `progress.json`. The 39 original archived files match
  their recorded SHA-256 hashes. The report and raw results are unmodified.
- The original report's `hours` column uses the saved invocation's timer.
  For resumed runs, notably the large table's 2.2 hours, this is not total
  training time. Do not compare those values as end-to-end run costs.
- The active source at the final interruption was `88e9d57`. The large-table
  and dense takeover runs finished under `580b491`; the table also contains
  updates from the initial schedule recorded in the launch history. The
  later change to five-minute checkpoints did not change training settings.
- The launch and manifest retain historical validation records. Publication
  validation is recorded separately in [publication.json](publication.json).
  Checkpoints are hashed in the original manifest but remain local; no
  weights, training corpus or optimizer state are published here.
- The 16-sentence ordinary-text check is a small side-effect diagnostic,
  not evidence that general reasoning is intact. There is no RAG control,
  capacity-scaling result, selective-lookup test or knowledge-shedding result
  in this archive.

The results support investigating retention further. They do not establish
that core-plus-memory is generally superior to adapters or retrieval.
