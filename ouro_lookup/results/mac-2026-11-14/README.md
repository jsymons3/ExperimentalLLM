# Frozen Ouro with associative memory: completed Mac results

Four variants completed 8,400 training steps each on November 15, 2026,
between 01:21 and 08:45 Central. The directory date identifies the overnight
queue launched November 14. The model was Ouro-1.4B, with its original
1.435 billion parameters frozen, on an Apple M5 Pro with 48 GB memory.

The learned memory works, but this experiment does not demonstrate an
accuracy or capacity advantage over low-rank adapters. The strongest result
is the reliability of a written intermediate answer handed to a fresh
question; the same procedure also succeeds with adapters alone.

## Results

| Variant | One-step | Unseen chains | Held-out 1st | Held-out 2nd | Held-out both | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| Table, written handoff | 100.0% | 100.0% | 97.8% | 99.8% | 97.8% | 67,666,176 |
| Table + adapters, written handoff | 100.0% | 100.0% | 99.6% | 100.0% | 100.0% | 69,239,040 |
| Table + adapters, latent loops | 100.0% | 3.6% | 0.8% | 0.2% | 0.7% | 69,239,040 |
| Adapters only, written handoff | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1,572,864 |

The world contains 2,160 random facts about 360 people, using six relations
and single-token names. Every fact is taught as a one-step question; 216
facts never occur in any training chain. Thus "held-out both" means both
facts were excluded from chain training, not that their information was
never supplied. All variants share this world and split, with seed 0.
Final evaluation samples up to 500 questions per group; the both-held-out
group contains 136 questions. These are sampled recall scores, not an
exhaustive evaluation of all 2,160 facts.

## What the interventions establish

- Removing the table from the table-only model takes one-step recall from
  100% to 0%. The frozen original model cannot answer these invented facts.
- Shuffling written notes takes held-out second-hop accuracy to 1.0% for
  every written-note variant. Their answers depend on the handoff.
- The latent model gets 100% on trained chains but only 3.6% on unseen
  chains, although its loop-2 state reads as the correct intermediate name
  on 100% of those unseen chains. Both facts are individually known, and
  asking them separately reaches 100%. Readable intermediate states alone
  did not produce reusable internal composition under this training recipe.
- Adapters alone achieve the same task with about 43 times fewer trainable
  parameters than the table-only model. Frozen-core knowledge acquisition
  and the ability to disable the learned addition are shared benefits.

The code constructs a fresh one-step question from the written name and
remaining relation. The model does not discover that procedure. Written
handoffs require two full model calls; latent answers use one call with four
loops. The comparison changes the input format, resets the state and spends
more inference compute, so it does not isolate a unique benefit of discrete
tokens or establish an inherent limitation of continuous reasoning.

## Cost and side effects

| Variant | Peak training memory | Training ms/step | Ordinary-text loss |
|---|---:|---:|---:|
| Original frozen Ouro | — | — | 2.970 |
| Table | 9.2 GB | 693 | 4.037 |
| Table + adapters | 9.6 GB | 775 | 6.618 |
| Latent | 9.6 GB | 778 | 6.463 |
| Adapters | 7.0 GB | 726 | 3.311 |

The text check contains only sixteen everyday sentences; it is a diagnostic,
not a general ability benchmark. Every learned addition increases loss while
enabled, despite the original weights staying frozen. Adapters disturb this
check less and use less training memory. The table's slightly faster steps
do not establish an inference-speed or matched-capacity advantage.

No experiment here tests large-capacity scaling, successive knowledge
updates, selective lookup, moving pre-existing facts out of the core, or
making the core smaller or better at reasoning. Those benefits remain
hypotheses. A useful next comparison would measure acquisition and retention
over successive fact batches for memory and adapters, with controlled
parameter and compute budgets, before making larger architectural claims.

## Archive and validation

- [Original generated report](report.md) and [timing plan](plan.json).
- Each variant directory contains its unmodified `config.json`,
  `metrics.jsonl` (40 periodic evaluations), and `final.json`.
- [Manifest](manifest.json): SHA-256 hashes, source commit
  `c5609bf21cda6709dada0b59282b405966f442ac`, pinned pretrained-model revision
  and hashes, preflight checks, and local checkpoint hashes. No weights or
  checkpoints are included in this archive.
- The source includes the Apple GPU short-sequence attention gradient fix.
  The archived preflight records 38 passing Ouro tests and a 64-fact check
  with 100% table-on recall and 0% table-off recall.
- Before publication, all 127 tests across the four experiment packages
  passed. All 40 raw result files across both new archives matched their
  manifest hashes and live outputs; both reports regenerated exactly from
  saved diagnostics without rerunning model evaluation.

The [companion reading test](../../../phrase_scratchpad/results/reading-test-2026-11-14/README.md)
completed first, on the smaller model with multi-word names and two seeds.
