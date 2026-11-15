# Reading test: completed two-seed results

Eight runs completed 64,000 steps each between November 14, 2026 at 19:36
and November 15 at 01:21 Central on the Apple M5 Pro. Each of four variants
uses seeds 0 and 1 on the same controlled fact graph and split.

Teaching known facts through the scratchpad improves supervised chaining
in both seeds. It does not make useful intermediate writing emerge without
supervision. Hiding the original subject shows that interference from the
visible question accounts for much of the supervised control's deficit.

## Main results

These are means across two seeds on chains whose second fact never appeared
in a training chain. Every constituent fact was taught individually.

| Variant | Held-out 2nd, mean (seed range) | Bridge written | Correct bridge supplied | Notes shuffled | Original subject hidden |
|---|---:|---:|---:|---:|---:|
| Supervised control | 78.6% (78.5–78.8) | 99.2% | 78.9% | 7.6% | 92.1% |
| Supervised + reader training | 90.5% (89.1–91.8) | 99.3% | 90.7% | 6.3% | 92.7% |
| Unsupervised control | 26.3% (25.1–27.5) | 1.4% | 34.8% | 22.4% | 26.9% |
| Unsupervised + reader training | 31.0% (29.5–32.5) | 2.2% | 79.5% | 17.2% | 29.9% |

"Unsupervised" here means no intermediate-answer supervision on chains;
ordinary one-step facts and final chain answers are still supervised.
Reader training replaces half of eligible one-step questions with the same
fact asked through an injected note in an unrelated chain-shaped question.
It does not add real training chains or reveal held-out chain answers, but
it explicitly trains facts through the interface later used in chains.
Correct-bridge scores use the answer key and are diagnostic ceilings, not
unaided performance.

## Interpretation

1. **Reading improves reproducibly.** Supervised accuracy rises from 78.8%
   to 89.1% in seed 0 and from 78.5% to 91.8% in seed 1. Notes remain causal:
   shuffling them reduces the reader runs to 6.3% on average. Held-out-both
   accuracy also improves, from 67.9% to 78.3% on average.
2. **The original question competes with the note.** Hiding its subject
   raises the supervised control to 92.1%, near the reader model's 92.7%
   under the same intervention. Direct held-out fact lookup through a note
   with the original question visible rises from 72.6% to 90.8% with reader
   training. The control can already use notes when distracting context is
   removed; the failure is not simply that note-based lookup is impossible.
3. **Reading does not teach writing by itself.** Without intermediate
   supervision, reader training raises correct-bridge-supplied accuracy
   from 34.8% to 79.5%, while the bridge is written on only 2.2% of questions.
   Unaided held-out second-hop accuracy improves modestly to 31.0%, and
   unseen-chain accuracy is lower (61.5% versus 69.2%). This is not discovery
   of the intended two-step procedure.

These are two initializations of one graph and split, not two independent
datasets. The facts use multi-word names and controlled Wikipedia-derived
relations; this is not raw Wikipedia language modeling. Reader training
tests reuse of facts through a practiced interface, not transfer to an
interface never used during training.

## Archive and validation

- [Original generated report](report.md) and [timing plan](plan.json).
- Each run contains its unmodified `config.json`, `metrics.jsonl` (40
  periodic evaluations), and `final.json`.
- [Manifest](manifest.json): file hashes, source commit
  `c5609bf21cda6709dada0b59282b405966f442ac`, command, preflight validation
  and hashes of checkpoints retained locally. No checkpoints are committed.
- The archive records 31 passing reading-package tests before launch.
- Before publication, all 127 tests across the four experiment packages
  passed. The 26 raw files in this archive matched their recorded hashes and
  live outputs, and its report regenerated exactly from the saved diagnostics.

The [following Ouro sweep](../../../ouro_lookup/results/mac-2026-11-14/README.md)
tests a pretrained frozen model and includes the adapters-only control.
