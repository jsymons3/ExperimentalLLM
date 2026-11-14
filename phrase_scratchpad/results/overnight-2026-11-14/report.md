# Multi-token scratchpad results: phrase_scratchpad/results/overnight-2026-11-14

## Accuracy

| run | task | steps | 1-step | 1-step, never in a chain | trained chains | unseen chains | held-out 1st | held-out 2nd | held-out both | guessing, held-out 2nd |
|---|---|---|---|---|---|---|---|---|---|---|
| maths | maths | 64000 | 99.2% | 98.1% | 81.5% | 82.3% | 82.3% | 83.2% | 80.2% | 0.4% |
| clutrr | clutrr | 64000 | 99.5% | - | 100.0% | 99.6% | - | - | - | - |
| facts | facts | 64000 | 99.2% | 95.5% | 99.0% | 95.9% | 84.5% | 78.3% | 68.5% | 13.5% |
| mix_split | maths | 64000 | 99.5% | 99.9% | 89.0% | 88.0% | 88.3% | 88.5% | 86.2% | 0.4% |
| mix_split | clutrr | 64000 | 98.8% | - | 98.9% | 97.7% | - | - | - | - |
| mix_split | facts | 64000 | 98.9% | 98.5% | 91.8% | 62.0% | 56.2% | 24.6% | 20.9% | 13.5% |
| facts_unsupervised | facts | 64000 | 99.4% | 99.1% | 95.8% | 67.8% | 58.1% | 24.2% | 18.9% | 13.5% |

An answer counts only if every token is right. Every one-step question is trained. A held-out chain needs a fact that never appeared in a training chain, as its first step, its second step, or both. "Guessing" always gives the commonest training answer for the final relation or operation, and is the floor to compare against. Each task is scored the way it was trained, with the table on or off.

## Is the scratchpad used?

| run | task | questions | bridge written at step 1 | normal | scratchpad cleared | scratchpad shuffled | true bridge supplied |
|---|---|---|---|---|---|---|---|
| maths | maths | held-out 2nd | 99.5% | 83.2% | 52.1% | 81.5% | 83.2% |
| clutrr | clutrr | unseen chains | 99.6% | 99.6% | 99.5% | 99.6% | 99.6% |
| facts | facts | held-out 2nd | 99.1% | 78.3% | 14.8% | 7.4% | 78.6% |
| mix_split | maths | held-out 2nd | 99.5% | 88.5% | 80.4% | 87.9% | 88.5% |
| mix_split | clutrr | unseen chains | 93.0% | 97.7% | 93.3% | 95.9% | 97.7% |
| mix_split | facts | held-out 2nd | 4.2% | 24.6% | 19.1% | 17.9% | 34.7% |
| facts_unsupervised | facts | held-out 2nd | 1.6% | 24.2% | 23.1% | 20.2% | 30.9% |

"Bridge written" is the share of chains whose scratchpad holds exactly the intermediate answer after step 1. The other columns are accuracy when what step 2 reads is replaced. If the answer goes through the scratchpad, clearing or shuffling it should destroy accuracy. Supplying the true bridge uses the answer key and is a diagnostic, not a score.

## Two-call ceiling (held-out 2nd chains)

| run | task | one question | two calls, 1 step each | two calls, 2 steps each | both facts known | of those, answered as one question |
|---|---|---|---|---|---|---|
| maths | maths | 83.2% | 55.9% | 99.1% | 1,982 of 2,000 | 1,654 |
| facts | facts | 78.3% | 0.9% | 93.3% | 1,865 of 2,000 | 1,534 |
| mix_split | maths | 88.5% | 97.9% | 99.5% | 1,989 of 2,000 | 1,763 |
| mix_split | facts | 24.6% | 93.0% | 95.0% | 1,894 of 2,000 | 484 |
| facts_unsupervised | facts | 24.2% | 98.0% | 98.2% | 1,962 of 2,000 | 481 |

Two calls ask the first step as its own question, then ask the second step about whatever the model answered. It shows what the model has stored, whether or not it can chain it. A run without a scratchpad has no outer steps, so it has one kind of call.

## What depends on the table

| run | task | trained with table | 1-step: on / off | unseen chains: on / off |
|---|---|---|---|---|
| maths | maths | on | 99.2% / 0.3% | 82.3% / 0.6% |
| clutrr | clutrr | on | 99.5% / 40.3% | 99.6% / 19.2% |
| facts | facts | on | 99.2% / 44.0% | 95.9% / 5.1% |
| mix_split | maths | off | 99.5% / 99.5% | 87.6% / 88.0% |
| mix_split | clutrr | off | 98.2% / 98.8% | 97.2% / 97.7% |
| mix_split | facts | on | 98.9% / 51.2% | 62.0% / 14.2% |
| facts_unsupervised | facts | on | 99.4% / 64.2% | 67.8% / 33.8% |

Each cell is accuracy with the lookup table on, then off. A split between a reasoning core and a fact table would show arithmetic and the story questions surviving with the table off, and facts collapsing.

## Accuracy by number of outer steps at test time

| run | task | questions | 1 step | 2 steps | 3 steps | 4 steps |
|---|---|---|---|---|---|---|
| maths | maths | held-out 2nd | 0.9% | 83.2% | 95.5% | 96.5% |
| clutrr | clutrr | unseen chains | 0.0% | 99.6% | 99.6% | 99.6% |
| facts | facts | held-out 2nd | 0.0% | 78.3% | 75.9% | 74.8% |
| mix_split | maths | held-out 2nd | 1.0% | 88.5% | 97.4% | 98.5% |
| mix_split | clutrr | unseen chains | 2.0% | 97.7% | 97.7% | 97.7% |
| mix_split | facts | held-out 2nd | 11.7% | 24.6% | 24.9% | 24.9% |
| facts_unsupervised | facts | held-out 2nd | 21.4% | 24.2% | 24.5% | 24.7% |

## Readout

- Stage 1, arithmetic with the intermediate result taught: held-out 2nd 83.2%.
- Stage 1, family stories with the intermediate link taught: unseen stories 99.6%.
- Stage 2, facts with the intermediate answer taught: held-out 2nd 78.3%.
- Stage 3 control, facts alone with no intermediate answer taught: unseen chains 67.8%, held-out 2nd 24.2%, bridge written on unseen chains 0.7%.
- Stage 3 (table off on arithmetic and stories), fact chains with the intermediate taught only on arithmetic and stories: unseen chains 62.0%, held-out 2nd 24.6%, bridge written on unseen chains 2.8%.

Stage 3 worked if its fact chains beat the control and the bridge is written for fact chains, where it was never taught. One seed per run: small gaps are noise.
