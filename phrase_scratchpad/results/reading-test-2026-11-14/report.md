# Multi-token scratchpad results: runs/reading-test

## Accuracy

| run | task | steps | 1-step | 1-step, never in a chain | trained chains | unseen chains | held-out 1st | held-out 2nd | held-out both | guessing, held-out 2nd |
|---|---|---|---|---|---|---|---|---|---|---|
| facts, seed 0 | facts | 64000 | 98.8% | 96.4% | 98.8% | 95.4% | 84.7% | 78.8% | 67.2% | 13.5% |
| facts, seed 1 | facts | 64000 | 99.2% | 96.5% | 99.0% | 96.5% | 88.3% | 78.5% | 68.7% | 13.5% |
| facts_reader, seed 0 | facts | 64000 | 98.9% | 96.6% | 98.4% | 97.2% | 86.9% | 89.1% | 77.6% | 13.5% |
| facts_reader, seed 1 | facts | 64000 | 98.6% | 96.4% | 99.2% | 98.7% | 90.2% | 91.8% | 79.1% | 13.5% |
| facts_unsupervised, seed 0 | facts | 64000 | 99.0% | 98.4% | 95.2% | 68.0% | 58.5% | 25.1% | 23.5% | 13.5% |
| facts_unsupervised, seed 1 | facts | 64000 | 99.5% | 98.9% | 97.0% | 70.5% | 62.1% | 27.5% | 22.3% | 13.5% |
| facts_unsupervised_reader, seed 0 | facts | 64000 | 98.0% | 96.7% | 85.2% | 56.3% | 54.5% | 29.5% | 29.5% | 13.5% |
| facts_unsupervised_reader, seed 1 | facts | 64000 | 98.9% | 98.4% | 91.4% | 66.6% | 61.4% | 32.5% | 34.0% | 13.5% |

An answer counts only if every token is right. Every one-step question is trained. A held-out chain needs a fact that never appeared in a training chain, as its first step, its second step, or both. "Guessing" always gives the commonest training answer for the final relation or operation, and is the floor to compare against. Each task is scored the way it was trained, with the table on or off.

## Is the scratchpad used?

| run | task | questions | bridge written at step 1 | normal | scratchpad cleared | scratchpad shuffled | true bridge supplied | final answer written at step 1 | subject hidden after step 1 | own note in a fresh question |
|---|---|---|---|---|---|---|---|---|---|---|
| facts, seed 0 | facts | held-out 2nd | 99.1% | 78.8% | 17.0% | 8.2% | 79.0% | 0.0% | 92.6% | 96.9% |
| facts, seed 1 | facts | held-out 2nd | 99.4% | 78.5% | 20.0% | 7.1% | 78.8% | 0.0% | 91.6% | 92.2% |
| facts_reader, seed 0 | facts | held-out 2nd | 99.2% | 89.1% | 12.7% | 6.9% | 89.5% | 0.0% | 92.0% | 95.1% |
| facts_reader, seed 1 | facts | held-out 2nd | 99.5% | 91.8% | 10.4% | 5.7% | 92.0% | 0.0% | 93.2% | 90.5% |
| facts_unsupervised, seed 0 | facts | held-out 2nd | 1.7% | 25.1% | 22.7% | 20.2% | 33.1% | 22.1% | 26.0% | 28.5% |
| facts_unsupervised, seed 1 | facts | held-out 2nd | 1.1% | 27.5% | 25.8% | 24.5% | 36.5% | 23.7% | 27.7% | 28.3% |
| facts_unsupervised_reader, seed 0 | facts | held-out 2nd | 3.1% | 29.5% | 22.0% | 18.0% | 74.7% | 6.3% | 24.7% | 26.2% |
| facts_unsupervised_reader, seed 1 | facts | held-out 2nd | 1.4% | 32.5% | 22.6% | 16.4% | 84.2% | 17.4% | 35.2% | 33.5% |

"Bridge written" is the share of chains whose scratchpad holds exactly the intermediate answer after step 1. The other columns are accuracy when what step 2 reads is replaced. If the answer goes through the scratchpad, clearing or shuffling it should destroy accuracy. Supplying the true bridge uses the answer key and is a diagnostic, not a score.

"Final answer written at step 1" is the share of chains whose scratchpad already holds the final answer after step 1: the chain answered in one step, with the scratchpad only carrying the result forward. "Subject hidden" hides the question's subject from step 2 on, as half the training fact chains do. "Own note in a fresh question" takes the note the model wrote, writes it into a new one-step question about the remaining relation, and asks that on its own: the handoff done by rewriting the question in code, not by reading the scratchpad.

## Reading a known fact from the note

| run | task | facts | count | asked as a question | asked through the note | through the note, question in view |
|---|---|---|---|---|---|---|
| facts, seed 0 | facts | used in training chains | 526 | 97.5% | 97.5% | 80.2% |
| facts, seed 0 | facts | never in a training chain | 445 | 96.2% | 92.6% | 71.9% |
| facts, seed 1 | facts | used in training chains | 526 | 98.1% | 97.3% | 84.0% |
| facts, seed 1 | facts | never in a training chain | 445 | 94.2% | 91.5% | 73.3% |
| facts_reader, seed 0 | facts | used in training chains | 526 | 98.1% | 97.9% | 95.1% |
| facts_reader, seed 0 | facts | never in a training chain | 445 | 93.9% | 92.8% | 89.0% |
| facts_reader, seed 1 | facts | used in training chains | 526 | 96.8% | 98.7% | 97.9% |
| facts_reader, seed 1 | facts | never in a training chain | 445 | 93.9% | 95.5% | 92.6% |
| facts_unsupervised, seed 0 | facts | used in training chains | 526 | 98.7% | 75.1% | 9.3% |
| facts_unsupervised, seed 0 | facts | never in a training chain | 445 | 97.5% | 62.2% | 6.7% |
| facts_unsupervised, seed 1 | facts | used in training chains | 526 | 99.6% | 77.6% | 9.9% |
| facts_unsupervised, seed 1 | facts | never in a training chain | 445 | 98.4% | 59.6% | 9.0% |
| facts_unsupervised_reader, seed 0 | facts | used in training chains | 526 | 96.4% | 94.5% | 86.9% |
| facts_unsupervised_reader, seed 0 | facts | never in a training chain | 445 | 93.0% | 93.0% | 81.3% |
| facts_unsupervised_reader, seed 1 | facts | used in training chains | 526 | 97.9% | 96.6% | 88.8% |
| facts_unsupervised_reader, seed 1 | facts | never in a training chain | 445 | 96.6% | 93.7% | 82.5% |

One-step facts that can be the second step of a chain. "Through the note" puts the fact's subject in the scratchpad of an unrelated two-step question and asks for the fact's answer, first with that question's subject hidden after step 1 and then with it in view. A held-out second step needs a fact from the "never in a training chain" rows to be read this way.

## Two-call ceiling (held-out 2nd chains)

| run | task | one question | two calls, 1 step each | two calls, 2 steps each | both facts known | of those, answered as one question |
|---|---|---|---|---|---|---|
| facts, seed 0 | facts | 78.8% | 0.3% | 96.7% | 1,926 of 2,000 | 1,551 |
| facts, seed 1 | facts | 78.5% | 0.2% | 92.3% | 1,844 of 2,000 | 1,548 |
| facts_reader, seed 0 | facts | 89.1% | 0.1% | 95.1% | 1,898 of 2,000 | 1,750 |
| facts_reader, seed 1 | facts | 91.8% | 0.3% | 90.1% | 1,795 of 2,000 | 1,742 |
| facts_unsupervised, seed 0 | facts | 25.1% | 97.6% | 96.5% | 1,921 of 2,000 | 498 |
| facts_unsupervised, seed 1 | facts | 27.5% | 97.8% | 98.6% | 1,962 of 2,000 | 546 |
| facts_unsupervised_reader, seed 0 | facts | 29.5% | 81.5% | 93.2% | 1,856 of 2,000 | 573 |
| facts_unsupervised_reader, seed 1 | facts | 32.5% | 93.5% | 95.0% | 1,891 of 2,000 | 642 |

Two calls ask the first step as its own question, then ask the second step about whatever the model answered. It shows what the model has stored, whether or not it can chain it. A run without a scratchpad has no outer steps, so it has one kind of call.

## What depends on the table

| run | task | trained with table | 1-step: on / off | unseen chains: on / off |
|---|---|---|---|---|
| facts, seed 0 | facts | on | 98.8% / 40.8% | 95.4% / 5.3% |
| facts, seed 1 | facts | on | 99.2% / 14.3% | 96.5% / 2.2% |
| facts_reader, seed 0 | facts | on | 98.9% / 41.2% | 97.2% / 7.2% |
| facts_reader, seed 1 | facts | on | 98.6% / 14.1% | 98.7% / 5.5% |
| facts_unsupervised, seed 0 | facts | on | 99.0% / 67.8% | 68.0% / 34.2% |
| facts_unsupervised, seed 1 | facts | on | 99.5% / 47.4% | 70.5% / 14.1% |
| facts_unsupervised_reader, seed 0 | facts | on | 98.0% / 38.6% | 56.3% / 22.7% |
| facts_unsupervised_reader, seed 1 | facts | on | 98.9% / 43.9% | 66.6% / 20.2% |

Each cell is accuracy with the lookup table on, then off. A split between a reasoning core and a fact table would show arithmetic and the story questions surviving with the table off, and facts collapsing.

## Accuracy by number of outer steps at test time

| run | task | questions | 1 step | 2 steps | 3 steps | 4 steps |
|---|---|---|---|---|---|---|
| facts, seed 0 | facts | held-out 2nd | 0.0% | 78.8% | 75.3% | 74.5% |
| facts, seed 1 | facts | held-out 2nd | 0.0% | 78.5% | 74.7% | 73.2% |
| facts_reader, seed 0 | facts | held-out 2nd | 0.0% | 89.1% | 90.2% | 90.2% |
| facts_reader, seed 1 | facts | held-out 2nd | 0.0% | 91.8% | 91.6% | 91.5% |
| facts_unsupervised, seed 0 | facts | held-out 2nd | 22.1% | 25.1% | 25.3% | 25.3% |
| facts_unsupervised, seed 1 | facts | held-out 2nd | 23.7% | 27.5% | 28.0% | 28.1% |
| facts_unsupervised_reader, seed 0 | facts | held-out 2nd | 6.3% | 29.5% | 31.6% | 31.8% |
| facts_unsupervised_reader, seed 1 | facts | held-out 2nd | 17.4% | 32.5% | 33.1% | 33.0% |

## The reading test

| run | seeds | held-out 2nd | bridge written | true bridge supplied | scratchpad shuffled | subject hidden after step 1 | own note in a fresh question | held-out fact read through the note |
|---|---|---|---|---|---|---|---|---|
| facts | 2 | 78.6% (78.5 to 78.8) | 99.2% (99.1 to 99.4) | 78.9% (78.8 to 79.0) | 7.6% (7.1 to 8.2) | 92.1% (91.6 to 92.6) | 94.5% (92.2 to 96.9) | 92.0% (91.5 to 92.6) |
| facts_reader | 2 | 90.5% (89.1 to 91.8) | 99.3% (99.2 to 99.5) | 90.7% (89.5 to 92.0) | 6.3% (5.7 to 6.9) | 92.7% (92.0 to 93.2) | 92.8% (90.5 to 95.1) | 94.2% (92.8 to 95.5) |
| facts_unsupervised | 2 | 26.3% (25.1 to 27.5) | 1.4% (1.1 to 1.7) | 34.8% (33.1 to 36.5) | 22.4% (20.2 to 24.5) | 26.9% (26.0 to 27.7) | 28.4% (28.3 to 28.5) | 60.9% (59.6 to 62.2) |
| facts_unsupervised_reader | 2 | 31.0% (29.5 to 32.5) | 2.2% (1.4 to 3.1) | 79.5% (74.7 to 84.2) | 17.2% (16.4 to 18.0) | 29.9% (24.7 to 35.2) | 29.8% (26.2 to 33.5) | 93.4% (93.0 to 93.7) |

Fact chains whose second fact never appeared in a training chain; the mean over seeds, with the range. A `_reader` run is the run above it with half of its one-step questions about second-step facts asked through the note instead. It sees each fact as often and sees no extra chain. The last column is the direct check that the teaching took: one-step facts never used in a training chain, asked through the note.

- Reading is fixed if, with the intermediate answer taught, `facts_reader` raises "true bridge supplied" and held-out 2nd toward "own note in a fresh question".
- If `facts` scores far higher with the subject hidden than without, its note is read correctly when nothing else is in view, and the loss comes from the question still being visible at step 2.
- The note appears by itself if `facts_unsupervised_reader` writes the bridge, where no run was ever taught to, and its held-out 2nd rises above `facts_unsupervised` while shuffling still destroys it. A high "true bridge supplied" with the bridge not written means step 2 can read a name and step 1 does not supply one.

## Readout

- Stage 2, facts with the intermediate answer taught: held-out 2nd 78.6%.
- Stage 3 control, facts alone with no intermediate answer taught: unseen chains 69.2%, held-out 2nd 26.3%, bridge written on unseen chains 0.5%.
- Reading test, facts with the intermediate answer taught and known facts also asked through the note: unseen chains 97.9%, held-out 2nd 90.5%, bridge written on unseen chains 99.1%.
- Reading test, facts with no intermediate answer taught and known facts also asked through the note: unseen chains 61.5%, held-out 2nd 31.0%, bridge written on unseen chains 1.6%.

Stage 3 worked if its fact chains beat the control and the bridge is written for fact chains, where it was never taught. These lines give the mean over seeds.
