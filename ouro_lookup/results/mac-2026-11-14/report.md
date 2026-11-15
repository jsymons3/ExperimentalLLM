# Ouro with a lookup table: runs/ouro-lookup

Quick check before the runs (64 facts, 300 steps): one-step recall 100.0% with the table, 0.0% without. Passed.

2160 made-up facts about 360 people (for example "The mentor of Ernst is ..."), 7346 two-step training questions (for example "The mentor of the mentor of Ernst is ..."). 216 facts never appear in a two-step training question.

## Runs

| run | what it is | two-step questions | steps | minutes | trained numbers |
|---|---|---|---|---|---|
| table | frozen Ouro plus the table; two-step questions go through a written note | written note | 8400 | 102 | 67,666,176 |
| table_adapters | table plus adapters; two-step questions go through a written note | written note | 8400 | 114 | 69,239,040 |
| latent | table plus adapters; two-step questions answered in one pass through Ouro's loops, with the intermediate answer taught at loop 2 | one pass | 8400 | 113 | 69,239,040 |
| adapters | control: adapters only, no table, written note | written note | 8400 | 107 | 1,572,864 |

## Accuracy

| run | one-step | two-step, trained | two-step, unseen | held-out 1st | held-out 2nd | held-out both | guessing |
|---|---|---|---|---|---|---|---|
| table | 100.0% | 100.0% | 100.0% | 97.8% | 99.8% | 97.8% | 0.6% |
| table_adapters | 100.0% | 100.0% | 100.0% | 99.6% | 100.0% | 100.0% | 0.6% |
| latent | 100.0% | 100.0% | 3.6% | 0.8% | 0.2% | 0.7% | 0.6% |
| adapters | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.6% |

An answer counts only if it is the model's single most likely next token. Every fact is trained as a one-step question. A held-out two-step question needs a fact that never appeared in a two-step training question, as its first step, its second step, or both. "Guessing" always gives the commonest answer.

## What each addition contributes

| run | questions | everything on | table off | adapters off | original Ouro |
|---|---|---|---|---|---|
| table | one-step | 100.0% | - | - | 0.0% |
| table | two-step, unseen | 100.0% | - | - | 0.0% |
| table_adapters | one-step | 100.0% | 0.0% | 0.4% | 0.0% |
| table_adapters | two-step, unseen | 100.0% | 0.0% | 0.2% | 0.0% |
| latent | one-step | 100.0% | 0.0% | 26.4% | 0.0% |
| latent | two-step, unseen | 3.6% | 0.2% | 0.0% | 0.0% |
| adapters | one-step | 100.0% | - | - | 0.0% |
| adapters | two-step, unseen | 100.0% | - | - | 0.0% |

"Original Ouro" is the frozen model with nothing added: what it knew beforehand, which should be nothing. For a run with only one addition, that column is also the run with its addition switched off.

## What limits two-step accuracy

| run | questions | accuracy | both facts known | accuracy when both known | two separate questions |
|---|---|---|---|---|---|
| table | two-step, unseen | 100.0% | 100.0% | 100.0% | 100.0% |
| table | held-out 1st | 97.8% | 100.0% | 97.8% | 100.0% |
| table | held-out 2nd | 99.8% | 100.0% | 99.8% | 100.0% |
| table | held-out both | 97.8% | 100.0% | 97.8% | 100.0% |
| table_adapters | two-step, unseen | 100.0% | 100.0% | 100.0% | 100.0% |
| table_adapters | held-out 1st | 99.6% | 100.0% | 99.6% | 100.0% |
| table_adapters | held-out 2nd | 100.0% | 100.0% | 100.0% | 100.0% |
| table_adapters | held-out both | 100.0% | 100.0% | 100.0% | 100.0% |
| latent | two-step, unseen | 3.6% | 100.0% | 3.6% | 100.0% |
| latent | held-out 1st | 0.8% | 100.0% | 0.8% | 100.0% |
| latent | held-out 2nd | 0.2% | 100.0% | 0.2% | 100.0% |
| latent | held-out both | 0.7% | 100.0% | 0.7% | 100.0% |
| adapters | two-step, unseen | 100.0% | 100.0% | 100.0% | 100.0% |
| adapters | held-out 1st | 100.0% | 100.0% | 100.0% | 100.0% |
| adapters | held-out 2nd | 100.0% | 100.0% | 100.0% | 100.0% |
| adapters | held-out both | 100.0% | 100.0% | 100.0% | 100.0% |

"Both facts known" is the share where each fact is answered correctly as a one-step question. "Two separate questions" asks the first step outright, then asks the second step about whatever came back: what chaining could reach with the facts as they are recalled.

## The written note

| run | questions | note is right | own notes | shuffled notes | correct notes supplied |
|---|---|---|---|---|---|
| table | two-step, unseen | 100.0% | 100.0% | 0.2% | 100.0% |
| table | held-out 1st | 97.8% | 97.8% | 0.8% | 100.0% |
| table | held-out 2nd | 99.8% | 99.8% | 1.0% | 100.0% |
| table | held-out both | 97.8% | 97.8% | 0.7% | 100.0% |
| table_adapters | two-step, unseen | 100.0% | 100.0% | 0.2% | 100.0% |
| table_adapters | held-out 1st | 99.6% | 99.6% | 0.6% | 100.0% |
| table_adapters | held-out 2nd | 100.0% | 100.0% | 1.0% | 100.0% |
| table_adapters | held-out both | 100.0% | 100.0% | 0.7% | 100.0% |
| adapters | two-step, unseen | 100.0% | 100.0% | 0.2% | 100.0% |
| adapters | held-out 1st | 100.0% | 100.0% | 0.8% | 100.0% |
| adapters | held-out 2nd | 100.0% | 100.0% | 1.0% | 100.0% |
| adapters | held-out both | 100.0% | 100.0% | 0.7% | 100.0% |

The model writes one name as a note, and a fresh one-step question is asked about that name. Shuffled notes are swapped between questions; if accuracy holds up with them, the note is not being used.

## What the state reads as after each loop

| run | reads as | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|---|
| table | intermediate answer | 0.2% | 0.4% | 8.0% | 100.0% |
| table | final answer | 0.4% | 0.2% | 0.4% | 0.0% |
| table_adapters | intermediate answer | 0.0% | 0.0% | 30.2% | 100.0% |
| table_adapters | final answer | 0.0% | 0.0% | 0.4% | 0.0% |
| latent | intermediate answer | 23.8% | 100.0% | 59.8% | 10.4% |
| latent | final answer | 0.0% | 0.0% | 0.0% | 3.6% |
| adapters | intermediate answer | 0.0% | 0.4% | 11.2% | 100.0% |
| adapters | final answer | 0.0% | 0.0% | 0.0% | 0.0% |

Unseen two-step questions. After every loop Ouro's state can be read out as a next token; this is how often that token is the intermediate answer, and how often the final one.

## Side effects and cost

| run | loss on ordinary text, original | with additions | table entries used | ms per step | peak memory |
|---|---|---|---|---|---|
| table | 2.970 | 4.037 | 31.9% | 693 | 9.2 GB |
| table_adapters | 2.970 | 6.618 | 42.1% | 775 | 9.6 GB |
| latent | 2.970 | 6.463 | 57.6% | 778 | 9.6 GB |
| adapters | 2.970 | 3.311 | - | 726 | 7.0 GB |

Loss on ordinary text is the mean next-token loss on sixteen everyday sentences; a rise means the additions disturb what Ouro already did.
