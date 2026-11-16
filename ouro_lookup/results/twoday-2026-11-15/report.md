# Ouro, two-day run: runs/ouro-twoday

Quick check before the runs: passed.

## Taking over the removed layers

The MLP of layers 8 to 15 is removed: 276,824,064 numbers, 19.3% of the model. Each stand-in is trained to imitate the original model on ordinary text. Held-out loss is on 97,280 tokens of articles never trained on. Known facts are 238 sentence starts, of 269 tried, that the original model completes correctly. Nothing is trained on them.

| what stands in | trained numbers | computation, as a share of the removed MLPs | tokens trained on | held-out loss | share of the lost ground recovered | known facts right | share of known facts |
|---|---|---|---|---|---|---|---|
| original Ouro | - | 100% | - | 2.865 | - | 238 | 100.0% |
| nothing put back | 0 | 0% | - | 4.945 | 0.0% | 38 | 16.0% |
| table | 541,329,408 | 2.0% | 5.7M | 3.277 | 80.2% | 174 | 73.1% |
| dense | 5,505,024 | 2.0% | 5.7M | 3.231 | 82.4% | 193 | 81.1% |
| adapters | 1,572,864 | 0.0% | 5.7M | 3.302 | 79.0% | 164 | 68.9% |
| table_small | 138,545,152 | 1.9% | 5.7M | 3.270 | 80.5% | 157 | 66.0% |

"Lost ground" is the rise in held-out loss when the MLPs are removed and nothing is put back. Computation counts multiplications per token for what stands in for one removed MLP; adapters add none once folded into the weights.
These are arithmetic operation counts, not measured inference latency. The large table has more parameters than the removed MLPs. Scores on probes measure the first answer token only; recovery does not establish where facts were stored or that general reasoning was preserved.

- `table`: a table of 16,384 entries stands in for each removed MLP.
- `dense`: a small MLP with the table's computation stands in for each removed MLP.
- `adapters`: nothing stands in; adapters on every attention layer.
- `table_small`: a table of 4,096 entries stands in for each removed MLP.

### Known facts by kind

| what stands in | capitals | currencies | authors | elements | opposites | places | sequences | general |
|---|---|---|---|---|---|---|---|---|
| original Ouro | 99 | 4 | 30 | 20 | 28 | 12 | 8 | 37 |
| nothing put back | 0 | 1 | 3 | 5 | 9 | 0 | 8 | 12 |
| table | 72 | 4 | 19 | 19 | 23 | 2 | 8 | 27 |
| dense | 84 | 3 | 26 | 20 | 23 | 1 | 7 | 29 |
| adapters | 62 | 2 | 23 | 19 | 22 | 1 | 7 | 28 |
| table_small | 51 | 3 | 22 | 19 | 24 | 2 | 8 | 28 |

Counts of sentence starts completed correctly. "Sequences" are patterns to continue ("Monday, Tuesday, Wednesday,"), which need no stored fact.

### Side measures

| what stands in | distance from the original's predictions | loss on sixteen sentences | table entries used | ms per step | peak memory | hours |
|---|---|---|---|---|---|---|
| nothing put back | 2.164 | 3.996 | - | - | - | - |
| table | 0.466 | 3.269 | 34.8% | 1069 | 19.6 GB | 2.2 |
| dense | 0.417 | 3.288 | - | 619 | 6.8 GB | 2.6 |
| adapters | 0.482 | 3.276 | - | 641 | 6.9 GB | 2.7 |
| table_small | 0.460 | 3.307 | 71.8% | 905 | 15.3 GB | 3.8 |

Distance is the mean divergence of the next-token distribution from the original model's, on the held-out text. The original scores 2.970 on the sixteen sentences.

## Adding facts batch by batch

2160 made-up facts about 360 people, split by person into 6 batches. Each batch is trained alone until it is recalled, and no earlier batch is shown again.

| run | seeds | recall of a batch just after learning it | earlier batches at the end | first batch at the end | all facts at the end | steps per batch | every batch learned | loss on sixteen sentences at the end |
|---|---|---|---|---|---|---|---|---|
| table | 1 | 99.2% | 30.9% | 32.5% | 42.3% | 133 | yes | 3.114 |
| adapters | 1 | 99.3% | 7.7% | 0.8% | 23.0% | 133 | yes | 3.617 |
| adapters_matched | 1 | 99.8% | 14.2% | 10.3% | 28.5% | 117 | yes | 3.594 |
| table_fixed_keys | 1 | 99.8% | 53.5% | 59.2% | 61.2% | 167 | yes | 3.026 |

The mean over seeds, with the range. "Earlier batches at the end" is recall, after the last batch, of every batch but the last: what survives later learning. The original model scores 2.970 on the sixteen sentences.

- `table`: the table alone.
- `adapters`: adapters alone, rank 8.
- `adapters_matched`: adapters alone, with as many trained numbers as the table (rank 344).
- `table_fixed_keys`: the table, with the part that forms and matches queries frozen after the first phase.

### Recall of each batch after each phase

`table`, seed 0:

|  | steps | batch 1 | batch 2 | batch 3 | batch 4 | batch 5 | batch 6 |
|---|---|---|---|---|---|---|---|
| after batch 1 | 300 | 98.9% | 0.6% | 0.0% | 0.6% | 0.6% | 0.6% |
| after batch 2 | 100 | 84.2% | 99.4% | 0.6% | 0.3% | 0.3% | 0.3% |
| after batch 3 | 100 | 65.6% | 71.1% | 99.2% | 0.6% | 0.6% | 0.3% |
| after batch 4 | 100 | 54.7% | 39.4% | 69.2% | 99.7% | 0.3% | 0.6% |
| after batch 5 | 100 | 38.9% | 20.8% | 32.2% | 62.2% | 98.9% | 0.0% |
| after batch 6 | 100 | 32.5% | 10.3% | 16.7% | 25.8% | 69.2% | 99.2% |

`adapters`, seed 0:

|  | steps | batch 1 | batch 2 | batch 3 | batch 4 | batch 5 | batch 6 |
|---|---|---|---|---|---|---|---|
| after batch 1 | 200 | 99.7% | 0.8% | 0.3% | 0.3% | 0.6% | 0.0% |
| after batch 2 | 200 | 6.4% | 100.0% | 0.0% | 0.0% | 0.0% | 0.6% |
| after batch 3 | 100 | 1.9% | 58.1% | 99.4% | 0.0% | 0.0% | 0.8% |
| after batch 4 | 100 | 0.6% | 15.8% | 27.2% | 98.9% | 0.6% | 0.3% |
| after batch 5 | 100 | 1.1% | 8.9% | 7.5% | 25.3% | 98.1% | 0.8% |
| after batch 6 | 100 | 0.8% | 4.4% | 3.1% | 6.1% | 24.2% | 99.4% |

`adapters_matched`, seed 0:

|  | steps | batch 1 | batch 2 | batch 3 | batch 4 | batch 5 | batch 6 |
|---|---|---|---|---|---|---|---|
| after batch 1 | 200 | 100.0% | 0.3% | 0.0% | 0.3% | 0.3% | 0.0% |
| after batch 2 | 100 | 79.7% | 100.0% | 0.0% | 0.3% | 0.0% | 0.8% |
| after batch 3 | 100 | 47.5% | 65.6% | 100.0% | 0.3% | 0.0% | 0.0% |
| after batch 4 | 100 | 31.1% | 26.1% | 59.2% | 99.7% | 0.3% | 0.6% |
| after batch 5 | 100 | 24.2% | 12.2% | 28.1% | 53.1% | 99.4% | 0.6% |
| after batch 6 | 100 | 10.3% | 5.3% | 6.1% | 10.0% | 39.4% | 99.7% |

`table_fixed_keys`, seed 0:

|  | steps | batch 1 | batch 2 | batch 3 | batch 4 | batch 5 | batch 6 |
|---|---|---|---|---|---|---|---|
| after batch 1 | 500 | 99.7% | 0.3% | 0.3% | 0.0% | 0.6% | 0.0% |
| after batch 2 | 100 | 97.5% | 100.0% | 0.6% | 0.0% | 0.3% | 0.0% |
| after batch 3 | 100 | 89.4% | 94.7% | 99.7% | 0.3% | 0.0% | 0.0% |
| after batch 4 | 100 | 77.5% | 71.4% | 93.9% | 100.0% | 0.0% | 0.0% |
| after batch 5 | 100 | 69.7% | 42.8% | 63.6% | 87.8% | 99.7% | 0.6% |
| after batch 6 | 100 | 59.2% | 28.9% | 38.6% | 59.2% | 81.7% | 99.4% |

## Changing facts

All the facts are trained together. Then 100 of them get a new answer and only those are trained, for 200 steps. The other 2060 are not shown again.

| run | seeds | all facts before the change | steps until 95% of new answers | untouched facts kept at that point | new answers given at the end | old answers still given | untouched facts kept at the end | loss on sixteen sentences at the end |
|---|---|---|---|---|---|---|---|---|
| table | 1 | 98.8% | 40 | 93.5% | 100.0% | 0.0% | 79.9% | 3.106 |
| adapters | 1 | 99.2% | 40 | 51.6% | 100.0% | 0.0% | 27.4% | 3.362 |
| adapters_matched | 1 | 99.7% | 40 | 91.9% | 100.0% | 0.0% | 84.2% | 3.648 |
| table_fixed_keys | 1 | 98.2% | 40 | 93.3% | 100.0% | 0.0% | 78.7% | 3.334 |

The mean over seeds, with the range. "At that point" is the first check at which 95% of the changed facts give their new answer.
