# Overnight results: multi-token scratchpads

Five independent runs completed 64,000 optimizer steps each on an Apple GPU,
seed 0. Training started November 13, 2026 at 21:33 Central and finished
November 14 at 03:55. The trained source is commit
`9a043378a1e0f0c807e3ee6d2e499f6ac81e12cf`.

The supervised fact result extends the hard scratchpad mechanism to multi-word
names. The attempted transfer from arithmetic and family stories did not
improve held-out second-step fact chaining over its control.

## Evidence

The [complete report](report.md) contains all splits, scratchpad interventions,
table interventions, two-call diagnostics, and depth comparisons. Each run
directory contains its exact `config.json`, all 40 periodic evaluations in
`metrics.jsonl`, and its complete `final.json`. The [manifest](manifest.json)
records their hashes, the source, data hashes, completion times, and local
checkpoint hashes. Checkpoints and the downloaded CLUTRR data are not committed.

| Fact-training setup | Held-out second step | Notes shuffled | Correct bridge written | True bridge supplied |
|---|---:|---:|---:|---:|
| Intermediate answer supervised | 78.3% | 7.4% | 99.1% | 78.6% |
| Mixed training; fact bridge unsupervised | 24.6% | 17.9% | 4.2% | 34.7% |
| Facts-only; bridge unsupervised | 24.2% | 20.2% | 1.6% | 30.9% |

Every constituent one-step fact was trained. "Held-out second step" means
the second fact never appeared in a training chain. Exact-match scoring
requires every answer token to be correct. The majority-answer baseline is
13.5%, not a uniform random-guess probability.

The supervised facts run reaches 95.9% on unseen combinations of facts that
were used in other training chains, and 68.5% when both constituent facts
were excluded from chain training. This is a controlled table with word-level
tokens, including merged names; it is not raw Wikipedia or unrestricted
natural-language question answering.

## Why transfer may have failed

The first two points are directly supported by interventions. The precise
internal shortcut and the contribution of the remaining factors are hypotheses.

1. **The source tasks did not establish reliable use of the note.** In the
   mixed run, arithmetic accuracy changes from 88.5% to 87.9% when notes are
   shuffled; family stories change from 97.7% to 95.9%. The model writes
   plausible intermediate answers but can usually answer without their
   correct contents. The strong arithmetic dependence seen in a tiny pilot
   did not survive at full size. We therefore cannot say that a successful
   general note-reading procedure was learned and then failed to transfer.
2. **Producing a bridge and reading it are separate problems.** The mixed
   fact model writes the expected bridge on only 4.2% of held-out second-step
   questions. Giving it the correct bridge raises accuracy to only 34.7%.
   Yet two separate calls reach 95.0%, or 93.0% with one outer step per call
   (the same total block count as the native two-step answer). This points
   to a problem applying accessible knowledge through the scratchpad
   interface, beyond merely discovering the correct intermediate name.
3. **The interfaces differ.** A one-step subject is in the question prefix;
   a chain's second-step subject is in the scratchpad, with both relations
   still in the question. Arithmetic, stories, and facts also have separate
   token namespaces, different question layouts, and different bridge
   meanings. Shared weights do not force the same operation at those
   positions. This is a plausible explanation, not a measured causal result.
4. **The recipe permits alternative strategies.** The subject is hidden in
   only half the fact-chain training questions and is visible during normal
   evaluation. Hiding it can force information through the note without
   forcing that information to be the canonical intermediate entity. The
   dense core and learned step embeddings also allow task- and step-specific
   behavior. The current interventions do not identify the exact shortcut.
5. **The comparison has optimization differences.** Both stage-3 runs have
   a width-512 dense core, but the mixed model uses a larger vocabulary and
   averages three task losses. It sees the same number of fact examples per
   update as the control, but fact gradients have a different relative
   weight and interact with the other tasks. This could contribute to the
   lower unseen-chain accuracy; the results do not isolate its effect.

More recurrent steps alone do little for the mixed facts model: held-out
second-step accuracy is 24.6% at two steps and 24.9% at three or four. This
does not rule out longer training, but it gives a more specific next target
than increasing compute alone.

## What did separate between core and table

With the table disabled, the mixed model scores 88.0% on unseen arithmetic
chains and 97.7% on unseen family stories. Its unseen fact chains fall from
62.0% with the table to 14.2% without it. Those tasks were explicitly trained
with their respective table settings. The result demonstrates task-dependent
reliance, not knowledge removal or a fully modular decomposition: one-step
fact accuracy remains 51.2% with the table disabled.

The five runs start independently. In particular, `mix_split` is joint
training, not "pretrain the core, freeze it, then learn facts in the table."

## Proposed next comparison (not launched)

Start with a small diagnostic comparison that keeps the fact graph, splits,
vocabulary, model width, fact exposure, and fact-loss weighting matched.

1. **Teach the reading interface using only existing one-step labels.**
   Present a one-step fact both in the ordinary question format and with its
   subject in the scratchpad and its relation in the second-step position.
   Hide the original subject for the latter examples. Train the same answer
   in both formats, including the second-step embedding used by chains.
   Match the control's number of fact presentations by substituting these
   examples, rather than simply adding more training. This exposes no new
   facts or held-out chain answers. It tests whether making a known fact
   readable through the note fixes the low correct-bridge diagnostic.
2. **Establish actual source-note use before testing transfer.** For
   arithmetic, after the first step retain only the note, remaining operator,
   and remaining operand. Teach the intermediate answer as before. Require
   that shuffled notes strongly damage answers at the intended model size
   before treating arithmetic as a source of a chaining procedure. Keep
   stories as an optional core task, not evidence of note use unless their
   own interventions support it.
3. **Separate the contributions.** Compare the current fact recipe,
   the fact recipe with the reading-interface training, and that same
   interface training plus arithmetic that must use its note. Use the same
   vocabulary and architecture for all arms and report at least two seeds.
   A small fully supervised fact arm is a positive control. Measure bridge
   writing, shuffled/cleared/correct notes, table off, and two-call accuracy
   conditioned on both facts being individually answerable.

The decision rule is explicit: first improve the correct-bridge score, then
ask whether source-task training improves *generated*-bridge composition
beyond the matched fact-only reader control. A manually decomposed two-call
wrapper remains a useful ceiling; it is not evidence that decomposition was
learned. A frozen-core/table-only curriculum should follow a demonstrably
working interface, rather than freezing the current failure.

These are proposed changes, not results. The overnight sweep has one seed
per run and does not establish that transfer is impossible.
