"""Synthetic knowledge graph and its train/test splits.

Every (entity, relation) pair has exactly one tail entity, drawn uniformly at
random, so a fact cannot be guessed: it has to be stored somewhere.

Question formats (token ids; the answer is read at the last position):

    one-hop:  [h, r]        ->  tail[h, r]
    two-hop:  [h, r1, r2]   ->  tail[tail[h, r1], r2]

Splits

* Every fact is trained as a one-hop question.
* A fraction of facts (``held_out_frac``) is "held out": those facts never
  take part in a two-hop *training* chain. The model has seen them, but only
  on their own.
* Chains built only from the remaining facts are split into two-hop training
  chains and unseen "id" test chains.
* Chains that need a held-out fact are test-only, grouped by which hop needs
  it: ``held_first``, ``held_second``, ``held_both``.

``held_second`` and ``held_both`` are the important ones. A causal model sees
the same prefix [h, r1] in a one-hop and a two-hop question, so the first hop
is reused for free. The second hop is only right if the model fetches a fact
it has never been trained to use in second position.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphConfig:
    n_entities: int = 5000
    n_relations: int = 20
    held_out_frac: float = 0.05
    # Fraction of the chains built from non-held-out facts that go into the
    # two-hop training set. 0.4 gives about 7 training chains per fact when
    # there are 20 relations.
    two_hop_train_frac: float = 0.4
    eval_size: int = 2000
    seed: int = 0


EVAL_SPLITS = (
    "one_hop",
    "one_hop_held",
    "two_hop_train",
    "two_hop_id",
    "two_hop_held_first",
    "two_hop_held_second",
    "two_hop_held_both",
)


class MultiHopData:
    def __init__(self, cfg: GraphConfig):
        self.cfg = cfg
        E, R = cfg.n_entities, cfg.n_relations
        rng = np.random.default_rng(cfg.seed)

        self.n_entities = E
        self.n_relations = R
        self.vocab_size = E + R  # entity tokens, then relation tokens

        self.tail = rng.integers(0, E, size=(E, R), dtype=np.int32)

        n_facts = E * R
        n_held = int(round(cfg.held_out_frac * n_facts))
        held = np.zeros(n_facts, dtype=bool)
        held[rng.permutation(n_facts)[:n_held]] = True
        self.held = held.reshape(E, R)

        # All facts as (h, r), in a fixed order.
        h, r = np.divmod(np.arange(n_facts, dtype=np.int32), R)
        self.facts = np.stack([h, r], axis=1).astype(np.int32)

        # All chains as (h, r1, r2).
        idx = np.arange(E * R * R, dtype=np.int64)
        ch = (idx // (R * R)).astype(np.int32)
        cr1 = ((idx // R) % R).astype(np.int32)
        cr2 = (idx % R).astype(np.int32)
        bridge = self.tail[ch, cr1]
        first_held = self.held[ch, cr1]
        second_held = self.held[bridge, cr2]
        chains = np.stack([ch, cr1, cr2], axis=1)

        clean = chains[~first_held & ~second_held]
        clean = clean[rng.permutation(len(clean))]
        n_train = int(round(cfg.two_hop_train_frac * len(clean)))
        self.two_hop_train = clean[:n_train]

        def sample(rows):
            rows = rows[rng.permutation(len(rows))]
            return rows[: cfg.eval_size]

        self.eval_sets = {
            "one_hop": sample(self.facts[~held]),
            "one_hop_held": sample(self.facts[held]),
            "two_hop_train": sample(self.two_hop_train),
            "two_hop_id": sample(clean[n_train:]),
            "two_hop_held_first": sample(chains[first_held & ~second_held]),
            "two_hop_held_second": sample(chains[~first_held & second_held]),
            "two_hop_held_both": sample(chains[first_held & second_held]),
        }

    # -- encoding ---------------------------------------------------------

    def encode_one_hop(self, facts):
        """facts: (n, 2) of (h, r). Returns tokens (n, 2) and targets (n,)."""
        h, r = facts[:, 0], facts[:, 1]
        tokens = np.stack([h, self.n_entities + r], axis=1).astype(np.int32)
        return tokens, self.tail[h, r]

    def encode_two_hop(self, chains):
        """chains: (n, 3) of (h, r1, r2). Returns tokens (n, 3), targets (n,)."""
        h, r1, r2 = chains[:, 0], chains[:, 1], chains[:, 2]
        E = self.n_entities
        tokens = np.stack([h, E + r1, E + r2], axis=1).astype(np.int32)
        return tokens, self.tail[self.tail[h, r1], r2]

    def encode(self, rows):
        """Encode one-hop (n, 2) or two-hop (n, 3) rows by their width."""
        if rows.shape[1] == 2:
            return self.encode_one_hop(rows)
        return self.encode_two_hop(rows)

    # -- training batches -------------------------------------------------

    def sample_batch(self, rng, n_one, n_two):
        """Uniform samples of one-hop facts and two-hop training chains."""
        facts = self.facts[rng.integers(0, len(self.facts), n_one)]
        chains = self.two_hop_train[rng.integers(0, len(self.two_hop_train), n_two)]
        return self.encode_one_hop(facts), self.encode_two_hop(chains)

    def summary(self):
        out = {
            "entities": self.n_entities,
            "relations": self.n_relations,
            "facts": len(self.facts),
            "held_out_facts": int(self.held.sum()),
            "two_hop_train_chains": len(self.two_hop_train),
        }
        out.update({f"eval_{k}": len(v) for k, v in self.eval_sets.items()})
        return out
