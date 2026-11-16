from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from ouro_lookup import mechanism, takeover
from ouro_lookup.download import load_tokenizer
from ouro_lookup.model import AddonConfig, LookupOuro
from ouro_lookup.ouro import load_ouro
from ouro_lookup.probes import Probes, all_probes


def test_subject_filter_is_case_normalized_and_word_bounded():
    pattern = mechanism.mention_pattern({'France': ['France', 'French'],
                                        'New Zealand': ['New Zealand']})
    assert pattern.search(mechanism.normalize('The FRENCH government.'))
    assert pattern.search(mechanism.normalize('In New\nZealand.'))
    assert not pattern.search(mechanism.normalize('Frances and Frenchman'))
    spec = mechanism.filter_spec()
    capitals = [p for p in all_probes() if p[0] == 'capitals']
    assert len(spec) == len(capitals)
    assert all(name in aliases for name, aliases in spec.items())


def test_random_removal_matches_length_bins_without_filtering_mentions():
    paragraphs = [('France. ' if i % 3 == 0 else 'A boat. ') + 'long words ' * (20 + i % 9)
                  for i in range(120)]
    spec = {'France': ['France']}
    chosen = mechanism.select_paragraphs(paragraphs, spec)
    repeated = mechanism.select_paragraphs(paragraphs, spec)
    assert all(np.array_equal(chosen[k], repeated[k]) for k in chosen)
    assert all('France' not in paragraphs[i] for i in chosen['filtered'])
    assert any('France' in paragraphs[i] for i in chosen['random_control'])
    bins = lambda indices: sorted(len(paragraphs[i]).bit_length() for i in indices)
    assert bins(chosen['filtered']) == bins(chosen['random_control'])


def test_answer_ranks_ties_and_margins():
    logits = mx.array([[4., 3., 2.], [2., 2., 1.], [-2., 0., 3.]])
    scores = mechanism.score_logits(logits, np.array([1, 1, 2]))
    np.testing.assert_array_equal(scores['rank_min'], [2, 1, 1])
    np.testing.assert_array_equal(scores['rank_max'], [2, 2, 1])
    np.testing.assert_array_equal(scores['margin'], [-1., 0., 3.])
    np.testing.assert_array_equal(scores['correct'], [False, False, True])
    expected = np.array([3., 2., 3.]) - np.log(np.exp(np.array(logits)).sum(axis=1))
    np.testing.assert_allclose(scores['log_probability'], expected, atol=1e-6)


def test_rank_batches_agree_with_existing_probe_accuracy(tiny_dir):
    encode, start = load_tokenizer(tiny_dir)
    probes = Probes(encode, start, [('names', 'The mentor of Daniel is', 'Alice'),
                                  ('names', 'Alice', 'Greg')])
    model = LookupOuro(load_ouro(tiny_dir, mx.float32),
                       AddonConfig(remove_layers=(1,), table_layers=(1,), n_subkeys=8))
    model.eval()
    for switches in ({'original': True}, {'table': False}, {}):
        scores = mechanism.rank_probes(model, probes, **switches)
        np.testing.assert_array_equal([s['correct'] for s in scores], probes.right(model, **switches))
        assert all(s['rank_min'] <= s['rank_max'] for s in scores)


def test_short_runs_keep_the_long_runs_schedule(tmp_path):
    old = takeover.build_config('table', steps=14800, batch_size=4,
                                schedule_initial_steps=7800, schedule_transition_step=925)
    short = replace(old, steps=8000, schedule_steps=14800)
    points = mx.array([0, 199, 200, 924, 925, 1500, 7559, 7999], dtype=mx.uint64)
    for peak in (old.lr, old.adapter_lr):
        np.testing.assert_array_equal(
            np.array(takeover.takeover_learning_rate(old, peak)(points)),
            np.array(takeover.takeover_learning_rate(short, peak)(points)))
    command = mechanism.training_command(old, 'filtered', tmp_path, 1500)
    cfg = takeover.config_from_args(takeover.parse_args(command[4:]))
    assert cfg.steps == 1500 and cfg.schedule_steps == 14800
    assert cfg.schedule_initial_steps == 7800 and cfg.schedule_transition_step == 925
    assert cfg.batch_size == old.batch_size and cfg.seq_len == old.seq_len
    assert '--resume' not in command
    with pytest.raises(ValueError):
        takeover.takeover_learning_rate(replace(short, schedule_steps=7999), short.lr)


def test_rank_summary_separates_residual_from_recovered_access():
    def score(rank):
        return dict(correct=rank == 1, rank_min=rank, rank_max=rank)
    rows = [dict(category='capitals', original=score(1), nothing=score(2), trained=score(1)),
            dict(category='capitals', original=score(1), nothing=score(100), trained=score(1)),
            dict(category='capitals', original=score(2), nothing=score(1), trained=score(1))]
    summary = mechanism.rank_summary(rows)['capitals_known']
    assert summary['count'] == 2 and summary['nothing']['right'] == 0
    assert summary['restored'] == dict(count=2, already_definitely_top10_without_replacement=1)
