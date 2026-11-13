"""Checks on the multi-token scratchpad model."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from phrase_scratchpad.model import PhraseConfig, PhraseModel, count_parameters

QUESTION = mx.array([[5, 6, 21, 22, 0, 0], [7, 8, 9, 23, 24, 0]])
ANSWER = mx.array([[3, 4, 0], [5, 0, 0]])


def make(seed=0, **kw):
    mx.random.seed(seed)
    base = dict(vocab_size=30, n_output=12, question_len=8, answer_len=3, d_model=32,
                n_subkeys=8, d_key=16, top_k=4)
    base.update(kw)
    model = PhraseModel(PhraseConfig(**base))
    model.eval()
    return model


def loss_of(model, steps=2, **kw):
    def loss_fn(model):
        logits = model(QUESTION, steps=steps, **kw)
        return nn.losses.cross_entropy(logits, ANSWER, reduction="mean")

    return loss_fn


def gradients(model, steps=2, **kw):
    return dict(tree_flatten(nn.value_and_grad(model, loss_of(model, steps, **kw))(model)[1]))


def test_answer_shape_and_default_steps():
    model = make()
    assert model(QUESTION).shape == (2, 3, 12)
    logits, trace = model(QUESTION, steps=3, return_trace=True)
    assert len(trace["step_logits"]) == len(trace["slots"]) == 3
    assert len(trace["retrieved"]) == 3 * model.cfg.n_passes
    assert np.array_equal(np.array(logits), np.array(trace["step_logits"][-1]))
    assert model.cfg.default_steps == 2


def test_padding_changes_nothing():
    model = make()
    wider = mx.concatenate([QUESTION, mx.zeros((2, 2), dtype=QUESTION.dtype)], axis=1)
    np.testing.assert_allclose(np.array(model(QUESTION)), np.array(model(wider)), atol=1e-5)


def test_padding_is_left_out_of_the_query_statistics():
    wider = mx.concatenate([QUESTION, mx.zeros((2, 2), dtype=QUESTION.dtype)], axis=1)
    stats = []
    for question in (QUESTION, wider):
        model = make()
        model.train()
        model(question, steps=1)
        stats.append(np.array(model.ff.norm.running_mean))
    np.testing.assert_allclose(stats[0], stats[1], atol=1e-6)
    assert np.abs(stats[0]).max() > 0


def test_scratchpad_holds_exactly_the_predicted_tokens():
    model = make()
    _, trace = model(QUESTION, steps=2, return_trace=True)
    predicted = np.array(mx.argmax(trace["step_logits"][0], axis=-1))
    embeddings = np.array(model.embed.weight)[predicted]  # (B, K, d)
    np.testing.assert_allclose(np.array(trace["slots"][0]), embeddings, atol=1e-5)
    assert predicted.max() < model.cfg.n_output


def test_step_two_reads_only_the_question_and_the_scratchpad():
    model = make()
    logits, trace = model(QUESTION, steps=2, return_trace=True)
    # Supplying the same scratchpad reproduces the answer...
    same = model(QUESTION, steps=2, slot_override=trace["slots"][0])
    np.testing.assert_allclose(np.array(same), np.array(logits), atol=1e-5)
    # ...and different tokens in it change the answer.
    written = mx.argmax(trace["step_logits"][0], axis=-1)
    different = model.embed.weight[(written + 1) % model.cfg.n_output]
    other = model(QUESTION, steps=2, slot_override=different)
    assert not np.allclose(np.array(other), np.array(logits), atol=1e-4)
    model.train()
    with pytest.raises(ValueError):
        model(QUESTION, slot_override=trace["slots"][0])


def test_final_loss_reaches_step_one_through_the_scratchpad():
    model = make()
    grads = gradients(model, steps=3)
    step = np.abs(np.array(grads["step_emb"]))
    assert step[0].max() > 0 and step[2].max() > 0  # steps 1 and 3 both ran
    assert step[3].max() == 0  # step 4 did not
    assert float(mx.abs(grads["slot_empty"]).max()) > 0


def test_table_switch_removes_the_lookup_for_chosen_questions():
    model = make(core_ff=64)
    on = np.array(model(QUESTION))
    mixed = np.array(model(QUESTION, table_on=mx.array([0.0, 1.0])))
    assert not np.allclose(mixed[0], on[0], atol=1e-4)  # question 0: table off
    np.testing.assert_allclose(mixed[1], on[1], atol=1e-5)  # question 1: unchanged

    # With the table off, the table's contents cannot matter.
    off = mx.zeros((2,))
    before = np.array(model(QUESTION, table_on=off))
    model.ff.write = model.ff.write * 3 + 1
    np.testing.assert_allclose(np.array(model(QUESTION, table_on=off)), before, atol=1e-6)
    assert not np.allclose(np.array(model(QUESTION)), on, atol=1e-4)
    grads = gradients(model, table_on=off)
    assert float(mx.abs(grads["ff.write"]).max()) == 0
    assert float(mx.abs(grads["core.up.weight"]).max()) > 0


def test_questions_with_the_table_off_do_not_move_its_query_statistics():
    model = make(core_ff=64)
    model.train()
    before = np.array(model.ff.norm.running_mean)
    model(QUESTION, steps=1, table_on=mx.zeros((2,)))
    assert np.array_equal(before, np.array(model.ff.norm.running_mean))


def test_plain_model_has_no_scratchpad():
    model = make(scratchpad=False, n_passes=4, outer_steps_train=(1,))
    names = {k for k, _ in tree_flatten(model.parameters())}
    assert not any(k.startswith(("slot_", "step_emb")) for k in names)
    logits, trace = model(QUESTION, steps=5, return_trace=True)
    assert logits.shape == (2, 3, 12) and len(trace["step_logits"]) == 1
    assert len(trace["retrieved"]) == 4


def test_other_options_train():
    for option in (dict(slot_content="soft"), dict(ff="dense", d_ff=48), dict(core_ff=48)):
        model = make(**option)
        loss_fn = loss_of(model)
        before = loss_fn(model).item()
        import mlx.optimizers as optim

        optimizer = optim.AdamW(learning_rate=1e-2)
        for _ in range(5):
            optimizer.update(model, nn.value_and_grad(model, loss_fn)(model)[1])
        assert loss_fn(model).item() < before, option
        assert count_parameters(model)["total"] > 0


def test_hidden_subject_can_only_matter_through_the_scratchpad():
    model = make()
    hide = mx.array([2, 0])  # hide the first two tokens of question 0 from step 2 on
    logits, trace = model(QUESTION, steps=2, return_trace=True, hide_prefix=hide)
    plain, plain_trace = model(QUESTION, steps=2, return_trace=True)
    # Step 1 is unaffected, and so is the question with nothing hidden.
    np.testing.assert_allclose(
        np.array(trace["step_logits"][0]), np.array(plain_trace["step_logits"][0]), atol=1e-5
    )
    np.testing.assert_allclose(np.array(logits[1]), np.array(plain[1]), atol=1e-5)
    assert not np.allclose(np.array(logits[0]), np.array(plain[0]), atol=1e-4)

    # With its subject hidden, changing the subject changes the final answer
    # only if it changes what step 1 wrote. Fix the scratchpad and it cannot.
    other = mx.array([[9, 10, 21, 22, 0, 0], [7, 8, 9, 23, 24, 0]])
    fixed = trace["slots"][0]
    a = model(QUESTION, steps=2, slot_override=fixed, hide_prefix=hide)
    b = model(other, steps=2, slot_override=fixed, hide_prefix=hide)
    np.testing.assert_allclose(np.array(a[0]), np.array(b[0]), atol=1e-5)
    c = model(other, steps=2, slot_override=fixed)
    assert not np.allclose(np.array(a[0]), np.array(c[0]), atol=1e-4)
