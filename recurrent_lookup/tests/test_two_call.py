import numpy as np

from recurrent_lookup.two_call import diagnose, summarize


def test_conditioning_is_on_individual_facts_not_chain_success():
    rows = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]])
    result = summarize(
        rows,
        np.array([2, 3, 4]),
        np.array([5, 6, 7]),
        np.array([2, 9, 4]),
        np.array([5, 6, 8]),
        np.array([5, 6, 8]),
        {4: np.array([0, 6, 7])},
    )
    assert result["both_facts_known"]["correct"] == 1
    assert result["two_call"]["correct"] == 2  # can be right via an incorrect bridge
    assert result["on_both_known"]["native"]["4"]["accuracy"] == 0
    assert result["on_both_known"]["two_call"]["accuracy"] == 1
    assert result["examples_known_facts_native_wrong"][0]["head"] == 0


def test_explicit_second_query_uses_prediction_not_true_bridge():
    import mlx.core as mx

    from recurrent_lookup.data import GraphConfig, MultiHopData

    data = MultiHopData(GraphConfig(n_entities=10, n_relations=2, eval_size=10))
    data.tail[0, 0] = 1
    data.tail[1, 1] = 3
    data.tail[2, 1] = 4
    data.eval_sets = {"two_hop_held_both": np.array([[0, 0, 1]])}

    class FakeModel:
        training = True

        def eval(self):
            self.training = False

        def train(self, value):
            self.training = value

        def __call__(self, tokens, n_passes=None):
            tokens = np.array(tokens)
            pred = []
            for row in tokens:
                if len(row) == 3:
                    answer = 3
                elif row[0] == 0:
                    answer = 2  # deliberately wrong first hop
                else:
                    answer = data.tail[row[0], row[1] - data.n_entities]
                pred.append(answer)
            logits = np.zeros((len(tokens), 10), dtype=np.float32)
            logits[np.arange(len(tokens)), pred] = 1
            return mx.array(logits)

    model = FakeModel()
    result = diagnose(model, data, (4,), (4,), 2)["two_hop_held_both"]["4"]
    assert result["second_fact_with_true_bridge"]["accuracy"] == 1
    assert result["two_call"]["accuracy"] == 0  # must answer 4, not oracle answer 3
    assert result["on_both_known"]["native"]["4"]["accuracy"] is None
    assert model.training  # restore evaluation state
