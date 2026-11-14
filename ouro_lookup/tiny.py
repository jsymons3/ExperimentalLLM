"""A tiny random model with Ouro's shape, for tests and dry runs.

    python -m ouro_lookup.tiny data/ouro-tiny

Writes config.json, model.safetensors and tokenizer.json in the same layout
as the real download, so every other command works on it by passing
``--weights data/ouro-tiny``. The weights are random: it knows nothing, and
only shows that the code runs.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .data import NAMES, ORDINARY, RELATIONS
from .ouro import Ouro, OuroConfig

TINY = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
    num_key_value_heads=4, head_dim=16, total_ut_steps=4,
)


def make_tiny(directory, seed=0, **overrides):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    words = ["<|endoftext|>", "<unk>", "The", "the", "of", "is", *RELATIONS, *NAMES]
    for sentence in ORDINARY:
        words += [word for word, _ in Whitespace().pre_tokenize_str(sentence)]
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words))}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))

    values = {**TINY, **overrides, "vocab_size": len(vocab)}
    config = {**values, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu"}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    mx.random.seed(seed)
    model = Ouro(OuroConfig.from_json(directory / "config.json"))
    mx.save_safetensors(str(directory / "model.safetensors"), dict(tree_flatten(model.parameters())))
    return directory


if __name__ == "__main__":
    print("wrote", make_tiny(sys.argv[1] if len(sys.argv) > 1 else "data/ouro-tiny"))
