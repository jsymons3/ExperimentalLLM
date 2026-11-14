"""Fetch Ouro-1.4B (about 2.9 GB) and check that it loaded correctly.

    python -m ouro_lookup.download              # into data/ouro-1.4b
    python -m ouro_lookup.download --check-only # only check files already there

The files come from https://huggingface.co/ByteDance/Ouro-1.4B (Apache-2.0),
pinned to one revision and verified against known SHA-256 sums. An
interrupted download resumes. After downloading, the model is asked to
complete "The capital of France is", which fails loudly if the weights or
this port of the model are wrong.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "ByteDance/Ouro-1.4B"
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
FILES = {
    "config.json": "ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3",
    "tokenizer.json": "fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d",
    "model.safetensors": "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af",
}
DEFAULT_DIR = "data/ouro-1.4b"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name, directory):
    """Download one file, resuming a partial one, and verify it."""
    target = directory / name
    if target.exists():
        return
    part = directory / (name + ".part")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url)
    if done:
        request.add_header("Range", f"bytes={done}-")
    with urllib.request.urlopen(request) as response:
        if done and response.status != 206:  # the server sent the whole file again
            done = 0
        total = done + int(response.headers.get("Content-Length", 0))
        with open(part, "ab" if done else "wb") as f:
            shown = -1
            while block := response.read(1 << 22):
                f.write(block)
                done += len(block)
                percent = int(100 * done / total) if total else 0
                if percent != shown and total > 1 << 26:
                    print(f"\r{name}: {percent}% of {total / 1e9:.2f} GB", end="", flush=True)
                    shown = percent
    if total > 1 << 26:
        print()
    found = sha256(part)
    if found != FILES[name]:
        part.unlink()
        raise SystemExit(f"{name}: checksum {found} does not match the expected {FILES[name]}")
    part.rename(target)


def download(directory=DEFAULT_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch(name, directory)
    return directory


def load_tokenizer(directory):
    """(encode, start token): text to a list of token ids, and the id that
    starts every sequence."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
    start = tokenizer.token_to_id("<|endoftext|>")
    if start is None:
        raise ValueError("the tokenizer has no <|endoftext|> token to start sequences with")
    return (lambda text: tokenizer.encode(text).ids), start


def sanity_check(directory):
    """True if the loaded model completes a well-known sentence."""
    import mlx.core as mx

    from .ouro import load_ouro

    encode, start = load_tokenizer(directory)
    model = load_ouro(directory)
    tokens = mx.array([start] + encode("The capital of France is"))[None]
    top = int(mx.argmax(model(tokens)[0, -1].astype(mx.float32)).item())
    want = encode(" Paris")
    ok = [top] == want
    print(f"'The capital of France is' -> token {top}; expected {want}: {'ok' if ok else 'WRONG'}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--check-only", action="store_true", help="do not download")
    p.add_argument("--no-check", action="store_true", help="download without loading the model")
    args = p.parse_args(argv)
    directory = Path(args.dir) if args.check_only else download(args.dir)
    missing = [name for name in FILES if not (directory / name).exists()]
    if missing:
        raise SystemExit(f"missing from {directory}: {missing}")
    print(f"Ouro-1.4B is in {directory}")
    if not args.no_check and not sanity_check(directory):
        raise SystemExit("the model did not give the expected answer; do not train on it")


if __name__ == "__main__":
    sys.exit(main())
