"""Tests for loading tiktoken model files via ``Tokenizer.from_tiktoken``.

The core tests use a tiny hand-crafted model (no network / no ``tiktoken``
install needed). An optional parity test against the real ``tiktoken`` library
runs only when it is importable and its encodings can be loaded.
"""

import base64

import pytest

from fastokens._native import Tokenizer


def _write_model(tmp_path, entries):
    """Write ``(bytes, rank)`` entries in tiktoken's ``base64 rank`` format."""
    path = tmp_path / "tiny.model"
    lines = [f"{base64.b64encode(tok).decode('ascii')} {rank}" for tok, rank in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# A minimal byte-level BPE: single bytes 0..3 then merges "ab"=4, "abc"=5.
TINY = [(b"a", 0), (b"b", 1), (b"c", 2), (b" ", 3), (b"ab", 4), (b"abc", 5)]


def test_tiny_model_encode_decode(tmp_path):
    path = _write_model(tmp_path, TINY)
    tok = Tokenizer.from_tiktoken(
        path,
        pattern=r"\S+| +",
        special_tokens={"<|end|>": 6},
    )
    assert tok.encode("abc").ids == [5]
    assert tok.encode("ab").ids == [4]
    assert tok.encode("abab").ids == [4, 4]
    assert tok.encode("ab c").ids == [4, 3, 2]

    ids = tok.encode("ab<|end|>c").ids
    assert ids == [4, 6, 2]
    assert tok.token_to_id("<|end|>") == 6
    assert tok.id_to_token(6) == "<|end|>"
    assert tok.decode(ids, skip_special_tokens=False) == "ab<|end|>c"
    assert tok.decode(ids, skip_special_tokens=True) == "abc"


def test_missing_pattern_raises(tmp_path):
    path = _write_model(tmp_path, TINY)
    with pytest.raises(ValueError):
        Tokenizer.from_tiktoken(path)


def test_unknown_preset_raises(tmp_path):
    path = _write_model(tmp_path, TINY)
    with pytest.raises(ValueError):
        Tokenizer.from_tiktoken(path, encoding="bogus")


@pytest.mark.parametrize("name", ["cl100k_base", "o200k_base"])
def test_parity_with_tiktoken(tmp_path, name):
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding(name)
    except Exception as e:  # network / cache unavailable
        pytest.skip(f"could not load tiktoken encoding {name}: {e}")

    model_path = tmp_path / f"{name}.model"
    with open(model_path, "w", encoding="utf-8") as f:
        for token, rank in sorted(enc._mergeable_ranks.items(), key=lambda kv: kv[1]):
            f.write(f"{base64.b64encode(token).decode('ascii')} {rank}\n")

    tok = Tokenizer.from_tiktoken(str(model_path), encoding=name)

    corpus = [
        "Hello, world!",
        "café 你好 мир 😀",
        "def f(x):\n    return x * 2\n",
        "   mixed\t whitespace \n\n and 12345 numbers",
        "The quick brown fox jumps over the lazy dog. " * 25,
    ]
    for text in corpus:
        assert tok.encode(text).ids == enc.encode_ordinary(text), f"mismatch on {text!r}"

    # Special tokens: fastokens splits them like HF; compare to allowed_special.
    special = next(iter(enc._special_tokens))
    text = f"prefix {special} suffix"
    assert tok.encode(text).ids == enc.encode(text, allowed_special="all")
