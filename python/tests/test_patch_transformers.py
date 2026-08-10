"""Tests for fastokens.patch_transformers / unpatch_transformers."""

import pytest

transformers = pytest.importorskip("transformers")

from fastokens._compat import _TokenizerShim  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(autouse=True)
def _unpatch():
    """Ensure every test starts and ends in an unpatched state."""
    import fastokens

    yield
    fastokens.unpatch_transformers()


def test_patch_swaps_backend():
    """After patching, AutoTokenizer.from_pretrained should use _TokenizerShim."""
    import fastokens

    fastokens.patch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    assert isinstance(tok._tokenizer, _TokenizerShim), (
        f"expected _TokenizerShim, got {type(tok._tokenizer).__name__}"
    )


def test_encode_decode_through_shim():
    """Encoding and decoding should round-trip through the patched backend."""
    import fastokens

    fastokens.patch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    text = "Hello, world!"
    ids = tok(text)["input_ids"]
    assert len(ids) > 0, "encode returned empty ids"
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "Hello" in decoded, f"unexpected decode: {decoded!r}"


def test_decode_ignores_ids_outside_uint32():
    """Fast decode should drop invalid IDs instead of raising."""
    import fastokens

    fastokens.patch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    valid = tok("Hello, world!")["input_ids"]
    mixed = valid[:2] + [-1, 2**32] + valid[2:]

    assert tok.decode(mixed, skip_special_tokens=True) == tok.decode(
        valid, skip_special_tokens=True
    )

    batch_valid = [valid, valid]
    batch_mixed = [mixed, valid[:1] + [-1] + valid[1:]]
    assert tok.batch_decode(batch_mixed, skip_special_tokens=True) == tok.batch_decode(
        batch_valid, skip_special_tokens=True
    )


def test_unpatch_restores_backend():
    """After unpatching, from_pretrained should return the original backend."""
    import fastokens

    fastokens.patch_transformers()
    fastokens.unpatch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    assert not isinstance(tok._tokenizer, _TokenizerShim), (
        "backend should be original tokenizers.Tokenizer after unpatch"
    )
