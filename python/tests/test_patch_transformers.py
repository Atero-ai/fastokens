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


def test_unpatch_restores_backend():
    """After unpatching, from_pretrained should return the original backend."""
    import fastokens

    fastokens.patch_transformers()
    fastokens.unpatch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    assert not isinstance(tok._tokenizer, _TokenizerShim), (
        "backend should be original tokenizers.Tokenizer after unpatch"
    )


def test_save_pretrained_with_patched_tokenizer(tmp_path):
    """save_pretrained must work under the patched backend.

    HF fast tokenizers (e.g. Qwen2/Qwen3 family) call
    ``self._tokenizer.model.save(save_directory, name=filename_prefix)``
    inside ``save_vocabulary``. Regression for an earlier _ModelStub.save
    signature that only accepted ``prefix=``.
    """
    import fastokens

    fastokens.patch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    tok.save_pretrained(str(tmp_path))


def test_model_stub_save_accepts_both_kwargs(tmp_path):
    """_ModelStub.save must accept both ``name=`` (transformers callsite) and
    ``prefix=`` (upstream tokenizers .pyi spelling)."""
    from fastokens._compat import _ModelStub, _TokenizerShim  # noqa: PLC0415

    shim = _TokenizerShim.__new__(_TokenizerShim)
    shim.get_vocab = lambda: {"a": 1, "b": 2}  # type: ignore[attr-defined]
    stub = _ModelStub(shim)

    out_name = stub.save(str(tmp_path), name="qwen")
    out_prefix = stub.save(str(tmp_path), prefix="qwen")
    out_positional = stub.save(str(tmp_path), "qwen")
    assert out_name == out_prefix == out_positional
