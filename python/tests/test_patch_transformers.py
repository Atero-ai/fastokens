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


def test_apply_chat_template_not_routed_by_default():
    """Default patch must leave transformers' apply_chat_template untouched."""
    import fastokens

    original = transformers.PreTrainedTokenizerBase.apply_chat_template
    fastokens.patch_transformers()

    assert transformers.PreTrainedTokenizerBase.apply_chat_template is original


def test_apply_chat_template_opt_in_routes_native(monkeypatch):
    """With apply_chat_template=True, render-only calls go through the shim."""
    import fastokens

    calls = []
    original_shim_apply = _TokenizerShim.apply_chat_template

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("chat_template") is not None)
        return original_shim_apply(self, *args, **kwargs)

    monkeypatch.setattr(_TokenizerShim, "apply_chat_template", spy)
    fastokens.patch_transformers(apply_chat_template=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    messages = [{"role": "user", "content": "hi"}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    assert calls, "shim apply_chat_template was not called"

    fastokens.unpatch_transformers()
    reference = transformers.AutoTokenizer.from_pretrained(MODEL)
    expected = reference.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered == expected


def test_apply_chat_template_tokenize_true_falls_through(monkeypatch):
    """tokenize=True is unsupported natively and must use transformers' path."""
    import fastokens

    fastokens.patch_transformers(apply_chat_template=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    encoded = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True)
    # v4 returns a list of ids, v5 returns a BatchEncoding mapping.
    ids = encoded["input_ids"] if not isinstance(encoded, list) else encoded
    assert len(ids) > 0


def test_apply_chat_template_env_var_overrides_arg(monkeypatch):
    """FASTOKENS_APPLY_CHAT_TEMPLATE=0 wins over apply_chat_template=True."""
    import fastokens

    original = transformers.PreTrainedTokenizerBase.apply_chat_template
    monkeypatch.setenv(fastokens.APPLY_CHAT_TEMPLATE_ENV, "0")
    fastokens.patch_transformers(apply_chat_template=True)

    assert transformers.PreTrainedTokenizerBase.apply_chat_template is original


def test_apply_chat_template_env_var_enables(monkeypatch):
    """FASTOKENS_APPLY_CHAT_TEMPLATE=1 enables routing without the arg."""
    import fastokens

    original = transformers.PreTrainedTokenizerBase.apply_chat_template
    monkeypatch.setenv(fastokens.APPLY_CHAT_TEMPLATE_ENV, "1")
    fastokens.patch_transformers()

    assert transformers.PreTrainedTokenizerBase.apply_chat_template is not original


def test_unpatch_restores_apply_chat_template():
    """Unpatch must restore the original transformers method."""
    import fastokens

    original = transformers.PreTrainedTokenizerBase.apply_chat_template
    fastokens.patch_transformers(apply_chat_template=True)
    assert transformers.PreTrainedTokenizerBase.apply_chat_template is not original

    fastokens.unpatch_transformers()
    assert transformers.PreTrainedTokenizerBase.apply_chat_template is original
