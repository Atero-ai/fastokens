from __future__ import annotations

from fastokens._compat import _TokenizerShim
from fastokens._native import Tokenizer

__all__ = ["Tokenizer", "patch_transformers", "unpatch_transformers"]


_patched = False
_originals: dict = {}
_patch_pcre2_options: dict[str, int | None] = {
    "pcre2_match_limit": None,
    "pcre2_depth_limit": None,
    "pcre2_heap_limit": None,
    "pcre2_max_jit_stack_size": None,
}


class _ConfiguredTokenizerShim(_TokenizerShim):
    def __init__(self, src, **kwargs):
        super().__init__(src, **{**_patch_pcre2_options, **kwargs})


def _swap_backend(tokenizer, shim_cls):
    """Replace the backend ``_tokenizer`` with a fastokens shim if needed."""
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None and not isinstance(backend, shim_cls):
        tokenizer._tokenizer = shim_cls(backend)
    return tokenizer


def patch_transformers(
    pcre2_match_limit: int | None = None,
    pcre2_depth_limit: int | None = None,
    pcre2_heap_limit: int | None = None,
    pcre2_max_jit_stack_size: int | None = None,
) -> None:
    """
    Monkey-patch ``tokenizers.Tokenizer`` so that the
    ``transformers`` library uses fastokens for encoding.

    Call this before any ``AutoTokenizer.from_pretrained``
    invocation::

        import fastokens
        fastokens.patch_transformers()

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B"
        )

    Supports both transformers v4 (``tokenization_utils_fast``)
    and v5+ (``tokenization_utils_tokenizers``).

    PCRE2 resource limits can be supplied to guard against tokenizer regexes
    with pathological backtracking on adversarial input.
    """
    global _patched
    if _patched:
        print("[fastokens] patch_transformers: already patched.")
        return

    from fastokens._native import DecodeStream

    _patch_pcre2_options.update(
        {
            "pcre2_match_limit": pcre2_match_limit,
            "pcre2_depth_limit": pcre2_depth_limit,
            "pcre2_heap_limit": pcre2_heap_limit,
            "pcre2_max_jit_stack_size": pcre2_max_jit_stack_size,
        }
    )

    import tokenizers.decoders as _td

    # ── v5+: wrap from_pretrained on TokenizersBackend ────────────────
    # In transformers v5, model-specific tokenizer classes (e.g.
    # LlamaTokenizer) build self._tokenizer directly via
    # `from tokenizers import Tokenizer` in their own __init__,
    # bypassing any module-level name we could patch.
    #
    # Wrapping from_pretrained is the most reliable approach: it runs
    # *after* all initialisation (vocab, normalizer, pre-tokenizer,
    # decoder, post-processor, truncation, padding, added tokens) is
    # complete, so our shim captures the fully-configured backend via
    # its to_str() JSON serialization.
    _v5_patched = False
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        _orig_fp = TokenizersBackend.from_pretrained

        @classmethod
        def _patched_from_pretrained(cls, *args, **kwargs):
            tokenizer = _orig_fp.__func__(cls, *args, **kwargs)
            return _swap_backend(tokenizer, _ConfiguredTokenizerShim)

        _originals["TokenizersBackend.from_pretrained"] = _orig_fp
        TokenizersBackend.from_pretrained = _patched_from_pretrained
        _v5_patched = True
    except ImportError:
        pass

    # ── v4: replace the module-level TokenizerFast name ───────────────
    # In transformers v4, PreTrainedTokenizerFast.__init__ loads the
    # backend via TokenizerFast.from_file() from this module.
    if not _v5_patched:
        try:
            import transformers.tokenization_utils_fast as _tuf

            _originals["tokenization_utils_fast"] = (_tuf, _tuf.TokenizerFast)
            _tuf.TokenizerFast = _ConfiguredTokenizerShim
        except ImportError:
            pass

    if not _v5_patched and "tokenization_utils_fast" not in _originals:
        raise ImportError(
            "Could not import transformers.tokenization_utils_tokenizers "
            "(v5+) or transformers.tokenization_utils_fast (v4). "
            "Is transformers installed?"
        )

    # Replace tokenizers.decoders.DecodeStream so that vLLM's
    # FastIncrementalDetokenizer receives a stream that accepts our
    # _TokenizerShim rather than requiring a tokenizers.Tokenizer.
    _originals["DecodeStream"] = _td.DecodeStream
    _td.DecodeStream = DecodeStream

    _patched = True

    from importlib.metadata import version
    # Assuming transformers is installed.
    # If not, this will raise an error, which is fine since patching won't work without it.
    transformers_version = version("transformers")
    print(f"[fastokens] patch_transformers: successfully patched transformers v{transformers_version}")


def unpatch_transformers() -> None:
    """
    Reverse the monkey-patching applied by :func:`patch_transformers`,
    restoring the ``transformers`` library to its original state.
    """
    global _patched
    if not _patched:
        return

    import tokenizers.decoders as _td

    # v5 path
    if "TokenizersBackend.from_pretrained" in _originals:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        # `from_pretrained` is inherited from `PreTrainedTokenizerBase`, not
        # defined on `TokenizersBackend`. The value captured during patch
        # via attribute access is a bound `method`, not a classmethod
        # descriptor — assigning it back installs a stray attribute in
        # `TokenizersBackend.__dict__` that shadows the inherited
        # classmethod and breaks `cls` polymorphism for subclasses.
        # Removing our patch attribute restores plain inheritance.
        if "from_pretrained" in TokenizersBackend.__dict__:
            del TokenizersBackend.from_pretrained

    # v4 path
    if "tokenization_utils_fast" in _originals:
        mod, original_cls = _originals["tokenization_utils_fast"]
        mod.TokenizerFast = original_cls

    _td.DecodeStream = _originals["DecodeStream"]

    _originals.clear()
    _patch_pcre2_options.update(
        {
            "pcre2_match_limit": None,
            "pcre2_depth_limit": None,
            "pcre2_heap_limit": None,
            "pcre2_max_jit_stack_size": None,
        }
    )
    _patched = False
