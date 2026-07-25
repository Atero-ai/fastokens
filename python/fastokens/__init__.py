from __future__ import annotations

import os
import warnings

from fastokens._native import StructuralTokenConfig, Tokenizer
from fastokens.tiktoken import (
    tiktoken_model_to_tokenizer_json,
    tiktoken_to_tokenizer_json,
)

__all__ = [
    "StructuralTokenConfig",
    "Tokenizer",
    "patch_transformers",
    "tiktoken_model_to_tokenizer_json",
    "tiktoken_to_tokenizer_json",
    "unpatch_transformers",
]

APPLY_CHAT_TEMPLATE_ENV = "FASTOKENS_APPLY_CHAT_TEMPLATE"

_patched = False
_originals: dict = {}
_native_chat_fallback_warned = False


def _swap_backend(tokenizer, shim_cls):
    """Replace the backend ``_tokenizer`` with a fastokens shim if needed."""
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None and not isinstance(backend, shim_cls):
        tokenizer._tokenizer = shim_cls(backend)
    return tokenizer


def _env_flag(name: str) -> bool | None:
    """Parse a boolean-like environment variable; ``None`` when unset."""
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on", "native"):
        return True
    if normalized in ("", "0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be boolean-like (0/1/true/false), got {value!r}")


def _warn_native_chat_fallback(exc: Exception) -> None:
    global _native_chat_fallback_warned
    if not _native_chat_fallback_warned:
        _native_chat_fallback_warned = True
        warnings.warn(
            "[fastokens] native chat template rendering failed; falling back to "
            f"transformers' jinja2 renderer for this and future failures: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )


def _patched_apply_chat_template(self, conversation, *args, **kwargs):
    """Route render-only ``apply_chat_template`` calls to the fastokens shim.

    Anything the native path does not support byte-for-byte — positional
    arguments, ``tokenize=True``, dict outputs, batched conversations, a
    non-shim backend — falls through to transformers' own implementation,
    as does any error raised by the native renderer (warned once).
    """
    from fastokens._compat import _TokenizerShim

    original = _originals["PreTrainedTokenizerBase.apply_chat_template"]
    backend = getattr(self, "_tokenizer", None)
    if (
        args
        or kwargs.get("tokenize", True)
        or kwargs.get("return_dict")
        or kwargs.get("return_assistant_tokens_mask")
        or not isinstance(backend, _TokenizerShim)
        or (
            isinstance(conversation, (list, tuple))
            and conversation
            and isinstance(conversation[0], (list, tuple))
        )
    ):
        return original(self, conversation, *args, **kwargs)

    try:
        get_chat_template = getattr(self, "get_chat_template", None)
        if get_chat_template is not None:
            chat_template = get_chat_template(kwargs.get("chat_template"), kwargs.get("tools"))
        else:
            chat_template = kwargs.get("chat_template") or self.chat_template
        if not isinstance(chat_template, str):
            return original(self, conversation, **kwargs)

        native_kwargs = dict(kwargs)
        for consumed in (
            "tokenize",
            "chat_template",
            "tools",
            "documents",
            "add_generation_prompt",
            "continue_final_message",
        ):
            native_kwargs.pop(consumed, None)
        # transformers silently ignores kwargs the template does not read;
        # keep that contract on the patched path unless the caller opts in.
        native_kwargs.setdefault("fastokens_strict_template", False)
        special_tokens = {
            key: [str(item) for item in value] if isinstance(value, (list, tuple)) else str(value)
            for key, value in self.special_tokens_map.items()
        }
        return backend.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=kwargs.get("add_generation_prompt", False),
            continue_final_message=kwargs.get("continue_final_message") or False,
            tools=kwargs.get("tools"),
            documents=kwargs.get("documents"),
            special_tokens=special_tokens,
            **native_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - fallback must equal the status quo
        _warn_native_chat_fallback(exc)
        return original(self, conversation, **kwargs)


def patch_transformers(apply_chat_template: bool = False) -> None:
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

    ``apply_chat_template=True`` additionally routes render-only
    ``tokenizer.apply_chat_template(..., tokenize=False)`` calls to the
    fastokens native renderer (falling back to transformers' jinja2 on any
    unsupported input or error). The ``FASTOKENS_APPLY_CHAT_TEMPLATE``
    environment variable, when set, overrides this argument in either
    direction — it doubles as a fleet-wide kill switch.
    """
    global _patched
    if _patched:
        print("[fastokens] patch_transformers: already patched.")
        return

    env_override = _env_flag(APPLY_CHAT_TEMPLATE_ENV)
    route_chat_template = env_override if env_override is not None else apply_chat_template

    from fastokens._compat import _TokenizerShim
    from fastokens._native import DecodeStream

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
            return _swap_backend(tokenizer, _TokenizerShim)

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
            _tuf.TokenizerFast = _TokenizerShim
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

    if route_chat_template:
        from transformers import PreTrainedTokenizerBase

        _originals["PreTrainedTokenizerBase.apply_chat_template"] = (
            PreTrainedTokenizerBase.apply_chat_template
        )
        PreTrainedTokenizerBase.apply_chat_template = _patched_apply_chat_template

    _patched = True

    from importlib.metadata import version

    # Assuming transformers is installed.
    # If not, this will raise an error, which is fine since patching won't work without it.
    transformers_version = version("transformers")
    print(
        f"[fastokens] patch_transformers: successfully patched transformers v{transformers_version}"
    )
    if route_chat_template:
        source = "env" if env_override is not None else "arg"
        print(
            "[fastokens] chat template engine: fastokens-native for "
            f"tokenize=False (via {source})"
        )
    else:
        print(
            "[fastokens] chat template engine: transformers jinja2 "
            "(enable native with patch_transformers(apply_chat_template=True) "
            f"or {APPLY_CHAT_TEMPLATE_ENV}=1)"
        )


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

    if "PreTrainedTokenizerBase.apply_chat_template" in _originals:
        from transformers import PreTrainedTokenizerBase

        PreTrainedTokenizerBase.apply_chat_template = _originals[
            "PreTrainedTokenizerBase.apply_chat_template"
        ]

    _td.DecodeStream = _originals["DecodeStream"]

    _originals.clear()
    _patched = False
