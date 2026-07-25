from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, NamedTuple
from urllib.parse import urlparse

DEFAULT_TIKTOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
    r"\s+(?!\S)|\s+"
)
KIMI_TIKTOKEN_PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)
PRINTABLE_ASCII_START = 33
PRINTABLE_ASCII_END = 126
LATIN1_DIRECT_START = 0xA1
LATIN1_DIRECT_END_EXCLUSIVE = 0xAD
LATIN1_DIRECT_RESUME = 0xAE

logger = logging.getLogger(__name__)


class _MergeCandidate(NamedTuple):
    rank: int
    left_rank: int
    right_rank: int
    left: bytes
    right: bytes


class _AddedToken(NamedTuple):
    id: int
    content: str
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = False
    special: bool = True


def _byte_to_unicode() -> dict[int, str]:
    table: dict[int, str] = {}
    next_codepoint = 256
    for byte in range(256):
        # Match the GPT-2 byte-level alphabet: printable ASCII plus most
        # Latin-1 bytes are kept as-is, while whitespace/control bytes map
        # to code points above the byte range.
        should_use_direct_mapping = (
            PRINTABLE_ASCII_START <= byte <= PRINTABLE_ASCII_END
            or LATIN1_DIRECT_START <= byte < LATIN1_DIRECT_END_EXCLUSIVE
            or byte >= LATIN1_DIRECT_RESUME
        )
        if should_use_direct_mapping:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(next_codepoint)
            next_codepoint += 1
    return table


_BYTE_ENCODER = _byte_to_unicode()


def _token_bytes_to_string(token: bytes) -> str:
    return "".join(_BYTE_ENCODER[byte] for byte in token)


def _extract_encoding(encoding: Any) -> Any:
    if isinstance(encoding, str):
        try:
            import tiktoken
        except ImportError:
            return None
        return tiktoken.get_encoding(encoding)
    return encoding


def _get_rank_pair_for_sorting(split: tuple[int, int, bytes, bytes]) -> tuple[int, int]:
    left_rank, right_rank, _, _ = split
    return left_rank, right_rank


def _extract_vocab_and_merges(
    mergeable_ranks: dict[bytes, int],
) -> tuple[dict[str, int], list[list[str]]]:
    vocab = {
        _token_bytes_to_string(token): rank
        for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1])
    }

    merge_candidates: list[_MergeCandidate] = []
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        split_candidates: list[tuple[int, int, bytes, bytes]] = []
        for index in range(1, len(token)):
            left = token[:index]
            right = token[index:]
            if left in mergeable_ranks and right in mergeable_ranks:
                split_candidates.append(
                    (mergeable_ranks[left], mergeable_ranks[right], left, right)
                )
        split_candidates.sort(key=_get_rank_pair_for_sorting)
        merge_candidates.extend(
            _MergeCandidate(
                rank=rank,
                left_rank=left_rank,
                right_rank=right_rank,
                left=left,
                right=right,
            )
            for left_rank, right_rank, left, right in split_candidates
        )

    merge_candidates.sort(key=lambda item: (item.rank, item.left_rank, item.right_rank))
    merges = [
        [
            _token_bytes_to_string(candidate.left),
            _token_bytes_to_string(candidate.right),
        ]
        for candidate in merge_candidates
    ]
    return vocab, merges


def _normalise_special_tokens(
    special_tokens: Mapping[str, int] | None,
) -> list[_AddedToken]:
    if special_tokens is None:
        return []
    return [
        _AddedToken(id=token_id, content=token, special=True)
        for token, token_id in sorted(
            dict(special_tokens).items(), key=lambda item: item[1]
        )
    ]


def _bool_config(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _auto_tokenizer_entries(auto_tokenizer: Any) -> list[str]:
    if isinstance(auto_tokenizer, str):
        return [auto_tokenizer]
    if isinstance(auto_tokenizer, list):
        return [item for item in auto_tokenizer if isinstance(item, str)]
    return []


def _is_kimi_tiktoken_config(config: dict[str, Any]) -> bool:
    auto_map = config.get("auto_map", {})
    if isinstance(auto_map, dict):
        if any(
            "tokenization_kimi.TikTokenTokenizer" in item
            for item in _auto_tokenizer_entries(auto_map.get("AutoTokenizer"))
        ):
            return True
    if config.get("tokenizer_class") != "TikTokenTokenizer":
        return False
    added_tokens_decoder = config.get("added_tokens_decoder", {})
    if not isinstance(added_tokens_decoder, dict):
        return False
    kimi_markers = {
        "<|open|>",
        "<|close|>",
        "<|sep|>",
        "<|end_of_msg|>",
        "<|im_user|>",
        "<|im_assistant|>",
        "<|im_middle|>",
        "<|tool_calls_section_begin|>",
    }
    for token_config in added_tokens_decoder.values():
        if (
            isinstance(token_config, dict)
            and token_config.get("content") in kimi_markers
        ):
            return True
    return False


def _has_tokenizer_identity(config: dict[str, Any]) -> bool:
    return "tokenizer_class" in config or "auto_map" in config


def _config_added_tokens(
    config: dict[str, Any], base_vocab_size: int
) -> list[_AddedToken]:
    added_tokens: dict[int, _AddedToken] = {}
    added_tokens_decoder = config.get("added_tokens_decoder", {})
    if isinstance(added_tokens_decoder, dict):
        for token_id, token_config in added_tokens_decoder.items():
            try:
                token_id_int = int(token_id)
            except (TypeError, ValueError):
                continue
            if isinstance(token_config, dict):
                content = token_config.get("content")
                special = _bool_config(token_config.get("special"), False)
                single_word = _bool_config(token_config.get("single_word"), False)
                lstrip = _bool_config(token_config.get("lstrip"), False)
                rstrip = _bool_config(token_config.get("rstrip"), False)
                normalized = _bool_config(token_config.get("normalized"), False)
            else:
                content = token_config
                special = False
                single_word = False
                lstrip = False
                rstrip = False
                normalized = False
            if isinstance(content, str):
                added_tokens[token_id_int] = _AddedToken(
                    id=token_id_int,
                    content=content,
                    single_word=single_word,
                    lstrip=lstrip,
                    rstrip=rstrip,
                    normalized=normalized,
                    special=special,
                )

    if _is_kimi_tiktoken_config(config) and added_tokens:
        max_token_id = max(added_tokens)
        for token_id in range(base_vocab_size, max_token_id + 1):
            added_tokens.setdefault(
                token_id,
                _AddedToken(
                    id=token_id, content=f"<|reserved_token_{token_id}|>", special=True
                ),
            )
    return [added_tokens[token_id] for token_id in sorted(added_tokens)]


def _load_model_config(model: str | Path) -> tuple[str, dict[str, Any]]:
    if isinstance(model, str) and urlparse(model).scheme in {
        "http",
        "https",
        "s3",
        "gs",
        "az",
        "file",
    }:
        return model, {}
    model_path = Path(model)
    if model_path.is_dir():
        config_path = model_path / "tokenizer_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        return str(model_path / "tiktoken.model"), config
    return str(model), {}


def _tokenizer_json_from_parts(
    mergeable_ranks: dict[bytes, int],
    pattern: str,
    added_tokens: Iterable[_AddedToken],
    *,
    pretty: bool = False,
) -> str:
    vocab, merges = _extract_vocab_and_merges(mergeable_ranks)
    added_tokens = [
        {
            "id": token.id,
            "content": token.content,
            "single_word": token.single_word,
            "lstrip": token.lstrip,
            "rstrip": token.rstrip,
            "normalized": token.normalized,
            "special": token.special,
        }
        for token in added_tokens
    ]

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": pattern},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": merges,
        },
    }
    if pretty:
        return json.dumps(tokenizer_json, indent=2, ensure_ascii=False)
    return json.dumps(tokenizer_json, ensure_ascii=False)


def tiktoken_model_to_tokenizer_json(
    model: str | Path,
    *,
    pattern: str | None = None,
    special_tokens: Mapping[str, int] | None = None,
    pretty: bool = False,
) -> str | None:
    """
    Convert a ``tiktoken.model`` BPE file to a Hugging Face ``tokenizer.json`` string.

    ``model`` may point to a local ``tiktoken.model`` file, a URL accepted by
    ``tiktoken.load.load_tiktoken_bpe``, or a local directory containing
    ``tiktoken.model`` and optionally ``tokenizer_config.json``. Passing a model
    path returns ``None`` if the optional ``tiktoken`` package is not installed.
    """
    try:
        from tiktoken.load import load_tiktoken_bpe
    except ImportError:
        return None

    model_path, config = _load_model_config(model)
    if pattern is None:
        config_pattern = config.get("pat_str")
        if not isinstance(config_pattern, str):
            config_pattern = config.get("pattern")
        if isinstance(config_pattern, str):
            pattern = config_pattern
        elif _is_kimi_tiktoken_config(config):
            pattern = KIMI_TIKTOKEN_PATTERN
        elif _has_tokenizer_identity(config):
            logger.warning(
                "fastokens does not support converting tiktoken tokenizer config "
                "%r: tokenizer_config.json does not define pat_str or pattern, "
                "and the tokenizer type is unknown",
                model_path,
            )
            return None
        else:
            pattern = DEFAULT_TIKTOKEN_PATTERN
    try:
        mergeable_ranks = load_tiktoken_bpe(model_path)
    except ValueError as exc:
        raise ValueError(f"invalid tiktoken model {model_path!r}: {exc}") from exc
    added_tokens = (
        _config_added_tokens(config, len(mergeable_ranks))
        if special_tokens is None
        else _normalise_special_tokens(special_tokens)
    )
    return _tokenizer_json_from_parts(
        mergeable_ranks, pattern, added_tokens, pretty=pretty
    )


def tiktoken_to_tokenizer_json(encoding: Any, *, pretty: bool = False) -> str | None:
    """
    Convert a ``tiktoken`` encoding to a Hugging Face ``tokenizer.json`` string.

    ``encoding`` may be either a ``tiktoken.Encoding`` instance or an encoding
    name accepted by ``tiktoken.get_encoding``. Passing an encoding name returns
    ``None`` if the optional ``tiktoken`` package is not installed. The returned
    JSON can be passed directly to ``fastokens.Tokenizer.from_json_str``.
    """
    encoding = _extract_encoding(encoding)
    if encoding is None:
        return None
    try:
        mergeable_ranks = encoding._mergeable_ranks
        pattern = encoding._pat_str
        special_tokens = encoding._special_tokens
    except AttributeError as exc:
        raise TypeError(
            "expected a tiktoken.Encoding or encoding name with "
            "_mergeable_ranks, _pat_str, and _special_tokens"
        ) from exc

    return _tokenizer_json_from_parts(
        mergeable_ranks,
        pattern,
        _normalise_special_tokens(special_tokens),
        pretty=pretty,
    )
