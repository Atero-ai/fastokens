"""Tests for the native ordinary-text encoding API."""

import json

from fastokens._native import Tokenizer


def test_encode_ordinary_bypasses_added_tokens_and_applies_length_settings():
    config = {
        "version": "1.0",
        "added_tokens": [
            {
                "id": 21,
                "content": "<control>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"Regex": "\\s+"},
            "behavior": "Removed",
            "invert": False,
        },
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {
                "<unk>": 0,
                "h": 1,
                "e": 2,
                "l": 3,
                "o": 4,
                "w": 5,
                "r": 6,
                "d": 7,
                "he": 8,
                "ll": 9,
                "hell": 10,
                "hello": 11,
                "wo": 12,
                "wor": 13,
                "worl": 14,
                "world": 15,
                "<": 16,
                "c": 17,
                "n": 18,
                "t": 19,
                ">": 20,
            },
            "merges": [
                ["h", "e"],
                ["l", "l"],
                ["he", "ll"],
                ["hell", "o"],
                ["w", "o"],
                ["wo", "r"],
                ["wor", "l"],
                ["worl", "d"],
            ],
        },
    }
    tokenizer = Tokenizer.from_json_str(json.dumps(config))

    assert tokenizer.encode("<control>").ids == [21]
    assert tokenizer.encode_ordinary("<control>").ids == [16, 17, 4, 18, 19, 6, 4, 3, 20]

    tokenizer.enable_truncation(max_length=1)
    tokenizer.enable_padding(length=3, pad_id=99)
    encoding = tokenizer.encode_ordinary("hello world")

    assert encoding.ids == [11, 99, 99]
    assert encoding.attention_mask == [1, 0, 0]
