import json

from fastokens._native import Tokenizer


def _tokenizer() -> Tokenizer:
    """A tiny BPE tokenizer with one special added token (`<end>` -> id 5) whose
    literal characters are all in the model vocab, so encoding it as ordinary
    content yields the per-character ids [0, 1, 2, 3, 4]."""
    return Tokenizer.from_json_str(
        json.dumps(
            {
                "added_tokens": [
                    {
                        "id": 5,
                        "content": "<end>",
                        "special": True,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                    }
                ],
                "model": {
                    "type": "BPE",
                    "vocab": {"<": 0, "e": 1, "n": 2, "d": 3, ">": 4},
                    "merges": [],
                },
            }
        )
    )


def test_trusted_segment_recognizes_special_token():
    tok = _tokenizer()
    assert tok.encode_segments([("<end>", True)]).ids == [5]


def test_untrusted_segment_encodes_special_text_as_ordinary():
    tok = _tokenizer()
    ids = tok.encode_segments([("<end>", False)]).ids
    assert ids == [0, 1, 2, 3, 4]  # literal characters, not the special id
    assert ids == tok.encode_ordinary("<end>").ids


def test_segments_are_encoded_independently_and_concatenated():
    tok = _tokenizer()
    got = tok.encode_segments([("<end>", True), ("<end>", False)]).ids
    assert got == [5, 0, 1, 2, 3, 4]


def test_empty_segments():
    tok = _tokenizer()
    assert tok.encode_segments([]).ids == []
    assert tok.encode_segments([("", False)]).ids == []
    assert tok.encode_segments([("", True), ("", False)]).ids == []
