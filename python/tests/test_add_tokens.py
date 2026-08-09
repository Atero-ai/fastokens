"""Tests for vocabulary extension and special-token splitting in the shim.

These exercise `_TokenizerShim` directly — the object `patch_transformers`
installs as the `transformers` backend — so they run without transformers.
"""

import copy
import json
import pickle

import pytest

from fastokens._compat import _TokenizerShim

# Byte-level alphabet characters, one model token each, so arbitrary ASCII
# encodes to one id per character.
BASE_VOCAB = {chr(code): code - 33 for code in range(33, 127)}
BASE_VOCAB[" "] = len(BASE_VOCAB)
BASE_SIZE = len(BASE_VOCAB)

ADDED = [("<|endoftext|>", True), ("<think>", False)]


def added_entry(id, content, special):
    return {
        "id": id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": special,
    }


def tokenizer_json():
    return json.dumps({
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            added_entry(BASE_SIZE + offset, content, special)
            for offset, (content, special) in enumerate(ADDED)
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": BASE_VOCAB,
            "merges": [],
        },
    })


@pytest.fixture
def shim():
    return _TokenizerShim(tokenizer_json())


DECLARED_SIZE = BASE_SIZE + len(ADDED)


def test_declared_vocabulary_is_counted(shim):
    assert shim.get_vocab_size() == DECLARED_SIZE


def test_add_tokens_pads_up_to_an_embedding_count(shim):
    """The case this exists for: a checkpoint with a padded embedding matrix.

    A loader appends placeholder tokens until the tokenizer matches the
    embedding count, then indexes every id below it.
    """
    num_model_embeddings = DECLARED_SIZE + 24
    placeholders = [
        f"<|padding_token_{i}|>"
        for i in range(num_model_embeddings - shim.get_vocab_size())
    ]

    assert shim.add_tokens(placeholders) == 24
    assert shim.get_vocab_size() == num_model_embeddings

    # Contiguous from the old size, so they land on the padded rows.
    assert [shim.token_to_id(t) for t in placeholders] == list(
        range(DECLARED_SIZE, num_model_embeddings)
    )
    for id in range(num_model_embeddings):
        assert shim.id_to_token(id) is not None, f"id {id} has no token"


def test_added_tokens_appear_in_the_vocabulary_views(shim):
    shim.add_tokens(["<|pad|>"])
    id = shim.token_to_id("<|pad|>")

    assert shim.get_vocab()["<|pad|>"] == id
    assert len(shim.get_vocab()) == shim.get_vocab_size()

    decoder = shim.get_added_tokens_decoder()
    assert decoder[id].content == "<|pad|>"
    assert decoder[id].special is False


def test_add_tokens_is_idempotent(shim):
    assert shim.add_tokens(["<|pad|>"]) == 1
    size = shim.get_vocab_size()

    assert shim.add_tokens(["<|pad|>"]) == 0
    assert shim.get_vocab_size() == size


def test_add_tokens_recognizes_the_new_token_when_encoding(shim):
    shim.add_tokens(["<|pad|>"])
    pad = shim.token_to_id("<|pad|>")

    assert pad in shim.encode("a<|pad|>b", add_special_tokens=False).ids
    assert shim.decode([pad], skip_special_tokens=False) == "<|pad|>"


def test_add_special_tokens_promotes_an_existing_token(shim):
    """What makes `split_special_tokens=True` cover ordinary added tokens.

    Callers re-register every added token as special before encoding untrusted
    text, since the flag only skips tokens marked special.
    """
    think = shim.token_to_id("<think>")
    size = shim.get_vocab_size()
    assert shim.get_added_tokens_decoder()[think].special is False

    assert shim.add_special_tokens(["<think>"]) == 1

    assert shim.token_to_id("<think>") == think, "promotion must not move the id"
    assert shim.get_vocab_size() == size, "no new string, no new entry"
    assert shim.get_added_tokens_decoder()[think].special is True


def test_added_tokens_survive_serialization(shim):
    shim.add_tokens(["<|pad|>"])
    id = shim.token_to_id("<|pad|>")

    assert json.loads(shim.to_str())["added_tokens"][-1]["content"] == "<|pad|>"
    for clone in (pickle.loads(pickle.dumps(shim)), copy.deepcopy(shim)):
        assert clone.get_vocab_size() == shim.get_vocab_size()
        assert clone.token_to_id("<|pad|>") == id
        assert clone.get_added_tokens_decoder()[id].content == "<|pad|>"


def test_encode_special_tokens_splits_only_special_tokens(shim):
    """`transformers` sets this flag for `split_special_tokens=True`."""
    eot = shim.token_to_id("<|endoftext|>")
    think = shim.token_to_id("<think>")
    text = "a<|endoftext|>b<think>c"

    matched = shim.encode(text, add_special_tokens=False).ids
    assert eot in matched and think in matched

    shim.encode_special_tokens = True
    split = shim.encode(text, add_special_tokens=False).ids
    assert eot not in split, "a control token in untrusted text must not survive"
    assert think in split, "only special tokens are split"
    assert shim.decode(split, skip_special_tokens=False) == text

    # Batch encoding honors the flag the same way.
    assert list(shim.encode_batch([text])[0].ids) == list(split)
