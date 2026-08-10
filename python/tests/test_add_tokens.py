"""Tests for vocabulary extension and special-token splitting in the shim.

These exercise `_TokenizerShim` directly — the object `patch_transformers`
installs as the `transformers` backend — so they run without transformers.
"""

import copy
import json
import pickle

import pytest

from fastokens._compat import _AddedTokenInfo, _TokenizerShim

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


def test_get_vocab_without_added_tokens_excludes_them(shim):
    """`transformers.get_added_vocab()` diffs get_vocab(False) vs get_vocab(True);
    the added tokens must show up in that difference."""
    full = shim.get_vocab(with_added_tokens=True)
    base = shim.get_vocab(with_added_tokens=False)
    added = {content for content, _ in ADDED}

    assert added <= set(full), "added tokens must be in the full vocab"
    assert added.isdisjoint(base), "added tokens must be absent from the base vocab"
    assert {tok for tok in full if tok not in base} == added


def test_encode_batch_flat_honors_split_special_tokens(shim):
    """The bulk flat path must suppress control tokens like the other paths."""
    import array

    eot = shim.token_to_id("<|endoftext|>")
    text = "a<|endoftext|>b"

    def flat_ids(split):
        raw, _offsets = shim._fast.encode_batch_flat([text], split_special_tokens=split)
        return list(array.array("I", bytes(raw)))

    assert eot in flat_ids(False)
    assert eot not in flat_ids(True), (
        "a control token in untrusted text must not survive the flat path"
    )


def test_add_special_tokens_forces_special_on_addedtoken_object(shim):
    """`add_special_tokens` must mark its arguments special even when handed an
    ``AddedToken(..., special=False)``, matching HuggingFace."""
    token = _AddedTokenInfo(content="<ctrl>", special=False)
    assert shim.add_special_tokens([token]) == 1

    tid = shim.token_to_id("<ctrl>")
    assert shim.get_added_tokens_decoder()[tid].special is True

    shim.encode_special_tokens = True
    ids = shim.encode("a<ctrl>b", add_special_tokens=False).ids
    assert tid not in ids, "a promoted special token must be split like the rest"


def test_added_token_missing_normalized_defaults_to_true_and_round_trips():
    """A source entry that omits ``normalized`` is True (HF default), and a
    load/save round-trip must not rewrite it to False."""
    cfg = json.loads(tokenizer_json())
    entry = cfg["added_tokens"][0]
    del entry["normalized"]
    tid = entry["id"]
    shim = _TokenizerShim(json.dumps(cfg))

    assert shim.get_added_tokens_decoder()[tid].normalized is True

    out_entry = next(e for e in json.loads(shim.to_str())["added_tokens"] if e["id"] == tid)
    assert "normalized" not in out_entry, "round-trip must not inject normalized=false"
