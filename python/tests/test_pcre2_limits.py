import copy
import json
import pickle

import pytest

from fastokens._compat import _TokenizerShim
from fastokens._native import Tokenizer


def _tokenizer_json() -> str:
    return json.dumps(
        {
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "!": 1},
                "merges": [],
            },
            "pre_tokenizer": {
                "type": "Split",
                "pattern": {"Regex": "^(a+)+$"},
                "behavior": "Isolated",
                "invert": False,
            },
        }
    )


def test_native_from_json_str_applies_pcre2_match_limit():
    tok = Tokenizer.from_json_str(_tokenizer_json(), pcre2_match_limit=1)

    with pytest.raises(ValueError, match="match limit"):
        tok.encode("aaaaaaaaaaaaaaaa!")


def test_shim_deepcopy_preserves_pcre2_limits():
    shim = _TokenizerShim(_tokenizer_json(), pcre2_match_limit=1)
    cloned = copy.deepcopy(shim)

    with pytest.raises(ValueError, match="match limit"):
        cloned.encode("aaaaaaaaaaaaaaaa!")


def test_shim_pickle_preserves_pcre2_limits():
    shim = _TokenizerShim(_tokenizer_json(), pcre2_match_limit=1)
    restored = pickle.loads(pickle.dumps(shim))

    with pytest.raises(ValueError, match="match limit"):
        restored.encode("aaaaaaaaaaaaaaaa!")
