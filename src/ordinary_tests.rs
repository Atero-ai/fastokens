use serde_json::{Value, json};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use super::Tokenizer;

const REGULAR_TOKEN: &str = "<|regular|>";
const SPECIAL_TOKEN: &str = "<|special|>";

fn tokenizer_json(fused: bool, with_added_tokens: bool) -> Value {
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocab = alphabet
        .into_iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), json!(id)))
        .collect::<serde_json::Map<_, _>>();

    let pre_tokenizer = if fused {
        json!({
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": "\\S+|\\s+"},
                    "behavior": "Isolated",
                    "invert": false
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true,
                    "use_regex": false
                }
            ]
        })
    } else {
        json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        })
    };

    let added_tokens = with_added_tokens.then(|| {
        json!([
            {
                "id": 256,
                "content": REGULAR_TOKEN,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": true,
                "special": false
            },
            {
                "id": 257,
                "content": SPECIAL_TOKEN,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ])
    });

    json!({
        "version": "1.0",
        "added_tokens": added_tokens.unwrap_or_else(|| json!([])),
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": pre_tokenizer,
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": []
        }
    })
}

#[test]
fn encode_ordinary_matches_added_empty_pipeline() {
    for fused in [false, true] {
        let tokenizer = Tokenizer::from_json(tokenizer_json(fused, true)).expect("build tokenizer");
        let added_empty =
            Tokenizer::from_json(tokenizer_json(fused, false)).expect("build added-empty");

        assert_eq!(tokenizer.split_only.is_some(), fused);
        assert_eq!(tokenizer.encode(REGULAR_TOKEN).unwrap(), vec![256]);
        assert_eq!(tokenizer.encode(SPECIAL_TOKEN).unwrap(), vec![257]);

        for text in [
            "",
            "hello",
            "Cafe\u{301}",
            REGULAR_TOKEN,
            SPECIAL_TOKEN,
            "hello <|regular|> Cafe\u{301} <|special|> tail",
        ] {
            assert_eq!(
                tokenizer.encode_ordinary(text).unwrap(),
                added_empty.encode(text).unwrap(),
                "fused={fused}, text={text:?}",
            );
        }
    }
}
