use serde_json::Value;

use super::Error;
use crate::normalizers::Replace;

/// Replace decoder.
///
/// Applies the [`Replace`](crate::normalizers::Replace) transform to each token
/// in the decode chain. The replacement semantics (literal vs. regex, literal
/// replacement content) are identical to the normalizer; the only difference is
/// that the decoder maps the transform over a list of tokens.
#[derive(Clone, Debug)]
pub struct ReplaceDecoder {
    inner: Replace,
}

impl ReplaceDecoder {
    pub fn from_config(pattern: Value, content: String) -> Result<Self, Error> {
        Ok(Self {
            inner: Replace::from_config(pattern, content)?,
        })
    }

    pub fn decode_chain(&self, tokens: Vec<String>) -> Vec<String> {
        tokens
            .into_iter()
            .map(|token| self.inner.normalize(&token).into_owned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn literal_replace_decoder() {
        let dec = ReplaceDecoder::from_config(json!("▁"), " ".to_string()).unwrap();
        let out = dec.decode_chain(vec!["▁Hello".to_string(), "▁world".to_string()]);
        assert_eq!(out, vec![" Hello", " world"]);
    }

    #[test]
    fn regex_replace_decoder() {
        let dec = ReplaceDecoder::from_config(json!({"Regex": "[0-9]+"}), "#".to_string()).unwrap();
        let out = dec.decode_chain(vec!["a12b".to_string(), "34".to_string()]);
        assert_eq!(out, vec!["a#b", "#"]);
    }

    #[test]
    fn literal_empty_pattern_is_not_noop() {
        let dec = ReplaceDecoder::from_config(json!(""), "-".to_string()).unwrap();
        let out = dec.decode_chain(vec!["ab".to_string()]);
        assert_eq!(out, vec!["-a-b-"]);
    }

    #[test]
    fn regex_replacement_content_is_literal() {
        let dec =
            ReplaceDecoder::from_config(json!({"Regex": "([a-z]+)"}), "$1".to_string()).unwrap();
        let out = dec.decode_chain(vec!["abc".to_string()]);
        assert_eq!(out, vec!["$1"]);
    }
}
