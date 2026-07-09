use fancy_regex::{NoExpand, Regex};
use serde::de::Error as _;
use serde_json::Value;

use super::Error;

#[derive(Clone, Debug)]
enum Pattern {
    Literal(String),
    Regex(Regex),
}

impl Pattern {
    fn from_json(value: Value) -> Result<Self, Error> {
        if let Some(s) = value.as_str() {
            return Ok(Self::Literal(s.to_string()));
        }

        let obj = value.as_object().ok_or_else(|| {
            serde_json::Error::custom("Replace.pattern must be a string or an object")
        })?;

        if let Some(literal) = obj.get("String").and_then(Value::as_str) {
            return Ok(Self::Literal(literal.to_string()));
        }

        if let Some(regex) = obj.get("Regex").and_then(Value::as_str) {
            return Ok(Self::Regex(Regex::new(regex)?));
        }

        Err(serde_json::Error::custom("Replace.pattern object must contain String or Regex").into())
    }
}

/// Replace decoder.
///
/// Replaces each occurrence of a literal string or regex pattern with
/// `content` for each token in the decode chain.
#[derive(Clone, Debug)]
pub struct ReplaceDecoder {
    pattern: Pattern,
    content: String,
}

impl ReplaceDecoder {
    pub fn from_config(pattern: Value, content: String) -> Result<Self, Error> {
        Ok(Self {
            pattern: Pattern::from_json(pattern)?,
            content,
        })
    }

    pub fn decode_chain(&self, tokens: Vec<String>) -> Vec<String> {
        match &self.pattern {
            Pattern::Literal(needle) => tokens
                .into_iter()
                .map(|token| replace_literal(token, needle, &self.content))
                .collect(),
            Pattern::Regex(re) => tokens
                .into_iter()
                .map(|token| re.replace_all(&token, NoExpand(&self.content)).into_owned())
                .collect(),
        }
    }
}

fn replace_literal(token: String, needle: &str, replacement: &str) -> String {
    if needle.is_empty() {
        let mut output = String::new();
        output.push_str(replacement);
        for ch in token.chars() {
            output.push(ch);
            output.push_str(replacement);
        }
        return output;
    }

    let Some(first_match) = token.find(needle) else {
        return token;
    };

    let mut output = String::with_capacity(token.len());
    output.push_str(&token[..first_match]);
    output.push_str(replacement);

    let mut tail = &token[first_match + needle.len()..];
    while let Some(next_match) = tail.find(needle) {
        output.push_str(&tail[..next_match]);
        output.push_str(replacement);
        tail = &tail[next_match + needle.len()..];
    }

    output.push_str(tail);
    output
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
        let dec = ReplaceDecoder::from_config(json!({"Regex": "([a-z]+)"}), "$1".to_string())
            .unwrap();
        let out = dec.decode_chain(vec!["abc".to_string()]);
        assert_eq!(out, vec!["$1"]);
    }
}
