use std::borrow::Cow;

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

/// Replace normalizer.
///
/// Replaces each occurrence of a literal string or regex pattern with
/// `content`.
#[derive(Clone, Debug)]
pub struct Replace {
    pattern: Pattern,
    content: String,
}

impl Replace {
    pub fn from_config(pattern: Value, content: String) -> Result<Self, Error> {
        Ok(Self {
            pattern: Pattern::from_json(pattern)?,
            content,
        })
    }

    pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        match &self.pattern {
            Pattern::Literal(needle) => replace_literal(input, needle, &self.content),
            Pattern::Regex(re) => re.replace_all(input, NoExpand(&self.content)),
        }
    }
}

fn replace_literal<'a>(input: &'a str, needle: &str, replacement: &str) -> Cow<'a, str> {
    if needle.is_empty() {
        let mut output = String::new();
        output.push_str(replacement);
        for ch in input.chars() {
            output.push(ch);
            output.push_str(replacement);
        }
        return Cow::Owned(output);
    }

    let Some(first_match) = input.find(needle) else {
        return Cow::Borrowed(input);
    };

    let mut output = String::with_capacity(input.len());
    output.push_str(&input[..first_match]);
    output.push_str(replacement);

    let mut tail = &input[first_match + needle.len()..];
    while let Some(next_match) = tail.find(needle) {
        output.push_str(&tail[..next_match]);
        output.push_str(replacement);
        tail = &tail[next_match + needle.len()..];
    }

    output.push_str(tail);
    Cow::Owned(output)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use serde_json::json;

    use super::*;

    #[test]
    fn literal_replace() {
        let repl = Replace::from_config(json!({"String": " "}), "▁".to_string()).unwrap();
        assert_eq!(repl.normalize("a b c"), "a▁b▁c");
    }

    #[test]
    fn literal_no_change_borrowed() {
        let repl = Replace::from_config(json!({"String": "x"}), "y".to_string()).unwrap();
        let out = repl.normalize("abc");
        assert_eq!(out, "abc");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn regex_replace() {
        let repl = Replace::from_config(json!({"Regex": "\\s+"}), " ".to_string()).unwrap();
        assert_eq!(repl.normalize("hello   world"), "hello world");
    }

    #[test]
    fn accepts_plain_string_pattern() {
        let repl = Replace::from_config(json!("."), " ".to_string()).unwrap();
        assert_eq!(repl.normalize("hello.world"), "hello world");
    }

    #[test]
    fn literal_empty_pattern_is_not_noop() {
        let repl = Replace::from_config(json!(""), "-".to_string()).unwrap();
        assert_eq!(repl.normalize("ab"), "-a-b-");
    }

    #[test]
    fn regex_replacement_content_is_literal() {
        let repl = Replace::from_config(json!({"Regex": "([a-z]+)"}), "$1".to_string()).unwrap();
        assert_eq!(repl.normalize("abc"), "$1");
    }
}
