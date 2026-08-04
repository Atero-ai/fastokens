mod byte_fallback;
mod byte_level;
mod replace;

use crate::json_structs::{DecoderConfig, DecoderKind};

pub use self::byte_fallback::ByteFallbackDecoder;
pub use self::byte_level::ByteLevelDecoder;
pub use self::replace::ReplaceDecoder;

/// Errors from constructing or running a decoder.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid config value: {0}")]
    Json(#[from] serde_json::Error),

    #[error("regex error: {0}")]
    Regex(#[from] fancy_regex::Error),

    #[error("unsupported decoder type: {0}")]
    Unsupported(String),
}

impl From<crate::normalizers::Error> for Error {
    fn from(e: crate::normalizers::Error) -> Self {
        match e {
            crate::normalizers::Error::Json(j) => Self::Json(j),
            crate::normalizers::Error::Regex(r) => Self::Regex(r),
            crate::normalizers::Error::Unsupported(s) => Self::Unsupported(s),
        }
    }
}

/// A compiled decoder ready for use.
#[derive(Debug)]
pub enum Decoder {
    ByteFallback(ByteFallbackDecoder),
    ByteLevel(ByteLevelDecoder),
    Replace(ReplaceDecoder),
    /// Fuses the token list into a single concatenated token. Must reduce the
    /// list (not just rely on the final join) so that any following step — e.g.
    /// a `Strip` — operates on the whole string, not each token.
    Fuse,
    /// Strips up to `start` leading and `stop` trailing occurrences of
    /// `content` from each token (e.g. SentencePiece `Strip(" ", 1, 0)`
    /// drops the leading metaspace space).
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
    Sequence(Vec<Decoder>),
}

/// Strip up to `start` leading and `stop` trailing occurrences of `content`
/// from `token`. Mirrors HuggingFace's `Strip` decoder.
fn strip_token(token: &str, content: char, start: usize, stop: usize) -> String {
    let chars: Vec<char> = token.chars().collect();

    let mut start_cut = 0;
    for (i, &c) in chars.iter().enumerate().take(start) {
        if c == content {
            start_cut = i + 1;
        } else {
            break;
        }
    }

    let mut stop_cut = chars.len();
    for (i, &c) in chars.iter().rev().enumerate().take(stop) {
        if c == content {
            stop_cut = chars.len() - (i + 1);
        } else {
            break;
        }
    }

    if start_cut >= stop_cut {
        return String::new();
    }
    chars[start_cut..stop_cut].iter().collect()
}

impl Decoder {
    /// Build a decoder from its JSON configuration.
    pub fn from_config(config: DecoderConfig) -> Result<Self, Error> {
        match config {
            DecoderConfig::ByteFallback => Ok(Self::ByteFallback(ByteFallbackDecoder)),
            DecoderConfig::ByteLevel => Ok(Self::ByteLevel(ByteLevelDecoder)),
            DecoderConfig::Replace { pattern, content } => Ok(Self::Replace(
                ReplaceDecoder::from_config(pattern, content)?,
            )),
            DecoderConfig::Strip {
                content,
                start,
                stop,
            } => Ok(Self::Strip {
                content,
                start,
                stop,
            }),
            DecoderConfig::Fuse => Ok(Self::Fuse),
            DecoderConfig::Sequence { decoders } => {
                let steps = decoders
                    .into_iter()
                    .map(Self::from_config)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            DecoderConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
            other => {
                let kind = DecoderKind::from(&other);
                Err(Error::Unsupported(kind.to_string()))
            }
        }
    }

    /// Apply this decoder step to a list of token strings, returning the
    /// transformed list.
    ///
    /// Follows HuggingFace's `decode_chain` semantics: each decoder step
    /// transforms the token list, and the final result is joined.
    pub fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>, Error> {
        match self {
            Self::ByteFallback(bf) => Ok(bf.decode_chain(tokens)),
            Self::ByteLevel(bl) => Ok(bl.decode_chain(tokens)),
            Self::Replace(repl) => Ok(repl.decode_chain(tokens)),
            Self::Fuse => Ok(vec![tokens.concat()]),
            Self::Strip {
                content,
                start,
                stop,
            } => Ok(tokens
                .into_iter()
                .map(|t| strip_token(&t, *content, *start, *stop))
                .collect()),
            Self::Sequence(steps) => {
                let mut current = tokens;
                for step in steps {
                    current = step.decode_chain(current)?;
                }
                Ok(current)
            }
        }
    }

    /// High-level decode: apply `decode_chain` then join.
    pub fn decode(&self, tokens: Vec<String>) -> Result<String, Error> {
        let result = self.decode_chain(tokens)?;
        Ok(result.join(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn toks(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn strip_token_removes_leading_and_trailing() {
        assert_eq!(strip_token("  hi ", ' ', 1, 0), " hi ");
        assert_eq!(strip_token("  hi ", ' ', 2, 1), "hi");
        assert_eq!(strip_token("hi", ' ', 1, 1), "hi");
        // Count-based: 3 spaces, strip one from each end -> one remains.
        assert_eq!(strip_token("   ", ' ', 1, 1), " ");
        // Strip more than present -> empty.
        assert_eq!(strip_token("  ", ' ', 3, 0), "");
    }

    /// The SentencePiece decode chain (Llama-2/Gemma-v1): `Fuse` must reduce the
    /// token list to one string *before* `Strip`, so only the single leading
    /// metaspace space is removed — not one per token.
    #[test]
    fn fuse_then_strip_matches_sentencepiece() {
        let dec = Decoder::from_config(DecoderConfig::Sequence {
            decoders: vec![
                DecoderConfig::Replace {
                    pattern: json!({ "String": "\u{2581}" }),
                    content: " ".to_string(),
                },
                DecoderConfig::Fuse,
                DecoderConfig::Strip {
                    content: ' ',
                    start: 1,
                    stop: 0,
                },
            ],
        })
        .unwrap();
        // "▁Hello" "▁world" -> " Hello world" -> "Hello world"
        assert_eq!(
            dec.decode(toks(&["\u{2581}Hello", "\u{2581}world"]))
                .unwrap(),
            "Hello world"
        );
    }

    #[test]
    fn fuse_alone_concatenates() {
        let dec = Decoder::from_config(DecoderConfig::Fuse).unwrap();
        assert_eq!(
            dec.decode_chain(toks(&["a", "b", "c"])).unwrap(),
            vec!["abc"]
        );
    }
}
