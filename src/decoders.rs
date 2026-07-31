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
    Sequence(Vec<Decoder>),
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
            DecoderConfig::Fuse => Ok(Self::Sequence(vec![])), // identity/no-op
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
