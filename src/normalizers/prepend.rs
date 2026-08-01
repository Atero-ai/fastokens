use std::borrow::Cow;

/// `Prepend` normalizer: prepends a fixed string to the (non-empty) input.
///
/// Mirrors HuggingFace `tokenizers.normalizers.Prepend`. SentencePiece-style
/// tokenizers (e.g. gemma) use `Prepend("▁")` so the first word receives the
/// same metaspace prefix as interior words. HF prepends only when the input is
/// non-empty; an empty input is left untouched.
#[derive(Debug)]
pub struct Prepend {
    prepend: String,
}

impl Prepend {
    /// Build a `Prepend` normalizer from its prefix string.
    pub fn new(prepend: String) -> Self {
        Self { prepend }
    }

    /// Apply the prepend to `input`.
    ///
    /// Returns `Cow::Borrowed` when the prefix is empty or the input is empty,
    /// avoiding allocation.
    pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        if self.prepend.is_empty() || input.is_empty() {
            Cow::Borrowed(input)
        } else {
            let mut out = String::with_capacity(self.prepend.len() + input.len());
            out.push_str(&self.prepend);
            out.push_str(input);
            Cow::Owned(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepends_metaspace() {
        let p = Prepend::new("\u{2581}".to_string());
        assert_eq!(p.normalize("hello"), "\u{2581}hello");
    }

    #[test]
    fn empty_input_unchanged() {
        let p = Prepend::new("\u{2581}".to_string());
        let out = p.normalize("");
        assert_eq!(out, "");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn empty_prefix_unchanged() {
        let p = Prepend::new(String::new());
        let out = p.normalize("hello");
        assert_eq!(out, "hello");
        assert!(matches!(out, Cow::Borrowed(_)));
    }
}
