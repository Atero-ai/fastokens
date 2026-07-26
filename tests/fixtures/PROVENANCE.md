# corpus_multilingual.txt — provenance & license

A ~2 MB differential-test corpus: real prose mixing English with eight
non-Latin scripts, used by the specialized split scanners' large-corpus
equivalence tests (`fast_split*`), which assert the hand-written scanner
produces byte-identical token spans to `fancy-regex` on the exact pattern.

The mix (≈75% English, ≈25% split across Han, kana, Hangul, Cyrillic, Arabic,
Devanagari, Thai, Greek, Hebrew) is deliberate: English exercises the
case/contraction/digit/punctuation logic that dominates serving traffic, and
the non-Latin scripts exercise the Han-run, mark, and class-boundary paths that
English cannot reach (notably the Kimi `[\p{Han}]+` and `&&[^\p{Han}]` rules).

## Sources

- **English documents** — OpenWebText (`Skylion007/openwebtext`), a public
  reproduction of the WebText corpus (web pages linked from Reddit). Sampled via
  the HuggingFace datasets-server rows API. OpenWebText is distributed for
  research without a single clear redistribution license; treat as
  web-scraped content.
- **Non-Latin documents** — Wikipedia (`wikimedia/wikipedia`, 2023-11-01
  snapshots for zh, ja, ko, ru, ar, hi, th, el, he), sampled via the same API.
  Wikipedia text is licensed **CC BY-SA 4.0**; reuse requires attribution to
  Wikipedia contributors and share-alike of the text itself.

Documents are separated by the `<|endoftext|>` marker (the common training
separator) and truncated per document. This file is a test fixture only; it is
not part of the shipped library.
