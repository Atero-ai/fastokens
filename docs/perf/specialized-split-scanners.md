# Specialized split scanners

Hand-written pre-tokenization scanners for split regexes known ahead of time,
dispatched by byte-for-byte pattern match at `Split` construction
(`src/pre_tokenizers/fast_split.rs`). Technique adapted from
[gigatoken](https://github.com/marcelroed/gigatoken) (MIT): first-byte
dispatch + SWAR ASCII-letter runs + a packed 2-bit-per-codepoint Unicode class
table, no regex engine.

Tier order in `Split::pre_tokenize` (fastest first):
specialized scanner → PCRE2 JIT (parallel + incremental cache) → fancy-regex.
The scanner activates only on an exact recognized pattern with `Isolated`
behavior and `invert == false`; everything else is untouched.

## Measured (2026-07-25, EPYC 208-core box, `taskset`-pinned, Qwen3)

The benchmark that matters for serving is **long-context, per-request, cold
cache**: each iteration encodes a *distinct* context so the PCRE2 tier's
incremental split cache (keyed on shared prefix with the previous call) never
hits. That cache helps streaming/growing sequences, not independent requests —
re-encoding one fixed string, as an earlier draft of this benchmark did,
overstates PCRE2 by making its split nearly free and hides the real per-request
cost. Reproduce with `examples/serving_split_bench.rs`, A/B by checking out
`main` (PCRE2) vs this branch (scanner).

Full `encode()`, median, distinct inputs:

| context | scanner (8 core) | PCRE2 (8 core) | scanner (1 core) | PCRE2 (1 core) |
|---------|-----------------:|---------------:|-----------------:|---------------:|
| 10k ch  | 0.11 ms          | 0.20 ms (1.8x) | 0.079 ms         | 0.170 ms (2.15x) |
| 50k ch  | 0.29 ms          | 0.59 ms (2.0x) | 0.39 ms          | *panic\**      |
| 200k ch | 0.9 ms           | 1.8 ms (1.9x)  | 1.54 ms          | *panic\**      |

**~1.8–2.15x faster full encode across 10k–200k char prompts**, single- or
multi-threaded. The gain is per-core split throughput; at full encode the BPE
stage is shared, so this is a split-bound-workload / serving-latency win, not a
bulk-throughput one. On a single re-encoded string (warm incremental cache) the
two tiers tie — that regime is not representative of independent requests.

\* PCRE2's parallel authority-zone path (`split.rs`) panics on >16 KiB inputs
under `RAYON_NUM_THREADS=1` — a pre-existing bug on `main`, unrelated to this
change. The scanner replaces that path for recognized patterns, so Qwen3
sidesteps it; other patterns still hit it.

### Kimi (o200k-family) — `fast_split_o200k.rs`

The Kimi (moonshotai K2) pattern uses `[\p{Han}]+` runs and `&&[^\p{Han}]`
class intersections that PCRE2 can't parse, so the PCRE2 tier runs a
hand-rewritten pattern with a per-codepoint `(?!\p{Han})` negative lookahead on
every letter-bracket char — slow on both ASCII and CJK. The scanner classifies
via one packed table load. A/B, distinct inputs, full `encode()` on the
vendored Kimi tokenizer:

| workload    | scanner  | PCRE2    | speedup |
|-------------|---------:|---------:|---------|
| ASCII 10k   | 0.095 ms | 0.215 ms | ~2.2x   |
| ASCII 50k   | 0.22 ms  | 0.61 ms  | ~2.7x   |
| CJK 10k     | 0.14 ms  | 0.33 ms  | ~2.3x   |
| CJK 50k     | 0.26 ms  | 0.74 ms  | ~2.8x   |

Enabling Kimi also lets the hand-maintained PCRE2 intersection-rewrite special
case in `split.rs` eventually retire (once o200k/Nemotron are ported too — they
share `fast_split_o200k`'s parameterized `advance_pos`).

## Correctness

Differential vs `fancy-regex` on the exact pattern (fancy-regex handles the
`&&` intersections, so it is a faithful oracle): Qwen2 has 60+ hand-picked edge
cases + a 4000-round fuzz; Kimi has 30+ cases (camelCase phase automaton,
`HTTPResponse`, contractions incl. U+017F, Han runs, Han numerals U+3007, Han
symbols/marks U+16FF0, mixed CJK/Latin, combining marks) + a 6000-round fuzz
over a CJK-heavy pool. Both have an integration test asserting the scanner path
— including the parallel chunked path — equals the regex path through
`Split::pre_tokenize`. Parallel chunk boundaries are placed only after a `\n`
followed by a non-whitespace ASCII byte, a point no alternative can cross.

### Qwen2N3 (GLM / dolma2 / OLMo)

Same as Qwen2 but with `\p{N}{1,3}` number runs. Enabled. Full `encode()` on
the vendored GLM-5 tokenizer, distinct inputs, A/B vs PCRE2:

| context | scanner  | PCRE2    | speedup |
|---------|---------:|---------:|---------|
| 10k ch  | 0.085 ms | 0.170 ms | ~2.0x   |
| 50k ch  | 0.315 ms | 0.651 ms | ~2.1x   |
| 200k ch | 0.926 ms | 2.098 ms | ~2.3x   |

An earlier draft gated this off after a same-text-4 MiB microbenchmark showed
the GLM scanner ~5x slower than the Qwen2 scanner. That was a **first-touch /
page-warming artifact of re-encoding one fixed buffer**: swapping the A/B order
reversed which scheme was "slow", and on distinct inputs (real serving) the two
scanners are within noise in both the sequential and parallel paths. No
scheme-specific slowdown exists.

## Status

Every vendored tokenizer is now scanner-backed, one scheme each:

- **`Qwen2`** (Qwen2/Qwen3 and derivatives): **enabled**.
- **`Qwen2N3`** (GLM / dolma2 / OLMo, `\p{N}{1,3}`): **enabled**.
- **`Kimi`** (moonshotai K2 family, o200k-family with Han runs): **enabled**.
  Also catches **knext / Kimi K3**, whose tiktoken `pat_str` is byte-identical
  and reaches `from_pattern` verbatim through the tiktoken→tokenizer.json
  conversion (`python/fastokens/tiktoken.py`) — a `recognizes_only_known_patterns`
  assertion pins that exact string so an upstream regex tweak fails loudly
  instead of silently falling back to PCRE2.
- **`O200k`** (o200k_base: GPT-4o, gpt-oss): **enabled**. Same walker as Kimi
  with `HAN=false` and a `[\r\n/]*` punct tail (`SLASH=true`).

Recognition is byte-exact: a model whose split regex differs by one character
silently uses PCRE2 (safe — correctness is never at risk — but unaccelerated).

## Next steps (ranked)

1. Nemotron-3 scheme (`fast_split_o200k::advance_pos::<false,false,true,false>`)
   — a pattern const + one dispatch line + fuzz. With o200k done this also lets
   the hand-maintained PCRE2 intersection-rewrite special case in `split.rs`
   retire once nothing else needs it.
2. Fix the PCRE2 `RAYON_NUM_THREADS=1` panic independently (affects all
   non-specialized patterns).
3. Later scalar/SIMD tiers from gigatoken once split is the bottleneck:
   dual-cursor ILP (+25% measured by them), NEON/AVX-512 mask scanner.
