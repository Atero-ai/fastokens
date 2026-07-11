## Context Hand-Off: Adapting `fastokens` for `GemmaTokenizer`

### 1. Objective and Current Status

* **The Goal:** Adapt `fastokens` to support `GemmaTokenizer` while keeping the large speedups seen on byte-level models like `GPT-OSS-120b`.
* **Correctness:** 100% parity with HuggingFace on `google/gemma-4-E2B` across ShareGPT52K (45,332 samples) and LongBench-v2 (503 samples).
* **Performance (measured, release build, vocab-bigram splitting):**
  * ShareGPT52K: **8.92x** — 48.52 MB/s vs HF 5.44 MB/s
  * LongBench-v2: **27.88x** — 82.24 MB/s vs HF 2.95 MB/s
  * **Previous (newline split):** 3.85x / 8.61x
  * **Improvement:** 2.3x on ShareGPT, 3.2x on LongBench over newline splitting

### 2. The Root Cause (recap)

Gemma is an SPM/metaspace pipeline:

1. **Normalizer** (`Replace`) rewrites every space `" "` → `"▁"`.
2. **Pre-tokenizer** is `Split(pattern=" ", behavior=MergedWithPrevious)`.

Because the normalizer runs *before* the pre-tokenizer, all spaces are gone by the
time the space-`Split` runs, so it matches nothing and is a **no-op**. The entire
document reaches the BPE engine as one giant chunk: the word-level cache
(`TL_BPE_CACHE`) never hits, the ≥16-split rayon parallelism never triggers, and
the priority-queue BPE runs `O(N log N)` over the whole document.

The BPE math itself is char-based (`merge_all_encoded_into`), using raw UTF-8 with
`<0xXX>` byte-fallback — i.e. `byte_encode = false`, exactly as llama.cpp specifies
for Gemma. That part was already correct; the problem was purely chunking.

### 3. Evolution: From Newline to Vocab-Aware Splitting

**Initial approach (deprecated):** Newline pre-splitting was inspired by llama.cpp's
Gemma4 PR (https://github.com/ggml-org/llama.cpp/pull/21343), which used
`regex_exprs = { "[^\\n]+|[\\n]+" }`. This provided 3.85x/8.61x speedup by creating
per-line chunks.

**Current approach:** Vocab-aware bigram splitting (see section 5) **replaced** newline
splitting entirely. It's more generic, model-agnostic, and delivers superior
performance (8.92x/27.88x). The newline approach was removed from the codebase as it
was strictly inferior.

**Why the earlier `▁`-splitting attempts failed** (still true — do not retry):
Gemma's vocab has cross-word merges like `>▁`, and `▁` merges rightward into the
following word. Splitting on `▁` severs those merges and breaks parity.

### 4. Other Applied Micro-Optimizations

* **Removed the debug `println!` in `run_merge_loop`** (`bpe.rs`). It fired on every
  cache-miss chunk; under rayon, `stdout`'s global lock **serialized all worker
  threads**. This was a major parallel bottleneck.
* **ASCII fast-path char→token lookup** (`bpe.rs`, `single_char_token: [u32; 128]`).
  The char-based merge engine previously did a HashMap probe per character; ASCII
  characters (the bulk of most text) now use a flat array lookup. Non-ASCII (incl.
  `▁`, U+2581) still uses the HashMap. Measured: small positive gain with parity
  intact (3.74→3.85x ShareGPT, 8.04→8.61x LongBench).

### 5. Vocab-Bigram Safe Splitting (CURRENT IMPLEMENTATION ✅)

**Status:** Shipped, validated, and now the **only** splitting strategy. Newline splitting
has been removed from the codebase.

**The insight:** A split between input bytes `x | y` can never be crossed by BPE if
no vocabulary token contains the adjacent byte pair `xy`. Proof: any output token's
byte string is a contiguous substring of the input and is itself in the vocab (all
merges produce vocab tokens). If `x` and `y` ended up in the same token, that token
would contain substring `xy` — contradiction. Splitting there is therefore
output-preserving.

**Implementation** (`src/models/bpe.rs`, `src/lib.rs`):

1. **`BigramBridgeTable`** — 64 KB lookup table `[bool; 256*256]` built at
   initialization by scanning all vocab token byte strings. Marks which byte pairs
   appear in any vocab token.

2. **Diagnostic output** — On first Bpe initialization, reports split potential:
   ```
   Bigram bridge table: 12070 bridgeable pairs (18.4%), 53466 unbridgeable (81.6%)
   ```
   For Gemma, **81.6% of byte pairs are unbridgeable** — huge splitting opportunity.

3. **`split_on_unbridgeable_bigrams()`** — Scans input, splits at every unbridgeable
   byte pair that's also a UTF-8 character boundary (checking `(cur & 0xC0) != 0x80`).
   Runtime cost: one array lookup + one bitwise op per byte.

4. **Byte-fallback tokens** — Confirmed these are literal strings like `"<0x00>"`
   (containing angle brackets and hex digits), not the actual byte values. The bigram
   scan naturally handles them; no special logic needed.

**Results:**
- ShareGPT52K: **8.92x** (up from 3.85x with newline splitting)
- LongBench-v2: **27.88x** (up from 8.61x with newline splitting)
- 100% parity maintained on both datasets

**Why it works:** The algorithm is model-agnostic — it adapts to each tokenizer's
vocabulary automatically. For Gemma specifically, after normalization (spaces → `▁`),
most word boundaries become unbridgeable. Long lines break into multi-word chunks →
per-chunk cache hits + rayon parallelism. The LongBench speedup (27.88x) is
particularly strong because those samples have very long lines that benefit most from
fine-grained splitting.

**Generality:** This approach works for **all BPE models** without model-specific logic.
Each model's bigram bridge table is computed at initialization by scanning its
vocabulary, so the split points automatically adapt to that model's merge patterns.

### 6. Further Optimization Opportunities

At **8.92x / 27.88x**, Gemma is now competitive with byte-level models on realistic
workloads. The LongBench result is approaching the theoretical byte-level ceiling.
Remaining micro-optimizations have diminishing returns but are documented for
completeness:

* **Pre-computed const table for Gemma.** The bigram table is deterministic per vocab.
  For production, generate it at build time and embed as `const GEMMA_BIGRAM_TABLE:
  [bool; 65536]`. Zero initialization cost, same results.

* **Bypass `shared_cache` for low-reuse chunks.** The cross-thread `SharedCache`
  (sharded `Mutex<HashMap<String, Vec<u32>>>`) is inserted on *every* miss with a
  `String` + `Vec` allocation and a lock. Consider gating shared-cache population by
  chunk length / observed hit rate, or relying on the thread-local `FlatCache` alone.

* **Bulk heapify in the encoded path.** `merge_all_raw_into` seeds initial pairs into
  `heap_buf` then bulk-extends; `merge_all_encoded_into` calls `init_merge_heap`.
  Verify the encoded path gets the same bulk-heapify treatment.

* **`▁` single-token fast path.** One HashMap probe per word is spent on the metaspace
  char; a dedicated cached id would remove it (minor).

### 7. Ground Truth / How to Validate

* `cd examples && ./validate_model.sh google/gemma-4-E2B` — parity + speed on
  ShareGPT52K and LongBench-v2. **Parity is the gate**; any splitting change must
  keep it green.
* `cargo test --lib --no-default-features` — offline unit tests (vocab-aware splitting
  is tested via the full correctness test suite across multiple models).
* The 30x figure is a byte-level, word-split ceiling on short-document benchmarks.
  For Gemma on **long documents** (LongBench), we've achieved **27.88x** — near the
  theoretical limit. On mixed workloads (ShareGPT), **8.92x** is strong and leaves
  limited headroom without structural changes (e.g., rewriting the BPE engine).
