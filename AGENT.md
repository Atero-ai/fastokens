## Context Hand-Off: Adapting `fastokens` for `GemmaTokenizer`

### 1. Objective and Current Status

* **The Goal:** Adapt `fastokens` to support `GemmaTokenizer` while keeping the large speedups seen on byte-level models like `GPT-OSS-120b`.
* **Correctness:** 100% parity with HuggingFace on `google/gemma-4-E2B` across ShareGPT52K (45,332 samples) and LongBench-v2 (503 samples).
* **Performance (measured, release build, after newline split + debug-print
  removal + ASCII fast-path):**
  * ShareGPT52K: **3.85x** — 19.82 MB/s vs HF 5.15 MB/s (was ~1.5x)
  * LongBench-v2: **8.61x** — 24.95 MB/s vs HF 2.90 MB/s (was ~1.5x)

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

### 3. The Fix That Worked: newline pre-splitting

Key insight from the llama.cpp Gemma4 PR:

```
regex_exprs = { "[^\\n]+|[\\n]+" };  // split ONLY on newlines
byte_encode = false;                  // raw UTF-8, not GPT-2 byte encoding
```

`split_on_newlines()` in `src/lib.rs` re-slices each text split at every transition
between newline and non-newline characters, so each resulting split is either a run
of `\n` or a run of non-`\n`. It runs in the non-fused encode path right after
pre-tokenization.

**Why it is output-preserving:** Gemma's BPE never merges across the
newline/non-newline boundary, so splitting there loses no merge. Consecutive
newlines stay grouped (the `[\n]+` run), so `\n\n` still merges into its own token
(id 108) — this was exactly the failure mode of the earlier "Attempt 3". It is a
**no-op for byte-level models**, whose buffers contain no raw `0x0A` byte after byte
encoding, so other models are unaffected.

**Why the earlier `▁`-splitting attempts failed** (still true — do not retry):
Gemma's vocab has cross-word merges like `>▁`, and `▁` merges rightward into the
following word. Splitting on `▁` severs those merges and breaks parity. Newline is
the only *trivially* safe boundary.

### 4. Other Applied Micro-Optimizations

* **Removed the debug `println!` in `run_merge_loop`** (`bpe.rs`). It fired on every
  cache-miss chunk; under rayon, `stdout`'s global lock **serialized all worker
  threads**. This was a major parallel bottleneck.
* **ASCII fast-path char→token lookup** (`bpe.rs`, `single_char_token: [u32; 128]`).
  The char-based merge engine previously did a HashMap probe per character; ASCII
  characters (the bulk of most text) now use a flat array lookup. Non-ASCII (incl.
  `▁`, U+2581) still uses the HashMap. Measured: small positive gain with parity
  intact (3.74→3.85x ShareGPT, 8.04→8.61x LongBench).

### 5. The Big Remaining Lever: vocab-bigram safe splitting

The structural ceiling today is that Gemma chunks are whole **lines**. Byte-level
30x models split at **word** level (GPT-2 regex) → tiny chunks → ~100% cache hits +
cheap per-chunk BPE. A long line (prose paragraph, minified JSON, long code line)
is still one big chunk with poor cache reuse.

**Idea — provably-safe fine-grained splitting via absent vocab bigrams:**

> A split between input bytes `x | y` can never be crossed by BPE if no vocabulary
> token contains the adjacent byte pair `xy`. Proof: any output token's byte string
> is a contiguous substring of the input and is itself in the vocab (all merges
> produce vocab tokens). If `x` and `y` ended up in the same token, that token would
> contain substring `xy` — contradiction. Splitting there is therefore
> output-preserving.

Build a `[bool; 256*256]` table `bridgeable[x*256+y]` = "some vocab token contains
adjacent bytes x,y". At encode time, insert a split at every position where
`!bridgeable[prev*256+cur]`. This generalizes the newline trick and would break long
lines into near-word-level chunks **for free, with provable parity**, likely closing
much of the gap to the byte-level models. Runtime cost is one flat-array lookup per
byte — far cheaper than the BPE it saves.

**Caveat to handle before shipping:** byte-fallback. Out-of-vocab characters are
emitted as `<0xXX>` tokens whose *strings* are not the raw bytes, so the bigram
argument needs to either (a) restrict split points to boundaries where both sides
are in-vocab single-char tokens, or (b) confirm Gemma's `<0xXX>` fallback tokens are
terminal (never merge). Validate with `./validate_model.sh google/gemma-4-E2B`
(it checks parity, not just speed).

### 6. Smaller Opportunities

* **Bypass `shared_cache` for low-reuse chunks.** On Gemma the cross-thread
  `SharedCache` (sharded `Mutex<HashMap<String, Vec<u32>>>`) is inserted on *every*
  miss with a `String` + `Vec` allocation and a lock. With mostly-unique line chunks
  under parallelism this is contention + allocation for near-zero hit rate. Consider
  gating shared-cache population by chunk length / observed hit rate, or relying on
  the thread-local `FlatCache` alone.
* **Bulk heapify in the encoded path.** `merge_all_raw_into` seeds initial pairs into
  `heap_buf` then bulk-extends; `merge_all_encoded_into` calls `init_merge_heap`.
  Verify the encoded path gets the same bulk-heapify treatment.
* **`▁` single-token fast path.** One HashMap probe per word is spent on the
  metaspace char; a dedicated cached id would remove it (minor).

### 7. Ground Truth / How to Validate

* `./validate_model.sh google/gemma-4-E2B` — parity + speed on ShareGPT52K and
  LongBench-v2. **Parity is the gate**; any splitting change must keep it green.
* `cargo test --lib --no-default-features` — offline unit tests (includes
  `newline_split_tests`).
* The 30x figure is a byte-level, word-split ceiling. For Gemma, §5 is the realistic
  path to substantially beyond 8x while preserving exact parity.
