## Context Hand-Off: Adapting `fastokens` for `GemmaTokenizer`

### 1. Objective and Current Status

* **The Goal:** Adapt the `fastokens` library to support `GemmaTokenizer` while maintaining the massive 6x–33x speedups seen with models like `GPT-OSS-120b`.
* **The Additions:** Added the `Replace` normalizer, `ByteFallback` decoder, and `byte_fallback=True` initialization in the BPE engine.
* **Current Status:** Tokenization is 100% mathematically accurate but only achieves a **1.5x to 1.6x speedup**.

### 2. The Core Bottleneck: Priority Queue Thrashing

Through debugging (adding a counter to `run_merge_loop`), we discovered that the time is being lost in the BPE priority-queue heap.

* **GPT-OSS:** Processes chunks in **10–60 iterations**.
* **Gemma:** Processes chunks in **5,000–9,000 iterations**.
* **The Cause:** `fastokens` achieves its insane speed by relying on a word-level cache (`TL_BPE_CACHE`). For Gemma, this cache is being completely starved, forcing the tokenizer to process entire documents through the slow BPE priority-queue fallback.

### 3. The Architectural Root Cause: The Normalizer Quirk

The reason the cache is starved lies in a quirk of how Hugging Face configured Gemma:

1. **The Normalizer:** Replaces all spaces (`" "`) with a replacement character (`"▁"`).
2. **The Pre-Tokenizer:** Is configured to split on spaces (`pattern=" "`, `behavior="merged_with_previous"`).
3. **The Flaw:** Because the normalizer runs *before* the pre-tokenizer, all spaces are destroyed before the pre-tokenizer can see them. The pre-tokenizer finds zero matches and acts as a complete no-op.
4. **The Result:** A 10,000-word document is passed to the BPE loop as a single, massive, unbroken string. The `TL_BPE_CACHE` never hits, and the BPE priority queue suffers $O(N \log N)$ thrashing.

### 4. Attempted Mitigations and Why They Failed

We attempted to artificially chunk the string before it hit the BPE engine to feed the cache, but ran into strict token-parity failures due to Gemma's BPE training data (which includes a lot of code and HTML).

* **Attempt 1: Greedy Prefix Matching (in `bpe.rs`)**
* *Approach:* Scan ahead up to 16 characters to group existing vocab tokens directly, bypassing the heap.
* *Why it failed:* BPE is rank-based, not length-based. Greedy matching stole characters that were supposed to be merged across boundaries based on higher BPE ranks.


* **Attempt 2: `split_inclusive('▁')` (in `lib.rs`)**
* *Approach:* Emulate Hugging Face's intended pre-tokenizer behavior by splitting on `▁`, leaving the character at the *end* of the chunk (e.g., `"[human]: ▁"`).
* *Why it failed:* It separated `▁` from the word that followed it (e.g., `root`), preventing necessary cross-boundary merges like `▁` + `root` -> `▁root` (token `5989`).


* **Attempt 3: Custom Split *Before* `▁` and `\n**`
* *Approach:* Split the string right before `▁` to keep it attached to the target word, and split after `\n` to keep chunks small.
* *Why it failed:* Splitting unconditionally on `\n` separated consecutive newlines, preventing the BPE from merging `\n` + `\n` into the double-newline token (token `108`).


* **Attempt 4: Strict Split *Before* `▁` Only**
* *Approach:* Removed newline splitting; split strictly right before `▁` if it preceded a word.
* *Why it failed:* **Cross-boundary code/HTML merges.** In HTML strings (e.g., `</p>▁Op`), slicing the string between `>` and `▁` built a wall between the two characters. Gemma's vocabulary contains highly ranked cross-boundary tokens like `>▁`. The artificial boundary prevented these exact merges, failing the parity tests.



### 5. Key Takeaways for the Next Agent

1. **Do not attempt manual string chunking:** Gemma's BPE vocabulary relies heavily on cross-boundary merges (punctuation + space, tags + space). Any artificial string slicing will inevitably break tokenization parity.
2. **The 1.5x Speedup is accurate:** The current implementation is outperforming Hugging Face's raw BPE loop by 50%. The lack of a 30x speedup is a structural reality of Gemma passing massive strings to the BPE engine, not a bug in the Rust port.
3. **Future Optimization Vectors:** If optimization continues, it must happen *inside* `run_merge_loop` or `init_merge_heap` in a way that respects mathematical BPE ranks.
* *Idea:* Pre-filtering unmergeable byte-fallback sequences before pushing them to the priority queue to reduce stale-entry heap bloat.
* *Idea:* Optimizing the memory layout or stale-entry cleanup logic of the BinaryHeap itself.