# Tokenizing million-token prompts up to 6.9x faster than tiktoken

Long-context inference starts with a CPU problem: the model cannot process a
prompt until the tokenizer has turned all of it into token IDs. At 200,000
tokens that work is noticeable. At one million tokens it is directly on the
time-to-first-token path.

The next fastokens release reduces that latency in two layers:

1. The BPE merge loop does less allocation and less synchronization.
2. On cold, long o200k-family inputs, equal pretokens go to the same worker so
   one worker computes a repeated word once and then serves it from its local
   cache.

Together, these changes make fastokens **4.9x faster at 200k tokens and 6.9x
faster at one million tokens than tiktoken** when using eight CPU cores. On one
core, the same implementation is 1.3x and 1.9x faster respectively.

## Results

We compared the public Python APIs using tiktoken 0.13.0 and the checked-in
gpt-oss o200k tokenizer. Both tokenizers received the same multilingual text,
and the benchmark asserted that every output token ID was identical before
recording a result.

| CPU allocation | Tokens | fastokens | tiktoken | Speedup |
|---|---:|---:|---:|---:|
| 1 core | 204,056 | 54.67 ms | 71.63 ms | 1.31x |
| 1 core | 1,022,785 | 149.44 ms | 287.75 ms | 1.93x |
| 8 cores | 204,056 | 13.19 ms | 64.09 ms | 4.86x |
| 8 cores | 1,022,785 | 37.65 ms | 261.43 ms | 6.94x |

These are median encode times on an Intel Xeon Platinum 8480+ host. Physical
cores were pinned with `taskset`; `RAYON_NUM_THREADS` fixed the fastokens worker
count. Tokenizer loading was excluded. Every measured fastokens encode used a
new tokenizer instance, so its BPE caches began cold. The first iteration was
discarded, the remaining call order alternated, and tiktoken used
`encode_ordinary` because the corpus contains no special tokens.

The benchmark is checked in as
[`examples/tiktoken_long_context.py`](examples/tiktoken_long_context.py):

```bash
uv pip install --python .venv/bin/python 'tiktoken>=0.13.0'
.venv/bin/maturin develop --release

taskset -c 0 env RAYON_NUM_THREADS=1 \
  .venv/bin/python examples/tiktoken_long_context.py

taskset -c 0,2,4,6,8,10,12,14 env RAYON_NUM_THREADS=8 \
  .venv/bin/python examples/tiktoken_long_context.py
```

## Making cold BPE cheaper

Most pretokens are short. The old merge path still allocated heap storage and
used the same machinery as an unusually long piece of text. Short pretokens
now use a stack-resident merge representation and a linear minimum-rank scan.
The general heap path remains the fallback above 32 bytes.

The first merge round also uses a dense byte-pair table, avoiding a hash lookup
for the most common case. Finally, the shared BPE cache is now bounded and
sharded, with flat storage and allocation-free inserts. That keeps
synchronization local and prevents a long-running process from accumulating an
unbounded cache.

In our Rust cold-context harness, this first layer alone reduced encode latency
by 15.6% at roughly 200k Kimi tokens and 20.3% at one million.

## Scheduling repeated work once

Long prompts repeat small pretokens constantly: whitespace, punctuation,
common words, and code fragments. Contiguous parallel chunks can send the same
text to several workers, which makes each thread's cold local cache repeat the
same BPE work.

For the first long encode, fastokens now hashes each pretoken into a worker
bucket. Equal text therefore reaches one worker and becomes a local-cache hit
after its first occurrence. Each bucket records output boundaries, and a final
linear pass restores the original token order without allocating one vector
per pretoken.

This scheduler is deliberately narrow. It activates only when all of these are
true:

- The tokenizer uses an exactly recognized Kimi or o200k split pattern.
- The input is at least 64 KiB.
- This is the tokenizer instance's first eligible encode.

Everything else uses the existing contiguous scheduler. Limiting affinity to
the first cold pass matters: grouping by content helps an empty cache, but it
can disrupt locality once the normal caches are warm. With the one-shot gate,
the existing Kimi warm benchmark is unchanged within noise.

On top of the BPE improvements, affinity reduced cold Kimi latency by another
13.0% at roughly 200k tokens and 11.5% at one million. Relative to the current
main branch, the combined reduction was 26.6% and 29.4%.

## What we did not ship

Two attractive shortcuts failed the benchmark and stayed out:

- Returning a vocabulary ID whenever an entire pretoken exists in the vocab is
  not generally correct. Some vocabulary entries are unreachable through the
  configured merge graph.
- Pre-seeding the cache from the vocabulary increased tokenizer load cost and
  did not improve steady encode latency enough to justify it.

The useful pattern, also visible in recent fastokens work and in gigatoken, is
specialization behind a proof-sized gate: recognize an exact tokenizer family,
remove work from its hot path, and retain the general implementation as the
fallback.
