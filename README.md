# ⚡ fastokens

fastokens is a fast [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer for use with
popular open-weight LLMs, built on top of a high-performance Rust backend. It loads both the
HuggingFace `tokenizer.json` format and [tiktoken](https://github.com/openai/tiktoken) model files.

`fastokens` can be installed from source:
```
git clone https://github.com/atero-ai/fast-tokens
uv pip install fast-tokens/python
```

The Python API lives in the `python` directory. To use `fastokens` as a drop-in replacement with
[transformers](https://github.com/huggingface/transformers), or with [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo), see the
[usage examples](#usage) below.

## Performance

`fastokens` on average achieves a 10x+ faster tokenization compared to the `tokenizers` library.
The gap widens as prompt sizes scale, as shown in the graphs below.

![OSS Speedup on various processors](assets/speedup_oss.png)

![Average Speedup](assets/speedup_average.png)

Faster tokenization directly impacts live workloads. Tested using SGLang's benchmark suite, `fastokens` reduces time-to-first-token (TTFT) across prompt sizes:

![TTFT P50 comparison](assets/ttft_p50.png)

Note that `fastokens` is focused on inference and does not support all features of `tokenizers`.
In particular, additional encoding outputs, and some normalizers/pretokenizers are not available.

## Tested models

The following models have been tested, but `fastokens` should generally work with most BPE tokenizers supported by the `transformers` library, including:

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `openai/gpt-oss-120b`
- `deepseek-ai/DeepSeek-V3.2`
- `deepseek-ai/DeepSeek-V3`
- `deepseek-ai/DeepSeek-R1`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `Qwen/Qwen3-Next-80B-A3B-Instruct`
- `Qwen/Qwen3-235B-A22B-Instruct-2507`
- `Qwen/Qwen3.5-397B-A17B`
- `MiniMaxAI/MiniMax-M2.1`
- `MiniMaxAI/MiniMax-M2.5`
- `mistralai/Devstral-Small-2-24B-Instruct-2512`
- `zai-org/GLM-4.7`
- `zai-org/GLM-5`


## Usage

### Using with transformers

Supports transformers v4 (e.g. 4.57.1 used by current sglang) and v5+ (e.g. 5.3.0).

```python
import fastokens
fastokens.patch_transformers()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
tokens = tokenizer("Hello, world!")
assert tokens["input_ids"] == [22177, 1044, 4304, 1033]
```

### Standalone usage

```python
from fastokens._native import Tokenizer
tokenizer = Tokenizer.from_model("deepseek-ai/DeepSeek-V3.2")
tokens = tokenizer.encode("A very long prompt that is now lightning fast.")
```

### Loading a tiktoken model

`fastokens` can also load [tiktoken](https://github.com/openai/tiktoken)
model files (`tiktoken.model`, or OpenAI's `.tiktoken` files) in addition to
`tokenizer.json`.

A tiktoken model file only contains the byte-level BPE ranks — the
pre-tokenization regex (`pat_str`) and the special tokens live in companion
code, so they are supplied separately. For OpenAI's standard encodings, pass
`encoding=` to use the built-in defaults:

```python
from fastokens import Tokenizer

# cl100k_base (GPT-3.5 / GPT-4) or o200k_base (GPT-4o):
tok = Tokenizer.from_tiktoken("cl100k_base.tiktoken", encoding="cl100k_base")
tok.encode("Hello, world!").ids  # matches tiktoken's encode_ordinary
```

For any other model that ships a `tiktoken.model` (e.g. Kimi-K2), pass the
model's own pattern and special tokens explicitly:

```python
tok = Tokenizer.from_tiktoken(
    "tiktoken.model",
    pattern=r"...the model's pat_str...",
    special_tokens={"<|im_end|>": 163842, "<|im_user|>": 163843},
)
```

The same is available in Rust via `Tokenizer::from_tiktoken_file`,
`from_tiktoken_str`, and `from_tiktoken_ranks`, with `TiktokenConfig::cl100k_base()`
/ `o200k_base()` presets. Special tokens are treated like HuggingFace added
tokens (split out before the model, skippable on decode).

For the `o200k` and Kimi pattern families, pre-tokenization uses a hand-written,
parallelized Unicode scanner instead of a regex engine (its classification
tables are built from the same `regex-syntax` data the reference matcher uses,
so results are identical). This makes single-document ("1M context")
tokenization several times faster than the regex path on those models.

### Prefix cache (shared system prompts)

For serving workloads where many requests share a long prefix — a common system
prompt or a large shared context — an opt-in prefix cache tokenizes the shared
prefix once and reuses its token ids, tokenizing only each request's unique
tail. Enable it with `FASTOKENS_INPUT_CACHE=<capacity>` (number of recent
prefixes to retain) or `Tokenizer::enable_input_cache(capacity)` in Rust; it is
off by default. Reuse cuts are only ever made at hard pretoken boundaries, so
results are bit-identical to tokenizing from scratch. On a ~1M-token shared
prefix this takes per-request encoding from ~2.9 ms to ~0.6 ms; an exact repeat
reuses the whole encoding.

### PCRE2 resource limits

PCRE2 resource limits can be set when constructing a tokenizer to guard against
pathological regex/input combinations in `tokenizer.json`:

```python
from fastokens._native import Tokenizer

tokenizer = Tokenizer.from_model(
    "deepseek-ai/DeepSeek-V3.2",
    pcre2_match_limit=1_000_000,
)
```

If a limit is reached during pre-tokenization, encoding returns an error instead
of continuing an expensive regex match. The same keyword arguments are accepted
by `Tokenizer(...)`, `Tokenizer.from_file(...)`, `Tokenizer.from_json_str(...)`,
and `fastokens.patch_transformers(...)`.

### Dynamo usage

`fastokens` is integrated with NVIDIA Dynamo's frontend, and can be used by passing the flag `--tokenizer fastokens` to the latest version (either build from source or wait for the official release, coming in the next few days).

## Acknowledgements

This library builds on the well-known and widely used Hugging Face tokenizers library and uses code written for HF tokenizers in several flows.
