"""Compare cold-cache fastokens and tiktoken on long o200k contexts.

The benchmark asserts identical token IDs before timing. Tokenizer construction
is excluded, and each fastokens round uses a new tokenizer so its BPE caches
start cold.

    taskset -c 0 env RAYON_NUM_THREADS=1 \
        .venv/bin/python examples/tiktoken_long_context.py
    taskset -c 0,2,4,6,8,10,12,14 env RAYON_NUM_THREADS=8 \
        .venv/bin/python examples/tiktoken_long_context.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import fastokens
import tiktoken

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = REPO / "vendored_tokenizers/gpt-oss-120b/tokenizer.json"
DEFAULT_CORPUS = REPO / "tests/fixtures/corpus_multilingual.txt"


def build_context(
    corpus_bytes: bytes,
    corpus_tokens: int,
    target_tokens: int,
) -> str:
    estimated_bytes = (
        len(corpus_bytes) * target_tokens + corpus_tokens - 1
    ) // corpus_tokens
    repeats = (estimated_bytes + len(corpus_bytes) - 1) // len(corpus_bytes)
    text_bytes = (corpus_bytes * repeats)[:estimated_bytes]
    while True:
        try:
            return text_bytes.decode()
        except UnicodeDecodeError:
            text_bytes = text_bytes[:-1]


def timed_encode(callable_encode, text: str) -> tuple[object, float]:
    start = perf_counter()
    output = callable_encode(text)
    return output, (perf_counter() - start) * 1_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    reference = tiktoken.get_encoding("o200k_base")
    tokenizer_json = args.tokenizer.read_text()
    corpus = args.corpus.read_text().replace("<|endoftext|>", " ")
    corpus_bytes = corpus.encode()
    corpus_tokens = len(reference.encode_ordinary(corpus))
    threads = os.environ.get("RAYON_NUM_THREADS", "default")

    print(
        f"tiktoken {tiktoken.__version__}, RAYON_NUM_THREADS={threads}, "
        f"{args.rounds} measured rounds"
    )
    for target in (200_000, 1_000_000):
        text = build_context(corpus_bytes, corpus_tokens, target)
        expected = reference.encode_ordinary(text)
        actual = fastokens.Tokenizer.from_json_str(tokenizer_json).encode(text).ids
        if actual != expected:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(actual, expected))
                    if left != right
                ),
                min(len(actual), len(expected)),
            )
            raise AssertionError(
                f"token IDs differ at {mismatch}: "
                f"fastokens={len(actual)}, tiktoken={len(expected)}"
            )

        fastokens_times: list[float] = []
        tiktoken_times: list[float] = []
        for round_index in range(args.rounds + 1):
            tokenizer = fastokens.Tokenizer.from_json_str(tokenizer_json)
            if round_index % 2:
                reference_ids, tiktoken_ms = timed_encode(
                    reference.encode_ordinary, text
                )
                encoding, fastokens_ms = timed_encode(tokenizer.encode, text)
            else:
                encoding, fastokens_ms = timed_encode(tokenizer.encode, text)
                reference_ids, tiktoken_ms = timed_encode(
                    reference.encode_ordinary, text
                )
            if len(encoding) != len(reference_ids):
                raise AssertionError("timed encodes produced different token counts")
            if round_index:
                fastokens_times.append(fastokens_ms)
                tiktoken_times.append(tiktoken_ms)

        fastokens_ms = median(fastokens_times)
        tiktoken_ms = median(tiktoken_times)
        print(
            f"{target:>9} target | {len(expected):>9} actual | "
            f"fastokens {fastokens_ms:>7.2f} ms | "
            f"tiktoken {tiktoken_ms:>7.2f} ms | "
            f"{tiktoken_ms / fastokens_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
