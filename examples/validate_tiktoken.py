#!/usr/bin/env python3
"""Validate fastokens' tiktoken loader + encoder bit-for-bit against tiktoken.

For tiktoken models the reference tokenizer is `tiktoken` itself (not the
HuggingFace `tokenizers` library, which cannot load a bare `tiktoken.model`).
This exercises exactly the scanner + BPE encode path that the recent
performance work touches (o200k and Kimi pretokenizer families), so a mismatch
here means a regression in that path.

Usage:
    python examples/validate_tiktoken.py [MODEL ...]

MODEL is one of: o200k_base, cl100k_base, kimi   (default: o200k_base kimi)

Requires: tiktoken, fastokens, and (for `kimi`) huggingface_hub.
Exit status is non-zero if any model's output diverges from tiktoken.
"""
from __future__ import annotations

import base64
import random
import sys
import tempfile
import time

import tiktoken
from tiktoken.load import load_tiktoken_bpe

import fastokens

# The exact Kimi (moonshotai `tokenization_kimi.py`) pat_str.
KIMI_PATTERN = "|".join([
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
])

# ── Test corpus ──────────────────────────────────────────────────────────────

ADVERSARIAL = [
    "", " ", "\n", "\n\n", "  \n  \n", "\t\t\n",
    "hello world", " hello  world ", "Hello, World!",
    "don't can't they're we've I'm you'll he'd it's",
    "camelCaseIdentifier HTTPResponse XMLHttpRequest iPhone",
    "ALLCAPS Mixed lower UPPER123mix",
    "one two three four five six seven eight nine ten",
    "  leading and  double   spaces   between ",
    "trailing spaces at end     ",
    "line1\nline2\r\nline3\n\n\nline4",
    "def f(x, y): return x*y + 17  # comment\n    pass\n",
    "a=1; b=2.5; c=3e10; nums 12 345 6789 0 999999",
    "path/to/file.txt http://example.com/a/b?x=1&y=2",
    "!!!??? ...---___ (){}[]<> @#$%^&*",
    "你好世界 中文模型 日本語のテキスト 漢字とかな",
    "mixed 中English文 abc中文def ABC中文 中文ABC",
    "中'se 中's 中文's 123中文 中文123 1〇2 〇〇 中〇文",
    "emoji 😀🔥🎉 and symbols ™®©€£¥",
    "Ünïcödé àccénts café naïve résumé Zürich",
    "русский текст ελληνικά العربية עברית",
    "    nbsp and unicode ws",
    "'starting with apostrophe' 'quoted' \"double\"",
    "tab\tseparated\tvalues\there",
    "a" * 200, " " * 50 + "word", "word" + "!" * 40,
    "1234567890" * 10, "aB" * 60, "x\n" * 40,
]

# Character pools for random fuzzing, spanning the classes the scanner splits on.
POOLS = [
    list("abcdefghijklmnopqrstuvwxyz"),
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    list("0123456789"),
    list(" \t\n\r"),
    list(".,!?;:'\"()[]{}<>/\\|@#$%^&*-_=+~`"),
    list("éèêëàâäßñüöçøåÉÀÜ"),
    list("你好世界中文模型日本語漢字かなカナ"),
    list("〇〡〢㆒々〆㐀"),
    list("'s 't 're 've 'm 'll 'd"),  # contraction fragments
    list("😀🔥™®©€£¥  "),
]


def build_corpus(n_random: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    corpus = list(ADVERSARIAL)
    for _ in range(n_random):
        length = rng.randint(1, 60)
        s = []
        for _ in range(length):
            pool = rng.choice(POOLS)
            s.append(rng.choice(pool))
        corpus.append("".join(s))
    # A few large documents to exercise the parallel scanner and cache growth.
    words = ("the quick brown fox jumps over the lazy dog while 42 apples "
             "fall from a Tall Tree near the River. def compute(x): return x*2\n"
             "SECTION: values (a, b, c) -- 中文模型 don't stop 你好\n").split(" ")
    for mult in (2000, 8000):
        r = random.Random(seed + mult)
        corpus.append("".join(r.choice(words) + r.choice(" \n  ") for _ in range(mult)))
    return corpus


# ── Model loaders (fastokens + tiktoken reference) ───────────────────────────

def ranks_to_file(ranks: dict[bytes, int]) -> str:
    """Write mergeable ranks to a temporary `tiktoken.model` file."""
    fd, path = tempfile.mkstemp(suffix=".tiktoken")
    with open(fd, "w") as f:
        for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
            f.write(f"{base64.b64encode(token).decode()} {rank}\n")
    return path


def load_openai(name: str):
    """o200k_base / cl100k_base: reference from tiktoken, fastokens from the
    same ranks written to a file (using tiktoken's own pat_str + specials)."""
    ref = tiktoken.get_encoding(name)
    path = ranks_to_file(ref._mergeable_ranks)
    fast = fastokens.Tokenizer.from_tiktoken(
        path, pattern=ref._pat_str, special_tokens=dict(ref._special_tokens)
    )
    return fast, ref


def load_kimi():
    """Kimi-K2: download the tiktoken.model from the Hub; reference is a
    tiktoken.Encoding with the Kimi pat_str + reserved specials."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("moonshotai/Kimi-K2-Instruct", "tiktoken.model")
    ranks = load_tiktoken_bpe(path)
    nb = len(ranks)
    specials = {f"<|reserved_token_{i}|>": i for i in range(nb, nb + 256)}
    ref = tiktoken.Encoding(
        name="kimi-k2", pat_str=KIMI_PATTERN, mergeable_ranks=ranks, special_tokens=specials
    )
    fast = fastokens.Tokenizer.from_tiktoken(path, pattern=KIMI_PATTERN, special_tokens=specials)
    return fast, ref


LOADERS = {
    "o200k_base": load_openai,
    "cl100k_base": load_openai,
    "kimi": load_kimi,
}


# ── Validation ───────────────────────────────────────────────────────────────

def validate(name: str) -> bool:
    loader = LOADERS[name]
    fast, ref = (loader(name) if loader is load_openai else loader())
    corpus = build_corpus(n_random=5000, seed=0x5EED)

    fails = 0
    first = []
    t_fast = t_ref = 0.0
    for s in corpus:
        r0 = time.perf_counter()
        exp = ref.encode_ordinary(s)
        r1 = time.perf_counter()
        got = fast.encode(s).ids
        r2 = time.perf_counter()
        t_ref += r1 - r0
        t_fast += r2 - r1
        if got != exp:
            fails += 1
            if len(first) < 3:
                idx = next((i for i, (a, b) in enumerate(zip(got, exp)) if a != b), 0)
                first.append((s[:60], idx, got[idx:idx + 4], exp[idx:idx + 4]))

    total_tok = sum(len(ref.encode_ordinary(s)) for s in corpus)
    status = "OK" if fails == 0 else f"FAIL ({fails}/{len(corpus)})"
    print(f"  [{name:11s}] {len(corpus):5d} cases, {total_tok:>9d} tokens  "
          f"identical-to-tiktoken: {status}")
    for ctx, idx, g, e in first:
        print(f"      mismatch @tok {idx}: fastokens {g} != tiktoken {e}  ctx={ctx!r}")

    # Speed on a representative large document (single-string encode, i.e. the
    # per-request/serving path). tiktoken encodes one string on a single thread;
    # fastokens parallelizes within the document.
    doc = "".join(build_corpus(0, 0)[-1] for _ in range(4))  # ~a few MB of realistic text
    mb = len(doc.encode()) / 1e6
    fast.encode(doc); ref.encode_ordinary(doc)  # warm
    best = lambda f: min((lambda t: (f(), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(5))
    tf, tr = best(lambda: fast.encode(doc)), best(lambda: ref.encode_ordinary(doc))
    print(f"      speed ({mb:.1f} MB doc): fastokens {tf*1e3:.1f} ms vs tiktoken {tr*1e3:.1f} ms "
          f"-> {tr/tf:.1f}x")
    return fails == 0


def main() -> int:
    models = sys.argv[1:] or ["o200k_base", "kimi"]
    print(f"Validating fastokens vs tiktoken on: {', '.join(models)}")
    ok = True
    for m in models:
        if m not in LOADERS:
            print(f"  [{m}] unknown model (choose from {list(LOADERS)})")
            ok = False
            continue
        try:
            ok &= validate(m)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"  [{m}] ERROR: {type(e).__name__}: {e}")
            ok = False
    print("All tiktoken models identical to tiktoken." if ok else "TIKTOKEN VALIDATION FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
