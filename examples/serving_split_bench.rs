//! Long-context, single-request encode benchmark — the serving latency that
//! matters (10k+ char prompts, ms scale).
//!
//! Each iteration encodes a DISTINCT context so the PCRE2 tier's incremental
//! split cache (keyed on shared prefix with the previous call) never hits —
//! that cache helps streaming/growing sequences, not independent requests, so
//! measuring repeated identical input overstates it. Runs single-threaded
//! (`RAYON_NUM_THREADS=1`) to model a loaded server where each concurrent
//! request effectively gets one core.
//!
//! A/B by checking out `main` (PCRE2 tier) vs this branch (scanner):
//!   RAYON_NUM_THREADS=1 cargo run --release --example serving_split_bench

use std::time::Instant;

fn build_context(chars: usize, seed: u64) -> String {
    let mut rng = seed | 1;
    let words = [
        "the",
        "tokenizer",
        "performance",
        "streaming",
        "request",
        "context",
        "attention",
        "under",
        "load",
        "server",
        "batch",
        "and",
        "for",
        "with",
        "It's",
        "don't",
        "2024",
        "1.5e-7",
        "GPU",
        "throughput,",
        "(latency)",
        "https://example.test/path",
        "fn",
        "encode()",
        "->",
        "Vec<u32>",
        "0x7f",
        "многоязычный",
        "文脈",
        "コンテキスト",
    ];
    let mut s = String::with_capacity(chars + 32);
    while s.chars().count() < chars {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push_str(words[(rng >> 33) as usize % words.len()]);
        s.push(if rng & 0xF == 0 { '\n' } else { ' ' });
    }
    s
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "vendored_tokenizers/qwen3-8b/tokenizer.json".to_string());
    let raw = std::fs::read_to_string(&path).expect("read tokenizer.json");
    let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let tok = fastokens::Tokenizer::from_json(value).unwrap();

    for chars in [10_000usize, 50_000, 200_000] {
        // Pool of distinct contexts; rotate so no two consecutive encodes
        // share a prefix (defeats the incremental split cache).
        let pool: Vec<String> = (0..16).map(|i| build_context(chars, 0x1234 + i)).collect();
        for c in &pool {
            let _ = tok.encode(c);
        }
        let iters = (40_000_000 / chars).max(64);
        let mut times = Vec::with_capacity(iters);
        for i in 0..iters {
            let text = &pool[i % pool.len()];
            let t = Instant::now();
            std::hint::black_box(tok.encode(text).unwrap());
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        let bytes = pool[0].len();
        println!(
            "{:>7} chars ({:>8} bytes): {:7.3} ms  ({:6.1} MiB/s)",
            chars,
            bytes,
            median,
            bytes as f64 / (1 << 20) as f64 / (median / 1e3)
        );
    }
}
