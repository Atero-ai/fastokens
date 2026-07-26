//! Cold-cache benchmark for single 200k- and 1M-token contexts.
//!
//! The input is the repository's multilingual corpus repeated to the requested
//! token count. A separate tokenizer calibrates input size; timed iterations
//! construct a fresh tokenizer before encoding so BPE caches start cold.
//!
//!   RAYON_NUM_THREADS=8 cargo run --release --example long_context_bench
//!   RAYON_NUM_THREADS=8 cargo run --release --example long_context_bench -- kimi-k2.5 5

use std::time::Instant;

fn tokenizer_json(name: &str) -> serde_json::Value {
    let repo = env!("CARGO_MANIFEST_DIR");
    let path = format!("{repo}/vendored_tokenizers/{name}/tokenizer.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("read tokenizer.json"))
        .expect("parse tokenizer.json")
}

fn build_context(tokenizer: &fastokens::Tokenizer, corpus: &str, target_tokens: usize) -> String {
    let corpus_tokens = tokenizer.encode(corpus).expect("calibrate corpus").len();
    let estimated_bytes = corpus
        .len()
        .saturating_mul(target_tokens)
        .div_ceil(corpus_tokens);
    let repeats = estimated_bytes.div_ceil(corpus.len()).max(1);
    let mut text = corpus.repeat(repeats);
    let mut cutoff = estimated_bytes.min(text.len());
    while !text.is_char_boundary(cutoff) {
        cutoff -= 1;
    }
    text.truncate(cutoff);
    text
}

fn percentile(sorted: &[f64], percentile: usize) -> f64 {
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn main() {
    let repo = env!("CARGO_MANIFEST_DIR");
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "kimi-k2.5".to_string());
    let rounds = std::env::args()
        .nth(2)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(5usize);
    let raw_corpus =
        std::fs::read_to_string(format!("{repo}/tests/fixtures/corpus_multilingual.txt"))
            .expect("read multilingual corpus");
    let corpus = raw_corpus.replace("<|endoftext|>", " ");
    let json = tokenizer_json(&name);
    let calibration =
        fastokens::Tokenizer::from_json(json.clone()).expect("construct calibration tokenizer");

    println!(
        "{name}, RAYON_NUM_THREADS={}, {rounds} cold rounds",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string())
    );
    for target in [200_000usize, 1_000_000] {
        let text = build_context(&calibration, &corpus, target);
        let actual_tokens = calibration.encode(&text).expect("count tokens").len();
        let mib = text.len() as f64 / (1 << 20) as f64;
        let mut times = Vec::with_capacity(rounds);

        for _ in 0..rounds {
            let tokenizer =
                fastokens::Tokenizer::from_json(json.clone()).expect("construct tokenizer");
            let start = Instant::now();
            let ids = tokenizer.encode(&text).expect("encode context");
            times.push(start.elapsed().as_secs_f64() * 1e3);
            assert_eq!(ids.len(), actual_tokens);
        }

        times.sort_by(f64::total_cmp);
        let median = percentile(&times, 50);
        let p90 = percentile(&times, 90);
        println!(
            "{target:>9} target | {actual_tokens:>9} actual | {mib:>6.2} MiB | \
             p50 {median:>8.2} ms | p90 {p90:>8.2} ms | {throughput:>6.1} MiB/s",
            throughput = mib / (median / 1e3),
        );
    }
}
