#!/usr/bin/env bash
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_MODELS=(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    "deepseek-ai/DeepSeek-V3.2"
    "MiniMaxAI/MiniMax-M2.1"
    "openai/gpt-oss-120b"
    "mistralai/Mistral-Nemo-Instruct-2407"
)
DEFAULT_DATASETS=(
    "zai-org/LongBench-v2"
    "RyokoAI/ShareGPT52K"
)
DEFAULT_BATCH_SIZES=("seq" "1" "8" "32" "128")
DEFAULT_MAX_SAMPLES=50
# Tiktoken models are validated for output identicalness against `tiktoken`
# (the reference for tiktoken models — HuggingFace `tokenizers` cannot load a
# bare `tiktoken.model`). This exercises the o200k / Kimi scanner encode path.
DEFAULT_TIKTOKEN_MODELS=("o200k_base" "kimi")
PYTHON="${PYTHON:-python3}"

# ── Parse arguments ───────────────────────────────────────────────────
MODELS=()
DATASETS=()
BATCH_SIZES=()
TIKTOKEN_MODELS=()
MAX_SAMPLES="$DEFAULT_MAX_SAMPLES"
OUTPUT_DIR=""
TIKTOKEN_ONLY=0
RUN_TIKTOKEN=1
TIKTOKEN_FAILED=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run simple_bench across models, datasets, and batch sizes.

Options:
  --models MODELS        Comma-separated model list (default: 5 representative models)
  --datasets DATASETS    Comma-separated dataset list (default: LongBench-v2,ShareGPT52K)
  --batch-sizes SIZES    Comma-separated batch sizes; use "seq" for sequential (default: seq,1,8,32,128)
  -n, --max-samples N    Max samples per run (default: $DEFAULT_MAX_SAMPLES)
  --tiktoken-models M    Comma-separated tiktoken models to validate vs tiktoken
                         (default: o200k_base,kimi; choices: o200k_base,cl100k_base,kimi)
  --tiktoken-only        Only run the tiktoken identicalness check (skip HF matrix)
  --no-tiktoken          Skip the tiktoken identicalness check
  --output-dir DIR       Output directory (default: ablation_results_TIMESTAMP)
  -h, --help             Show this help

Environment:
  PYTHON                 Python interpreter for the tiktoken check (default: python3;
                         must have fastokens + tiktoken installed, huggingface_hub for kimi)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)     IFS=',' read -ra MODELS <<< "$2"; shift 2 ;;
        --datasets)   IFS=',' read -ra DATASETS <<< "$2"; shift 2 ;;
        --batch-sizes) IFS=',' read -ra BATCH_SIZES <<< "$2"; shift 2 ;;
        --tiktoken-models) IFS=',' read -ra TIKTOKEN_MODELS <<< "$2"; shift 2 ;;
        --tiktoken-only) TIKTOKEN_ONLY=1; shift ;;
        --no-tiktoken)   RUN_TIKTOKEN=0; shift ;;
        -n|--max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

[[ ${#MODELS[@]} -eq 0 ]]          && MODELS=("${DEFAULT_MODELS[@]}")
[[ ${#DATASETS[@]} -eq 0 ]]        && DATASETS=("${DEFAULT_DATASETS[@]}")
[[ ${#BATCH_SIZES[@]} -eq 0 ]]     && BATCH_SIZES=("${DEFAULT_BATCH_SIZES[@]}")
[[ ${#TIKTOKEN_MODELS[@]} -eq 0 ]] && TIKTOKEN_MODELS=("${DEFAULT_TIKTOKEN_MODELS[@]}")

# ── Tiktoken identicalness check (fastokens vs tiktoken) ──────────────
# tiktoken models can't be loaded by HuggingFace `tokenizers`, so their
# reference is `tiktoken`. Run before the HF matrix so a scanner/encode-path
# regression is caught fast.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_tiktoken_validation() {
    echo "Tiktoken identicalness — fastokens vs tiktoken"
    echo "  models: ${TIKTOKEN_MODELS[*]}"
    echo "══════════════════════════════════════════════════"
    if ! "$PYTHON" -c "import fastokens, tiktoken" 2>/dev/null; then
        echo "  SKIPPED: $PYTHON lacks fastokens+tiktoken (build with 'maturin develop --release'," \
             "'pip install tiktoken huggingface_hub'); set PYTHON=<interp> to choose one."
        return
    fi
    if "$PYTHON" "$SCRIPT_DIR/validate_tiktoken.py" "${TIKTOKEN_MODELS[@]}"; then
        echo "  -> PASS: all tiktoken models identical to tiktoken"
    else
        TIKTOKEN_FAILED=1
        echo "  -> FAIL: a tiktoken model diverged (see output above)"
    fi
    echo
}

if [[ $RUN_TIKTOKEN -eq 1 ]]; then
    run_tiktoken_validation
fi
if [[ $TIKTOKEN_ONLY -eq 1 ]]; then
    exit $TIKTOKEN_FAILED
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="ablation_results_$(date +%Y%m%d_%H%M%S)"
fi

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#BATCH_SIZES[@]} ))

# ── Print configuration ──────────────────────────────────────────────
echo "Ablation Benchmark"
echo "══════════════════════════════════════════════════"
echo "  Models:       ${MODELS[*]}"
echo "  Datasets:     ${DATASETS[*]}"
echo "  Batch sizes:  ${BATCH_SIZES[*]}"
echo "  Max samples:  $MAX_SAMPLES"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Total runs:   $TOTAL"
echo "══════════════════════════════════════════════════"
echo

# ── Build ─────────────────────────────────────────────────────────────
echo "Building simple_bench (release)..."
cargo build --example simple_bench --release 2>&1
echo

BENCH_BIN="target/release/examples/simple_bench"
if [[ ! -x "$BENCH_BIN" ]]; then
    echo "ERROR: $BENCH_BIN not found after build" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Combined CSV header ──────────────────────────────────────────────
COMBINED_CSV="$OUTPUT_DIR/combined.csv"
echo "model,dataset,batch_size,input_index,input_char_len,output_token_len,hf_duration_ms,fastokens_duration_ms" \
    > "$COMBINED_CSV"

# ── Run matrix ────────────────────────────────────────────────────────
run_idx=0
failed=0

# Short name for filenames (replace / with -)
short_name() { echo "$1" | tr '/' '-'; }

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            run_idx=$((run_idx + 1))

            tag="$(short_name "$model")_$(short_name "$dataset")_bs${bs}"
            csv_file="$OUTPUT_DIR/${tag}.csv"
            log_file="$OUTPUT_DIR/${tag}.log"

            echo "[$run_idx/$TOTAL] model=$model dataset=$dataset batch=$bs"

            # Build command
            cmd=("$BENCH_BIN" "$model" --dataset "$dataset" -n "$MAX_SAMPLES" -o "$csv_file")
            if [[ "$bs" != "seq" ]]; then
                cmd+=(-b "$bs")
            fi

            # Run and capture output
            if "${cmd[@]}" > "$log_file" 2>&1; then
                # Append per-row CSV data with model/dataset/batch_size prefix
                if [[ -f "$csv_file" ]]; then
                    tail -n +2 "$csv_file" | while IFS= read -r line; do
                        echo "$model,$dataset,$bs,$line" >> "$COMBINED_CSV"
                    done
                fi
                # Extract speedup from log
                speedup=$(awk '/Speedup:/{gsub(/x/,"",$NF); print $NF}' "$log_file" 2>/dev/null)
                speedup="${speedup:-?}"
                echo "  -> done (speedup: ${speedup}x)"
            else
                failed=$((failed + 1))
                echo "  -> FAILED (see $log_file)"
            fi
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────
echo
echo "══════════════════════════════════════════════════"
echo "  Ablation complete: $run_idx runs, $failed failed"
echo "  Combined CSV:  $COMBINED_CSV"
echo "  Per-run logs:  $OUTPUT_DIR/*.log"
echo "  Per-run CSVs:  $OUTPUT_DIR/*.csv"
echo "══════════════════════════════════════════════════"

# ── Summary table from logs ───────────────────────────────────────────
echo
printf "%-50s %-20s %-6s %10s %10s %8s\n" "MODEL" "DATASET" "BATCH" "HF(ms)" "FT(ms)" "SPEEDUP"
printf '%.0s─' {1..110}; echo

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            tag="$(short_name "$model")_$(short_name "$dataset")_bs${bs}"
            log_file="$OUTPUT_DIR/${tag}.log"
            if [[ -f "$log_file" ]]; then
                hf_total=$(awk '/HF total:/{print $(NF-1)}' "$log_file" 2>/dev/null)
                ft_total=$(awk '/fastokens total:/{print $(NF-1)}' "$log_file" 2>/dev/null)
                speedup=$(awk '/Speedup:/{gsub(/x/,"",$NF); print $NF}' "$log_file" 2>/dev/null)
                hf_total="${hf_total:--}"
                ft_total="${ft_total:--}"
                speedup="${speedup:--}"
                # Shorten model/dataset for display
                short_model="${model##*/}"
                short_dataset="${dataset##*/}"
                printf "%-50s %-20s %-6s %10s %10s %8s\n" \
                    "$short_model" "$short_dataset" "$bs" "$hf_total" "$ft_total" "${speedup}x"
            fi
        done
    done
done

if [[ $RUN_TIKTOKEN -eq 1 ]]; then
    if [[ $TIKTOKEN_FAILED -eq 0 ]]; then
        echo "  Tiktoken identicalness: PASS (${TIKTOKEN_MODELS[*]})"
    else
        echo "  Tiktoken identicalness: FAIL"
    fi
fi

if [[ $failed -gt 0 || $TIKTOKEN_FAILED -gt 0 ]]; then
    exit 1
fi
