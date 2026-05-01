#!/usr/bin/env bash
# Sweep prefill TTFT: Triton vs PTO-megakernel across models.
#
# Writes JSONL files under OUT_DIR/<model_label>/{triton,pto_mega}.jsonl.
# PTO-staged (pto) is skipped by default; set WITH_PTO_STAGED=1 to include it.
#
# Usage:
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   bash benchmarks/vllm_prefill/run_prefill_sweep.sh
#
# Customise:
#   MODELS="qwen35_0_8b qwen35_9b qwen36_27b_w8a8 qwen36_35b_a3b_w8a8"
#   SEQ_LENS="512 1024 2048 4096 8192 16384 32768"
#   WARMUP=2  REPEATS=10  OUT_DIR=outputs/data/prefill
#   WITH_PTO_STAGED=1  # also benchmark PTO staged (slower, optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${SCRIPT_DIR}/benchmark_prefill.py"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-2}"
REPEATS="${REPEATS:-10}"
SEQ_LENS="${SEQ_LENS:-512 1024 2048 4096 8192 16384 32768 65536}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/data/prefill_$(date +%Y%m%d_%H%M%S)}"

# Model specs: preset_name:quantization (use "" for none)
# Presets resolve to paths defined in benchmark_prefill.py
declare -A MODEL_QUANT
MODEL_QUANT=(
    [qwen35_0_8b]=""
    [qwen35_9b]=""
    [qwen36_27b_w8a8]="ascend"
    [qwen36_35b_a3b_w8a8]="ascend"
)
MODELS="${MODELS:-qwen35_0_8b qwen35_9b qwen36_27b_w8a8 qwen36_35b_a3b_w8a8}"
WITH_PTO_STAGED="${WITH_PTO_STAGED:-0}"

mkdir -p "$OUT_DIR"
echo "[prefill_sweep] output dir:      $OUT_DIR"
echo "[prefill_sweep] models:          $MODELS"
echo "[prefill_sweep] seq_lens:        $SEQ_LENS"
echo "[prefill_sweep] with_pto_staged: ${WITH_PTO_STAGED} (set WITH_PTO_STAGED=1 to enable)"

# Build the list of cases to run
CASES="pto_mega triton"
if [[ "${WITH_PTO_STAGED}" == "1" ]]; then
    CASES="pto_mega pto triton"
fi

for MODEL in $MODELS; do
    QUANT="${MODEL_QUANT[$MODEL]:-}"
    SUB="${OUT_DIR}/${MODEL}"
    mkdir -p "$SUB"
    echo "=== model: $MODEL ==="

    EXTRA=()
    if [[ -n "$QUANT" ]]; then
        EXTRA+=(--quantization "$QUANT")
    fi

    for CASE in $CASES; do
        echo "  [${MODEL}] case=${CASE} ..."
        python3 "$PY" \
            --case "$CASE" \
            --model "$MODEL" \
            --seq-len $SEQ_LENS \
            --warmup "$WARMUP" \
            --repeats "$REPEATS" \
            --device "$ASCEND_RT_VISIBLE_DEVICES" \
            "${EXTRA[@]}" \
            --output-jsonl "${SUB}/${CASE}.jsonl" \
            2>&1 | tee "${SUB}/${CASE}.log"
        echo "  [${MODEL}/${CASE}] done → ${SUB}/${CASE}.jsonl"
    done
done

echo "[prefill_sweep] all done. Results under ${OUT_DIR}/"
echo "[prefill_sweep] plot: python scripts/plot_results.py --prefill-dir ${OUT_DIR}"
