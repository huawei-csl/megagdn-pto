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
#   SEQ_LENS="512 1024 ..."  base list (default ends at 65536 for large models)
#   WARMUP=2  REPEATS=10  OUT_DIR=outputs/data/prefill
#   WITH_PTO_STAGED=1  # also benchmark PTO staged (slower, optional)
#
# Automatically applied shorter lists (unless you bypass this script logic):
#   - qwen35_9b: lengths above 32768 are skipped (preset tops out before 65536 prompts).
#   - triton qwen36_27b_w8a8: 65536 is skipped only for this quantized model —
#       avoids ACL stream sync / device timeout on long Triton prefill.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${SCRIPT_DIR}/benchmark_prefill.py"

# shellcheck disable=SC2329
effective_seq_lens() {
    local model="$1"
    local case_kind="$2"
    local lens=""
    for L in ${SEQ_LENS}; do
        if [[ "${model}" == "qwen35_9b" && "${L}" -gt 32768 ]]; then
            continue
        fi
        # Long Triton prompts at 65536 tokens destabilised Qwen3.6-27B-w8a8; other models keep it.
        if [[ "${case_kind}" == "triton" && "${model}" == "qwen36_27b_w8a8" && "${L}" -eq 65536 ]]; then
            continue
        fi
        lens="${lens} ${L}"
    done
    echo "${lens# }"
}

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
echo "[prefill_sweep] base seq_lens (per model/case pruning may apply): ${SEQ_LENS}"
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
        RUN_SEQ="$(effective_seq_lens "${MODEL}" "${CASE}")"
        echo "  [${MODEL}] case=${CASE} seq_lens=${RUN_SEQ} ..."
        python3 "$PY" \
            --case "$CASE" \
            --model "$MODEL" \
            --seq-len ${RUN_SEQ} \
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
