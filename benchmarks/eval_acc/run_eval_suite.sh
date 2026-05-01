#!/usr/bin/env bash
# Run lm-eval accuracy suite for all four models with PTO megakernel backend.
#
# Compares PTO results against the Triton baseline for the same models.
# Results are written under OUT_DIR/<preset>/{pto_mega,triton}/eval.json.
#
# Usage:
#   export ASCEND_RT_VISIBLE_DEVICES=0
#   bash benchmarks/eval_acc/run_eval_suite.sh
#
# Options (env vars):
#   PRESETS    Space-separated preset names (default: all 4 models)
#   BACKENDS   Space-separated backends (default: "pto_mega triton")
#   OUT_DIR    Output directory (default: outputs/data/eval_<stamp>)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${SCRIPT_DIR}/run_lm_eval.py"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
PRESETS="${PRESETS:-qwen35_0_8b qwen35_9b qwen36_27b_w8a8 qwen36_35b_a3b_w8a8}"
BACKENDS="${BACKENDS:-pto_mega triton}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/data/eval_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"
echo "[eval_suite] output dir: $OUT_DIR"
echo "[eval_suite] presets:  $PRESETS"
echo "[eval_suite] backends: $BACKENDS"

for PRESET in $PRESETS; do
    # Tier settings: smaller models can afford larger contexts
    if [[ "$PRESET" == qwen35_0_8b || "$PRESET" == qwen35_9b ]]; then
        MAX_LEN=8192
        GPU_MEM=0.52
        BATCH=8
    else
        MAX_LEN=4096
        GPU_MEM=0.82
        BATCH=4
    fi
    echo "=== preset: $PRESET (max_len=$MAX_LEN gpu_mem=$GPU_MEM batch=$BATCH) ==="

    for BACKEND in $BACKENDS; do
        OUT="${OUT_DIR}/${PRESET}/${BACKEND}"
        mkdir -p "$OUT"
        echo "  [${PRESET}/${BACKEND}] running ..."
        python3 "$PY" \
            --preset "$PRESET" \
            --backend "$BACKEND" \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization "$GPU_MEM" \
            --max-batch-size "$BATCH" \
            --device "$ASCEND_RT_VISIBLE_DEVICES" \
            --output-json "${OUT}/eval.json" \
            2>&1 | tee "${OUT}/eval.log"
        echo "  [${PRESET}/${BACKEND}] done → ${OUT}/eval.json"
    done
done

echo "[eval_suite] all done. Results under ${OUT_DIR}/"
