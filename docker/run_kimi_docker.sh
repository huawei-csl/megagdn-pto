#!/usr/bin/env bash
# Run KimiLinear-48B evaluation inside vllm-ascend with PTO KDA megakernel.
#
# Env:
#   KIMI_IMAGE       — override image (default v0.18.0rc1)
#   KIMI_MODEL_DIR   — host path to model weights (default /scratch); mounted
#                      read-only at /scratch inside the container, so the
#                      hardcoded /scratch/model_weights/... preset paths still
#                      resolve regardless of the host location
#   KIMI_NO_MOUNTS=1 — skip the v0.18-specific file overrides (use for v0.19+)
#
# v0.18 mounts (vendored upstream overrides in docker/overrides/):
#   model_runner_v1.py — fixes isinstance crash + MLA hybrid-block reshape
#   worker_init.py     — worker/__init__.py with PTO hook pre-injected
#   kv_cache_utils.py  — debug prints (harmless, safe to keep)
#
# Usage (run from the repo root):
#   bash docker/run_kimi_docker.sh [CMD...]
#   bash docker/run_kimi_docker.sh python tests/test_kimi_load.py
#   bash docker/run_kimi_docker.sh python benchmarks/eval_acc/run_lm_eval.py \
#       --preset kimi_linear_48b --backend kda_mega

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${KIMI_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0rc1}"
MODEL_DIR="${KIMI_MODEL_DIR:-/scratch}"

# Ascend devices for TP=4 (NPUs 0-3)
DEVICES=(
    --device=/dev/davinci0
    --device=/dev/davinci1
    --device=/dev/davinci2
    --device=/dev/davinci3
    --device=/dev/davinci_manager
    --device=/dev/hisi_hdc
)

MOUNTS=(
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware
    -v /var/queue_schedule:/var/queue_schedule
    -v "${MODEL_DIR}:/scratch:ro"
    -v "${REPO_ROOT}:/sources"
)
if [ -z "${KIMI_NO_MOUNTS:-}" ]; then
    MOUNTS+=(
        -v "${REPO_ROOT}/docker/overrides/model_runner_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro"
        -v "${REPO_ROOT}/docker/overrides/worker_init.py:/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/__init__.py:ro"
        -v "${REPO_ROOT}/docker/overrides/kv_cache_utils.py:/vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py:ro"
    )
fi
if [ -n "${KIMI_V19_HOOK:-}" ]; then
    MOUNTS+=(
        -v "${REPO_ROOT}/docker/overrides/worker_init_v19.py:/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/__init__.py:ro"
    )
fi

# Allocate a TTY only when run interactively, so background/CI runs work too.
TTY_FLAGS=()
if [ -t 0 ]; then
    TTY_FLAGS=(-it)
fi

docker run "${TTY_FLAGS[@]}" --rm --privileged --network=host --ipc=host --shm-size=32g \
    "${DEVICES[@]}" \
    "${MOUNTS[@]}" \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -w /sources \
    "${IMAGE}" \
    "${@:-bash}"
