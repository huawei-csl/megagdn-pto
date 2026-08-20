#!/usr/bin/env bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
#
# Compile one or more PTO kernels for A2/A3 (dav-2201, memory-based PTO) into
# shared libraries. Useful for development and debugging of individual kernels.
#
# Usage:
#   scripts/compile_kernel.sh <kernel> [<kernel> ...]
#
# Example:
#   scripts/compile_kernel.sh chunk_h        # -> build/lib/libkernel_chunk_h.so
#
# Environment variables:
#   ASCEND_TOOLKIT_HOME  Ascend toolkit install (required, set by set_env.sh)
#   PTO_LIB_PATH         PTO library prefix     (default: $ASCEND_TOOLKIT_HOME)
#   KERNEL_DIR           Kernel sources         (default: kernels/pto)
#   BUILD_DIR            Output directory       (default: build/lib)
#   BISHENG              Compiler binary        (default: bisheng)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PTO_LIB_PATH="${PTO_LIB_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
KERNEL_DIR="${KERNEL_DIR:-${REPO_ROOT}/kernels/pto}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/lib}"
BISHENG="${BISHENG:-bisheng}"

if [ "$#" -eq 0 ]; then
    echo "usage: $(basename "$0") <kernel> [<kernel> ...]" >&2
    exit 2
fi

if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    echo "error: ASCEND_TOOLKIT_HOME is not set; source the Ascend set_env.sh first" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

for kernel in "$@"; do
    # Accept both layouts: <kernel>.cpp (megagdn-pto) and kernel_<kernel>.cpp (pto-kernels).
    src="${KERNEL_DIR}/${kernel}.cpp"
    if [ ! -f "${src}" ]; then
        src="${KERNEL_DIR}/kernel_${kernel}.cpp"
    fi
    if [ ! -f "${src}" ]; then
        echo "error: no source found for kernel '${kernel}' in ${KERNEL_DIR}" >&2
        exit 1
    fi

    out="${BUILD_DIR}/libkernel_${kernel}.so"
    echo "[A2A3 PTO kernel compilation] ${src} -> ${out}"

    "${BISHENG}" -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 \
        -I"${KERNEL_DIR}" \
        -I"${KERNEL_DIR}/include" \
        -I"${PTO_LIB_PATH}/include" \
        -I"${ASCEND_TOOLKIT_HOME}/pkg_inc" \
        -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime" \
        -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/profiling" \
        --npu-arch=dav-2201 \
        -Wno-ignored-attributes \
        "${src}" \
        -o "${out}"
done
