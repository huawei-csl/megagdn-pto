#!/usr/bin/env python3
"""Benchmark PTO chunk-GDN kernels vs FLA Triton baseline.

Measures latency (ms) for all six computation stages:
  chunk_cumsum, scaled_dot_kkt, solve_tril, wy_fast, chunk_h, chunk_o,
  and the fused mega_kernel.

PTO always uses chunk_size=128 (C_PTO=128).
Each Triton stage is timed at both BT=64 and BT=128 where applicable:
  - chunk_cumsum : Triton BT=64 and BT=128 both supported
  - scaled_dot_kkt: Triton BT=64 (default); BT=128 may fail for some H
  - solve_tril   : Triton BT=64 only (BT=128 not supported by Triton). PTO time is only
                   the ``tri_inverse`` ``lib.call_kernel`` launch (buffers + stream ptr
                   fixed outside the timed region; excludes ``solve_tril`` wrapper and fp16 ``copy_``).
  - wy_fast      : Triton BT=64 and BT=128 both supported
  - chunk_h      : Triton BT=64 and BT=128 both supported
  - chunk_o      : Triton BT=64 (default); BT=128 may fail for H=64

Results are printed to stdout and optionally saved to ``--output-json``.

Usage::

    python benchmarks/kernel/bench_gdn_kernels.py --device npu:0
    python benchmarks/kernel/bench_gdn_kernels.py --device npu:0 --H-list 32,64
    python benchmarks/kernel/bench_gdn_kernels.py --device npu:0 --mega
    python benchmarks/kernel/bench_gdn_kernels.py --device npu:0 --mega \\
        --l-seg 8192 --output-json outputs/data/kernel_bench_L8192.json

Environment (``--n-seq`` / ``--l-seg`` override these when passed):
    GDN_NPU_DEVICE    Default NPU device (default: npu:0)
    GDN_BENCH_N_SEQ   Number of sequences (default: 16)
    GDN_BENCH_L_SEG   Tokens per sequence (default: 16384)

    Large ``T = N_seq * L_seg`` makes the Triton ``solve_tril`` launch exceed the
    Ascend runtime grid limit (≤65536 blocks for ``NT × H``, where NT is roughly
    the total number of 64-token blocks across sequences). Prefer ``--l-seg 8192``
    (Triton parity for smaller H; see README) or ``--l-seg 4096`` when you need a
    larger head count together with all six Triton stages.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

# Add triton baseline to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_TRITON_BASELINE = os.path.join(_REPO_ROOT, "kernels", "triton_baseline")
if _TRITON_BASELINE not in sys.path:
    sys.path.insert(0, _TRITON_BASELINE)

from megagdn_pto.compile import BLOCK_DIM
from megagdn_pto.fast_inverse import launch_tri_inverse_kernel, load_tri_inverse, solve_tril
from megagdn_pto.kernel_libs import (
    load_chunk_cumsum,
    load_chunk_h,
    load_chunk_o,
    load_scaled_dot_kkt,
    load_wy_fast,
    precomputed_minus_identity,
    total_chunks,
    transpose_beta,
    transpose_gates,
)
from megagdn_pto.mega_kernel import run_mega_kernel

import ctypes

C_PTO = 128
D = 128


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _bench_npu(fn, warmup: int = 5, iters: int = 15) -> float:
    """Time an NPU function using Event pairs (ms)."""
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8).npu()
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    for i in range(iters):
        cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def _bench_triton(fn, warmup: int = 5, iters: int = 15) -> float:
    """Time a Triton NPU function (uses event.synchronize() for correctness)."""
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8).npu()
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        cache.zero_()
        torch.npu.synchronize()
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return sum(times) / len(times)


def _vp(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _ratio(ms_triton: float | None, ms_pto: float) -> str:
    return f"{ms_triton / ms_pto:.2f}x" if ms_triton and ms_pto > 0 else "—"


# ---------------------------------------------------------------------------
# Stage benchmarks
# ---------------------------------------------------------------------------

def _try_triton_cumsum(cu_seqlens, BT, dev, T, H) -> float | None:
    try:
        from fla_vendor.cumsum import chunk_local_cumsum
        from fla_vendor.utils import prepare_chunk_indices
        cu_long = cu_seqlens.long()
        g = torch.randn(1, T, H, device=dev, dtype=torch.float32)
        fn = lambda: chunk_local_cumsum(g=g, chunk_size=BT, cu_seqlens=cu_long)
        fn(); torch.npu.synchronize()
        return _bench_triton(fn)
    except Exception as exc:
        print(f"    [Triton cumsum BT={BT}: {str(exc).split(chr(10))[0][:80]}]")
        gc.collect()
        return None


def _try_triton_kkt(cu_seqlens, BT, dev, T, H, HG) -> float | None:
    try:
        from fla_vendor.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from fla_vendor.utils import prepare_chunk_indices
        cu_long = cu_seqlens.long()
        chunk_indices = prepare_chunk_indices(cu_long, BT)
        k = torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16)
        beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
        g = torch.randn(1, T, H, device=dev, dtype=torch.float32)
        fn = lambda: chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g, cu_seqlens=cu_long,
                                               chunk_indices=chunk_indices, chunk_size=BT,
                                               output_dtype=torch.float32)
        fn(); torch.npu.synchronize()
        return _bench_triton(fn)
    except Exception as exc:
        print(f"    [Triton kkt BT={BT}: {str(exc).split(chr(10))[0][:80]}]")
        gc.collect()
        return None


def _try_triton_solve_tril(cu_seqlens, BT, dev, T, H) -> float | None:
    if BT > 64:
        print(f"    [Triton solve_tril BT={BT}: not supported by Triton (max BT=64)]")
        return None
    # Triton solve_tril requires NT * H ≤ ~65536 (NT ≈ Σ ceil(Lᵢ / BT) merge chunks).
    # Do NOT set TRITON_ALL_BLOCKS_PARALLEL here — it corrupts later Triton timings.
    try:
        from fla_vendor.solve_tril import solve_tril as triton_solve_tril
        cu_long = cu_seqlens.long()
        A = torch.randn(1, T, H, BT, device=dev, dtype=torch.float32).tril(-1)
        fn = lambda: triton_solve_tril(A=A, cu_seqlens=cu_long)
        fn(); torch.npu.synchronize()
        return _bench_triton(fn)
    except Exception as exc:
        msg = str(exc).split(chr(10))[0][:80]
        if "grid" in msg or "65536" in msg:
            print(f"    [Triton solve_tril BT={BT}: grid exceeds 65536 (T={T}, H too large for this L_seg)]")
        else:
            print(f"    [Triton solve_tril BT={BT}: {msg}]")
        gc.collect()
        return None


def _try_triton_chunk_h(cu_seqlens, BT, dev, T, H, HG) -> float | None:
    # H=64 with BT=64 triggers an aicore exception on this NPU, corrupting device state.
    if H >= 64 and BT <= 64:
        print(f"    [Triton chunk_h BT={BT} H={H}: known aicore failure, skip]")
        return None
    try:
        from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
        cu_long = cu_seqlens.long()
        CI = prepare_chunk_indices(cu_long, BT)
        CO = prepare_chunk_offsets(cu_long, BT)
        k_tr = torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16)
        w_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        u_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)
        fn = lambda: chunk_gated_delta_rule_fwd_h(k=k_tr, w=w_tr, u=u_tr, g=g_tr,
                                                   initial_state=None, output_final_state=False,
                                                   cu_seqlens=cu_long, chunk_indices=CI,
                                                   chunk_offsets=CO, chunk_size=BT)
        fn(); torch.npu.synchronize()
        return _bench_triton(fn)
    except Exception as exc:
        print(f"    [Triton chunk_h BT={BT}: {str(exc).split(chr(10))[0][:80]}]")
        gc.collect()
        return None


def _try_triton_wy_fast(cu_seqlens, BT, dev, T, H, HG) -> float | None:
    try:
        from fla_vendor.wy_fast import recompute_w_u_fwd
        from fla_vendor.utils import prepare_chunk_indices
        cu_long = cu_seqlens.long()
        CI = prepare_chunk_indices(cu_long, BT)
        k_tr = torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16)
        v_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        beta_tr = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
        A_tr = torch.randn(1, T, H, BT, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)
        fn = lambda: recompute_w_u_fwd(k=k_tr, v=v_tr, beta=beta_tr, g_cumsum=g_tr,
                                        A=A_tr, cu_seqlens=cu_long, chunk_indices=CI)
        fn(); torch.npu.synchronize()
        return _bench_triton(fn)
    except Exception as exc:
        print(f"    [Triton wy_fast BT={BT}: {str(exc).split(chr(10))[0][:80]}]")
        gc.collect()
        return None


def _try_triton_chunk_o(cu_seqlens, BT, dev, T, H, HG) -> float | None:
    # H=64 with BT=64 is a known aicore failure; skip to protect NPU state.
    if H >= 64 and BT <= 64:
        print(f"    [Triton chunk_o BT={BT} H={H}: known aicore failure, skip]")
        return None
    try:
        from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla_vendor.chunk_o import chunk_fwd_o
        from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
        cu_long = cu_seqlens.long()
        CI = prepare_chunk_indices(cu_long, BT)
        CO = prepare_chunk_offsets(cu_long, BT)
        scale = D ** -0.5
        q_tr = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16), dim=-1, p=2)
        k_tr = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16), dim=-1, p=2)
        w_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        u_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)
        h_tr, v_new_tr, _ = chunk_gated_delta_rule_fwd_h(
            k=k_tr, w=w_tr, u=u_tr, g=g_tr, initial_state=None, output_final_state=False,
            cu_seqlens=cu_long, chunk_indices=CI, chunk_offsets=CO, chunk_size=BT)
        torch.npu.synchronize()
        fn = lambda: chunk_fwd_o(q=q_tr, k=k_tr, v=v_new_tr, h=h_tr, g=g_tr,
                                  scale=scale, cu_seqlens=cu_long, chunk_size=BT)
        fn(); torch.npu.synchronize()
        return _bench_triton(fn)
    except Exception as exc:
        print(f"    [Triton chunk_o BT={BT}: {str(exc).split(chr(10))[0][:80]}]")
        gc.collect()
        return None


def _print_stage(name, ms_pto, ms_t64, ms_t128):
    sp64 = _ratio(ms_t64, ms_pto)
    sp128 = _ratio(ms_t128, ms_pto)
    print(f"\n  {name}  (PTO C={C_PTO})")
    print(f"    PTO        : {ms_pto:.2f} ms")
    print(f"    Triton BT=64 : {ms_t64:.2f} ms  → {sp64}" if ms_t64 else
          f"    Triton BT=64 : fail")
    print(f"    Triton BT=128: {ms_t128:.2f} ms  → {sp128}" if ms_t128 else
          f"    Triton BT=128: n/a")


def bench_chunk_cumsum(H, T, cu_seqlens, dev, stream, bd):
    lib = load_chunk_cumsum(H, D, C_PTO)
    g = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_sum = torch.empty_like(g)
    batch = len(cu_seqlens) - 1
    cu32 = cu_seqlens.to(torch.int32)

    def run_pto():
        lib.call_kernel(bd, stream, _vp(g), _vp(g_sum), _vp(cu32), batch, T)

    run_pto(); torch.npu.synchronize()
    ms_pto = _bench_npu(run_pto)
    ms_t64 = _try_triton_cumsum(cu_seqlens, 64, dev, T, H)
    ms_t128 = _try_triton_cumsum(cu_seqlens, 128, dev, T, H)
    _print_stage("chunk_cumsum", ms_pto, ms_t64, ms_t128)
    return ms_pto, ms_t64, ms_t128


def bench_kkt(H, HG, T, cu_seqlens, dev, stream, bd):
    lib_k = load_scaled_dot_kkt(H, D, C_PTO, key_heads=HG)
    k = torch.randn(1, T, HG, D, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    g_sum = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_t, beta_t = transpose_gates(g_sum), transpose_beta(beta)
    msk = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=-1).float()
    ws = torch.zeros(bd * 2, C_PTO, C_PTO, device=dev, dtype=torch.float16)
    A = torch.empty(1, T, H, C_PTO, device=dev, dtype=torch.float16)
    batch = len(cu_seqlens) - 1

    def run_pto():
        lib_k.call_kernel(bd, stream, _vp(k), _vp(beta_t), _vp(g_t), _vp(msk),
                          _vp(ws), _vp(A), _vp(cu_seqlens), batch, T, T)

    run_pto(); torch.npu.synchronize()
    ms_pto = _bench_npu(run_pto)
    ms_t64 = _try_triton_kkt(cu_seqlens, 64, dev, T, H, HG)
    ms_t128 = _try_triton_kkt(cu_seqlens, 128, dev, T, H, HG)
    _print_stage("scaled_dot_kkt", ms_pto, ms_t64, ms_t128)
    return ms_pto, ms_t64, ms_t128


def bench_solve_tril(H, T, cu_seqlens, dev, tri_inv):
    """PTO solve_tril (C=128) vs Triton solve_tril (BT=64; BT=128 unsupported).

    PTO timing uses :func:`megagdn_pto.fast_inverse.launch_tri_inverse_kernel` inside the loop:
    one ctypes dispatch to ``lib.call_kernel``, with buffers, ``minus_identity``, ``num_matrices``,
    and ``stream_ptr = torch.npu.current_stream()._as_parameter_`` captured once outside.
    Does not invoke :func:`~megagdn_pto.fast_inverse.solve_tril` (no tile-count Python sync,
    no fp32→fp16 ``copy_``). Caller still passes ``tri_inv`` so the DLL is warmed with the rest of the harness.

    Note: Ascend rejects launches when NT * H exceeds roughly 65536 (see README).
    Reduce ``L_seg`` (e.g. 8192 for H≤16 full parity; 4096 for H≤32) to benchmark
    the Triton reference without grid overflow.
    """
    _ = tri_inv  # preload contract shared with staged/mega callers

    A = torch.zeros(1, T, H, C_PTO, device=dev, dtype=torch.float16).tril(-1)
    cu32 = cu_seqlens.to(torch.int32)
    workspace_fp32 = torch.zeros_like(A, dtype=torch.float32)
    batch = int(cu32.numel()) - 1
    tc = total_chunks(batch, T, C_PTO, cu32)
    num_matrices = tc * H
    dty = dev.type
    dix = dev.index if dev.index is not None else -1
    minus_identity = precomputed_minus_identity(dty, dix, C_PTO)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    def run_tri_inverse_kernel():
        launch_tri_inverse_kernel(
            workspace_fp32,
            A,
            minus_identity,
            C_PTO,
            num_matrices,
            H,
            cu_seqlens=cu32,
            block_dim=BLOCK_DIM,
            stream_ptr=stream_ptr,
            is_lower=True,
        )

    run_tri_inverse_kernel()
    torch.npu.synchronize()
    ms_pto = _bench_npu(run_tri_inverse_kernel)
    ms_t64 = _try_triton_solve_tril(cu_seqlens, 64, dev, T, H)
    ms_t128 = None  # BT=128 not supported by Triton (max BT=64)
    print(f"    [Triton solve_tril BT=128: not supported (max BT=64)]")
    _print_stage("solve_tril", ms_pto, ms_t64, ms_t128)
    return ms_pto, ms_t64, ms_t128


def bench_wy_fast(H, HG, T, cu_seqlens, dev, stream, bd):
    lib = load_wy_fast(H, D, C_PTO, key_heads=HG)
    k = torch.randn(1, T, HG, D, device=dev, dtype=torch.float16)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    A = torch.randn(1, T, H, C_PTO, device=dev, dtype=torch.float16)
    g_sum = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_t, beta_t = transpose_gates(g_sum), transpose_beta(beta)
    ws1 = torch.zeros(bd, C_PTO, C_PTO, device=dev, dtype=torch.float16)
    ws2 = torch.zeros_like(ws1)
    w_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    batch = len(cu_seqlens) - 1

    def run_pto():
        lib.call_kernel(bd, stream, _vp(k), _vp(v), _vp(beta_t), _vp(g_t), _vp(A),
                        _vp(ws1), _vp(ws2), _vp(w_out), _vp(u_out), _vp(cu_seqlens),
                        batch, T, T)

    run_pto(); torch.npu.synchronize()
    ms_pto = _bench_npu(run_pto)
    ms_t64 = _try_triton_wy_fast(cu_seqlens, 64, dev, T, H, HG)
    ms_t128 = _try_triton_wy_fast(cu_seqlens, 128, dev, T, H, HG)
    _print_stage("wy_fast", ms_pto, ms_t64, ms_t128)
    return ms_pto, ms_t64, ms_t128


def bench_chunk_h(H, HG, T, tc, cu_seqlens, dev, stream, bd):
    lib = load_chunk_h(H, D, C_PTO, key_heads=HG)
    k = torch.randn(1, T, HG, D, device=dev, dtype=torch.float16)
    w = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    u = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    g_sum = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_t = transpose_gates(g_sum)
    ws = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)
    s = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)
    v_new = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs = torch.empty((len(cu_seqlens) - 1) * H, D, D, device=dev, dtype=torch.float16)
    batch = len(cu_seqlens) - 1

    def run_pto():
        lib.call_kernel(bd, stream, _vp(k), _vp(w), _vp(u), _vp(g_t),
                        _vp(s), _vp(v_new), _vp(fs), _vp(ws), _vp(cu_seqlens),
                        batch, T, T)

    run_pto(); torch.npu.synchronize()
    ms_pto = _bench_npu(run_pto)
    ms_t64 = _try_triton_chunk_h(cu_seqlens, 64, dev, T, H, HG)
    ms_t128 = _try_triton_chunk_h(cu_seqlens, 128, dev, T, H, HG)
    _print_stage("chunk_h", ms_pto, ms_t64, ms_t128)
    return ms_pto, ms_t64, ms_t128


def bench_chunk_o(H, HG, T, tc, cu_seqlens, dev, stream, bd):
    lib_h = load_chunk_h(H, D, C_PTO, key_heads=HG)
    lib_o = load_chunk_o(H, D, C_PTO, key_heads=HG)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    q = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    w = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    u = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    g_sum = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_t = transpose_gates(g_sum)
    ws_h = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)
    s = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)
    v_new = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs = torch.empty((len(cu_seqlens) - 1) * H, D, D, device=dev, dtype=torch.float16)
    batch = len(cu_seqlens) - 1
    # Populate s and v_new via chunk_h warmup
    lib_h.call_kernel(bd, stream, _vp(k), _vp(w), _vp(u), _vp(g_t),
                      _vp(s), _vp(v_new), _vp(fs), _vp(ws_h), _vp(cu_seqlens),
                      batch, T, T)
    torch.npu.synchronize()
    msk = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=0).float()
    ws1 = torch.zeros(bd, C_PTO, C_PTO, device=dev, dtype=torch.float16)
    ws2 = torch.zeros(bd, C_PTO, D, device=dev, dtype=torch.float16)
    ws3 = torch.zeros(bd, C_PTO, C_PTO, device=dev, dtype=torch.float16)
    o = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)

    def run_pto():
        lib_o.call_kernel(bd, stream, _vp(q), _vp(k), _vp(v_new), _vp(s), _vp(g_t),
                          _vp(msk), _vp(ws1), _vp(ws2), _vp(ws3), _vp(o), _vp(cu_seqlens),
                          batch, T, T)

    run_pto(); torch.npu.synchronize()
    ms_pto = _bench_npu(run_pto)
    ms_t64 = _try_triton_chunk_o(cu_seqlens, 64, dev, T, H, HG)
    ms_t128 = _try_triton_chunk_o(cu_seqlens, 128, dev, T, H, HG)
    _print_stage("chunk_o", ms_pto, ms_t64, ms_t128)
    return ms_pto, ms_t64, ms_t128


def bench_mega(H, HG, T, cu_seqlens, dev, stream, tri_inv):
    """Mega-kernel vs staged PTO (aggregated)."""
    q = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    g_in = torch.randn(1, T, H, device=dev, dtype=torch.float32).sigmoid().log()
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    scale = D ** -0.5

    def run_mega():
        run_mega_kernel(q, k, v, g_in, beta, cu_seqlens, stream=stream,
                        chunk_size=C_PTO, scale=scale, key_heads=HG)

    run_mega(); torch.npu.synchronize()
    ms_mega = _bench_npu(run_mega)

    # Staged PTO aggregated (all 6 stages)
    from megagdn_pto.kernel_libs import run_scaled_dot_kkt, run_wy_fast, run_chunk_h, run_chunk_o
    N_seq = int(cu_seqlens.numel()) - 1
    tc_n = total_chunks(N_seq, T, C_PTO, cu_seqlens)
    cu_cpu = cu_seqlens.cpu().tolist()

    def ref_cumsum_torch():
        out = torch.zeros(1, T, H, device=dev, dtype=torch.float32)
        for i in range(len(cu_cpu) - 1):
            bos, eos = cu_cpu[i], cu_cpu[i + 1]
            for j in range(0, eos - bos, C_PTO):
                s, e = bos + j, min(bos + j + C_PTO, eos)
                out[:, s:e, :] = g_in.float()[:, s:e, :].cumsum(dim=1)
        return out

    def run_staged():
        g_sum = ref_cumsum_torch()
        g_t = transpose_gates(g_sum)
        beta_t = transpose_beta(beta)
        torch.npu.synchronize()
        msk_l = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=-1).float()
        msk_f = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=0).float()
        A = torch.zeros(1, T, H, C_PTO, device=dev, dtype=torch.float16)
        run_scaled_dot_kkt(k, beta, g_sum, msk_l, A, stream=stream,
                           g_t=g_t, beta_t=beta_t, chunk_size=C_PTO,
                           cu_seqlens=cu_seqlens, batch_size_override=N_seq, key_heads=HG)
        torch.npu.synchronize()
        A_inv = solve_tril(A, cu_seqlens, C_PTO, H, tri_inv)
        torch.npu.synchronize()
        w = torch.empty_like(v)
        u = torch.empty_like(v)
        run_wy_fast(k, v, beta, g_sum, A_inv, w, u, stream=stream,
                    g_t=g_t, beta_t=beta_t, chunk_size=C_PTO,
                    cu_seqlens=cu_seqlens, batch_size_override=N_seq, key_heads=HG)
        torch.npu.synchronize()
        s = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
        v_new = torch.empty_like(v)
        fs = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
        run_chunk_h(k, w, u, g_sum, s, v_new, fs, stream=stream,
                    g_t=g_t, chunk_size=C_PTO,
                    cu_seqlens=cu_seqlens, batch_size_override=N_seq, key_heads=HG)
        torch.npu.synchronize()
        o = torch.empty_like(v)
        run_chunk_o(q, k, v_new, s, g_sum, msk_f, o, stream=stream,
                    g_t=g_t, chunk_size=C_PTO,
                    cu_seqlens=cu_seqlens, batch_size_override=N_seq, key_heads=HG)
        torch.npu.synchronize()

    run_staged()
    ms_staged = _bench_npu(run_staged)

    print(f"\n  mega_kernel vs staged PTO  (H={H} Hg={HG})")
    print(f"    Mega:   {ms_mega:.2f} ms")
    print(f"    Staged: {ms_staged:.2f} ms  →  mega speedup {_ratio(ms_staged, ms_mega)}")
    return ms_mega, ms_staged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--n-seq", type=int, default=None,
                        help="Sequences in batch (default: GDN_BENCH_N_SEQ or 16).")
    parser.add_argument("--l-seg", type=int, default=None,
                        help="Tokens per sequence before concat (default: GDN_BENCH_L_SEG or 16384).")
    parser.add_argument("--H-list", default="16,32,48,64",
                        help="Comma-separated value head counts H.")
    parser.add_argument("--hg", type=int, default=16, help="Key head count Hg.")
    parser.add_argument("--stage",
                        default="cumsum,kkt,solve_tril,wy_fast,chunk_h,chunk_o",
                        help="Comma-separated stages to benchmark.")
    parser.add_argument("--mega", action="store_true", help="Also benchmark mega-kernel.")
    parser.add_argument("--output-json", default=None,
                        help="Save results as JSON to this path.")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    stream = torch.npu.current_stream()._as_parameter_

    env_n = int(os.getenv("GDN_BENCH_N_SEQ", "16"))
    env_l = int(os.getenv("GDN_BENCH_L_SEG", "16384"))
    N_seq = args.n_seq if args.n_seq is not None else env_n
    L_seg = args.l_seg if args.l_seg is not None else env_l
    T = N_seq * L_seg
    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
    tc = total_chunks(N_seq, T, C_PTO, cu_seqlens)
    bd = BLOCK_DIM
    stages = {s.strip() for s in args.stage.split(",") if s.strip()}
    heads_list = [int(x) for x in args.H_list.split(",") if x.strip()]
    HG = args.hg

    tri_inv_needed = args.mega or "solve_tril" in {s.strip() for s in args.stage.split(",")}
    tri_inv = load_tri_inverse() if tri_inv_needed else None

    print(f"Workload: N_seq={N_seq}  L_seg={L_seg}  T={T}  D={D}  C_PTO={C_PTO}  BLOCK_DIM={bd}")
    print(f"Stages: {stages}  H_list={heads_list}  Hg={HG}")

    all_results: list[dict] = []
    out_path = Path(args.output_json) if args.output_json else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_results() -> None:
        if not out_path:
            return
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "N_seq": N_seq, "L_seg": L_seg, "D": D, "C_pto": C_PTO,
            "results": all_results,
        }
        out_path.write_text(json.dumps(meta, indent=2))
        print(f"  [saved: {out_path}]")

    for H in heads_list:
        assert H % HG == 0, f"H={H} must be divisible by Hg={HG}"
        print(f"\n{'='*68}")
        print(f"H={H}  Hg={HG}")
        print(f"{'='*68}")
        row: dict = {"H": H, "Hg": HG, "D": D, "N_seq": N_seq, "L_seg": L_seg, "C_pto": C_PTO}

        if "cumsum" in stages:
            ms_pto, ms_t64, ms_t128 = bench_chunk_cumsum(H, T, cu_seqlens, dev, stream, bd)
            row["cumsum_pto_ms"] = ms_pto
            row["cumsum_triton64_ms"] = ms_t64
            row["cumsum_triton128_ms"] = ms_t128
            row["cumsum_speedup_vs64"] = ms_t64 / ms_pto if ms_t64 else None
            row["cumsum_speedup_vs128"] = ms_t128 / ms_pto if ms_t128 else None
            gc.collect()
        if "kkt" in stages:
            ms_pto, ms_t64, ms_t128 = bench_kkt(H, HG, T, cu_seqlens, dev, stream, bd)
            row["kkt_pto_ms"] = ms_pto
            row["kkt_triton64_ms"] = ms_t64
            row["kkt_triton128_ms"] = ms_t128
            row["kkt_speedup_vs64"] = ms_t64 / ms_pto if ms_t64 else None
            row["kkt_speedup_vs128"] = ms_t128 / ms_pto if ms_t128 else None
            gc.collect()
        if "solve_tril" in stages:
            ms_pto, ms_t64, ms_t128 = bench_solve_tril(H, T, cu_seqlens, dev, tri_inv)
            row["solve_tril_pto_ms"] = ms_pto
            row["solve_tril_triton64_ms"] = ms_t64
            row["solve_tril_triton128_ms"] = ms_t128  # always None
            row["solve_tril_speedup_vs64"] = ms_t64 / ms_pto if ms_t64 else None
            gc.collect()
        if "wy_fast" in stages:
            ms_pto, ms_t64, ms_t128 = bench_wy_fast(H, HG, T, cu_seqlens, dev, stream, bd)
            row["wy_fast_pto_ms"] = ms_pto
            row["wy_fast_triton64_ms"] = ms_t64
            row["wy_fast_triton128_ms"] = ms_t128
            row["wy_fast_speedup_vs64"] = ms_t64 / ms_pto if ms_t64 else None
            row["wy_fast_speedup_vs128"] = ms_t128 / ms_pto if ms_t128 else None
            gc.collect()
        if "chunk_h" in stages:
            ms_pto, ms_t64, ms_t128 = bench_chunk_h(H, HG, T, tc, cu_seqlens, dev, stream, bd)
            row["chunk_h_pto_ms"] = ms_pto
            row["chunk_h_triton64_ms"] = ms_t64
            row["chunk_h_triton128_ms"] = ms_t128
            row["chunk_h_speedup_vs64"] = ms_t64 / ms_pto if ms_t64 else None
            row["chunk_h_speedup_vs128"] = ms_t128 / ms_pto if ms_t128 else None
            gc.collect()
        if "chunk_o" in stages:
            ms_pto, ms_t64, ms_t128 = bench_chunk_o(H, HG, T, tc, cu_seqlens, dev, stream, bd)
            row["chunk_o_pto_ms"] = ms_pto
            row["chunk_o_triton64_ms"] = ms_t64
            row["chunk_o_triton128_ms"] = ms_t128
            row["chunk_o_speedup_vs64"] = ms_t64 / ms_pto if ms_t64 else None
            row["chunk_o_speedup_vs128"] = ms_t128 / ms_pto if ms_t128 else None
            gc.collect()
        if args.mega:
            ms_mega, ms_staged = bench_mega(H, HG, T, cu_seqlens, dev, stream, tri_inv)
            row["mega_ms"] = ms_mega
            row["staged_ms"] = ms_staged
            row["mega_speedup"] = ms_staged / ms_mega if ms_mega else None
            gc.collect()

        all_results.append(row)
        _save_results()  # write after each H in case of later failure

    if out_path:
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
