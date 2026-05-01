#!/usr/bin/env python3
"""Benchmark PTO chunk-GDN kernels vs FLA Triton baseline.

Measures latency (ms) for each of the five computation stages:
  scaled_dot_kkt, wy_fast, chunk_h, chunk_o, and the fused mega_kernel.

For each stage, the Triton baseline uses chunk_size=64 (matching the vLLM
default). PTO always uses chunk_size=128, which yields ~2× more compute per
launch while still outperforming Triton — see outputs/data/ for saved results.

Triton notes:
  - ``chunk_o`` with H=64, BT=64 and ``scaled_dot_kkt`` BT=128 may fail to
    compile for certain head counts; those entries are marked as "fail" in output.
  - BT=128 for ``scaled_dot_kkt`` is optional (set GDN_TRITON_KKT_TRY128=1).

Results are printed to stdout and optionally saved to ``--output-json``.

Usage::

    python benchmarks/kernel/bench_gdn_kernels.py --device npu:0
    python benchmarks/kernel/bench_gdn_kernels.py --device npu:0 --H-list 32,64 --stage kkt,chunk_h
    python benchmarks/kernel/bench_gdn_kernels.py --device npu:0 --mega
    python benchmarks/kernel/bench_gdn_kernels.py --device npu:0 --mega \\
        --output-json outputs/data/kernel_bench.json

Environment:
    GDN_NPU_DEVICE             Default NPU device (default: npu:0)
    GDN_BENCH_N_SEQ            Number of sequences (default: 16)
    GDN_BENCH_L_SEG            Tokens per sequence (default: 16384)
    GDN_TRITON_KKT_TRY128      Set to 1 to attempt BT=128 for kkt Triton baseline
    GDN_TRITON_CHUNK_O_CHUNK   Chunk size for Triton chunk_o (default: 64)
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
from megagdn_pto.fast_inverse import load_tri_inverse, solve_tril
from megagdn_pto.kernel_libs import (
    load_chunk_h,
    load_chunk_o,
    load_scaled_dot_kkt,
    load_wy_fast,
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
        print(f"    [Triton kkt BT={BT} unavailable: {str(exc).split(chr(10))[0][:80]}]")
        gc.collect()
        return None


# ---------------------------------------------------------------------------
# Stage benchmarks
# ---------------------------------------------------------------------------

def bench_kkt(H, HG, T, cu_seqlens, dev, stream, bd, try_128=False):
    C_triton = int(os.getenv("GDN_TRITON_KKT_CHUNK", "64"))
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
    ms_t64 = _try_triton_kkt(cu_seqlens, C_triton, dev, T, H, HG)
    ms_t128 = _try_triton_kkt(cu_seqlens, 128, dev, T, H, HG) if try_128 else None

    print(f"\n  scaled_dot_kkt  (PTO C={C_PTO}  vs  Triton BT={C_triton})")
    print(f"    PTO:         {ms_pto:.2f} ms")
    if ms_t64 is not None:
        print(f"    Triton BT={C_triton}: {ms_t64:.2f} ms  →  speedup {_ratio(ms_t64, ms_pto)}")
    else:
        print(f"    Triton BT={C_triton}: fail")
    if try_128:
        if ms_t128 is not None:
            print(f"    Triton BT=128: {ms_t128:.2f} ms  →  speedup {_ratio(ms_t128, ms_pto)}")
        else:
            print(f"    Triton BT=128: fail")
    return ms_pto, ms_t64


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
    ms_tri = None
    try:
        from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
        cu_long = cu_seqlens.long()
        CI = prepare_chunk_indices(cu_long, C_PTO)
        CO = prepare_chunk_offsets(cu_long, C_PTO)
        k_tr = torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16)
        w_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        u_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)
        fn_t = lambda: chunk_gated_delta_rule_fwd_h(k=k_tr, w=w_tr, u=u_tr, g=g_tr,
                                                    initial_state=None, output_final_state=False,
                                                    cu_seqlens=cu_long, chunk_indices=CI,
                                                    chunk_offsets=CO, chunk_size=C_PTO)
        fn_t(); torch.npu.synchronize()
        ms_tri = _bench_triton(fn_t)
    except Exception as exc:
        print(f"    [Triton chunk_h unavailable: {str(exc).split(chr(10))[0][:80]}]")

    print(f"\n  chunk_h  (PTO C={C_PTO}  vs  Triton BT={C_PTO})")
    print(f"    PTO:    {ms_pto:.2f} ms")
    if ms_tri is not None:
        print(f"    Triton: {ms_tri:.2f} ms  →  speedup {_ratio(ms_tri, ms_pto)}")
    return ms_pto, ms_tri


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
    ms_tri = None
    try:
        from fla_vendor.wy_fast import recompute_w_u_fwd
        from fla_vendor.utils import prepare_chunk_indices
        cu_long = cu_seqlens.long()
        CI = prepare_chunk_indices(cu_long, C_PTO)
        k_tr = torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16)
        v_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        beta_tr = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
        A_tr = torch.randn(1, T, H, C_PTO, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)
        fn_t = lambda: recompute_w_u_fwd(k=k_tr, v=v_tr, beta=beta_tr, g_cumsum=g_tr,
                                          A=A_tr, cu_seqlens=cu_long, chunk_indices=CI)
        fn_t(); torch.npu.synchronize()
        ms_tri = _bench_triton(fn_t)
    except Exception as exc:
        print(f"    [Triton wy_fast unavailable: {str(exc).split(chr(10))[0][:80]}]")

    print(f"\n  wy_fast  (PTO C={C_PTO}  vs  Triton BT={C_PTO})")
    print(f"    PTO:    {ms_pto:.2f} ms")
    if ms_tri is not None:
        print(f"    Triton: {ms_tri:.2f} ms  →  speedup {_ratio(ms_tri, ms_pto)}")
    return ms_pto, ms_tri


def bench_chunk_o(H, HG, T, tc, cu_seqlens, dev, stream, bd):
    C_triton_o = int(os.getenv("GDN_TRITON_CHUNK_O_CHUNK", "64"))
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
    # Warmup chunk_h to populate s and v_new
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
    ms_tri = None

    # H=64 with BT=64 is a known Triton failure (aicore exception); skip to avoid
    # poisoning the NPU device state for subsequent benchmarks.
    if H >= 64 and C_triton_o <= 64:
        print(f"    [Triton chunk_o BT={C_triton_o} known failure for H={H}: skip]")
    else:
        try:
            from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
            from fla_vendor.chunk_o import chunk_fwd_o
            from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
            cu_long = cu_seqlens.long()
            CI = prepare_chunk_indices(cu_long, C_triton_o)
            CO = prepare_chunk_offsets(cu_long, C_triton_o)
            scale = D ** -0.5
            q_tr = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16), dim=-1, p=2)
            k_tr = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16), dim=-1, p=2)
            w_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
            u_tr = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
            g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)
            h_tr, v_new_tr, _ = chunk_gated_delta_rule_fwd_h(k=k_tr, w=w_tr, u=u_tr, g=g_tr,
                                                              initial_state=None, output_final_state=False,
                                                              cu_seqlens=cu_long, chunk_indices=CI,
                                                              chunk_offsets=CO, chunk_size=C_triton_o)
            torch.npu.synchronize()
            fn_t = lambda: chunk_fwd_o(q=q_tr, k=k_tr, v=v_new_tr, h=h_tr, g=g_tr,
                                        scale=scale, cu_seqlens=cu_long, chunk_size=C_triton_o)
            fn_t(); torch.npu.synchronize()
            ms_tri = _bench_triton(fn_t)
        except Exception as exc:
            reason = str(exc).split(chr(10))[0][:100]
            print(f"    [Triton chunk_o BT={C_triton_o} unavailable: {reason}]")

    print(f"\n  chunk_o  (PTO C={C_PTO}  vs  Triton BT={C_triton_o})")
    print(f"    PTO:    {ms_pto:.2f} ms")
    if ms_tri is not None:
        print(f"    Triton: {ms_tri:.2f} ms  →  speedup {_ratio(ms_tri, ms_pto)}")
    else:
        print(f"    Triton: fail (known H={H} BT={C_triton_o} incompatibility)" if H >= 64 else
              f"    Triton: fail")
    return ms_pto, ms_tri


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
    parser.add_argument("--H-list", default="16,32,48,64",
                        help="Comma-separated value head counts H.")
    parser.add_argument("--hg", type=int, default=16, help="Key head count Hg.")
    parser.add_argument("--stage", default="kkt,chunk_h,wy_fast,chunk_o",
                        help="Comma-separated stages to benchmark.")
    parser.add_argument("--mega", action="store_true", help="Also benchmark mega-kernel.")
    parser.add_argument("--try-kkt-128", action="store_true",
                        help="Try Triton BT=128 for kkt (may fail for some H).")
    parser.add_argument("--output-json", default=None,
                        help="Save results as JSON to this path.")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    stream = torch.npu.current_stream()._as_parameter_

    N_seq = int(os.getenv("GDN_BENCH_N_SEQ", "16"))
    L_seg = int(os.getenv("GDN_BENCH_L_SEG", "16384"))
    T = N_seq * L_seg
    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
    tc = total_chunks(N_seq, T, C_PTO, cu_seqlens)
    bd = BLOCK_DIM
    stages = {s.strip() for s in args.stage.split(",") if s.strip()}
    heads_list = [int(x) for x in args.H_list.split(",") if x.strip()]
    HG = args.hg

    tri_inv = load_tri_inverse() if args.mega else None

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

        if "kkt" in stages:
            ms_pto, ms_tri = bench_kkt(H, HG, T, cu_seqlens, dev, stream, bd, try_128=args.try_kkt_128)
            row["kkt_pto_ms"] = ms_pto
            row["kkt_triton_ms"] = ms_tri
            row["kkt_speedup"] = ms_tri / ms_pto if ms_tri else None
            gc.collect()
        if "chunk_h" in stages:
            ms_pto, ms_tri = bench_chunk_h(H, HG, T, tc, cu_seqlens, dev, stream, bd)
            row["chunk_h_pto_ms"] = ms_pto
            row["chunk_h_triton_ms"] = ms_tri
            row["chunk_h_speedup"] = ms_tri / ms_pto if ms_tri else None
            gc.collect()
        if "wy_fast" in stages:
            ms_pto, ms_tri = bench_wy_fast(H, HG, T, cu_seqlens, dev, stream, bd)
            row["wy_fast_pto_ms"] = ms_pto
            row["wy_fast_triton_ms"] = ms_tri
            row["wy_fast_speedup"] = ms_tri / ms_pto if ms_tri else None
            gc.collect()
        if "chunk_o" in stages:
            ms_pto, ms_tri = bench_chunk_o(H, HG, T, tc, cu_seqlens, dev, stream, bd)
            row["chunk_o_pto_ms"] = ms_pto
            row["chunk_o_triton_ms"] = ms_tri
            row["chunk_o_speedup"] = ms_tri / ms_pto if ms_tri else None
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
