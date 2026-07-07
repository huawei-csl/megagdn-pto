#!/usr/bin/env python3
"""Benchmark KDA vs GDN PTO kernels — per-stage and end-to-end.

Pipeline stages:
  KDA: gate_cumsum_kda → kkt_kda → solve_tril → wy_kda → chunk_h_kda → chunk_o_kda
  GDN: chunk_cumsum    → scaled_dot_kkt → solve_tril → wy_fast → chunk_h  → chunk_o

For each pair, reports KDA ms, GDN ms, and KDA/GDN ratio (>1 = KDA slower).
E2E: KDA staged pipeline  vs  GDN megakernel.

Tensor-shape differences between the two pipelines:
  KDA: q/k [B,T,HV,D] fp16, v [B,T,HV,D] fp16, g [B,T,HV,D] fp16, β [B,T,HV] fp16
  GDN: q/k [B,T,HG,D] fp16, v [B,T,H,D]  fp16, g [B,T,H]    fp32, β [B,T,H]  fp16
  Mapping: KDA HV ↔ GDN H (value heads),  KDA HG ↔ GDN HG (key heads),  D=128, C=128

--HV-list sweeps value heads; --hg sets the key-head count for both pipelines.

solve_tril is the same tri_inverse kernel for both; both pipelines pass fp16 input.

Usage::

    python benchmarks/kernel/bench_kda_vs_gdn.py --device npu:0
    python benchmarks/kernel/bench_kda_vs_gdn.py --device npu:0 --HV-list 16,32
    python benchmarks/kernel/bench_kda_vs_gdn.py --device npu:0 --e2e \\
        --l-seg 8192 --output-json outputs/data/kda_vs_gdn.json

Environment (--n-seq / --l-seg override these):
    GDN_NPU_DEVICE    Default NPU device (default: npu:0)
    GDN_BENCH_N_SEQ   Number of sequences (default: 16)
    GDN_BENCH_L_SEG   Tokens per sequence (default: 16384)
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megagdn_pto.compile import BLOCK_DIM
from megagdn_pto.fast_inverse import launch_tri_inverse_kernel, load_tri_inverse
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
from megagdn_pto.kda_kernel_libs import (
    load_chunk_h_kda,
    load_chunk_o_kda,
    load_gate_cumsum_kda,
    load_kkt_kda,
    load_wy_kda,
)
from megagdn_pto.mega_kernel import run_mega_kernel

C = 128   # chunk size (both pipelines)
D = 128   # head dim / key-gate dim / value dim (both pipelines)


def _vp(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _bench_npu(fn, warmup: int = 5, iters: int = 15) -> float:
    """Time an NPU function using Event pairs (ms)."""
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    cache  = torch.empty(256 * 1024 * 1024, dtype=torch.int8).npu()
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


def _ratio(ms_num: float | None, ms_den: float | None) -> str:
    if ms_num and ms_den and ms_den > 0:
        return f"{ms_num / ms_den:.2f}x"
    return "—"


def _print_stage(name: str, ms_kda: float, ms_gdn: float | None) -> None:
    kda_over_gdn = _ratio(ms_kda, ms_gdn)
    print(f"\n  {name}")
    print(f"    KDA : {ms_kda:.2f} ms")
    if ms_gdn is not None:
        print(f"    GDN : {ms_gdn:.2f} ms  →  KDA/GDN = {kda_over_gdn}")
    else:
        print(f"    GDN : —")


# ---------------------------------------------------------------------------
# Stage 1 — gate_cumsum_kda  [B,T,HV,D] fp32  vs  chunk_cumsum  [B,T,HV] fp32
# ---------------------------------------------------------------------------

def bench_gate_cumsum(HV: int, T: int, cu_seqlens, dev, stream, bd: int):
    cu32  = cu_seqlens.to(torch.int32)
    batch = len(cu_seqlens) - 1

    # KDA: [B, T, HV, D] fp16 in — D-dimensional gate per head per token;
    # cumsum output is fp32 (overflow-safe gate path).
    lib_kda  = load_gate_cumsum_kda(HV, D, C)
    g_kda    = torch.randn(1, T, HV, D, device=dev, dtype=torch.float16)
    gs_kda   = torch.empty(1, T, HV, D, device=dev, dtype=torch.float32)

    def run_kda():
        lib_kda.call_kernel(bd, stream, _vp(g_kda), _vp(gs_kda), _vp(cu32), batch, T, HV)

    run_kda()
    torch.npu.synchronize()
    ms_kda = _bench_npu(run_kda)

    # GDN: [B, T, HV] fp32 — scalar gate per head per token
    lib_gdn  = load_chunk_cumsum(C)
    g_gdn    = torch.randn(1, T, HV, device=dev, dtype=torch.float32)
    gs_gdn   = torch.empty_like(g_gdn)

    def run_gdn():
        lib_gdn.call_kernel(bd, stream, _vp(g_gdn), _vp(gs_gdn), _vp(cu32), batch, T, HV)

    run_gdn()
    torch.npu.synchronize()
    ms_gdn = _bench_npu(run_gdn)

    _print_stage(
        f"gate_cumsum_kda [B,T,{HV},{D}] fp16  vs  chunk_cumsum [B,T,{HV}] fp32",
        ms_kda, ms_gdn,
    )
    return ms_kda, ms_gdn


# ---------------------------------------------------------------------------
# Stage 2 — kkt_kda  vs  scaled_dot_kkt
# ---------------------------------------------------------------------------

def bench_kkt(HV: int, HG: int, T: int, cu_seqlens, dev, stream, bd: int):
    cu32  = cu_seqlens.to(torch.int32)
    batch = len(cu_seqlens) - 1
    rows  = torch.arange(C, device=dev).unsqueeze(1)
    cols  = torch.arange(C, device=dev).unsqueeze(0)

    # KDA: k head-major [B, HV, T, D] fp16; g_cs head-major fp32; β [B, HV, T] fp16
    lib_kda    = load_kkt_kda(HV, D, C)
    k_kda_hm   = torch.randn(1, HV, T, D, device=dev, dtype=torch.float16)
    g_cs_hm    = torch.randn(1, HV, T, D, device=dev, dtype=torch.float32)
    beta_kda_hm = torch.rand(1, HV, T,    device=dev, dtype=torch.float16)
    mask_kda   = (rows > cols).to(torch.float32)          # strictly lower tri
    ws_in_kda  = torch.zeros(bd * 2, 2 * C, D, device=dev, dtype=torch.float32)
    ws_out_kda = torch.zeros(bd * 2, C,      C, device=dev, dtype=torch.float32)
    b_ws_kda   = torch.zeros(bd * 2, C,         device=dev, dtype=torch.float32)
    L_kda      = torch.zeros(1, T, HV, C,    device=dev, dtype=torch.float16)

    def run_kda():
        lib_kda.call_kernel(bd, stream,
                            _vp(k_kda_hm), _vp(g_cs_hm), _vp(beta_kda_hm),
                            _vp(mask_kda), _vp(ws_in_kda), _vp(ws_out_kda),
                            _vp(b_ws_kda), _vp(L_kda),
                            _vp(cu32), batch, T, T, HV)

    run_kda()
    torch.npu.synchronize()
    ms_kda = _bench_npu(run_kda)

    # GDN: k [B,T,HG,D] fp16; β [B,HV,T] fp16 (head-major); g [B,HV,T] fp32 (head-major)
    lib_gdn    = load_scaled_dot_kkt(D, C)
    k_gdn      = torch.randn(1, T, HG, D, device=dev, dtype=torch.float16)
    beta_gdn   = torch.rand(1, T, HV,    device=dev, dtype=torch.float16)
    g_gdn      = torch.randn(1, T, HV,   device=dev, dtype=torch.float32)
    g_t_gdn    = transpose_gates(g_gdn)
    beta_t_gdn = transpose_beta(beta_gdn)
    mask_gdn   = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).float()
    ws_gdn     = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    A_gdn      = torch.empty(1, T, HV, C,  device=dev, dtype=torch.float16)

    def run_gdn():
        lib_gdn.call_kernel(bd, stream,
                            _vp(k_gdn), _vp(beta_t_gdn), _vp(g_t_gdn),
                            _vp(mask_gdn), _vp(ws_gdn), _vp(A_gdn),
                            _vp(cu32), batch, T, T, HV, HG)

    run_gdn()
    torch.npu.synchronize()
    ms_gdn = _bench_npu(run_gdn)

    _print_stage(
        f"kkt_kda k[{HV},T,D] fp16  vs  scaled_dot_kkt k[T,{HG},D] fp16",
        ms_kda, ms_gdn,
    )
    return ms_kda, ms_gdn


# ---------------------------------------------------------------------------
# Stage 3 — solve_tril (shared tri_inverse kernel)
#   KDA: fp32→fp16 conversion + tri_inverse
#   GDN: tri_inverse only (input already fp16)
# ---------------------------------------------------------------------------

def bench_solve_tril(HV: int, T: int, cu_seqlens, dev, stream, bd: int):
    cu32    = cu_seqlens.to(torch.int32)
    batch   = len(cu_seqlens) - 1
    tc      = total_chunks(batch, T, C, cu32)
    n_mat   = tc * HV
    dt, di  = dev.type, dev.index if dev.index is not None else -1
    minus_I = precomputed_minus_identity(dt, di, C)

    # KDA: L is fp16 from kkt_kda — pass directly to tri_inverse
    L_fp16  = torch.zeros(1, T, HV, C, device=dev, dtype=torch.float16).tril(-1)
    ws_kda  = torch.zeros_like(L_fp16, dtype=torch.float32)

    def run_kda():
        launch_tri_inverse_kernel(ws_kda, L_fp16, minus_I, C, n_mat, HV,
                                  cu_seqlens=cu32, block_dim=bd,
                                  stream_ptr=stream, is_lower=True)

    run_kda()
    torch.npu.synchronize()
    ms_kda = _bench_npu(run_kda)

    # GDN: A already fp16
    A_gdn   = torch.zeros(1, T, HV, C, device=dev, dtype=torch.float16).tril(-1)
    ws_gdn  = torch.zeros_like(A_gdn, dtype=torch.float32)

    def run_gdn():
        launch_tri_inverse_kernel(ws_gdn, A_gdn, minus_I, C, n_mat, HV,
                                  cu_seqlens=cu32, block_dim=bd,
                                  stream_ptr=stream, is_lower=True)

    run_gdn()
    torch.npu.synchronize()
    ms_gdn = _bench_npu(run_gdn)

    _print_stage(
        "solve_tril  KDA=(fp16 tri_inv)  vs  GDN=(fp16 tri_inv)",
        ms_kda, ms_gdn,
    )
    return ms_kda, ms_gdn


# ---------------------------------------------------------------------------
# Stage 4 — wy_kda  vs  wy_fast
# ---------------------------------------------------------------------------

def bench_wy(HV: int, HG: int, T: int, cu_seqlens, dev, stream, bd: int):
    cu32  = cu_seqlens.to(torch.int32)
    batch = len(cu_seqlens) - 1

    # KDA: k head-major fp16; g_cs head-major fp32; v BSND fp16; β head-major fp16; INV BSND fp16
    lib_kda    = load_wy_kda(HV, D, C)
    k_hm       = torch.randn(1, HV, T, D, device=dev, dtype=torch.float16)
    g_cs_hm    = torch.randn(1, HV, T, D, device=dev, dtype=torch.float32)
    beta_hm    = torch.rand(1, HV, T,    device=dev, dtype=torch.float16)
    v_fp16     = torch.randn(1, T, HV, D, device=dev, dtype=torch.float16)
    INV_kda    = torch.randn(1, T, HV, C, device=dev, dtype=torch.float16)
    ws_a2      = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    ws_keff    = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
    u_kda      = torch.empty(1, T, HV, D, device=dev, dtype=torch.float16)
    w_kda      = torch.empty(1, T, HV, D, device=dev, dtype=torch.float16)

    def run_kda():
        lib_kda.call_kernel(bd, stream,
                            _vp(k_hm), _vp(v_fp16), _vp(beta_hm), _vp(g_cs_hm), _vp(INV_kda),
                            _vp(ws_a2), _vp(ws_keff),
                            _vp(u_kda), _vp(w_kda),
                            _vp(cu32), batch, T, T, HV)

    run_kda()
    torch.npu.synchronize()
    ms_kda = _bench_npu(run_kda)

    # GDN: k [B,T,HG,D] fp16; v/A_inv [B,T,HV,D/C] fp16; β/g head-major
    lib_gdn    = load_wy_fast(D, C)
    k_gdn      = torch.randn(1, T, HG, D, device=dev, dtype=torch.float16)
    v_gdn      = torch.randn(1, T, HV, D, device=dev, dtype=torch.float16)
    beta_gdn   = torch.rand(1, T, HV,    device=dev, dtype=torch.float16)
    g_gdn      = torch.randn(1, T, HV,   device=dev, dtype=torch.float32)
    g_t_gdn    = transpose_gates(g_gdn)
    beta_t_gdn = transpose_beta(beta_gdn)
    A_inv_gdn  = torch.randn(1, T, HV, C, device=dev, dtype=torch.float16)
    ws1_gdn    = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    ws2_gdn    = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    w_gdn      = torch.empty(1, T, HV, D, device=dev, dtype=torch.float16)
    u_gdn      = torch.empty(1, T, HV, D, device=dev, dtype=torch.float16)

    def run_gdn():
        lib_gdn.call_kernel(bd, stream,
                            _vp(k_gdn), _vp(v_gdn), _vp(beta_t_gdn), _vp(g_t_gdn), _vp(A_inv_gdn),
                            _vp(ws1_gdn), _vp(ws2_gdn),
                            _vp(w_gdn), _vp(u_gdn),
                            _vp(cu32), batch, T, T, HV, HG)

    run_gdn()
    torch.npu.synchronize()
    ms_gdn = _bench_npu(run_gdn)

    _print_stage("wy_kda fp16  vs  wy_fast fp16", ms_kda, ms_gdn)
    return ms_kda, ms_gdn


# ---------------------------------------------------------------------------
# Stage 5 — chunk_h_kda  vs  chunk_h
# ---------------------------------------------------------------------------

def bench_chunk_h(HV: int, HG: int, T: int, tc: int, cu_seqlens, dev, stream, bd: int):
    cu32  = cu_seqlens.to(torch.int32)
    batch = len(cu_seqlens) - 1

    # KDA: g_cs head-major fp32, rest fp16; state [tc,HV,D,D] fp16
    lib_kda     = load_chunk_h_kda(HV, D, C)
    k_hm        = torch.randn(1, HV, T, D,    device=dev, dtype=torch.float16)
    g_cs_hm     = torch.randn(1, HV, T, D,    device=dev, dtype=torch.float32)
    w_kda       = torch.randn(1, T, HV, D,    device=dev, dtype=torch.float16)
    u_kda       = torch.randn(1, T, HV, D,    device=dev, dtype=torch.float16)
    s_kda       = torch.zeros(tc, HV, D, D,   device=dev, dtype=torch.float16)
    vcorr_kda   = torch.empty(1, T, HV, D,    device=dev, dtype=torch.float16)
    ws_kda      = torch.zeros(bd * 5, D, D,   device=dev, dtype=torch.float16)

    def run_kda():
        lib_kda.call_kernel(bd, stream,
                            _vp(k_hm), _vp(w_kda), _vp(u_kda), _vp(g_cs_hm),
                            _vp(s_kda), _vp(vcorr_kda), _vp(ws_kda), _vp(cu32),
                            batch, T, T, HV)

    run_kda()
    torch.npu.synchronize()
    ms_kda = _bench_npu(run_kda)

    # GDN: k [B,T,HG,D] fp16; w/u BSND fp16; g head-major fp32; state [tc*HV,D,D] fp16
    lib_gdn     = load_chunk_h(D, C)
    k_gdn       = torch.randn(1, T, HG, D,    device=dev, dtype=torch.float16)
    w_gdn       = torch.randn(1, T, HV, D,    device=dev, dtype=torch.float16)
    u_gdn       = torch.randn(1, T, HV, D,    device=dev, dtype=torch.float16)
    g_t_gdn     = torch.randn(1, HV, T,       device=dev, dtype=torch.float32)
    s_gdn       = torch.zeros(tc * HV, D, D,  device=dev, dtype=torch.float16)
    vnew_gdn    = torch.empty(1, T, HV, D,    device=dev, dtype=torch.float16)
    fs_gdn      = torch.empty(batch * HV, D, D, device=dev, dtype=torch.float16)
    ws_gdn      = torch.zeros(bd * 4, D, D,   device=dev, dtype=torch.float16)

    def run_gdn():
        # final_state written to fs_gdn (output_final_state=1); no initial state (h0 null).
        lib_gdn.call_kernel(bd, stream,
                            _vp(k_gdn), _vp(w_gdn), _vp(u_gdn), _vp(g_t_gdn),
                            _vp(s_gdn), _vp(vnew_gdn), _vp(fs_gdn),
                            ctypes.c_void_p(0), 0, 1, _vp(ws_gdn), _vp(cu32),
                            batch, T, T, HV, HG)

    run_gdn()
    torch.npu.synchronize()
    ms_gdn = _bench_npu(run_gdn)

    _print_stage(
        "chunk_h_kda fp16 state [tc,HV,D,D]  vs  chunk_h fp16 state [tc*HV,D,D]",
        ms_kda, ms_gdn,
    )
    return ms_kda, ms_gdn


# ---------------------------------------------------------------------------
# Stage 6 — chunk_o_kda  vs  chunk_o
# ---------------------------------------------------------------------------

def bench_chunk_o(HV: int, HG: int, T: int, tc: int, cu_seqlens, dev, stream, bd: int):
    cu32  = cu_seqlens.to(torch.int32)
    batch = len(cu_seqlens) - 1
    rows  = torch.arange(C, device=dev).unsqueeze(1)
    cols  = torch.arange(C, device=dev).unsqueeze(0)

    # KDA: pre-populate s and v_corr via a chunk_h_kda warmup call (all fp16)
    lib_h_kda   = load_chunk_h_kda(HV, D, C)
    lib_o_kda   = load_chunk_o_kda(HV, D, C)
    k_hm        = F.normalize(torch.randn(1, HV, T, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    q_hm        = F.normalize(torch.randn(1, HV, T, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    g_cs_hm     = torch.randn(1, HV, T, D, device=dev, dtype=torch.float32)
    w_kda       = torch.randn(1, T, HV, D, device=dev, dtype=torch.float16)
    u_kda       = torch.randn(1, T, HV, D, device=dev, dtype=torch.float16)
    s_kda       = torch.zeros(tc, HV, D, D, device=dev, dtype=torch.float16)
    vcorr_kda   = torch.empty(1, T, HV, D,  device=dev, dtype=torch.float16)
    ws_h_kda    = torch.zeros(bd * 5, D, D,  device=dev, dtype=torch.float16)
    lib_h_kda.call_kernel(bd, stream,
                          _vp(k_hm), _vp(w_kda), _vp(u_kda), _vp(g_cs_hm),
                          _vp(s_kda), _vp(vcorr_kda), _vp(ws_h_kda), _vp(cu32),
                          batch, T, T, HV)
    torch.npu.synchronize()

    mask_kda    = (rows >= cols).to(torch.float32)   # inclusive diagonal
    ws_o_kda    = torch.zeros(bd * 7, D, D, device=dev, dtype=torch.float32)
    o_kda       = torch.empty(1, T, HV, D,  device=dev, dtype=torch.float16)

    def run_kda():
        lib_o_kda.call_kernel(bd, stream,
                              _vp(q_hm), _vp(k_hm), _vp(vcorr_kda), _vp(s_kda),
                              _vp(g_cs_hm), _vp(mask_kda),
                              _vp(ws_o_kda), _vp(o_kda), _vp(cu32),
                              batch, T, T, HV)

    run_kda()
    torch.npu.synchronize()
    ms_kda = _bench_npu(run_kda)

    # GDN: pre-populate s and v_new via chunk_h warmup
    lib_h_gdn   = load_chunk_h(D, C)
    lib_o_gdn   = load_chunk_o(D, C)
    k_gdn       = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    q_gdn       = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    w_gdn       = torch.randn(1, T, HV, D, device=dev, dtype=torch.float16)
    u_gdn       = torch.randn(1, T, HV, D, device=dev, dtype=torch.float16)
    g_t_gdn     = torch.randn(1, HV, T,   device=dev, dtype=torch.float32)
    s_gdn       = torch.zeros(tc * HV, D, D, device=dev, dtype=torch.float16)
    vnew_gdn    = torch.empty(1, T, HV, D,   device=dev, dtype=torch.float16)
    fs_gdn      = torch.empty(batch * HV, D, D, device=dev, dtype=torch.float16)
    ws_h_gdn    = torch.zeros(bd * 4, D, D,  device=dev, dtype=torch.float16)
    lib_h_gdn.call_kernel(bd, stream,
                          _vp(k_gdn), _vp(w_gdn), _vp(u_gdn), _vp(g_t_gdn),
                          _vp(s_gdn), _vp(vnew_gdn), _vp(fs_gdn),
                          ctypes.c_void_p(0), 0, 1, _vp(ws_h_gdn), _vp(cu32),
                          batch, T, T, HV, HG)
    torch.npu.synchronize()

    mask_gdn    = torch.tril(torch.ones(C, C, device=dev), diagonal=0).float()
    ws1_gdn     = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    ws2_gdn     = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
    ws3_gdn     = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_gdn       = torch.empty(1, T, HV, D, device=dev, dtype=torch.float16)

    def run_gdn():
        lib_o_gdn.call_kernel(bd, stream,
                              _vp(q_gdn), _vp(k_gdn), _vp(vnew_gdn), _vp(s_gdn), _vp(g_t_gdn),
                              _vp(mask_gdn), _vp(ws1_gdn), _vp(ws2_gdn), _vp(ws3_gdn), _vp(o_gdn),
                              _vp(cu32), batch, T, T, HV, HG)

    run_gdn()
    torch.npu.synchronize()
    ms_gdn = _bench_npu(run_gdn)

    _print_stage(
        "chunk_o_kda fp16  vs  chunk_o fp16",
        ms_kda, ms_gdn,
    )
    return ms_kda, ms_gdn


# ---------------------------------------------------------------------------
# E2E — KDA staged pipeline (pre-allocated)  vs  GDN megakernel
# ---------------------------------------------------------------------------

def bench_e2e(HV: int, HG: int, T: int, tc: int, cu_seqlens, dev, stream, bd: int):
    """KDA 6-stage pipeline with all buffers pre-allocated vs GDN run_mega_kernel.

    All workspace tensors, permuted views, and intermediates are allocated once
    before timing; the timed loop only calls lib.call_kernel + tensor copy ops.
    """
    cu32  = cu_seqlens.to(torch.int32)
    batch = len(cu_seqlens) - 1
    scale = D ** -0.5
    rows  = torch.arange(C, device=dev).unsqueeze(1)
    cols  = torch.arange(C, device=dev).unsqueeze(0)

    # ── KDA inputs (all fp16) ────────────────────────────────────────────────
    # q is pre-scaled; k/v/g/β are static across timing iterations
    q_bsnd   = F.normalize(torch.randn(1, T, HV, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    k_bsnd   = F.normalize(torch.randn(1, T, HV, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    v_bsnd   = torch.randn(1, T, HV, D, device=dev, dtype=torch.float16)
    g_log    = (-torch.rand(1, T, HV, D, device=dev, dtype=torch.float32) * 0.05).half()
    beta_bsnd = torch.sigmoid(torch.randn(1, T, HV, device=dev, dtype=torch.float32)).half()
    q_bsnd.mul_(scale)

    # Pre-computed head-major views of static inputs
    k_hm     = k_bsnd.permute(0, 2, 1, 3).contiguous()   # [B, HV, T, D] fp16
    q_hm     = q_bsnd.permute(0, 2, 1, 3).contiguous()
    beta_hm  = beta_bsnd.permute(0, 2, 1).contiguous()   # [B, HV, T] fp16

    # Intermediates (gate cumsum + tri_inverse workspaces are fp32; rest fp16)
    g_sum_bsnd = torch.empty(1, T, HV, D, device=dev, dtype=torch.float32)
    g_sum_hm   = torch.empty(1, HV, T, D, device=dev, dtype=torch.float32)
    L_kkt      = torch.zeros(1, T, HV, C, device=dev, dtype=torch.float16)  # kkt output
    ws_tri     = torch.zeros(1, T, HV, C, device=dev, dtype=torch.float32)  # tri_inverse fp32
    A_inv      = torch.empty(1, T, HV, C, device=dev, dtype=torch.float16)  # fp16 result
    u_out      = torch.zeros(1, T, HV, D, device=dev, dtype=torch.float16)
    w_out      = torch.zeros(1, T, HV, D, device=dev, dtype=torch.float16)
    s_snap     = torch.zeros(tc, HV, D, D, device=dev, dtype=torch.float16)
    v_corr     = torch.zeros(1, T, HV, D, device=dev, dtype=torch.float16)
    o_kda      = torch.zeros(1, T, HV, D, device=dev, dtype=torch.float16)

    # Workspaces (pre-allocated, not timed); kkt + chunk_o are fp32 (overflow-safe).
    ws_kkt_in  = torch.zeros(bd * 2, 2 * C, D, device=dev, dtype=torch.float32)
    ws_kkt_out = torch.zeros(bd * 2, C,      C, device=dev, dtype=torch.float32)
    ws_kkt_b   = torch.zeros(bd * 2, C,         device=dev, dtype=torch.float32)
    ws_wy_a2   = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    ws_wy_keff = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
    ws_h       = torch.zeros(bd * 5, D, D, device=dev, dtype=torch.float16)
    ws_o       = torch.zeros(bd * 7, D, D, device=dev, dtype=torch.float32)

    mask_strict = (rows > cols).to(torch.float32)
    mask_causal = (rows >= cols).to(torch.float32)

    dt, di     = dev.type, dev.index if dev.index is not None else -1
    minus_I    = precomputed_minus_identity(dt, di, C)
    n_mat      = tc * HV

    lib_cumsum = load_gate_cumsum_kda(HV, D, C)
    lib_kkt    = load_kkt_kda(HV, D, C)
    lib_wy     = load_wy_kda(HV, D, C)
    lib_h      = load_chunk_h_kda(HV, D, C)
    lib_o      = load_chunk_o_kda(HV, D, C)

    def run_kda():
        # 1. gate_cumsum_kda
        lib_cumsum.call_kernel(bd, stream, _vp(g_log), _vp(g_sum_bsnd), _vp(cu32), batch, T, HV)
        g_sum_hm.copy_(g_sum_bsnd.permute(0, 2, 1, 3))   # BSND → head-major (fp32)

        # 2. kkt_kda (all fp16)
        lib_kkt.call_kernel(bd, stream,
                            _vp(k_hm), _vp(g_sum_hm), _vp(beta_hm),
                            _vp(mask_strict), _vp(ws_kkt_in), _vp(ws_kkt_out),
                            _vp(ws_kkt_b), _vp(L_kkt),
                            _vp(cu32), batch, T, T, HV)

        # 3. solve_tril (fp16 in; tri_inverse writes fp32 to ws_tri; copy → A_inv fp16)
        launch_tri_inverse_kernel(ws_tri, L_kkt, minus_I, C, n_mat, HV,
                                  cu_seqlens=cu32, block_dim=bd,
                                  stream_ptr=stream, is_lower=True)
        A_inv.copy_(ws_tri)

        # 4. wy_kda (all fp16)
        lib_wy.call_kernel(bd, stream,
                           _vp(k_hm), _vp(v_bsnd), _vp(beta_hm), _vp(g_sum_hm), _vp(A_inv),
                           _vp(ws_wy_a2), _vp(ws_wy_keff),
                           _vp(u_out), _vp(w_out),
                           _vp(cu32), batch, T, T, HV)

        # 5. chunk_h_kda (all fp16; w_out already fp16)
        lib_h.call_kernel(bd, stream,
                          _vp(k_hm), _vp(w_out), _vp(u_out), _vp(g_sum_hm),
                          _vp(s_snap), _vp(v_corr), _vp(ws_h), _vp(cu32),
                          batch, T, T, HV)

        # 6. chunk_o_kda (all fp16)
        lib_o.call_kernel(bd, stream,
                          _vp(q_hm), _vp(k_hm), _vp(v_corr), _vp(s_snap),
                          _vp(g_sum_hm), _vp(mask_causal),
                          _vp(ws_o), _vp(o_kda), _vp(cu32),
                          batch, T, T, HV)

    run_kda()
    torch.npu.synchronize()
    ms_kda = _bench_npu(run_kda)

    # Free the KDA inputs/intermediates/workspaces before allocating the GDN
    # megakernel buffers — at the largest shapes both live sets together exceed
    # NPU HBM and OOM the run.
    del (q_bsnd, k_bsnd, v_bsnd, g_log, beta_bsnd, k_hm, q_hm, beta_hm,
         g_sum_bsnd, g_sum_hm, L_kkt, ws_tri, A_inv, u_out, w_out, s_snap,
         v_corr, o_kda, ws_kkt_in, ws_kkt_out, ws_wy_a2, ws_wy_keff, ws_h, ws_o)
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()

    # ── GDN megakernel ───────────────────────────────────────────────────────
    q_gdn  = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    k_gdn  = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    v_gdn  = torch.randn(1, T, HV, D,    device=dev, dtype=torch.float16)
    g_gdn  = torch.randn(1, T, HV,       device=dev, dtype=torch.float32).sigmoid().log()
    beta_gdn = torch.rand(1, T, HV,      device=dev, dtype=torch.float16)

    def run_gdn():
        run_mega_kernel(q_gdn, k_gdn, v_gdn, g_gdn, beta_gdn, cu_seqlens,
                        stream=stream, chunk_size=C, scale=scale, key_heads=HG)

    run_gdn()
    torch.npu.synchronize()
    ms_gdn = _bench_npu(run_gdn)

    kda_over_gdn = _ratio(ms_kda, ms_gdn)
    print(f"\n  E2E  (HV={HV} HG={HG})")
    print(f"    KDA staged    : {ms_kda:.2f} ms")
    print(f"    GDN megakernel: {ms_gdn:.2f} ms  →  KDA/GDN = {kda_over_gdn}")
    return ms_kda, ms_gdn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--n-seq", type=int, default=None,
                        help="Sequences in batch (default: GDN_BENCH_N_SEQ or 16).")
    parser.add_argument("--l-seg", type=int, default=None,
                        help="Tokens per sequence (default: GDN_BENCH_L_SEG or 16384).")
    parser.add_argument("--HV-list", default="16,32,48,64",
                        help="Comma-separated value head counts: KDA HV / GDN H.")
    parser.add_argument("--hg", type=int, default=16,
                        help="Key head count: KDA H / GDN HG (default 16).")
    parser.add_argument("--stage",
                        default="cumsum,kkt,solve_tril,wy,chunk_h,chunk_o",
                        help="Comma-separated stages to benchmark.")
    parser.add_argument("--e2e", action="store_true",
                        help="Also run end-to-end: KDA staged vs GDN megakernel.")
    parser.add_argument("--output-json", default=None,
                        help="Save results as JSON to this path.")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device(args.device)
    dev    = torch.device(args.device)
    stream = torch.npu.current_stream()._as_parameter_

    env_n  = int(os.getenv("GDN_BENCH_N_SEQ", "16"))
    env_l  = int(os.getenv("GDN_BENCH_L_SEG", "16384"))
    N_seq  = args.n_seq if args.n_seq is not None else env_n
    L_seg  = args.l_seg if args.l_seg is not None else env_l
    T      = N_seq * L_seg
    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
    tc     = total_chunks(N_seq, T, C, cu_seqlens)
    bd     = BLOCK_DIM
    HG     = args.hg
    stages = {s.strip() for s in args.stage.split(",") if s.strip()}
    HV_list = [int(x) for x in args.HV_list.split(",") if x.strip()]

    if args.e2e or "solve_tril" in stages:
        load_tri_inverse()   # warm up the DLL; launch_tri_inverse_kernel uses it internally

    print(f"Workload : N_seq={N_seq}  L_seg={L_seg}  T={T}  D={D}  C={C}  BLOCK_DIM={bd}")
    print(f"Stages   : {sorted(stages)}  HV_list={HV_list}  HG={HG}")

    all_results: list[dict] = []
    out_path = Path(args.output_json) if args.output_json else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save():
        if not out_path:
            return
        out_path.write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "N_seq": N_seq, "L_seg": L_seg, "D": D, "C": C,
            "results": all_results,
        }, indent=2))
        print(f"  [saved: {out_path}]")

    for HV in HV_list:
        assert HV % HG == 0, f"HV={HV} must be divisible by HG={HG}"
        print(f"\n{'='*68}")
        print(f"HV={HV}  HG={HG}  (KDA HV = GDN H;  KDA/GDN key heads = {HG})")
        print(f"{'='*68}")
        row: dict = {"HV": HV, "HG": HG, "D": D, "N_seq": N_seq, "L_seg": L_seg, "C": C}

        if "cumsum" in stages:
            ms_kda, ms_gdn = bench_gate_cumsum(HV, T, cu_seqlens, dev, stream, bd)
            row.update(cumsum_kda_ms=ms_kda, cumsum_gdn_ms=ms_gdn,
                       cumsum_kda_over_gdn=ms_kda / ms_gdn if ms_gdn else None)
            gc.collect()

        if "kkt" in stages:
            ms_kda, ms_gdn = bench_kkt(HV, HG, T, cu_seqlens, dev, stream, bd)
            row.update(kkt_kda_ms=ms_kda, kkt_gdn_ms=ms_gdn,
                       kkt_kda_over_gdn=ms_kda / ms_gdn if ms_gdn else None)
            gc.collect()

        if "solve_tril" in stages:
            ms_kda, ms_gdn = bench_solve_tril(HV, T, cu_seqlens, dev, stream, bd)
            row.update(solve_tril_kda_ms=ms_kda, solve_tril_gdn_ms=ms_gdn,
                       solve_tril_kda_over_gdn=ms_kda / ms_gdn if ms_gdn else None)
            gc.collect()

        if "wy" in stages:
            ms_kda, ms_gdn = bench_wy(HV, HG, T, cu_seqlens, dev, stream, bd)
            row.update(wy_kda_ms=ms_kda, wy_gdn_ms=ms_gdn,
                       wy_kda_over_gdn=ms_kda / ms_gdn if ms_gdn else None)
            gc.collect()

        if "chunk_h" in stages:
            ms_kda, ms_gdn = bench_chunk_h(HV, HG, T, tc, cu_seqlens, dev, stream, bd)
            row.update(chunk_h_kda_ms=ms_kda, chunk_h_gdn_ms=ms_gdn,
                       chunk_h_kda_over_gdn=ms_kda / ms_gdn if ms_gdn else None)
            gc.collect()

        if "chunk_o" in stages:
            ms_kda, ms_gdn = bench_chunk_o(HV, HG, T, tc, cu_seqlens, dev, stream, bd)
            row.update(chunk_o_kda_ms=ms_kda, chunk_o_gdn_ms=ms_gdn,
                       chunk_o_kda_over_gdn=ms_kda / ms_gdn if ms_gdn else None)
            gc.collect()

        if args.e2e:
            ms_kda, ms_gdn = bench_e2e(HV, HG, T, tc, cu_seqlens, dev, stream, bd)
            row.update(e2e_kda_staged_ms=ms_kda, e2e_gdn_mega_ms=ms_gdn,
                       e2e_kda_over_gdn=ms_kda / ms_gdn if ms_gdn else None)
            gc.collect()

        all_results.append(row)
        _save()

    if out_path:
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
