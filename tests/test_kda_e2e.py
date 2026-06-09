#!/usr/bin/env python3
"""End-to-end KDA pipeline test: all NPU PTO stages vs CPU float32 reference.

Stages (all on NPU):
  [1] gate_cumsum_kda — within-chunk prefix sum of g [B, T, HV, K]
  [2] kkt_kda        — L matrix (gated K×K product)
  [3] solve_tril      — (I + L)^{-1} via PTO-ISA tri_inverse kernel
  [4] wy_kda         — u and w transforms
  [5] chunk_h_kda    — sequential state pass (snapshots + v_corr)
  [6] chunk_o_kda    — output pass

Usage::

    python tests/test_kda_e2e.py --device npu:0
    python tests/test_kda_e2e.py --device npu:0 --H 4 --HV 8
    python tests/test_kda_e2e.py --device npu:0 --quick
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

from megagdn_pto.fast_inverse import load_tri_inverse, solve_tril

# Both helper modules are in the same tests/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from megagdn_pto.kda_kernel_libs import (
    run_gate_cumsum_kda,
    run_kkt_kda,
    run_wy_kda,
    run_chunk_h_kda,
    run_chunk_o_kda,
)
from tests.ref_kda import RefKDA
from megagdn_pto.kda_mega_kernel import run_mega_kernel_kda
from tests.utils import NumericalAccuracy

CHUNK = 128  # small chunk for fast CPU tests
K = 128  # key/query dimension
V_DIM = 128  # value dimension

# ---------------------------------------------------------------------------
# PTO pipeline (all 6 stages on NPU)
# ---------------------------------------------------------------------------


def pto_pipeline_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta_sig: torch.Tensor,
    cu_seqlens_list,
    scale: float,
    dev: torch.device,
    chunk_size: int = CHUNK,
    tri_inv_func=None,
) -> torch.Tensor:
    """KDA pipeline with all stages on NPU PTO kernels.

    Args:
        q:               [B, T, H,  K]  float32, L2-normalised queries
        k:               [B, T, H,  K]  float32, L2-normalised keys
        v:               [B, T, HV, V]  float32
        g_log:           [B, T, HV, K]  float32, log-space per-dim gate values
        beta_sig:        [B, T, HV]     float32, post-sigmoid beta in (0, 1)
        cu_seqlens_list: list[int] | None
        scale:           float, query scale (typically K**-0.5)
        dev:             NPU device
        chunk_size:      int, must match compiled kernel
        tri_inv_func:    pre-loaded tri_inverse callable (auto-loaded if None)

    Returns:
        o: [B, T, HV, V]  float32
    """

    B, T, H, Kd = q.shape
    HV = v.shape[2]
    G = HV // H
    Vd = v.shape[3]

    # GQA expansion + scale (mirrors cpu_pipeline_kda)
    qf = q.half().repeat_interleave(G, dim=2) * scale  # [B, T, HV, K] fp16
    kf = k.half().repeat_interleave(G, dim=2)  # [B, T, HV, K] fp16
    vf = v.half()
    bf = beta_sig.half()

    cu_dev = (
        torch.tensor(cu_seqlens_list, dtype=torch.int32, device=dev)
        if cu_seqlens_list
        else None
    )
    N_seq = len(cu_seqlens_list) - 1 if cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    # ── Stage 1: gate_cumsum_kda — NPU PTO kernel ────────────────────────────
    g_dev = g_log.half().to(dev)
    g_sum_dev = torch.empty_like(g_dev)
    run_gate_cumsum_kda(
        g_dev,
        g_sum_dev,
        stream=stream,
        chunk_size=chunk_size,
        cu_seqlens=cu_dev,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    # ── Stage 2: kkt_kda — NPU PTO kernel ────────────────────────────────────
    kf_dev = kf.to(dev)
    bf_dev = bf.to(dev)
    L_dev = torch.zeros(1, T, HV, chunk_size, device=dev, dtype=torch.float16)
    run_kkt_kda(
        kf_dev,
        g_sum_dev,
        bf_dev,
        L_dev,
        stream=stream,
        chunk_size=chunk_size,
        cu_seqlens=cu_dev,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    # ── Stage 3: solve_tril — PTO-ISA tri_inverse kernel ─────────────────────
    A_inv_dev = solve_tril(L_dev, cu_dev, chunk_size, HV, tri_inv_func)
    torch.npu.synchronize()

    # ── Stage 4: wy_kda — NPU PTO kernel ─────────────────────────────────────
    vf_dev = vf.to(dev)
    INV_dev = A_inv_dev  # [B, T, HV, C] float16
    u_dev = torch.zeros(1, T, HV, Vd, device=dev, dtype=torch.float16)
    w_dev = torch.zeros(1, T, HV, Kd, device=dev, dtype=torch.float16)
    run_wy_kda(
        kf_dev,
        vf_dev,
        g_sum_dev,
        bf_dev,
        INV_dev,
        u_dev,
        w_dev,
        stream=stream,
        chunk_size=chunk_size,
        cu_seqlens=cu_dev,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    # ── Stage 5: chunk_h_kda — NPU PTO kernel ────────────────────────────────
    if cu_seqlens_list is None:
        n_chunks = (T + chunk_size - 1) // chunk_size
    else:
        n_chunks = sum(
            (cu_seqlens_list[i + 1] - cu_seqlens_list[i] + chunk_size - 1) // chunk_size
            for i in range(len(cu_seqlens_list) - 1)
        )
    s_snapshots_dev = torch.zeros(n_chunks, HV, Kd, Vd, device=dev, dtype=torch.float16)
    v_corr_dev = torch.zeros(1, T, HV, Vd, device=dev, dtype=torch.float16)
    run_chunk_h_kda(
        kf_dev,
        w_dev,
        u_dev,
        g_sum_dev,
        s_snapshots_dev,
        v_corr_dev,
        stream=stream,
        chunk_size=chunk_size,
        cu_seqlens=cu_dev,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    # ── Stage 6: chunk_o_kda — NPU PTO kernel ────────────────────────────────
    qf_dev = qf.to(dev)
    o_dev = torch.zeros(1, T, HV, Vd, device=dev, dtype=torch.float16)
    run_chunk_o_kda(
        qf_dev,
        kf_dev,
        v_corr_dev,
        s_snapshots_dev,
        g_sum_dev,
        o_dev,
        stream=stream,
        chunk_size=chunk_size,
        cu_seqlens=cu_dev,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()
    return o_dev.cpu()


# ---------------------------------------------------------------------------
# Fused mega-kernel pipeline (all 6 stages in a single NPU launch)
# ---------------------------------------------------------------------------


def mega_pipeline_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta_sig: torch.Tensor,
    cu_seqlens_list,
    scale: float,
    dev: torch.device,
    chunk_size: int = CHUNK,
) -> torch.Tensor:
    """KDA pipeline fused into one NPU launch via ``run_mega_kernel_kda``.

    Input prep mirrors ``pto_pipeline_kda`` exactly (same GQA expansion + scale), so the
    only difference under test is staged dispatch vs. the fused mega-kernel.
    """

    H = q.shape[2]
    HV = v.shape[2]
    G = HV // H

    qf = q.half().repeat_interleave(G, dim=2) * scale  # [B, T, HV, K] fp16
    kf = k.half().repeat_interleave(G, dim=2)  # [B, T, HV, K] fp16
    vf = v.half()
    bf = beta_sig.half()
    g_dev = g_log.half().to(dev)

    cu_dev = (
        torch.tensor(cu_seqlens_list, dtype=torch.int32, device=dev)
        if cu_seqlens_list
        else None
    )
    N_seq = len(cu_seqlens_list) - 1 if cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    o_dev = run_mega_kernel_kda(
        qf.to(dev),
        kf.to(dev),
        vf.to(dev),
        g_dev,
        bf.to(dev),
        cu_dev,
        stream=stream,
        chunk_size=chunk_size,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()
    return o_dev.cpu()


# ---------------------------------------------------------------------------
# Test shapes
# ---------------------------------------------------------------------------

TEST_SHAPES: list[tuple] = [
    (256, None),
    (512, None),
    (1024, None),
    ([0, 256, 512], 512),
    ([0, 128, 384, 768], 768),
    ([0, 384, 512], 512),
    ([0, 128, 256, 512], 512),
]


# ---------------------------------------------------------------------------
# Single-shape runner
# ---------------------------------------------------------------------------


def run_one(
    T_or_cu,
    T_total,
    H: int,
    HV: int,
    dev: torch.device,
    scale: float,
    chunk_size: int,
    tri_inv_func,
) -> tuple[bool, str]:
    cu_list = T_or_cu if isinstance(T_or_cu, list) else None
    T = T_total if cu_list else T_or_cu
    label = f"varlen {cu_list}" if cu_list else f"T={T}"

    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    q = F.normalize(torch.randn(1, T, H, K), dim=-1, p=2)
    k = F.normalize(torch.randn(1, T, H, K), dim=-1, p=2)
    v = torch.randn(1, T, HV, V_DIM)
    # Values in (-0.05, 0): cumulative |g_cs| stays under ~7 so the fp16
    # workspaces in kkt_kda / chunk_o_kda (which stage exp(g_cs) and
    # exp(-g_cs) separately for Cube-core GEMMs) don't overflow.  Larger
    # magnitudes blow up exp(-g_cs) past fp16 max (~65504) -> inf -> NaN.
    g_log = -torch.rand(1, T, HV, K).to(torch.half) * 0.05
    beta_sig = torch.sigmoid(torch.randn(1, T, HV)).to(torch.half)

    # PTO staged pipeline (all stages on NPU)
    o_pto = pto_pipeline_kda(
        q, k, v, g_log, beta_sig, cu_list, scale, dev, chunk_size, tri_inv_func
    )

    # Fused mega-kernel pipeline (single NPU launch)
    o_mega = mega_pipeline_kda(
        q, k, v, g_log, beta_sig, cu_list, scale, dev, chunk_size
    )

    # CPU reference pipeline (all stages on CPU, double precision)
    cpu_ref_kda = RefKDA(torch.double)
    o_cpu = cpu_ref_kda.full_pipeline(
        q, k, v, g_log, beta_sig, cu_list, scale, chunk_size
    )

    e2e_accuracy = NumericalAccuracy(rtol=5e-3, atol=0, ftol=2e-3)
    ok_pto = e2e_accuracy.stats_ok(o_pto, o_cpu)
    ok_mega = e2e_accuracy.stats_ok(o_mega, o_cpu)
    ok_cross_check = e2e_accuracy.stats_ok(o_mega, o_pto)
    ok = ok_pto and ok_mega and ok_cross_check
    label = (
        f"{label}  [staged={'ok' if ok_pto else 'X'} "
        f"mega={'ok' if ok_mega else 'X'} "
        f"mega~staged={'ok' if ok_cross_check else 'X'}]"
    )
    return ok, label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--H", type=int, default=4, help="Number of Q/K heads")
    parser.add_argument(
        "--HV",
        type=int,
        default=None,
        help="Number of V/gate heads (default: same as H)",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK)
    parser.add_argument(
        "--quick", action="store_true", help="Run only T=16 fixed-length"
    )
    args = parser.parse_args()

    H = args.H
    HV = args.HV if args.HV is not None else H
    chunk_size = args.chunk_size
    assert HV % H == 0, f"HV={HV} must be a multiple of H={H}"

    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    scale = K**-0.5
    tri_inv_func = load_tri_inverse()

    shapes = [(256, None)] if args.quick else TEST_SHAPES

    print(f"device={args.device}  H={H}  HV={HV}  K={K}  V={V_DIM}  CHUNK={chunk_size}")
    print(
        f"NPU stages: [1] gate_cumsum_kda  [2] kkt_kda  [3] solve_tril  [4] wy_kda  [5] chunk_h_kda  [6] chunk_o_kda"
    )
    print(f"\n{'='*60}")

    all_pass = True
    for i, (T_or_cu, T_total) in enumerate(shapes):
        t0 = time.time()
        ok, label = run_one(
            T_or_cu, T_total, H, HV, dev, scale, chunk_size, tri_inv_func
        )
        dt = time.time() - t0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{i+1:2d}/{len(shapes)}] {status}  {label}  ({dt:.2f}s)")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
