#!/usr/bin/env python3
"""CPU float32 reference implementations and unit tests for each KDA stage.

Matches the math of ``kda_naive.naive_chunk_kda`` exactly, which is used as
ground truth in test_kda_e2e.py.

KDA pipeline stages:
  gate_cumsum → kkt (L matrix) → inversion → wy (u, w) → chunk_h_kda (snapshots + v_corr) → chunk_o_kda (output)

Key math (see kda_naive.py):
  - g is per-dimension log-space decay (natural exp applied internally)
  - beta is post-sigmoid scalar per (position, head)
  - L[r,c] = beta[r] * k_r·(k_c*exp(g_cs[r]-g_cs[c])) for r>c  (strictly lower tri)
  - (I+L)^{-1} via Neumann recursion: A=-L; for i: A[i,:i]+=A[i,:]@A[:,:i]; INV=A+I
  - A_final = (I+L)^{-1} @ diag(beta)  (column-scale after inversion)
  - u = A_final @ v = INV @ (beta*v),  w = A_final @ (exp(g)*k) = INV @ (beta*exp(g)*k)
  - Aqk[r,c] = q_r·(k_c*exp(g_cs[r]-g_cs[c])) for r>=c  (causal, includes diagonal)
  - output: (q*exp(g_cs)) @ S + Aqk @ (u - w @ S)
  - state:  S_new[k,:] = exp(g_total[k]) * S[k,:] + sum_c k_rest[c,k]*v_corr[c,:]

Stage device requirements:
  - cumsum: runs on NPU (requires --device); calls gate_cumsum_kda PTO kernel.
  - all others: CPU float32 reference only.

Usage:

    python tests/test_kda_single_kernels.py --device npu:0
    python tests/test_kda_single_kernels.py --device npu:0 --quick
    python tests/test_kda_single_kernels.py --device npu:0 --stage kkt,inv
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

from megagdn_pto.fast_inverse import load_tri_inverse, solve_tril
from megagdn_pto.kda_kernel_libs import (
    run_chunk_o_kda,
    run_chunk_h_kda,
    run_wy_kda,
    run_gate_cumsum_kda,
    run_kkt_kda,
)

from tests.utils import NumericalAccuracy, generate_random_inputs
from tests.ref_kda import RefKDA

ACCURACY = NumericalAccuracy()

C = 128  # PTO chunk size
D = 128  # head dimension

torch.manual_seed(42)

CHUNK = 128  # small chunk for fast CPU tests
K = 128  # key/query dimension
V_DIM = 128  # value dimension

RTOL = 1e-2
ATOL = 1e-5
MAX_RMSE_RATIO = 0.05
MIN_R2 = 0.99


# ---------------------------------------------------------------------------
# Test-case registry
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    label: str
    cu_seqlens_list: list[int] | None
    T: int
    dtype: torch.dtype = torch.double

    def __post_init__(self):
        self.ref_kda = RefKDA(self.dtype)


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu


def _rand_cu(n_seq: int, total: int, rng: random.Random) -> list[int]:
    if n_seq == 1:
        return [0, total]
    bnd = sorted(rng.sample(range(1, total), n_seq - 1))
    return [0] + bnd + [total]


def _align_cu(raw: list[int], cs: int) -> list[int]:
    aligned = [0]
    for i in range(1, len(raw) - 1):
        val = ((raw[i] + cs - 1) // cs) * cs
        aligned.append(max(val, aligned[-1] + cs))
    total = max(raw[-1], aligned[-1] + cs)
    aligned.append(((total + cs - 1) // cs) * cs)
    return aligned


def _build_test_cases(quick: bool) -> list[TestCase]:
    if quick:
        return [TestCase("quick T=128", None, 128)]
    cases = []
    for T in [128, 256, 385, 512, 1024]:
        cases.append(TestCase(f"fixed T={T}", None, T))
    for seqlens in [
        [128],
        [256],
        [384],
        [512],
        [256, 256],
        [128, 256],
        [384, 128],
        [128, 128, 128],
        [256, 128, 384],
    ]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
    for seqlens in [
        [1, 63, 64, 65, 127, 128, 129, 447],
        [1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367],
    ]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
    rng = random.Random(42)
    for n_seq, total in [(3, 768), (7, 1792), (10, 2560)]:
        cu = _align_cu(_rand_cu(n_seq, total, rng), CHUNK)
        cases.append(TestCase(f"varlen rand {n_seq}seq T={cu[-1]}", cu, cu[-1]))
    return cases


def _make_inputs(tc: TestCase, H: int, HV: int | None = None):
    HV = HV or H
    T = tc.T
    torch.manual_seed(42)
    q = torch.randn(1, T, H, K)
    k = torch.randn(1, T, H, K)
    # L2-normalize q and k to match actual KDA usage (model always normalizes).
    # Unnormalized keys with K=128 produce L matrices with condition numbers ~1e6,
    # making float32 linalg.inv inaccurate. Normalized keys give L entries ~0
    # (random orthogonal vectors), keeping (I+L) well-conditioned.
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = torch.randn(1, T, HV, V_DIM)
    # Keep cumulative gate magnitudes within fp16 range:
    # for CHUNK=128 the kernel computes exp(-g_cs); with values in (-0.05, 0)
    # the worst-case cumsum is bounded by ~6.4 so exp stays within fp16 (~600).
    g_log = -torch.rand(1, T, HV, K) * 0.05  # values in (-0.05, 0)
    beta_sig = torch.sigmoid(torch.randn(1, T, HV))
    scale = K**-0.5
    return q, k, v, g_log, beta_sig, scale


# ---------------------------------------------------------------------------
# Per-stage test functions
# ---------------------------------------------------------------------------


def test_gate_cumsum(tc: TestCase, H: int, dev: "torch.device | None" = None) -> bool:
    """Compare gate_cumsum_kda NPU kernel output to CPU float32 reference.

    Runs the PTO kernel on ``dev`` and checks that it matches ``tc.ref_kda.gate_cumsum``
    element-wise within tolerance.

    Args:
        tc:  Test case (defines T and optional cu_seqlens).
        H:   Number of value/gate heads HV.
        dev: NPU device (e.g. ``torch.device("npu:0")``).  Required.
    """
    if dev is None:
        raise ValueError("test_gate_cumsum requires --device (NPU device).")

    _, _, _, g_log, _, _ = _make_inputs(tc, H)
    # g_log: [1, T, H, K]
    g_dev = g_log.half().to(dev)
    g_sum = torch.empty_like(g_dev)
    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_gate_cumsum_kda(
        g_dev,
        g_sum,
        stream=stream,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    ref = tc.ref_kda.gate_cumsum(g_dev.cpu(), CHUNK, tc.cu_seqlens_list)
    acc = NumericalAccuracy(rtol=5e-3, atol=0, ftol=2e-3)
    return acc.stats_ok(g_sum.cpu(), ref)


def test_kkt(tc: TestCase, H: int, dev=None) -> bool:
    """Compare kkt_kda NPU kernel output to tc.ref_kda.kkt_kda CPU reference."""
    if dev is None:
        raise ValueError("test_kkt_npu requires --device (NPU device).")

    _, k, _, g_log, beta_sig, _ = _make_inputs(tc, H)

    g_cs = tc.ref_kda.gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)

    L_npu = torch.zeros(1, tc.T, H, CHUNK, device=dev, dtype=torch.float16)
    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_
    torch.npu.synchronize()

    run_kkt_kda(
        k.half().to(dev),
        g_cs.half().to(dev),
        beta_sig.half().to(dev),
        L_npu,
        stream=stream,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    ref = tc.ref_kda.kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    return ACCURACY.stats_ok(L_npu.cpu(), ref)


def test_inv(tc: TestCase, H: int, dev=None) -> bool:
    """Compare PTO-ISA tri_inverse output to ref_inversion_kda CPU reference."""
    if dev is None:
        raise ValueError("test_inv requires --device (NPU device).")

    _, k, _, g_log, beta_sig, _ = _make_inputs(tc, H)

    g_cs = tc.ref_kda.gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref = tc.ref_kda.kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)

    L_fp16 = L_ref.to(torch.float16).to(dev)
    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )

    tri_inv = load_tri_inverse()
    A_inv = solve_tril(L_fp16, cu, CHUNK, H, tri_inv)
    torch.npu.synchronize()

    ref = tc.ref_kda.inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    return ACCURACY.stats_ok(A_inv.float().cpu(), ref)


def test_wy(tc: TestCase, H: int, dev=None) -> bool:
    """Compare wy_kda NPU kernel output (u, w) to tc.ref_kda.wy_kda CPU reference.

    Feeds CPU-computed INV (via torch.linalg.inv) into the kernel so this
    isolates wy's correctness from the inversion stage — same strategy as
    test_inv at line 574.
    """
    if dev is None:
        raise ValueError("test_wy requires --device (NPU device).")

    _, k, v, g_log, beta_sig, _ = _make_inputs(tc, H)

    g_cs = tc.ref_kda.gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref = tc.ref_kda.kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV_ref = tc.ref_kda.inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    u_ref, w_ref = tc.ref_kda.wy_kda(
        k.float(), v.float(), g_cs, beta_sig, INV_ref, CHUNK, tc.cu_seqlens_list
    )

    k_d = k.half().to(dev)
    v_d = v.half().to(dev)
    g_cs_d = g_cs.half().to(dev)
    beta_d = beta_sig.half().to(dev)
    INV_d = INV_ref.half().to(dev)
    u_npu = torch.zeros(1, tc.T, H, V_DIM, device=dev, dtype=torch.float16)
    w_npu = torch.zeros(1, tc.T, H, K, device=dev, dtype=torch.float16)
    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_wy_kda(
        k_d,
        v_d,
        g_cs_d,
        beta_d,
        INV_d,
        u_npu,
        w_npu,
        stream=stream,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    return ACCURACY.stats_ok(u_npu.cpu(), u_ref) and ACCURACY.stats_ok(
        w_npu.cpu(), w_ref
    )


def test_chunk_h_kda(tc: TestCase, H: int, dev=None) -> bool:
    """Compare chunk_h_kda NPU kernel output (s_snapshots, v_corr) to ref."""
    if dev is None:
        raise ValueError("test_chunk_h_kda requires --device (NPU device).")

    _, k, v, g_log, beta_sig, _ = _make_inputs(tc, H)

    # Build inputs by chaining the CPU reference pipeline (matches test_wy).
    g_cs = tc.ref_kda.gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref = tc.ref_kda.kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV_ref = tc.ref_kda.inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    u_ref, w_ref = tc.ref_kda.wy_kda(
        k.float(), v.float(), g_cs, beta_sig, INV_ref, CHUNK, tc.cu_seqlens_list
    )
    s_ref, vcorr_ref = tc.ref_kda.chunk_h_kda(
        k.float(), u_ref, w_ref, g_cs, CHUNK, tc.cu_seqlens_list
    )

    s_npu = torch.zeros_like(s_ref, device=dev, dtype=torch.float16)
    vcorr_npu = torch.zeros(1, tc.T, H, V_DIM, device=dev, dtype=torch.float16)
    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_chunk_h_kda(
        k.half().to(dev),
        w_ref.half().to(dev),
        u_ref.half().to(dev),
        g_cs.half().to(dev),
        s_npu,
        vcorr_npu,
        stream=stream,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    return ACCURACY.stats_ok(s_npu.cpu(), s_ref) and ACCURACY.stats_ok(
        vcorr_npu.cpu(), vcorr_ref
    )


def test_chunk_o_kda(tc: TestCase, H: int, dev=None) -> bool:
    """Compare chunk_o_kda NPU kernel output to tc.ref_kda.chunk_o_kda CPU reference."""
    if dev is None:
        raise ValueError("test_chunk_o_kda requires --device (NPU device).")

    q, k, v, g_log, beta_sig, scale = _make_inputs(tc, H)

    # Apply scale (matches cpu_pipeline_kda lines 416-418).  No GQA expansion
    # needed since _make_inputs gives H == HV.
    qf = q.float() * scale
    kf = k.float()

    # Reference pipeline up through chunk_h_kda.
    g_cs = tc.ref_kda.gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref = tc.ref_kda.kkt_kda(kf, g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV_ref = tc.ref_kda.inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    u_ref, w_ref = tc.ref_kda.wy_kda(
        kf, v.float(), g_cs, beta_sig, INV_ref, CHUNK, tc.cu_seqlens_list
    )
    s_ref, vcorr_ref = tc.ref_kda.chunk_h_kda(
        kf, u_ref, w_ref, g_cs, CHUNK, tc.cu_seqlens_list
    )
    # Round-trip through fp16 to match what the NPU kernel actually receives.
    # s_ref accumulates across chunks; the rounding error in s_ref.half() propagates
    # through q_eff @ S and causes the NPU output to diverge from a float32 reference.
    o_ref = tc.ref_kda.chunk_o_kda(
        qf,
        kf,
        vcorr_ref.half().float(),
        s_ref.half().float(),
        g_cs,
        CHUNK,
        tc.cu_seqlens_list,
    )

    # NPU run.
    o_npu = torch.zeros(1, tc.T, H, V_DIM, device=dev, dtype=torch.float16)
    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_chunk_o_kda(
        qf.half().to(dev),
        kf.half().to(dev),
        vcorr_ref.half().to(dev),
        s_ref.half().to(dev),
        g_cs.half().to(dev),
        o_npu,
        stream=stream,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    return ACCURACY.stats_ok(o_npu.cpu(), o_ref)


# ---------------------------------------------------------------------------
# Stage registry and runner
# ---------------------------------------------------------------------------

_STAGES = {
    "cumsum": ("Gate cumsum", test_gate_cumsum),
    "kkt": ("KKT NPU kernel", test_kkt),
    "inv": ("Linalg (I+L)^{-1}", test_inv),
    "wy": ("WY transform (u, w)", test_wy),
    "chunk_h": ("chunk_h_kda (snapshots)", test_chunk_h_kda),
    "chunk_o": ("chunk_o_kda (output)", test_chunk_o_kda),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--stage", default=",".join(_STAGES))
    parser.add_argument(
        "--device",
        default=os.getenv("GDN_NPU_DEVICE", "npu:0"),
        help="NPU device for stages that run on device (e.g. cumsum). "
        "CPU-only stages ignore this. Default: $GDN_NPU_DEVICE or npu:0.",
    )
    args = parser.parse_args()

    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    for s in stages:
        if s not in _STAGES:
            sys.exit(f"Unknown stage {s!r}; choose from {list(_STAGES)}")

    # Initialise NPU device only when a device stage is requested.
    dev = None
    _DEVICE_STAGES = {"cumsum", "kkt", "inv", "wy", "chunk_h", "chunk_o"}
    if any(s in _DEVICE_STAGES for s in stages):
        torch.npu.set_device(args.device)
        dev = torch.device(args.device)

    cases = _build_test_cases(args.quick)
    H = args.H
    print(
        f"device={args.device}  H={H}  K={K}  V={V_DIM}  CHUNK={CHUNK}  cases={len(cases)}"
    )

    all_pass = True
    for stage in stages:
        name, fn = _STAGES[stage]
        print(f"\n{'='*60}\nStage: {name}\n{'='*60}")
        for i, tc in enumerate(cases):
            t0 = time.time()
            ok = fn(tc, H, dev)
            dt = time.time() - t0
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{i+1:2d}/{len(cases)}] {status}  {tc.label}  ({dt:.2f}s)")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
