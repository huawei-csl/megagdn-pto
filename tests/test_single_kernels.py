#!/usr/bin/env python3
"""Numerical accuracy tests for all six PTO chunk-GDN kernels.

Tests each stage against CPU fp32 reference implementations across a wide
range of packed-varlen shapes, value-head counts H, and GQA key-head counts Hg:

  cumsum, scaled_dot_kkt, solve_tril, wy_fast, chunk_h, chunk_o

Usage::

    python tests/test_single_kernels.py --device npu:0
    python tests/test_single_kernels.py --device npu:0 --quick
    python tests/test_single_kernels.py --device npu:0 --H-list 32,64 --stage kkt,chunk_h
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass

import torch

from megagdn_pto.fast_inverse import load_tri_inverse, solve_tril
from megagdn_pto.kernel_libs import (
    run_chunk_cumsum,
    run_chunk_h,
    run_chunk_o,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
    transpose_beta,
    transpose_gates,
)
from tests.utils import NumericalAccuracy, generate_random_inputs
from tests.ref_gdn import RefGDN

ACCURACY = NumericalAccuracy()

C = 128  # PTO chunk size
D = 128  # head dimension

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    label: str
    cu_seqlens_list: list[int] | None
    T: int
    dtype: torch.dtype = torch.double

    def __post_init__(self):
        self.ref_gdn = RefGDN(self.dtype)


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


def build_test_cases() -> list[TestCase]:
    cases: list[TestCase] = []
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
    # Boundary-heavy cases
    for seqlens in [
        [1, 63, 64, 65, 127, 128, 129, 447],
        [1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367],
    ]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
    rng = random.Random(42)
    for n_seq, total in [(3, 768), (7, 1792), (10, 2560)]:
        cu = _align_cu(_rand_cu(n_seq, total, rng), C)
        cases.append(TestCase(f"varlen rand {n_seq}seq T={cu[-1]}", cu, cu[-1]))
    return cases


# ---------------------------------------------------------------------------
# Per-stage test runners
# ---------------------------------------------------------------------------


def test_kkt(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T

    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    _, k, _, beta, g_in = generate_random_inputs(T, H, HG, D)

    g_sum = tc.ref_gdn.cumsum(g_in, C, tc.cu_seqlens_list).to(g_in.dtype)
    k_npu, beta_npu, g_sum_npu = (
        k.to(torch.float16).to(dev),
        beta.to(torch.float16).to(dev),
        g_sum.to(torch.float32).to(dev),
    )
    g_t, beta_t = transpose_gates(g_sum_npu), transpose_beta(beta_npu)
    mask = torch.tril(torch.ones(C, C, device=dev, dtype=torch.float32), diagonal=-1)
    A_out = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    stream = torch.npu.current_stream()._as_parameter_
    torch.npu.synchronize()
    run_scaled_dot_kkt(
        k_npu,
        beta_npu,
        g_sum_npu,
        mask,
        A_out,
        stream=stream,
        g_t=g_t,
        beta_t=beta_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()
    ref = tc.ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)

    return ACCURACY.stats_ok(A_out.cpu().to(tc.dtype), ref.to(tc.dtype), chunk_size=C)


def test_solve_tril(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T

    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    _, k, _, beta, g_in = generate_random_inputs(T, H, HG, D)
    g_sum = tc.ref_gdn.cumsum(g_in.cpu(), C, tc.cu_seqlens_list).to(tc.dtype)
    A_raw = tc.ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)

    A_dev = A_raw.to(torch.float16).to(dev)
    tri_inv = load_tri_inverse()
    A_inv = solve_tril(A_dev, cu, C, H, tri_inv)
    torch.npu.synchronize()
    ref = tc.ref_gdn.solve_tril(A_dev.cpu(), C, tc.cu_seqlens_list)
    return ACCURACY.stats_ok(A_inv.cpu().to(tc.dtype), ref.to(tc.dtype), chunk_size=C)


def test_wy_fast(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T

    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1

    _, k, v, beta, g_in = generate_random_inputs(T, H, HG, D)
    g_sum = tc.ref_gdn.cumsum(g_in.cpu(), C, tc.cu_seqlens_list)
    A = tc.ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    A_inv = tc.ref_gdn.solve_tril(A, C, tc.cu_seqlens_list)
    stream = torch.npu.current_stream()._as_parameter_

    k_npu, v_npu, beta_npu, g_sum_npu, A_inv_npu = (
        k.to(torch.float16).npu(),
        v.to(torch.float16).npu(),
        beta.to(torch.float16).npu(),
        g_sum.to(torch.float32).npu(),
        A_inv.to(torch.float16).npu(),
    )
    g_t, beta_t = transpose_gates(g_sum_npu), transpose_beta(beta_npu)
    w_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)

    run_wy_fast(
        k_npu,
        v_npu,
        beta_npu,
        g_sum_npu,
        A_inv_npu,
        w_out,
        u_out,
        stream=stream,
        g_t=g_t,
        beta_t=beta_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()
    w_ref, u_ref = tc.ref_gdn.wy_fast(k, v, beta, A_inv, g_sum, C, tc.cu_seqlens_list)
    return ACCURACY.stats_ok(
        w_out.cpu().to(tc.dtype), w_ref.to(tc.dtype), chunk_size=C
    ) and ACCURACY.stats_ok(u_out.cpu().to(tc.dtype), u_ref.to(tc.dtype), chunk_size=C)


def test_chunk_h(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T

    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1

    _, k, v, beta, g_in = generate_random_inputs(T, H, HG, D)
    g_sum = tc.ref_gdn.cumsum(g_in.cpu(), C, tc.cu_seqlens_list)
    A = tc.ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    A_inv = tc.ref_gdn.solve_tril(A, C, tc.cu_seqlens_list)
    w, u = tc.ref_gdn.wy_fast(k, v, beta, A_inv, g_sum, C, tc.cu_seqlens_list)
    stream = torch.npu.current_stream()._as_parameter_

    k_npu, g_sum_npu, w_npu, u_npu = (
        k.to(torch.float16).npu(),
        g_sum.to(torch.float32).npu(),
        w.to(torch.float16).npu(),
        u.to(torch.float16).npu(),
    )
    g_t = transpose_gates(g_sum_npu)

    tc_n = total_chunks(N_seq, T, C, cu)
    h_out = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    run_chunk_h(
        k_npu,
        w_npu,
        u_npu,
        g_sum_npu,
        h_out,
        v_out,
        fs_out,
        stream=stream,
        g_t=g_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()
    h_ref, v_ref, _ = tc.ref_gdn.chunk_h(
        k, w.to(torch.float16), u.to(torch.float16), g_sum, C, tc.cu_seqlens_list
    )
    ok_h = ACCURACY.stats_ok(
        h_out.cpu().to(tc.dtype).view(tc_n, H, D, D), h_ref.to(tc.dtype), chunk_size=C
    )
    ok_v = ACCURACY.stats_ok(v_out.cpu().to(tc.dtype), v_ref.to(tc.dtype), chunk_size=C)
    return ok_h and ok_v


def test_chunk_o(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T

    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1

    q, k, v, beta, g_in = generate_random_inputs(T, H, HG, D)
    g_sum = tc.ref_gdn.cumsum(g_in.cpu(), C, tc.cu_seqlens_list)
    A = tc.ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    A_inv = tc.ref_gdn.solve_tril(A, C, tc.cu_seqlens_list)
    w, u = tc.ref_gdn.wy_fast(k, v, beta, A_inv, g_sum, C, tc.cu_seqlens_list)
    h_states, v_new, _ = tc.ref_gdn.chunk_h(k, w, u, g_sum, C, tc.cu_seqlens_list)

    stream = torch.npu.current_stream()._as_parameter_

    q_npu, k_npu, g_sum_npu, v_new_npu, h_states_npu = (
        q.to(torch.float16).npu(),
        k.to(torch.float16).npu(),
        g_sum.to(torch.float32).npu(),
        v_new.to(torch.float16).npu(),
        h_states.to(torch.float16).npu(),
    )

    o_ref = tc.ref_gdn.chunk_o(
        q_npu.cpu(),
        k_npu.cpu(),
        v_new_npu.cpu(),
        h_states_npu.cpu(),
        g_sum_npu.cpu(),
        C,
        tc.cu_seqlens_list,
    )
    mask_npu = torch.tril(torch.ones(C, C, device=dev), diagonal=0).float()
    o_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    g_t = transpose_gates(g_sum_npu)
    run_chunk_o(
        q_npu,
        k_npu,
        v_new_npu,
        h_states_npu,
        g_sum_npu,
        mask_npu,
        o_out,
        stream=stream,
        g_t=g_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()

    return ACCURACY.stats_ok(o_out.cpu().to(tc.dtype), o_ref.to(tc.dtype), chunk_size=C)


def test_cumsum(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T

    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1

    g = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_sum = torch.empty_like(g)
    stream = torch.npu.current_stream()._as_parameter_
    run_chunk_cumsum(
        g, g_sum, stream=stream, chunk_size=C, cu_seqlens=cu, batch_size_override=N_seq
    )
    torch.npu.synchronize()
    ref = tc.ref_gdn.cumsum(g.cpu(), C, tc.cu_seqlens_list)
    return ACCURACY.stats_ok(g_sum.cpu().to(tc.dtype), ref.to(tc.dtype))


_STAGES = {
    "cumsum": ("chunk_cumsum", test_cumsum),
    "kkt": ("scaled_dot_kkt", test_kkt),
    "solve_tril": ("solve_tril", test_solve_tril),
    "chunk_h": ("chunk_h", test_chunk_h),
    "wy_fast": ("wy_fast", test_wy_fast),
    "chunk_o": ("chunk_o", test_chunk_o),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument(
        "--quick", action="store_true", help="Run a single small test case."
    )
    parser.add_argument(
        "--H-list", default="16,32,48,64", help="Comma-separated value head counts."
    )
    parser.add_argument("--hg", type=int, default=16, help="Key head count Hg.")
    parser.add_argument(
        "--stage",
        default="cumsum,kkt,solve_tril,chunk_h,wy_fast,chunk_o",
        help="Comma-separated stages to test.",
    )
    args = parser.parse_args()

    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    for s in stages:
        if s not in _STAGES:
            sys.exit(f"Unknown stage {s!r}; choose from {list(_STAGES)}")

    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    heads_list = [int(x) for x in args.H_list.split(",") if x.strip()]
    HG = args.hg

    cases = [TestCase("quick T=128", None, 128)] if args.quick else build_test_cases()

    print(
        f"Device: {args.device}  stages={stages}  H={heads_list}  Hg={HG}  D={D}  C={C}"
    )
    all_pass = True

    for stage in stages:
        name, fn = _STAGES[stage]
        print(f"\n{'=' * 60}\nStage: {name}\n{'=' * 60}")
        for H in heads_list:
            assert H % HG == 0, f"H={H} must be divisible by Hg={HG}"
            print(f"\n  H={H} (Hg={HG})")
            for i, tc in enumerate(cases):
                t0 = time.time()
                ok = fn(tc, dev, H, HG)
                dt = time.time() - t0
                status = "PASS" if ok else "FAIL"
                if not ok:
                    all_pass = False
                print(f"    [{i+1:2d}/{len(cases)}] {status}  {tc.label}  ({dt:.2f}s)")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
