#!/usr/bin/env python3
"""End-to-end accuracy test: full PTO pipeline vs CPU fp32 reference and Triton.

Tests the complete GDN pipeline for all input shapes:
  PTO:    cumsum -> scaled_dot_kkt -> solve_tril -> wy_fast -> chunk_h -> chunk_o (C=128)
  Triton: cumsum -> kkt -> solve_tril -> wy_fast -> chunk_h -> chunk_o (C=64, bf16)

Both are checked against an independent CPU fp32 reference pipeline.
PTO and Triton outputs are also compared pairwise.

GQA layout: Q/K use ``Hg`` heads, V/gates use ``H`` heads (``H % Hg == 0``).

Usage::

    python tests/test_e2e.py --device npu:0
    python tests/test_e2e.py --device npu:0 --H 32 --hg 16
    python tests/test_e2e.py --device npu:0 --no-triton
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Add triton baseline to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_TRITON_BASELINE = os.path.join(_REPO_ROOT, "kernels", "triton_baseline")
if _TRITON_BASELINE not in sys.path:
    sys.path.insert(0, _TRITON_BASELINE)

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
from megagdn_pto.mega_kernel import run_mega_kernel
from tests.ref_gdn import RefGDN
from tests.utils import NumericalAccuracy

C_PTO = 128
C_TRITON = 64
D = 128

# Cross-backend agreement thresholds (tighter than vs-CPU)
MAX_RMSE_CROSS = 0.02
MIN_R2_CROSS = 0.999
MIN_PEARSON_CROSS = 0.999

ACCURACY = NumericalAccuracy()


# ---------------------------------------------------------------------------
# PTO pipeline
# ---------------------------------------------------------------------------


def pto_pipeline(
    q,
    k,
    v,
    g_in,
    beta,
    cu32,
    H,
    Hg,
    scale=1.0,
    tri_inv_func=None,
    initial_state=None,
    return_final_state=False,
):
    """Full six-stage PTO pipeline on NPU."""
    dev = q.device
    T = q.shape[1]
    N_seq = int(cu32.numel()) - 1

    if tri_inv_func is None:
        tri_inv_func = load_tri_inverse()

    # 1. Cumsum
    g_sum = torch.empty_like(g_in)
    run_chunk_cumsum(
        g_in,
        g_sum,
        chunk_size=C_PTO,
        cu_seqlens=cu32,
        batch_size_override=N_seq,
    )
    g_t = transpose_gates(g_sum)
    beta_t = transpose_beta(beta)

    # 2. scaled_dot_kkt
    msk_lower = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=-1).float()
    A = torch.zeros(1, T, H, C_PTO, device=dev, dtype=torch.float16)
    run_scaled_dot_kkt(
        k,
        beta,
        g_sum,
        msk_lower,
        A,
        g_t=g_t,
        beta_t=beta_t,
        chunk_size=C_PTO,
        cu_seqlens=cu32,
        batch_size_override=N_seq,
        key_heads=Hg,
    )

    # 3. solve_tril
    A_inv = solve_tril(A, cu32, C_PTO, H, tri_inv_func)

    # 4. wy_fast
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    run_wy_fast(
        k,
        v,
        beta,
        g_sum,
        A_inv,
        w,
        u,
        g_t=g_t,
        beta_t=beta_t,
        chunk_size=C_PTO,
        cu_seqlens=cu32,
        batch_size_override=N_seq,
        key_heads=Hg,
    )

    # 5. chunk_h
    tc_n = total_chunks(N_seq, T, C_PTO, cu32)
    s = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_new = torch.empty_like(v)
    fs = (
        torch.zeros(N_seq, H, D, D, device=dev, dtype=torch.float16)
        if return_final_state
        else None
    )
    run_chunk_h(
        k,
        w,
        u,
        g_sum,
        s,
        v_new,
        fs,
        g_t=g_t,
        chunk_size=C_PTO,
        cu_seqlens=cu32,
        batch_size_override=N_seq,
        key_heads=Hg,
        initial_state=initial_state,
    )

    # 6. chunk_o
    msk_full = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=0).float()
    o = torch.empty_like(v)
    run_chunk_o(
        q,
        k,
        v_new,
        s,
        g_sum,
        msk_full,
        o,
        g_t=g_t,
        chunk_size=C_PTO,
        cu_seqlens=cu32,
        batch_size_override=N_seq,
        key_heads=Hg,
    )

    out = (o * scale).to(q.dtype)
    return (out, fs) if return_final_state else out


# ---------------------------------------------------------------------------
# Triton pipeline
# ---------------------------------------------------------------------------


def _triton_available() -> bool:
    try:
        from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa

        return True
    except Exception:
        return False


def triton_pipeline(q, k, v, g_in, beta, cu_long, H, Hg, scale=1.0, initial_state=None):
    """Full six-stage Triton pipeline (C=64, bf16)."""
    from fla_vendor.chunk import chunk_gated_delta_rule_fwd

    _, o_tri, _, _, _, _, _ = chunk_gated_delta_rule_fwd(
        q, k, v, g_in, beta, scale, initial_state, False, cu_long
    )
    torch.npu.synchronize()
    return o_tri.float()


# ---------------------------------------------------------------------------
# Cross-backend check
# ---------------------------------------------------------------------------


def _r2(ref, pred):
    r = ref.numpy().ravel().astype(np.float64)
    p = pred.numpy().ravel().astype(np.float64)
    ss_res = np.sum((r - p) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    return float("nan") if ss_tot < 1e-30 else 1.0 - ss_res / ss_tot


def cross_ok(pto: torch.Tensor, triton: torch.Tensor) -> bool:
    diff = (pto - triton).abs()
    mean_abs = float(triton.flatten().abs().mean())
    rmse = float(torch.sqrt((diff.flatten() ** 2).mean()))
    ratio = rmse / max(mean_abs, 1e-15)
    r2 = _r2(triton, pto)
    return ratio <= MAX_RMSE_CROSS and r2 >= MIN_R2_CROSS


def _format_compare(name: str, actual: torch.Tensor, expected: torch.Tensor, ok: bool) -> str:
    diff = (actual - expected).abs()
    flat_diff = diff.flatten()
    max_pos = int(flat_diff.argmax().item())
    max_idx = tuple(int(i) for i in np.unravel_index(max_pos, tuple(diff.shape)))
    max_abs = float(flat_diff[max_pos])
    mean_abs = float(expected.float().flatten().abs().mean())
    rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()))
    ratio = rmse / max(mean_abs, 1e-15)
    r2 = _r2(expected, actual)
    return (
        f"    {name}={'PASS' if ok else 'FAIL'}"
        f" max_abs={max_abs:.4e}"
        f" rmse={rmse:.4e}"
        f" rmse/mean_ref_abs={ratio:.4e}"
        f" r2={r2:.8f}"
        f" max_idx={max_idx}"
        f" actual={float(actual.flatten()[max_pos]):.4e}"
        f" expected={float(expected.flatten()[max_pos]):.4e}"
    )


def _failure_details(
    label: str,
    *,
    pto_output: tuple[torch.Tensor, torch.Tensor, bool],
    pto_final_state: tuple[torch.Tensor, torch.Tensor, bool],
    mega_output: tuple[torch.Tensor, torch.Tensor, bool],
    mega_final_state: tuple[torch.Tensor, torch.Tensor, bool],
    cross_output: tuple[torch.Tensor, torch.Tensor, bool] | None,
) -> list[str]:
    lines = [f"    [details for {label}]"]
    for name, item in (
        ("pto.output", pto_output),
        ("pto.final_state", pto_final_state),
        ("mega.output", mega_output),
        ("mega.final_state", mega_final_state),
    ):
        actual, expected, ok = item
        if not ok:
            lines.append(_format_compare(name, actual, expected, ok))
    if cross_output is not None:
        actual, expected, ok = cross_output
        if not ok:
            lines.append(_format_compare("cross.output", actual, expected, ok))
    return lines if len(lines) > 1 else []


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


def run_one(T_or_cu, T_total, H, Hg, dev, scale, tri_inv_func, triton_ok):
    cu_list = T_or_cu if isinstance(T_or_cu, list) else None
    T = T_total if cu_list else T_or_cu
    N_seq = len(cu_list) - 1 if cu_list else 1
    label = f"varlen {cu_list}" if cu_list else f"T={T}"

    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    q = F.normalize(
        torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16), dim=-1, p=2
    )
    k = F.normalize(
        torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16), dim=-1, p=2
    )
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))

    cu32 = (
        torch.tensor(cu_list, dtype=torch.int32, device=dev)
        if cu_list
        else torch.tensor([0, T], dtype=torch.int32, device=dev)
    )
    cu_cpu = cu32.cpu().tolist() if cu_list else None
    h0 = (0.01 * torch.randn(N_seq, H, D, D, device=dev, dtype=torch.float16)).contiguous()

    # PTO pipeline
    o_pto, fs_pto = pto_pipeline(
        q,
        k,
        v,
        g_in,
        beta,
        cu32,
        H,
        Hg,
        scale,
        tri_inv_func,
        initial_state=h0,
        return_final_state=True,
    )

    # CPU reference pipeline
    cpu_ref_gdn = RefGDN(torch.double)
    o_cpu, fs_cpu = cpu_ref_gdn.run_full_pipeline(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        g_in.cpu(),
        beta.cpu(),
        cu_cpu,
        H,
        Hg,
        scale,
        initial_state=h0.cpu(),
        return_final_state=True,
    )

    # PTO mega kernel
    o_mega, fs_mega = run_mega_kernel(
        q,
        k,
        v,
        g_in,
        beta,
        cu32,
        chunk_size=C_PTO,
        scale=scale,
        key_heads=Hg,
        initial_state=h0,
        return_final_state=True,
    )

    o_pto_cpu = o_pto.double().cpu()
    fs_pto_cpu = fs_pto.double().cpu()
    o_mega_cpu = o_mega.double().cpu()
    fs_mega_cpu = fs_mega.double().cpu()
    o_cpu = o_cpu.double()
    fs_cpu = fs_cpu.double()

    ok_pto_output = ACCURACY.stats_ok(o_pto_cpu, o_cpu)
    ok_pto_final_state = ACCURACY.stats_ok(fs_pto_cpu, fs_cpu)
    ok_mega_output = ACCURACY.stats_ok(o_mega_cpu, o_cpu)
    ok_mega_final_state = ACCURACY.stats_ok(fs_mega_cpu, fs_cpu)
    ok_pto = ok_pto_output and ok_pto_final_state
    ok_mega = ok_mega_output and ok_mega_final_state

    ok_cross = True
    cross_debug = None
    if triton_ok:
        try:
            cu_long = cu32.long()
            o_tri = triton_pipeline(
                q, k, v, g_in, beta, cu_long, H, Hg, scale, initial_state=h0
            )
            o_tri_cpu = o_tri.double().cpu()
            ok_cross = cross_ok(o_pto_cpu.float(), o_tri_cpu.float())
            cross_debug = (o_pto_cpu, o_tri_cpu, ok_cross)
        except Exception as e:
            print(f"    [Triton skipped: {str(e).split(chr(10))[0][:80]}]")
            ok_cross = True  # not a failure, triton may not support all shapes

    failure_details = []
    if not (ok_pto and ok_cross and ok_mega):
        failure_details = _failure_details(
            label,
            pto_output=(o_pto_cpu, o_cpu, ok_pto_output),
            pto_final_state=(fs_pto_cpu, fs_cpu, ok_pto_final_state),
            mega_output=(o_mega_cpu, o_cpu, ok_mega_output),
            mega_final_state=(fs_mega_cpu, fs_cpu, ok_mega_final_state),
            cross_output=cross_debug,
        )

    return ok_pto and ok_cross and ok_mega, label, ok_pto, ok_cross, ok_mega, failure_details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--hg", type=int, default=16)
    parser.add_argument("--H-list", default=None, help="Comma-separated H values.")
    parser.add_argument("--no-triton", action="store_true")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    heads = [int(x) for x in args.H_list.split(",")] if args.H_list else [args.H]
    Hg = args.hg
    scale = D**-0.5
    triton_ok = not args.no_triton and _triton_available()
    tri_inv_func = load_tri_inverse()

    print(f"Device: {args.device}  Hg={Hg}  D={D}  C_PTO={C_PTO}  C_TRITON={C_TRITON}")
    print(f"Triton cross-check: {'enabled' if triton_ok else 'disabled'}")

    all_pass = True
    for H in heads:
        assert H % Hg == 0
        print(f"\n{'='*60}\nH={H}  Hg={Hg}\n{'='*60}")
        for T_or_cu, T_total in TEST_SHAPES:
            ok, label, ok_pto, ok_cross, ok_mega, failure_details = run_one(
                T_or_cu, T_total, H, Hg, dev, scale, tri_inv_func, triton_ok
            )
            if not ok:
                all_pass = False
            cross_str = f"  cross={'PASS' if ok_cross else 'FAIL'}" if triton_ok else ""
            status = "PASS" if ok else "FAIL"
            print(
                f"  {status}  {label}  pto_vs_cpu={'PASS' if ok_pto else 'FAIL'}  "
                f"mega_vs_cpu={'PASS' if ok_mega else 'FAIL'}{cross_str}"
            )
            for line in failure_details:
                print(line)

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
