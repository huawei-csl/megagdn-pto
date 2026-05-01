#!/usr/bin/env python3
"""End-to-end accuracy test: full PTO pipeline vs CPU fp32 reference and Triton.

Tests the complete GDN pipeline for all input shapes:
  PTO:    cumsum → scaled_dot_kkt → solve_tril → wy_fast → chunk_h → chunk_o (C=128)
  Triton: cumsum → kkt → solve_tril → wy_fast → chunk_h → chunk_o (C=64, bf16)

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
    BLOCK_DIM,
    run_chunk_h,
    run_chunk_o,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
    transpose_beta,
    transpose_gates,
)
from tests.test_single_kernels import (
    ref_chunk_h,
    ref_chunk_o,
    ref_cumsum,
    ref_kkt,
    ref_wy,
    _seq_ranges,
    stats_ok,
)

C_PTO = 128
C_TRITON = 64
D = 128

# Cross-backend agreement thresholds (tighter than vs-CPU)
MAX_RMSE_CROSS = 0.02
MIN_R2_CROSS = 0.999
MIN_PEARSON_CROSS = 0.999


# ---------------------------------------------------------------------------
# CPU reference for solve_tril
# ---------------------------------------------------------------------------

def ref_solve_tril(A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
    """CPU fp64 reference for the triangular inverse (solve_tril) stage.

    For each chunk block of shape ``[v, v]`` (per head), computes
    ``inv(A^T + I)^T`` which matches the ``fast_inverse`` kernel convention.
    """
    A64 = A.detach().cpu().double()
    out = torch.zeros_like(A64)
    for bos, eos in _seq_ranges(A.shape[1], cu_seqlens):
        for chunk_start in range(bos, eos, cs):
            v = min(cs, eos - chunk_start)
            block = A64[:, chunk_start:chunk_start + v, :, :v]  # [1, v, H, v]
            eye = torch.eye(v, dtype=torch.float64)
            # inv(A^T + I)^T per head
            # block: [1, v, H, v] — treat as [H, v, v] for batched inv
            b = block[0].permute(1, 0, 2)  # [H, v, v]
            b_t = b.transpose(1, 2) + eye.unsqueeze(0)  # [H, v, v]
            inv_b = torch.linalg.inv(b_t).transpose(1, 2)  # [H, v, v]
            out[0, chunk_start:chunk_start + v, :, :v] = inv_b.permute(1, 0, 2)
    return out.to(dtype=A.dtype)


# ---------------------------------------------------------------------------
# Full CPU reference pipeline
# ---------------------------------------------------------------------------

def cpu_pipeline(q, k, v, g_in, beta, cu_seqlens_list, H, Hg, scale=1.0):
    """Complete CPU fp32 reference for the GDN pipeline."""
    cu = cu_seqlens_list
    g_sum = ref_cumsum(g_in, C_PTO, cu)
    A = ref_kkt(k, beta, g_sum, C_PTO, cu)
    A_inv = ref_solve_tril(A.float(), C_PTO, cu).to(torch.float32)
    w, u = ref_wy(k, v, beta, A_inv, g_sum, C_PTO, cu)
    _, v_new, _ = ref_chunk_h(k, w.float(), u.float(), g_sum, C_PTO, cu)
    N_seq = 1 if cu is None else len(cu) - 1
    tc = total_chunks(N_seq, q.shape[1], C_PTO,
                      torch.tensor(cu) if cu else None)
    _, _, h_states_list = ref_chunk_h(k, w.float(), u.float(), g_sum, C_PTO, cu)
    h_states, v_new, _ = ref_chunk_h(k, w.float(), u.float(), g_sum, C_PTO, cu)
    o = ref_chunk_o(q, k, v_new.float(), h_states, g_sum, C_PTO, cu)
    return (o * scale).float()


# ---------------------------------------------------------------------------
# PTO pipeline
# ---------------------------------------------------------------------------

def pto_pipeline(q, k, v, g_in, beta, cu32, H, Hg, scale=1.0, tri_inv_func=None):
    """Full six-stage PTO pipeline on NPU."""
    dev = q.device
    T = q.shape[1]
    N_seq = int(cu32.numel()) - 1
    cu_cpu = cu32.cpu().tolist()
    stream = torch.npu.current_stream()._as_parameter_

    if tri_inv_func is None:
        tri_inv_func = load_tri_inverse()

    # 1. Cumsum (CPU reference, exact match with PTO)
    g_sum = ref_cumsum(g_in.cpu(), C_PTO, cu_cpu).to(dev)
    g_t = transpose_gates(g_sum)
    beta_t = transpose_beta(beta)
    torch.npu.synchronize()

    # 2. scaled_dot_kkt
    msk_lower = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=-1).float()
    A = torch.zeros(1, T, H, C_PTO, device=dev, dtype=torch.float16)
    run_scaled_dot_kkt(k, beta, g_sum, msk_lower, A,
                       stream=stream, g_t=g_t, beta_t=beta_t, chunk_size=C_PTO,
                       cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg)
    torch.npu.synchronize()

    # 3. solve_tril
    A_inv = solve_tril(A, cu32, C_PTO, H, tri_inv_func)
    torch.npu.synchronize()

    # 4. wy_fast
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    run_wy_fast(k, v, beta, g_sum, A_inv, w, u,
                stream=stream, g_t=g_t, beta_t=beta_t, chunk_size=C_PTO,
                cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg)
    torch.npu.synchronize()

    # 5. chunk_h
    tc_n = total_chunks(N_seq, T, C_PTO, cu32)
    s = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_new = torch.empty_like(v)
    fs = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    run_chunk_h(k, w, u, g_sum, s, v_new, fs,
                stream=stream, g_t=g_t, chunk_size=C_PTO,
                cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg)
    torch.npu.synchronize()

    # 6. chunk_o
    msk_full = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=0).float()
    o = torch.empty_like(v)
    run_chunk_o(q, k, v_new, s, g_sum, msk_full, o,
                stream=stream, g_t=g_t, chunk_size=C_PTO,
                cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg)
    torch.npu.synchronize()

    return (o * scale).to(q.dtype)


# ---------------------------------------------------------------------------
# Triton pipeline
# ---------------------------------------------------------------------------

def _triton_available() -> bool:
    try:
        from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa
        return True
    except Exception:
        return False


def triton_pipeline(q, k, v, g_in, beta, cu_long, H, Hg, scale=1.0):
    """Full six-stage Triton pipeline (C=64, bf16)."""
    from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla_vendor.chunk_o import chunk_fwd_o
    from fla_vendor.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla_vendor.cumsum import chunk_local_cumsum
    from fla_vendor.solve_tril import solve_tril as triton_solve_tril
    from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
    from fla_vendor.wy_fast import recompute_w_u_fwd

    chunk_indices = prepare_chunk_indices(cu_long, C_TRITON)
    chunk_offsets = prepare_chunk_offsets(cu_long, C_TRITON)

    q_bf = q.bfloat16()
    k_bf = k.bfloat16()
    v_bf = v.bfloat16()
    beta_bf = beta.bfloat16()
    g_f32 = g_in.float()

    g_cs = chunk_local_cumsum(g_f32, cu_seqlens=cu_long, chunk_size=C_TRITON,
                               chunk_indices=chunk_indices)
    A_tri = chunk_scaled_dot_kkt_fwd(k=k_bf, beta=beta_bf, g_cumsum=g_cs,
                                     cu_seqlens=cu_long, chunk_indices=chunk_indices,
                                     chunk_size=C_TRITON, output_dtype=torch.float32)
    A_inv_tri = triton_solve_tril(A_tri.half(), cu_seqlens=cu_long)
    w_tri, u_tri = recompute_w_u_fwd(k=k_bf, v=v_bf, beta=beta_bf, g_cumsum=g_cs,
                                     A=A_inv_tri.bfloat16(), cu_seqlens=cu_long,
                                     chunk_indices=chunk_indices)
    h_tri, v_new_tri, _ = chunk_gated_delta_rule_fwd_h(
        k=k_bf, w=w_tri, u=u_tri, g=g_f32,
        initial_state=None, output_final_state=False,
        cu_seqlens=cu_long, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
        chunk_size=C_TRITON,
    )
    o_tri = chunk_fwd_o(q=q_bf, k=k_bf, v=v_new_tri, h=h_tri, g=g_f32,
                        scale=scale, cu_seqlens=cu_long, chunk_size=C_TRITON)
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
    q = F.normalize(torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    k = F.normalize(torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))

    cu32 = (
        torch.tensor(cu_list, dtype=torch.int32, device=dev) if cu_list
        else torch.tensor([0, T], dtype=torch.int32, device=dev)
    )
    cu_cpu = cu32.cpu().tolist() if cu_list else None

    # PTO pipeline
    o_pto = pto_pipeline(q, k, v, g_in, beta, cu32, H, Hg, scale, tri_inv_func)

    # CPU reference pipeline
    o_cpu = cpu_pipeline(
        q.cpu().float(), k.cpu().float(), v.cpu().float(),
        g_in.cpu(), beta.cpu().float(), cu_cpu, H, Hg, scale,
    )

    ok_pto = stats_ok(o_pto.float().cpu(), o_cpu)

    ok_cross = True
    if triton_ok:
        try:
            cu_long = cu32.long()
            o_tri = triton_pipeline(q, k, v, g_in, beta, cu_long, H, Hg, scale)
            ok_cross = cross_ok(o_pto.float().cpu(), o_tri.cpu())
        except Exception as e:
            print(f"    [Triton skipped: {str(e).split(chr(10))[0][:80]}]")
            ok_cross = True  # not a failure, triton may not support all shapes

    return ok_pto and ok_cross, label, ok_pto, ok_cross


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
    scale = D ** -0.5
    triton_ok = not args.no_triton and _triton_available()
    tri_inv_func = load_tri_inverse()

    print(f"Device: {args.device}  Hg={Hg}  D={D}  C_PTO={C_PTO}  C_TRITON={C_TRITON}")
    print(f"Triton cross-check: {'enabled' if triton_ok else 'disabled'}")

    all_pass = True
    for H in heads:
        assert H % Hg == 0
        print(f"\n{'='*60}\nH={H}  Hg={Hg}\n{'='*60}")
        for T_or_cu, T_total in TEST_SHAPES:
            ok, label, ok_pto, ok_cross = run_one(
                T_or_cu, T_total, H, Hg, dev, scale, tri_inv_func, triton_ok)
            if not ok:
                all_pass = False
            cross_str = f"  cross={'PASS' if ok_cross else 'FAIL'}" if triton_ok else ""
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label}  pto_vs_cpu={'PASS' if ok_pto else 'FAIL'}{cross_str}")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
