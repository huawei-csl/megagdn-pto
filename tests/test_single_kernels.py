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

import numpy as np
import torch
import torch.nn.functional as F

from megagdn_pto.fast_inverse import load_tri_inverse, solve_tril
from megagdn_pto.kernel_libs import (
    BLOCK_DIM,
    run_chunk_cumsum,
    run_chunk_h,
    run_chunk_o,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
    transpose_beta,
    transpose_gates,
)

C = 128  # PTO chunk size
D = 128  # head dimension

# Accuracy thresholds
RTOL = 1e-2
ATOL = 1e-5
MAX_RMSE_RATIO = 0.05
MIN_R2 = 0.99
HARD_FAIL_MAX = 1.0


# ---------------------------------------------------------------------------
# CPU Reference implementations
# ---------------------------------------------------------------------------

def _seq_ranges(T: int, cu_seqlens=None) -> list[tuple[int, int]]:
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def ref_cumsum(g: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
    """Chunk-local cumulative sum of gates (fp32)."""
    B, T, H = g.shape
    out = torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            out[:, s:e, :] = g.float()[:, s:e, :].cumsum(dim=1)
    return out


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def ref_solve_tril(A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
    """CPU reference for solve_tril: computes (I + A)^{-1} per chunk submatrix.

    A is strictly lower triangular [B, T, H, cs] (PTO convention).
    """
    B, T, H, _ = A.shape
    out = torch.zeros(B, T, H, cs, dtype=torch.float32)
    Af = A.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            v = e - s
            for h in range(H):
                Ac = Af[0, s:e, h, :v]  # [v, v], strictly lower triangular
                inv = torch.linalg.inv(torch.eye(v) + Ac)
                out[0, s:e, h, :v] = inv
    return out


def ref_kkt(k: torch.Tensor, beta: torch.Tensor, g_cumsum: torch.Tensor,
            cs: int, cu_seqlens=None) -> torch.Tensor:
    """CPU reference for scaled_dot_kkt (GQA: k has Hg heads, beta/g have H heads)."""
    B, T, Hg, Dd = k.shape
    H = beta.shape[2]
    grp = H // Hg
    out = torch.zeros(B, T, H, cs, dtype=torch.float32)
    kf, bf, gf = k.float(), beta.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            v = e - s
            for h in range(H):
                hg = h // grp
                kc, gc = kf[0, s:e, hg, :], gf[0, s:e, h]
                blk = (kc @ kc.T) * _safe_exp(gc[:, None] - gc[None, :]) * bf[0, s:e, h, None]
                mask = torch.arange(v)[:, None] > torch.arange(v)[None, :]
                out[0, s:e, h, :v] = blk * mask.float()
    return out


def ref_chunk_h(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor,
                g_cumsum: torch.Tensor, cs: int, cu_seqlens=None):
    """CPU reference for chunk_h: states S, v_new, final states."""
    B, T, Hg, Dd = k.shape
    H = w.shape[2]
    grp = H // Hg
    kf, wf, uf, gf = k.float(), w.float(), u.float(), g_cumsum.float()
    ranges = _seq_ranges(T, cu_seqlens)
    tc = total_chunks(len(ranges), T, cs, torch.tensor(cu_seqlens) if cu_seqlens else None)
    h_out = torch.zeros(tc, H, Dd, Dd, dtype=torch.float32)
    v_new = torch.zeros_like(uf)
    final = torch.zeros(len(ranges), H, Dd, Dd, dtype=torch.float32)
    ci_base = 0
    for si, (bos, eos) in enumerate(ranges):
        nc = (eos - bos + cs - 1) // cs
        for h in range(H):
            hg = h // grp
            S = torch.zeros(Dd, Dd, dtype=torch.float32)
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                gc = gf[0, s:e, h]
                gl = gc[e - s - 1]
                h_out[ci_base + ci, h] = S.clone()
                vc = uf[0, s:e, h, :] - wf[0, s:e, h, :] @ S
                v_new[0, s:e, h, :] = vc
                kv = kf[0, s:e, hg, :].T @ (vc * torch.exp(gl - gc)[:, None])
                S = torch.exp(gl) * S + kv
            final[si, h] = S
        ci_base += nc
    return h_out, v_new, final


def ref_wy(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, A: torch.Tensor,
           g_cumsum: torch.Tensor, cs: int, cu_seqlens=None):
    """CPU reference for wy_fast."""
    B, T, Hg, Kd = k.shape
    H = v.shape[2]
    grp = H // Hg
    w = torch.zeros(B, T, H, Kd, dtype=torch.float32)
    u = torch.zeros(B, T, H, v.shape[-1], dtype=torch.float32)
    kf, vf, bf, Af, gf = k.float(), v.float(), beta.float(), A.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            valid = e - s
            for h in range(H):
                hg = h // grp
                Ab = Af[0, s:e, h, :valid]
                gc = gf[0, s:e, h]
                vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                kb = kf[0, s:e, hg, :] * bf[0, s:e, h, None] * torch.exp(gc)[:, None]
                u[0, s:e, h, :] = Ab @ vb
                w[0, s:e, h, :] = Ab @ kb
    return w.to(k.dtype), u.to(v.dtype)


def ref_chunk_o(q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor,
                h_states: torch.Tensor, g_cumsum: torch.Tensor, cs: int,
                cu_seqlens=None) -> torch.Tensor:
    """CPU reference for chunk_o (PTO gating convention: min(Δg, 0))."""
    B, T, Hg, Dd = q.shape
    H = v_new.shape[2]
    grp = H // Hg
    qf, kf, vf, gf = q.float(), k.float(), v_new.float(), g_cumsum.float()
    o = torch.zeros(B, T, H, Dd, dtype=torch.float32)
    ranges = _seq_ranges(T, cu_seqlens)
    ci_base = 0
    for bos, eos in ranges:
        nc = (eos - bos + cs - 1) // cs
        for h in range(H):
            hg = h // grp
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                vlen = e - s
                qc, kc, vc, gc = qf[0, s:e, hg, :], kf[0, s:e, hg, :], vf[0, s:e, h, :], gf[0, s:e, h]
                inter = (qc @ h_states[ci_base + ci, h]) * torch.exp(gc)[:, None]
                qk = qc @ kc.T
                causal = torch.arange(vlen)[:, None] >= torch.arange(vlen)[None, :]
                gate = torch.exp(torch.minimum(gc[:, None] - gc[None, :], torch.zeros(vlen, vlen)))
                o[0, s:e, h, :] = inter + (qk * gate * causal.float()) @ vc
        ci_base += nc
    return o


# ---------------------------------------------------------------------------
# Statistical pass/fail
# ---------------------------------------------------------------------------

def _r2(y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
    ref = y_ref.detach().cpu().numpy().ravel().astype(np.float64)
    pred = y_pred.detach().cpu().numpy().ravel().astype(np.float64)
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return float("nan") if ss_tot < 1e-30 else 1.0 - ss_res / ss_tot


def stats_ok(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    diff = (actual - expected).abs()
    mx = diff.max().item()
    if mx > HARD_FAIL_MAX:
        return False
    bound = ATOL + RTOL * expected.abs()
    if (diff <= bound).all():
        return True
    mean_abs = float(expected.float().flatten().abs().mean())
    rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()))
    ratio = rmse / max(mean_abs, 1e-15)
    r2 = _r2(expected, actual)
    if mean_abs < 1e-9:
        return rmse < 5e-4
    return ratio <= MAX_RMSE_RATIO and np.isfinite(r2) and r2 >= MIN_R2


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    label: str
    cu_seqlens_list: list[int] | None
    T: int


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
    for seqlens in [[128], [256], [384], [512], [256, 256], [128, 256], [384, 128],
                    [128, 128, 128], [256, 128, 384]]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
    # Boundary-heavy cases
    for seqlens in [[1, 63, 64, 65, 127, 128, 129, 447],
                    [1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367]]:
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
    cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev) if tc.cu_seqlens_list else None
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    torch.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), C, tc.cu_seqlens_list).to(dev)
    g_t, beta_t = transpose_gates(g_sum), transpose_beta(beta)
    msk = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).float()
    A_out = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    stream = torch.npu.current_stream()._as_parameter_
    run_scaled_dot_kkt(k, beta, g_sum, msk, A_out,
                       stream=stream, g_t=g_t, beta_t=beta_t, chunk_size=C,
                       cu_seqlens=cu, batch_size_override=N_seq, key_heads=HG)
    torch.npu.synchronize()
    ref = ref_kkt(k.cpu(), beta.cpu(), g_sum.cpu(), C, tc.cu_seqlens_list)
    return stats_ok(A_out.float().cpu(), ref)


def test_chunk_h(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T
    cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev) if tc.cu_seqlens_list else None
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    torch.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    w = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    u = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), C, tc.cu_seqlens_list).to(dev)
    g_t = transpose_gates(g_sum)
    stream = torch.npu.current_stream()._as_parameter_
    tc_n = total_chunks(N_seq, T, C, cu)
    s_out = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    run_chunk_h(k, w, u, g_sum, s_out, v_out, fs_out,
                stream=stream, g_t=g_t, chunk_size=C,
                cu_seqlens=cu, batch_size_override=N_seq, key_heads=HG)
    torch.npu.synchronize()
    h_ref, v_ref, fs_ref = ref_chunk_h(k.cpu(), w.cpu(), u.cpu(), g_sum.cpu(), C, tc.cu_seqlens_list)
    ok_h = stats_ok(s_out.float().cpu().view(tc_n, H, D, D), h_ref.float())
    ok_v = stats_ok(v_out.float().cpu(), v_ref.float())
    ok_fs = stats_ok(fs_out.float().cpu().view(N_seq, H, D, D), fs_ref.float())
    return ok_h and ok_v and ok_fs


def test_wy(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T
    cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev) if tc.cu_seqlens_list else None
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    torch.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    A = torch.randn(1, T, H, C, device=dev, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), C, tc.cu_seqlens_list).to(dev)
    g_t, beta_t = transpose_gates(g_sum), transpose_beta(beta)
    stream = torch.npu.current_stream()._as_parameter_
    w_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    run_wy_fast(k, v, beta, g_sum, A, w_out, u_out,
                stream=stream, g_t=g_t, beta_t=beta_t, chunk_size=C,
                cu_seqlens=cu, batch_size_override=N_seq, key_heads=HG)
    torch.npu.synchronize()
    w_ref, u_ref = ref_wy(k.cpu(), v.cpu(), beta.cpu(), A.cpu(), g_sum.cpu(), C, tc.cu_seqlens_list)
    return stats_ok(w_out.float().cpu(), w_ref.float()) and stats_ok(u_out.float().cpu(), u_ref.float())


def test_chunk_o(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T
    cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev) if tc.cu_seqlens_list else None
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    torch.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    q = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    w = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    u = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), C, tc.cu_seqlens_list).to(dev)
    g_t = transpose_gates(g_sum)
    stream = torch.npu.current_stream()._as_parameter_
    tc_n = total_chunks(N_seq, T, C, cu)
    s_out = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    run_chunk_h(k, w, u, g_sum, s_out, v_out, fs_out,
                stream=stream, g_t=g_t, chunk_size=C,
                cu_seqlens=cu, batch_size_override=N_seq, key_heads=HG)
    torch.npu.synchronize()
    msk = torch.tril(torch.ones(C, C, device=dev), diagonal=0).float()
    o_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    run_chunk_o(q, k, v_out, s_out, g_sum, msk, o_out,
                stream=stream, g_t=g_t, chunk_size=C,
                cu_seqlens=cu, batch_size_override=N_seq, key_heads=HG)
    torch.npu.synchronize()
    s_re = s_out.float().cpu().view(tc_n, H, D, D)
    o_ref = ref_chunk_o(q.cpu(), k.cpu(), v_out.cpu(), s_re, g_sum.cpu(), C, tc.cu_seqlens_list)
    return stats_ok(o_out.float().cpu(), o_ref.float())


def test_cumsum(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T
    cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev) if tc.cu_seqlens_list else None
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    torch.manual_seed(42)
    g = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_sum = torch.empty_like(g)
    stream = torch.npu.current_stream()._as_parameter_
    run_chunk_cumsum(g, g_sum, stream=stream, chunk_size=C,
                     cu_seqlens=cu, batch_size_override=N_seq)
    torch.npu.synchronize()
    ref = ref_cumsum(g.cpu(), C, tc.cu_seqlens_list)
    return stats_ok(g_sum.cpu(), ref)


def test_solve_tril(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T
    cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev) if tc.cu_seqlens_list else None
    torch.manual_seed(42)
    # Build A lower-triangular in the (time-within-chunk × chunk-col) space.
    # Position-within-chunk = (t - bos) % C for each sequence starting at bos.
    # This handles non-chunk-aligned sequence starts correctly.
    # Use std=0.1 to keep (I+A)^{-1} well-conditioned in fp16.
    t_within = torch.zeros(T, dtype=torch.long)
    if tc.cu_seqlens_list is not None:
        for i in range(len(tc.cu_seqlens_list) - 1):
            bos, eos = tc.cu_seqlens_list[i], tc.cu_seqlens_list[i + 1]
            t_within[bos:eos] = torch.arange(eos - bos) % C
    else:
        t_within = torch.arange(T) % C
    chunk_mask = (t_within[:, None] > torch.arange(C)[None, :]).float()  # [T, C]
    A_raw = torch.randn(1, T, H, C) * 0.1 * chunk_mask[None, :, None, :]
    A = A_raw.to(torch.float16).to(dev)
    tri_inv = load_tri_inverse()
    A_inv = solve_tril(A, cu, C, H, tri_inv)
    torch.npu.synchronize()
    ref = ref_solve_tril(A.cpu(), C, tc.cu_seqlens_list)
    return stats_ok(A_inv.float().cpu(), ref)


_STAGES = {
    "cumsum":     ("chunk_cumsum",   test_cumsum),
    "kkt":        ("scaled_dot_kkt", test_kkt),
    "solve_tril": ("solve_tril",     test_solve_tril),
    "chunk_h":    ("chunk_h",        test_chunk_h),
    "wy_fast":    ("wy_fast",        test_wy),
    "chunk_o":    ("chunk_o",        test_chunk_o),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--quick", action="store_true", help="Run a single small test case.")
    parser.add_argument("--H-list", default="16,32,48,64", help="Comma-separated value head counts.")
    parser.add_argument("--hg", type=int, default=16, help="Key head count Hg.")
    parser.add_argument("--stage",
                        default="cumsum,kkt,solve_tril,chunk_h,wy_fast,chunk_o",
                        help="Comma-separated stages to test.")
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

    print(f"Device: {args.device}  stages={stages}  H={heads_list}  Hg={HG}  D={D}  C={C}")
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
