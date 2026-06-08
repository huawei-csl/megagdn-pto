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

CHUNK = 128   # small chunk for fast CPU tests
K = 128      # key/query dimension
V_DIM = 128  # value dimension

RTOL = 1e-2
ATOL = 1e-5
MAX_RMSE_RATIO = 0.05
MIN_R2 = 0.99


# ---------------------------------------------------------------------------
# Sequence-range helper
# ---------------------------------------------------------------------------

def _seq_ranges(T: int, cu_seqlens=None) -> list[tuple[int, int]]:
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


# ---------------------------------------------------------------------------
# Stage 1 – within-chunk gate cumulative sum
# ---------------------------------------------------------------------------

def ref_gate_cumsum(g: torch.Tensor, chunk_size: int, cu_seqlens=None) -> torch.Tensor:
    """Within-chunk cumulative sum of g [B, T, HV, K].

    Resets at chunk boundaries and sequence boundaries.
    Matches naive_chunk_kda's per-chunk ``g.cumsum(-2)``.
    """
    B, T, HV, Kd = g.shape
    out = torch.zeros(B, T, HV, Kd, dtype=torch.float32)
    gf = g.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, chunk_size):
            s, e = bos + j, min(bos + j + chunk_size, eos)
            out[:, s:e] = gf[:, s:e].cumsum(dim=1)
    return out


# ---------------------------------------------------------------------------
# Stage 2 – L matrix (gated K @ K.T product, strictly lower triangular)
# ---------------------------------------------------------------------------
def ref_kkt_kda(
    k: torch.Tensor,
    g_cs: torch.Tensor,
    beta_sig: torch.Tensor,
    chunk_size: int,
    cu_seqlens=None,
) -> torch.Tensor:
    """L matrix across all chunks: L[r,c]=beta[r]*k_r·(k_c*exp(g_cs[r]-g_cs[c])) for r>c.

    Args:
        k:        [B, T, HV, K]  float32, key vectors (GQA-expanded)
        g_cs:     [B, T, HV, K]  float32, within-chunk cumulative gate sum
        beta_sig: [B, T, HV]     float32, post-sigmoid beta in (0, 1)
        chunk_size: int
        cu_seqlens: list[int] | None

    Returns:
        L: [B, T, HV, chunk_size]  float32, strictly lower triangular per chunk
           (column index padded to chunk_size; unused entries are 0)
    """
    B, T, HV, Kd = k.shape
    L_out = torch.zeros(B, T, HV, chunk_size, dtype=torch.float32)
    kf = k.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, chunk_size):
            s, e = bos + j, min(bos + j + chunk_size, eos)
            c_len = e - s
            for h in range(HV):
                kc = kf[0, s:e, h, :]           # [c_len, K]
                gc = g_cs[0, s:e, h, :]          # [c_len, K]
                bc = beta_sig[0, s:e, h]         # [c_len]
                A = kc * torch.exp(gc)
                B = kc * torch.exp(-gc)
                L_full = A @ B.T
                L_out[0, s:e, h, :c_len] = torch.tril(
                    L_full * bc.unsqueeze(-1), diagonal=-1)
    return L_out


# ---------------------------------------------------------------------------
# Stage 3 – Linalg inverse: (I + L)^{-1}
# ---------------------------------------------------------------------------
def ref_inversion_kda(A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
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
                Ac = Af[0, s:e, h, :v].double().numpy()  # [v, v], strictly lower triangular
                inv = np.linalg.inv(np.eye(v) + Ac)
                out[0, s:e, h, :v] = torch.from_numpy(inv).float()
    return out


# ---------------------------------------------------------------------------
# Stage 4 – WY transform: u and w
# ---------------------------------------------------------------------------

def ref_wy_kda(
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    beta_sig: torch.Tensor,
    INV: torch.Tensor,
    chunk_size: int,
    cu_seqlens=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute u = INV @ (beta*v)  and  w = INV @ (beta*exp(g_cs)*k)  for all chunks.

    naive_chunk_kda applies beta twice:
      - Row-scale when building L  (baked into INV = (I+L_beta_row)^{-1})
      - Column-scale after:  A_final = INV @ diag(beta)
    Hence u = A_final @ v = INV @ (beta * v).

    Args:
        k:        [B, T, HV, K]         float32 (GQA-expanded)
        v:        [B, T, HV, V_DIM]     float32
        g_cs:     [B, T, HV, K]         float32, within-chunk cumulative gate sum
        beta_sig: [B, T, HV]            float32, post-sigmoid beta
        INV:      [B, T, HV, chunk_size] float32, from ref_inversion_kda
        chunk_size, cu_seqlens: as above

    Returns:
        u: [B, T, HV, V_DIM]  float32
        w: [B, T, HV, K]      float32
    """
    B, T, HV, Kd = k.shape
    Vd = v.shape[-1]
    u_out = torch.zeros(B, T, HV, Vd, dtype=torch.float32)
    w_out = torch.zeros(B, T, HV, Kd, dtype=torch.float32)
    kf, vf = k.float(), v.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, chunk_size):
            s, e = bos + j, min(bos + j + chunk_size, eos)
            c_len = e - s
            for h in range(HV):
                kc  = kf[0, s:e, h, :]          # [c_len, K]
                gc  = g_cs[0, s:e, h, :]         # [c_len, K]
                vc  = vf[0, s:e, h, :]           # [c_len, V]
                bc  = beta_sig[0, s:e, h]        # [c_len]
                INVc = INV[0, s:e, h, :c_len]    # [c_len, c_len]
                beta_col = bc.unsqueeze(-1)       # [c_len, 1]
                u_out[0, s:e, h, :] = INVc @ (vc * beta_col)
                w_out[0, s:e, h, :] = INVc @ (kc * torch.exp(gc) * beta_col)
    return u_out, w_out


# ---------------------------------------------------------------------------
# [Deprecated] Stage 5 – Sequential recurrence: Aqk, state update, output
# ---------------------------------------------------------------------------

def ref_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    g_cs: torch.Tensor,
    chunk_size: int,
    cu_seqlens=None,
) -> torch.Tensor:
    """Sequential recurrence: state S propagates across chunks, producing output.

    For each chunk:
      v_corr = u - w @ S                          (inter-chunk correction)
      Aqk[r,c] = q_r·(k_c*exp(g_cs[r]-g_cs[c])) for r>=c  (causal Q-K kernel)
      o = (q*exp(g_cs)) @ S + Aqk @ v_corr
      S_new[k,:] = exp(g_total[k]) * S[k,:] + k_rest.T @ v_corr

    Args:
        q:    [B, T, HV, K]      float32, queries (scale already applied)
        k:    [B, T, HV, K]      float32, keys (GQA-expanded)
        u:    [B, T, HV, V_DIM]  float32, from ref_wy_kda
        w:    [B, T, HV, K]      float32, from ref_wy_kda
        g_cs: [B, T, HV, K]      float32, within-chunk cumulative gate sum
        chunk_size, cu_seqlens: as above

    Returns:
        o: [B, T, HV, V_DIM]  float32
    """
    B, T, HV, Kd = q.shape
    Vd = u.shape[-1]
    o = torch.zeros(B, T, HV, Vd, dtype=torch.float32)

    for bos, eos in _seq_ranges(T, cu_seqlens):
        nc = (eos - bos + chunk_size - 1) // chunk_size
        for h in range(HV):
            S = torch.zeros(Kd, Vd, dtype=torch.float32)
            for ci in range(nc):
                s = bos + ci * chunk_size
                e = min(s + chunk_size, eos)
                c_len = e - s

                gc      = g_cs[0, s:e, h, :]    # [c_len, K]
                g_total = gc[c_len - 1]          # [K]
                kc      = k[0, s:e, h, :]        # [c_len, K]
                qc      = q[0, s:e, h, :]        # [c_len, K]
                uc      = u[0, s:e, h, :]        # [c_len, V]
                wc      = w[0, s:e, h, :]        # [c_len, K]

                v_corr  = uc - wc @ S            # [c_len, V]

                delta_g = gc.unsqueeze(1) - gc.unsqueeze(0)              # [c_len, c_len, K]
                Aqk = torch.tril(
                    (qc.unsqueeze(1) * kc.unsqueeze(0)
                     * torch.exp(delta_g)).sum(-1),
                    diagonal=0)                                           # [c_len, c_len]

                o[0, s:e, h, :] = (qc * torch.exp(gc)) @ S + Aqk @ v_corr

                k_rest = kc * torch.exp(g_total.unsqueeze(0) - gc)      # [c_len, K]
                S = torch.exp(g_total).unsqueeze(-1) * S + k_rest.T @ v_corr

    return o


# ---------------------------------------------------------------------------
# Stage 5 – chunk_h_kda: sequential state pass (snapshot S + compute v_corr)
# ---------------------------------------------------------------------------

def ref_chunk_h_kda(
    k: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    g_cs: torch.Tensor,
    chunk_size: int,
    cu_seqlens=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential state pass: snapshot S entering each chunk, compute v_corr.

    Mirrors GDN's ref_chunk_h.  The state S [K, V] propagates sequentially
    across chunks within each sequence.  Both output terms in chunk_o depend
    on S *entering* the chunk, so snapshotting here decouples output computation.

    Args:
        k:    [B, T, HV, K]  float32, keys (GQA-expanded)
        u:    [B, T, HV, V]  float32, from ref_wy_kda
        w:    [B, T, HV, K]  float32, from ref_wy_kda
        g_cs: [B, T, HV, K]  float32, within-chunk cumulative gate sum
        chunk_size, cu_seqlens: as in other stages

    Returns:
        s_snapshots: [total_chunks, HV, K, V]  float32 — S entering each chunk
        v_corr:      [B, T, HV, V]             float32 — u - w @ S per position
    """
    B, T, HV, Kd = k.shape
    Vd = u.shape[-1]
    ranges = _seq_ranges(T, cu_seqlens)
    n_chunks = sum((eos - bos + chunk_size - 1) // chunk_size for bos, eos in ranges)
    s_snapshots = torch.zeros(n_chunks, HV, Kd, Vd, dtype=torch.float32)
    v_corr_out  = torch.zeros(B, T, HV, Vd, dtype=torch.float32)
    ci_base = 0
    for bos, eos in ranges:
        nc = (eos - bos + chunk_size - 1) // chunk_size
        for h in range(HV):
            S = torch.zeros(Kd, Vd, dtype=torch.float32)
            for ci in range(nc):
                s = bos + ci * chunk_size
                e = min(s + chunk_size, eos)
                gc      = g_cs[0, s:e, h, :]                       # [c_len, K]
                g_total = gc[-1]                                    # [K]
                kc      = k[0, s:e, h, :]                          # [c_len, K]
                uc      = u[0, s:e, h, :]                          # [c_len, V]
                wc      = w[0, s:e, h, :]                          # [c_len, K]
                s_snapshots[ci_base + ci, h] = S.clone()
                v_corr = uc - wc @ S                               # [c_len, V]
                v_corr_out[0, s:e, h, :] = v_corr
                k_rest = kc * torch.exp(g_total.unsqueeze(0) - gc) # [c_len, K]
                S = torch.exp(g_total).unsqueeze(-1) * S + k_rest.T @ v_corr
        ci_base += nc
    return s_snapshots, v_corr_out


# ---------------------------------------------------------------------------
# Stage 6 – chunk_o_kda: output pass (no sequential state dependency)
# ---------------------------------------------------------------------------

def ref_chunk_o_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v_corr: torch.Tensor,
    s_snapshots: torch.Tensor,
    g_cs: torch.Tensor,
    chunk_size: int,
    cu_seqlens=None,
) -> torch.Tensor:
    """Output pass: compute o from pre-computed state snapshots and v_corr.

    Mirrors GDN's ref_chunk_o.  Each chunk's output depends only on
    s_snapshots[ci] (the state *entering* that chunk) and v_corr — no
    sequential dependency between chunks within this pass.

    Args:
        q:           [B, T, HV, K]         float32, queries (scale already applied)
        k:           [B, T, HV, K]         float32, keys (GQA-expanded)
        v_corr:      [B, T, HV, V]         float32, from ref_chunk_h_kda
        s_snapshots: [total_chunks, HV, K, V]  float32, from ref_chunk_h_kda
        g_cs:        [B, T, HV, K]         float32, within-chunk cumulative gate sum
        chunk_size, cu_seqlens: as in other stages

    Returns:
        o: [B, T, HV, V]  float32
    """
    B, T, HV, Kd = q.shape
    Vd = v_corr.shape[-1]
    o = torch.zeros(B, T, HV, Vd, dtype=torch.float32)
    ci_base = 0
    for bos, eos in _seq_ranges(T, cu_seqlens):
        nc = (eos - bos + chunk_size - 1) // chunk_size
        for h in range(HV):
            for ci in range(nc):
                s = bos + ci * chunk_size
                e = min(s + chunk_size, eos)
                gc  = g_cs[0, s:e, h, :]             # [c_len, K]
                qc  = q[0, s:e, h, :]                # [c_len, K]
                kc  = k[0, s:e, h, :]                # [c_len, K]
                vc  = v_corr[0, s:e, h, :]           # [c_len, V]
                S   = s_snapshots[ci_base + ci, h]   # [K, V]
                q_eff = qc * torch.exp(gc)            # [c_len, K]
                k_eff = kc * torch.exp(-gc)           # [c_len, K]
                inter = q_eff @ S                     # [c_len, V]
                Aqk   = torch.tril(q_eff @ k_eff.T, diagonal=0)  # [c_len, c_len]
                o[0, s:e, h, :] = inter + Aqk @ vc
        ci_base += nc
    return o


# ---------------------------------------------------------------------------
# Full CPU pipeline
# ---------------------------------------------------------------------------

def cpu_pipeline_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta_sig: torch.Tensor,
    cu_seqlens_list,
    scale: float,
    chunk_size: int = CHUNK,
) -> torch.Tensor:
    """Complete CPU float32 KDA reference.  Matches naive_chunk_kda exactly.

    Args:
        q:            [B, T, H,  K]  float32  (NOT L2-normalised; naive doesn't)
        k:            [B, T, H,  K]  float32
        v:            [B, T, HV, V]  float32
        g_log:        [B, T, HV, K]  float32  log-space per-dim gates (exp used inside)
        beta_sig:     [B, T, HV]     float32  post-sigmoid beta in (0, 1)
        cu_seqlens_list: list[int] | None     cumulative sequence lengths
        scale:        float   query scale (typically K**-0.5)
        chunk_size:   int     must divide every sequence length

    Returns:
        o: [B, T, HV, V]  float32
    """
    B, T, H, Kd = q.shape
    HV = v.shape[2]
    G  = HV // H

    # GQA expansion + scale (mirrors naive's repeat_interleave * scale)
    qf = q.float().repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    kf = k.float().repeat_interleave(G, dim=2)           # [B, T, HV, K]
    vf = v.float()
    bf = beta_sig.float()

    g_cs = ref_gate_cumsum(g_log.float(), chunk_size, cu_seqlens_list)
    L    = ref_kkt_kda(kf, g_cs, bf, chunk_size, cu_seqlens_list)
    INV  = ref_inversion_kda(L, chunk_size, cu_seqlens_list)
    u, w = ref_wy_kda(kf, vf, g_cs, bf, INV, chunk_size, cu_seqlens_list)
    s_snapshots, v_corr = ref_chunk_h_kda(kf, u, w, g_cs, chunk_size, cu_seqlens_list)
    return ref_chunk_o_kda(qf, kf, v_corr, s_snapshots, g_cs, chunk_size, cu_seqlens_list)


# ---------------------------------------------------------------------------
# Statistical pass/fail helpers
# ---------------------------------------------------------------------------

def _r2(y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
    ref  = y_ref.detach().float().numpy().ravel().astype(np.float64)
    pred = y_pred.detach().float().numpy().ravel().astype(np.float64)
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return float("nan") if ss_tot < 1e-30 else 1.0 - ss_res / ss_tot


def stats_ok(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    diff = (actual.float() - expected.float()).abs()
    bound = ATOL + RTOL * expected.float().abs()
    if (diff <= bound).all():
        return True
    mean_abs = float(expected.float().abs().mean())
    rmse     = float(torch.sqrt((diff.float() ** 2).mean()))
    ratio    = rmse / max(mean_abs, 1e-15)
    r2       = _r2(expected, actual)
    if mean_abs < 1e-9:
        return rmse < 5e-4
    return ratio <= MAX_RMSE_RATIO and np.isfinite(r2) and r2 >= MIN_R2


# ---------------------------------------------------------------------------
# Test-case registry
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


def _build_test_cases(quick: bool) -> list[TestCase]:
    if quick:
        return [TestCase("quick T=128", None, 128)]
    cases = []
    for T in [128, 256, 385, 512, 1024]:
        cases.append(TestCase(f"fixed T={T}", None, T))
    for seqlens in [[128], [256], [384], [512], [256, 256], [128, 256], [384, 128],
                    [128, 128, 128], [256, 128, 384]]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
    for seqlens in [[1, 63, 64, 65, 127, 128, 129, 447],
                    [1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367]]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
    rng = random.Random(42)
    for n_seq, total in [(3, 768), (7, 1792), (10, 2560)]:
        cu = _align_cu(_rand_cu(n_seq, total, rng), CHUNK)
        cases.append(TestCase(f"varlen rand {n_seq}seq T={cu[-1]}", cu, cu[-1]))
    return cases


def _make_inputs(tc: TestCase, H: int, HV: int | None = None):
    HV = HV or H
    T  = tc.T
    torch.manual_seed(42)
    q       = torch.randn(1, T, H, K)
    k       = torch.randn(1, T, H, K)
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
    g_log   = -torch.rand(1, T, HV, K) * 0.05   # values in (-0.05, 0)
    beta_sig = torch.sigmoid(torch.randn(1, T, HV))
    scale   = K ** -0.5
    return q, k, v, g_log, beta_sig, scale


# ---------------------------------------------------------------------------
# Per-stage test functions
# ---------------------------------------------------------------------------

def test_gate_cumsum(tc: TestCase, H: int, dev: "torch.device | None" = None) -> bool:
    """Compare gate_cumsum_kda NPU kernel output to CPU float32 reference.

    Runs the PTO kernel on ``dev`` and checks that it matches ``ref_gate_cumsum``
    element-wise within tolerance.

    Args:
        tc:  Test case (defines T and optional cu_seqlens).
        H:   Number of value/gate heads HV.
        dev: NPU device (e.g. ``torch.device("npu:0")``).  Required.
    """
    if dev is None:
        raise ValueError("test_gate_cumsum requires --device (NPU device).")

    from megagdn_pto.kda_kernel_libs import run_gate_cumsum_kda

    _, _, _, g_log, _, _ = _make_inputs(tc, H)
    # g_log: [1, T, H, K]
    g_dev   = g_log.half().to(dev)
    g_sum   = torch.empty_like(g_dev)
    cu      = (torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
               if tc.cu_seqlens_list else None)
    N_seq   = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream  = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_gate_cumsum_kda(
        g_dev, g_sum,
        stream=stream,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    ref = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    return stats_ok(g_sum.cpu(), ref)


def test_kkt(tc: TestCase, H: int, dev=None) -> bool:
    """Compare kkt_kda NPU kernel output to ref_kkt_kda CPU reference."""
    if dev is None:
        raise ValueError("test_kkt_npu requires --device (NPU device).")

    from megagdn_pto.kda_kernel_libs import run_kkt_kda

    _, k, _, g_log, beta_sig, _ = _make_inputs(tc, H)

    g_cs = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)

    L_npu  = torch.zeros(1, tc.T, H, CHUNK, device=dev, dtype=torch.float16)
    cu     = (torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
              if tc.cu_seqlens_list else None)
    N_seq  = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_
    torch.npu.synchronize()

    run_kkt_kda(
        k.half().to(dev), g_cs.half().to(dev), beta_sig.half().to(dev), L_npu,
        stream=stream, chunk_size=CHUNK, cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    ref = ref_kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    return stats_ok(L_npu.cpu(), ref)


def test_inv(tc: TestCase, H: int, dev=None) -> bool:
    """Compare PTO-ISA tri_inverse output to ref_inversion_kda CPU reference."""
    if dev is None:
        raise ValueError("test_inv requires --device (NPU device).")

    _, k, _, g_log, beta_sig, _ = _make_inputs(tc, H)

    g_cs = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref = ref_kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)

    L_fp16 = L_ref.to(torch.float16).to(dev)
    cu = (torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
          if tc.cu_seqlens_list else None)

    tri_inv = load_tri_inverse()
    A_inv = solve_tril(L_fp16, cu, CHUNK, H, tri_inv)
    torch.npu.synchronize()

    ref = ref_inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    return stats_ok(A_inv.float().cpu(), ref)


def test_wy(tc: TestCase, H: int, dev=None) -> bool:
    """Compare wy_kda NPU kernel output (u, w) to ref_wy_kda CPU reference.

    Feeds CPU-computed INV (via torch.linalg.inv) into the kernel so this
    isolates wy's correctness from the inversion stage — same strategy as
    test_inv at line 574.
    """
    if dev is None:
        raise ValueError("test_wy requires --device (NPU device).")

    from megagdn_pto.kda_kernel_libs import run_wy_kda

    _, k, v, g_log, beta_sig, _ = _make_inputs(tc, H)

    g_cs    = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref   = ref_kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV_ref = ref_inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    u_ref, w_ref = ref_wy_kda(k.float(), v.float(), g_cs, beta_sig,
                              INV_ref, CHUNK, tc.cu_seqlens_list)

    k_d    = k.half().to(dev)
    v_d    = v.half().to(dev)
    g_cs_d = g_cs.half().to(dev)
    beta_d = beta_sig.half().to(dev)
    INV_d  = INV_ref.half().to(dev)
    u_npu  = torch.zeros(1, tc.T, H, V_DIM, device=dev, dtype=torch.float16)
    w_npu  = torch.zeros(1, tc.T, H, K,     device=dev, dtype=torch.float16)
    cu     = (torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
              if tc.cu_seqlens_list else None)
    N_seq  = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_wy_kda(
        k_d, v_d, g_cs_d, beta_d, INV_d, u_npu, w_npu,
        stream=stream, chunk_size=CHUNK, cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    return stats_ok(u_npu.cpu(), u_ref) and stats_ok(w_npu.cpu(), w_ref)


def test_chunk_h_kda(tc: TestCase, H: int, dev=None) -> bool:
    """Compare chunk_h_kda NPU kernel output (s_snapshots, v_corr) to ref."""
    if dev is None:
        raise ValueError("test_chunk_h_kda requires --device (NPU device).")

    from megagdn_pto.kda_kernel_libs import run_chunk_h_kda

    _, k, v, g_log, beta_sig, _ = _make_inputs(tc, H)

    # Build inputs by chaining the CPU reference pipeline (matches test_wy).
    g_cs    = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref   = ref_kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV_ref = ref_inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    u_ref, w_ref = ref_wy_kda(k.float(), v.float(), g_cs, beta_sig,
                              INV_ref, CHUNK, tc.cu_seqlens_list)
    s_ref, vcorr_ref = ref_chunk_h_kda(k.float(), u_ref, w_ref, g_cs,
                                       CHUNK, tc.cu_seqlens_list)

    s_npu     = torch.zeros_like(s_ref, device=dev, dtype=torch.float16)
    vcorr_npu = torch.zeros(1, tc.T, H, V_DIM, device=dev, dtype=torch.float16)
    cu = (torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
          if tc.cu_seqlens_list else None)
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_chunk_h_kda(
        k.half().to(dev), w_ref.half().to(dev), u_ref.half().to(dev), g_cs.half().to(dev),
        s_npu, vcorr_npu,
        stream=stream, chunk_size=CHUNK, cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    return stats_ok(s_npu.cpu(), s_ref) and stats_ok(vcorr_npu.cpu(), vcorr_ref)


def test_chunk_o_kda(tc: TestCase, H: int, dev=None) -> bool:
    """Compare chunk_o_kda NPU kernel output to ref_chunk_o_kda CPU reference."""
    if dev is None:
        raise ValueError("test_chunk_o_kda requires --device (NPU device).")

    from megagdn_pto.kda_kernel_libs import run_chunk_o_kda

    q, k, v, g_log, beta_sig, scale = _make_inputs(tc, H)

    # Apply scale (matches cpu_pipeline_kda lines 416-418).  No GQA expansion
    # needed since _make_inputs gives H == HV.
    qf = q.float() * scale
    kf = k.float()

    # Reference pipeline up through chunk_h_kda.
    g_cs    = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref   = ref_kkt_kda(kf, g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV_ref = ref_inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    u_ref, w_ref = ref_wy_kda(kf, v.float(), g_cs, beta_sig,
                              INV_ref, CHUNK, tc.cu_seqlens_list)
    s_ref, vcorr_ref = ref_chunk_h_kda(kf, u_ref, w_ref, g_cs,
                                       CHUNK, tc.cu_seqlens_list)
    # Round-trip through fp16 to match what the NPU kernel actually receives.
    # s_ref accumulates across chunks; the rounding error in s_ref.half() propagates
    # through q_eff @ S and causes the NPU output to diverge from a float32 reference.
    o_ref = ref_chunk_o_kda(qf, kf, vcorr_ref.half().float(), s_ref.half().float(), g_cs,
                            CHUNK, tc.cu_seqlens_list)

    # NPU run.
    o_npu = torch.zeros(1, tc.T, H, V_DIM, device=dev, dtype=torch.float16)
    cu = (torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
          if tc.cu_seqlens_list else None)
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_chunk_o_kda(
        qf.half().to(dev), kf.half().to(dev), vcorr_ref.half().to(dev),
        s_ref.half().to(dev), g_cs.half().to(dev),
        o_npu,
        stream=stream, chunk_size=CHUNK, cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    return stats_ok(o_npu.cpu(), o_ref)


# ---------------------------------------------------------------------------
# Stage registry and runner
# ---------------------------------------------------------------------------

_STAGES = {
    "cumsum":   ("Gate cumsum",                test_gate_cumsum),
    "kkt":      ("KKT NPU kernel",             test_kkt),
    "inv":      ("Linalg (I+L)^{-1}",         test_inv),
    "wy":       ("WY transform (u, w)",        test_wy),
    "chunk_h":  ("chunk_h_kda (snapshots)",    test_chunk_h_kda),
    "chunk_o":  ("chunk_o_kda (output)",       test_chunk_o_kda),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--stage", default=",".join(_STAGES))
    parser.add_argument(
        "--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"),
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
    print(f"device={args.device}  H={H}  K={K}  V={V_DIM}  CHUNK={CHUNK}  cases={len(cases)}")

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
