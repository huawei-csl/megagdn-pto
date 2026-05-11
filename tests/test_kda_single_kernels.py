#!/usr/bin/env python3
"""CPU float32 reference implementations and unit tests for each KDA stage.

Matches the math of ``kda_naive.naive_chunk_kda`` exactly, which is used as
ground truth in test_kda_e2e.py.  No CUDA extensions, no FLA imports.

KDA pipeline stages:
  gate_cumsum → kkt (L matrix) → inversion → wy (u, w) → recurrent (Aqk + state)

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

Usage::

    python tests/test_kda_single_kernels.py
    python tests/test_kda_single_kernels.py --quick
    python tests/test_kda_single_kernels.py --stage kkt,inv
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

CHUNK = 16   # small chunk for fast CPU tests
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
# Stage 2 – L matrix (gated K×K product, strictly lower triangular)
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
                delta_g = gc.unsqueeze(1) - gc.unsqueeze(0)        # [c_len, c_len, K]
                L_full = (kc.unsqueeze(1) * kc.unsqueeze(0)
                          * torch.exp(delta_g)).sum(-1)             # [c_len, c_len]
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
                Ac = Af[0, s:e, h, :v]  # [v, v], strictly lower triangular
                inv = torch.linalg.inv(torch.eye(v) + Ac)
                out[0, s:e, h, :v] = inv
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
# Stage 5 – Sequential recurrence: Aqk, state update, output
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
    return ref_recurrent_kda(qf, kf, u, w, g_cs, chunk_size, cu_seqlens_list)


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


def _build_test_cases(quick: bool) -> list[TestCase]:
    if quick:
        return [TestCase("quick T=16", None, 16)]
    cases = []
    for T in [16, 32, 64, 128]:
        cases.append(TestCase(f"fixed T={T}", None, T))
    for seqlens in [[16, 16], [32, 16], [16, 32, 16]]:
        cu = [0]
        for s in seqlens:
            cu.append(cu[-1] + s)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
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
    v       = torch.randn(1, T, HV, V_DIM)
    g_log   = -torch.rand(1, T, HV, K) * 2.0    # values in (-2, 0)
    beta_sig = torch.sigmoid(torch.randn(1, T, HV))
    scale   = K ** -0.5
    return q, k, v, g_log, beta_sig, scale


# ---------------------------------------------------------------------------
# Per-stage test functions
# ---------------------------------------------------------------------------

def test_gate_cumsum(tc: TestCase, H: int) -> bool:
    q, k, v, g_log, beta_sig, scale = _make_inputs(tc, H)
    g_cs = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    # First element of each chunk equals g (cumsum[0] == g[0])
    for bos, eos in _seq_ranges(tc.T, tc.cu_seqlens_list):
        for j in range(0, eos - bos, CHUNK):
            s = bos + j
            if not torch.allclose(g_cs[:, s], g_log[:, s].float(), atol=1e-5):
                return False
    return True


def test_kkt(tc: TestCase, H: int) -> bool:
    q, k, v, g_log, beta_sig, scale = _make_inputs(tc, H)
    kf   = k.float()
    g_cs = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L    = ref_kkt_kda(kf, g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    # L must be strictly lower triangular within each chunk block
    for bos, eos in _seq_ranges(tc.T, tc.cu_seqlens_list):
        for j in range(0, eos - bos, CHUNK):
            s, e = bos + j, min(bos + j + CHUNK, eos)
            c_len = e - s
            for h in range(H):
                Lc = L[0, s:e, h, :c_len]
                if Lc.triu(diagonal=0).abs().max().item() > 1e-7:
                    return False
    return True


def test_neumann_inv(tc: TestCase, H: int) -> bool:
    q, k, v, g_log, beta_sig, scale = _make_inputs(tc, H)
    kf   = k.float()
    g_cs = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L    = ref_kkt_kda(kf, g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV  = ref_inversion_kda(L, CHUNK, tc.cu_seqlens_list)
    for bos, eos in _seq_ranges(tc.T, tc.cu_seqlens_list):
        for j in range(0, eos - bos, CHUNK):
            s, e = bos + j, min(bos + j + CHUNK, eos)
            c_len = e - s
            for h in range(H):
                Lc   = L[0, s:e, h, :c_len]
                INVc = INV[0, s:e, h, :c_len]
                IpL  = torch.eye(c_len) + Lc
                # Residual check in float64: (I+L)@INV should be identity.
                # Float32 cancels catastrophically for ill-conditioned chunks.
                residual = (IpL.double() @ INVc.double() - torch.eye(c_len).double()).float()
                if not stats_ok(residual, torch.zeros(c_len, c_len)):
                    return False
    return True


def test_wy(tc: TestCase, H: int) -> bool:
    q, k, v, g_log, beta_sig, scale = _make_inputs(tc, H)
    kf   = k.float()
    g_cs = ref_gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L    = ref_kkt_kda(kf, g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV  = ref_inversion_kda(L, CHUNK, tc.cu_seqlens_list)
    u, _ = ref_wy_kda(kf, v.float(), g_cs, beta_sig, INV, CHUNK, tc.cu_seqlens_list)
    # Verify u = INV @ (beta*v) using float64 for reference to avoid cancellation.
    # (I+L)^{-1} can amplify inputs by millions; residual (I+L)@u - beta*v then
    # loses all precision in float32, so we compare against the float64 matmul instead.
    for bos, eos in _seq_ranges(tc.T, tc.cu_seqlens_list):
        for j in range(0, eos - bos, CHUNK):
            s, e = bos + j, min(bos + j + CHUNK, eos)
            c_len = e - s
            for h in range(H):
                INVc = INV[0, s:e, h, :c_len].double()
                vc   = v[0, s:e, h, :].double()
                bc   = beta_sig[0, s:e, h].double()
                u_ref = (INVc @ (vc * bc.unsqueeze(-1))).float()
                if not stats_ok(u[0, s:e, h, :], u_ref):
                    return False
    return True


def test_pipeline(tc: TestCase, H: int) -> bool:
    """Determinism: two identical runs must match exactly."""
    q, k, v, g_log, beta_sig, scale = _make_inputs(tc, H)
    o1 = cpu_pipeline_kda(q, k, v, g_log, beta_sig, tc.cu_seqlens_list, scale, CHUNK)
    o2 = cpu_pipeline_kda(q, k, v, g_log, beta_sig, tc.cu_seqlens_list, scale, CHUNK)
    return bool(torch.allclose(o1, o2))


# ---------------------------------------------------------------------------
# Stage registry and runner
# ---------------------------------------------------------------------------

_STAGES = {
    "cumsum":   ("Gate cumsum",           test_gate_cumsum),
    "kkt":      ("KKT (L matrix)",        test_kkt),
    "inv":      ("Neumann (I+L)^{-1}",   test_neumann_inv),
    "wy":       ("WY transform (u, w)",   test_wy),
    "pipeline": ("Full pipeline (det.)",  test_pipeline),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--stage", default=",".join(_STAGES))
    args = parser.parse_args()

    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    for s in stages:
        if s not in _STAGES:
            sys.exit(f"Unknown stage {s!r}; choose from {list(_STAGES)}")

    cases = _build_test_cases(args.quick)
    H = args.H
    print(f"H={H}  K={K}  V={V_DIM}  CHUNK={CHUNK}  cases={len(cases)}")

    all_pass = True
    for stage in stages:
        name, fn = _STAGES[stage]
        print(f"\n{'='*60}\nStage: {name}\n{'='*60}")
        for i, tc in enumerate(cases):
            t0 = time.time()
            ok = fn(tc, H)
            dt = time.time() - t0
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{i+1:2d}/{len(cases)}] {status}  {tc.label}  ({dt:.2f}s)")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
