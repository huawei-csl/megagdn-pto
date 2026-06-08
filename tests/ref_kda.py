# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from einops import rearrange
import numpy as np


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``.
        beta (torch.Tensor):
            Beta scalars of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.
        chunk_size (int):
            Chunk size for the chunked computation. Default: 64.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``S`` has shape ``[B, HV, K, V]`` if ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K**-0.5
    assert T % BT == 0

    # Rearrange into chunks: [B, head, NT, BT, ...]
    q, k = [
        rearrange(x, "b (n c) h ... -> b h n c ...", c=BT).to(torch.float)
        for x in [q, k]
    ]
    v, g, beta = [
        rearrange(x, "b (n c) h ... -> b h n c ...", c=BT).to(torch.float)
        for x in [v, g, beta]
    ]
    # Expand q/k to value head dim for GVA: [B, H, ...] -> [B, HV, ...]
    q = q.repeat_interleave(G, dim=1) * scale  # [B, HV, NT, BT, K]
    k = k.repeat_interleave(G, dim=1)  # [B, HV, NT, BT, K]
    g = g.cumsum(-2)

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    # Akk uses k (expanded to HV) and g (per value head)
    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i : i + 1, :]
        A[..., i] = torch.einsum("... c d, ... d -> ... c", k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (
            A[..., i, :, None].clone() * A[..., :, :i].clone()
        ).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        # [B, HV, BT, ...]
        q_i = q[:, :, i]  # [B, HV, BT, K]
        k_i = k[:, :, i]  # [B, HV, BT, K]
        u_i = u[:, :, i]  # [B, HV, BT, V]
        g_i = g[:, :, i]  # [B, HV, BT, K]
        w_i = w[:, :, i]  # [B, HV, BT, K]
        # Aqk: per value head (q from qk head, g from value head, k from qk head)
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j : j + 1, :]
            Aqk[..., j] = torch.einsum(
                "... c d, ... d -> ... c", q_i * (g_i - g_j).exp(), k_j
            )
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), "b h k -> b h k 1")
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, "b h c k -> b h k c") @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, "b h n c d -> b (n c) h d").to(dtype), S


def _seq_ranges(T: int, cu_seqlens=None) -> list[tuple[int, int]]:
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


class RefKDA:
    """CPU float32 reference implementations for each KDA stage.

    Matches the math of ``kda_naive.naive_chunk_kda`` exactly, which is used as
    ground truth in test_kda_e2e.py.

    KDA pipeline stages:
    gate_cumsum → kkt (L matrix) → inversion → wy (u, w) → chunk_h_kda (snapshots + v_corr) → chunk_o_kda (output)

    Key math (see kda_naive.py):
    - g is per-dimension log-space decay (natural exp applied internally)
    - beta is post-sigmoid scalar per (position, head)
    - L[r,c] = beta[r] * k_r·(k_c*exp(g_cs[r]-g_cs[c])) for r>c  (strictly lower tri)
    - (I+L)^{-1} via Neumann recursion: A=-L; for i: A[i,:i]+=A[i,:]@A[:,:i]; A_inv=A+I
    - A_final = (I+L)^{-1} @ diag(beta)  (column-scale after inversion)
    - u = A_final @ v = A_inv @ (beta*v),  w = A_final @ (exp(g)*k) = A_inv @ (beta*exp(g)*k)
    - Aqk[r,c] = q_r·(k_c*exp(g_cs[r]-g_cs[c])) for r>=c  (causal, includes diagonal)
    - output: (q*exp(g_cs)) @ S + Aqk @ (u - w @ S)
    - state:  S_new[k,:] = exp(g_total[k]) * S[k,:] + sum_c k_rest[c,k]*v_corr[c,:]

    Stage device requirements:
    - cumsum: runs on NPU (requires --device); calls gate_cumsum_kda PTO kernel.
    - all others: CPU float32 reference only.

    Usage::

        python tests/test_kda_single_kernels.py --device npu:0
        python tests/test_kda_single_kernels.py --device npu:0 --quick
        python tests/test_kda_single_kernels.py --device npu:0 --stage kkt,inv
    """

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def gate_cumsum(
        self, g: torch.Tensor, chunk_size: int, cu_seqlens=None
    ) -> torch.Tensor:
        """Stage 1 - Within-chunk cumulative sum of g [B, T, HV, K].

        Resets at chunk boundaries and sequence boundaries.
        Matches naive_chunk_kda's per-chunk ``g.cumsum(-2)``.
        """
        B, T, HV, Kd = g.shape
        out = torch.zeros(B, T, HV, Kd, dtype=self.dtype)
        gf = g.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, chunk_size):
                s, e = bos + j, min(bos + j + chunk_size, eos)
                out[:, s:e] = gf[:, s:e].cumsum(dim=1)
        return out

    def kkt_kda(
        self,
        k: torch.Tensor,
        g_cs: torch.Tensor,
        beta_sig: torch.Tensor,
        chunk_size: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        """Stage 2 - L matrix across all chunks: L[r,c]=beta[r]*k_r·(k_c*exp(g_cs[r]-g_cs[c])) for r>c.

        Args:
            k:        [B, T, HV, K]  key vectors (GQA-expanded)
            g_cs:     [B, T, HV, K]  within-chunk cumulative gate sum
            beta_sig: [B, T, HV]     post-sigmoid beta in (0, 1)
            chunk_size: int
            cu_seqlens: list[int] | None

        Returns:
            L: [B, T, HV, chunk_size], strictly lower triangular per chunk
            (column index padded to chunk_size; unused entries are 0)
        """
        B, T, HV, Kd = k.shape
        L_out = torch.zeros(B, T, HV, chunk_size, dtype=self.dtype)
        kf = k.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, chunk_size):
                s, e = bos + j, min(bos + j + chunk_size, eos)
                c_len = e - s
                for h in range(HV):
                    kc = kf[0, s:e, h, :]  # [c_len, K]
                    gc = g_cs[0, s:e, h, :]  # [c_len, K]
                    bc = beta_sig[0, s:e, h]  # [c_len]
                    A = kc * torch.exp(gc)
                    B = kc * torch.exp(-gc)
                    L_full = A @ B.T
                    L_out[0, s:e, h, :c_len] = torch.tril(
                        L_full * bc.unsqueeze(-1), diagonal=-1
                    )
        return L_out

    def inversion_kda(self, A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        """Stage 3 - CPU reference for solve_tril: computes (I + A)^{-1} per chunk submatrix.

        A is strictly lower triangular [B, T, H, cs] (PTO convention).
        The inverse is computed always with numpy at double-precision, which internally
        uses LAPACK.
        """
        B, T, H, _ = A.shape
        out = torch.zeros(B, T, H, cs, dtype=self.dtype)
        Af = A.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                for h in range(H):
                    Ac = Af[0, s:e, h, :v]  # [v, v], strictly lower triangular
                    M = np.linalg.inv((np.identity(v) + Ac.numpy()).astype(np.double))
                    inv = torch.from_numpy(M)
                    out[0, s:e, h, :v] = inv.to(self.dtype)
        return out

    def wy_kda(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cs: torch.Tensor,
        beta_sig: torch.Tensor,
        A_inv: torch.Tensor,
        chunk_size: int,
        cu_seqlens=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 4 - Compute u = A_inv @ (beta*v)  and  w = A_inv @ (beta*exp(g_cs)*k)  for all chunks.

        naive_chunk_kda applies beta twice:
        - Row-scale when building L  (baked into A_inv = (I+L_beta_row)^{-1})
        - Column-scale after:  A_final = A_inv @ diag(beta)
        Hence u = A_final @ v = A_inv @ (beta * v).

        Args:
            k:        [B, T, HV, K]         (GQA-expanded)
            v:        [B, T, HV, V_DIM]
            g_cs:     [B, T, HV, K]         within-chunk cumulative gate sum
            beta_sig: [B, T, HV]            post-sigmoid beta
            A_inv:      [B, T, HV, chunk_size]  from inversion_kda
            chunk_size, cu_seqlens: as above

        Returns:
            u: [B, T, HV, V_DIM]  self.dtype
            w: [B, T, HV, K]      self.dtype
        """
        B, T, HV, Kd = k.shape
        Vd = v.shape[-1]
        u_out = torch.zeros(B, T, HV, Vd, dtype=self.dtype)
        w_out = torch.zeros(B, T, HV, Kd, dtype=self.dtype)
        kf, vf = k.to(self.dtype), v.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, chunk_size):
                s, e = bos + j, min(bos + j + chunk_size, eos)
                c_len = e - s
                for h in range(HV):
                    kc = kf[0, s:e, h, :]  # [c_len, K]
                    gc = g_cs[0, s:e, h, :]  # [c_len, K]
                    vc = vf[0, s:e, h, :]  # [c_len, V]
                    bc = beta_sig[0, s:e, h]  # [c_len]
                    A_invc = A_inv[0, s:e, h, :c_len]  # [c_len, c_len]
                    beta_col = bc.unsqueeze(-1)  # [c_len, 1]
                    u_out[0, s:e, h, :] = A_invc @ (vc * beta_col)
                    w_out[0, s:e, h, :] = A_invc @ (kc * torch.exp(gc) * beta_col)
        return u_out, w_out

    def recurrent_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        g_cs: torch.Tensor,
        chunk_size: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        """Stage 5 (Deprecated) - Sequential recurrence: state S propagates across chunks, producing output.

        For each chunk:
        v_corr = u - w @ S  (inter-chunk correction)
        Aqk[r,c] = q_r·(k_c*exp(g_cs[r]-g_cs[c])) for r>=c  (causal Q-K kernel)
        o = (q*exp(g_cs)) @ S + Aqk @ v_corr
        S_new[k,:] = exp(g_total[k]) * S[k,:] + k_rest.T @ v_corr

        Args:
            q:    [B, T, HV, K]      queries (scale already applied)
            k:    [B, T, HV, K]      keys (GQA-expanded)
            u:    [B, T, HV, V_DIM]  from wy_kda
            w:    [B, T, HV, K]      from wy_kda
            g_cs: [B, T, HV, K]      within-chunk cumulative gate sum
            chunk_size, cu_seqlens: as above

        Returns:
            o: [B, T, HV, V_DIM]  self.dtype
        """
        B, T, HV, Kd = q.shape
        Vd = u.shape[-1]
        o = torch.zeros(B, T, HV, Vd, dtype=self.dtype)

        for bos, eos in _seq_ranges(T, cu_seqlens):
            nc = (eos - bos + chunk_size - 1) // chunk_size
            for h in range(HV):
                S = torch.zeros(Kd, Vd, dtype=self.dtype)
                for ci in range(nc):
                    s = bos + ci * chunk_size
                    e = min(s + chunk_size, eos)
                    c_len = e - s

                    gc = g_cs[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    g_total = gc[c_len - 1].to(self.dtype)  # [K]
                    kc = k[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    qc = q[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    uc = u[0, s:e, h, :].to(self.dtype)  # [c_len, V]
                    wc = w[0, s:e, h, :].to(self.dtype)  # [c_len, K]

                    v_corr = uc - wc @ S  # [c_len, V]

                    delta_g = gc.unsqueeze(1) - gc.unsqueeze(0)  # [c_len, c_len, K]
                    Aqk = torch.tril(
                        (qc.unsqueeze(1) * kc.unsqueeze(0) * torch.exp(delta_g)).sum(
                            -1
                        ),
                        diagonal=0,
                    )  # [c_len, c_len]

                    o[0, s:e, h, :] = (qc * torch.exp(gc)) @ S + Aqk @ v_corr

                    k_rest = kc * torch.exp(g_total.unsqueeze(0) - gc)  # [c_len, K]
                    S = torch.exp(g_total).unsqueeze(-1) * S + k_rest.T @ v_corr

        return o

    # ---------------------------------------------------------------------------
    # Stage 5 – chunk_h_kda: sequential state pass (snapshot S + compute v_corr)
    # ---------------------------------------------------------------------------

    def chunk_h_kda(
        self,
        k: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        g_cs: torch.Tensor,
        chunk_size: int,
        cu_seqlens=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential state pass: snapshot S entering each chunk, compute v_corr.

        Mirrors GDN's chunk_h.  The state S [K, V] propagates sequentially
        across chunks within each sequence.  Both output terms in chunk_o depend
        on S *entering* the chunk, so snapshotting here decouples output computation.

        Args:
            k:    [B, T, HV, K]  keys (GQA-expanded)
            u:    [B, T, HV, V]  from wy_kda
            w:    [B, T, HV, K]  from wy_kda
            g_cs: [B, T, HV, K]  within-chunk cumulative gate sum
            chunk_size, cu_seqlens: as in other stages

        Returns:
            s_snapshots: [total_chunks, HV, K, V]  S entering each chunk
            v_corr:      [B, T, HV, V]             u - w @ S per position
        """
        B, T, HV, Kd = k.shape
        Vd = u.shape[-1]
        ranges = _seq_ranges(T, cu_seqlens)
        n_chunks = sum(
            (eos - bos + chunk_size - 1) // chunk_size for bos, eos in ranges
        )
        s_snapshots = torch.zeros(n_chunks, HV, Kd, Vd, dtype=self.dtype)
        v_corr_out = torch.zeros(B, T, HV, Vd, dtype=self.dtype)
        ci_base = 0
        for bos, eos in ranges:
            nc = (eos - bos + chunk_size - 1) // chunk_size
            for h in range(HV):
                S = torch.zeros(Kd, Vd, dtype=self.dtype)
                for ci in range(nc):
                    s = bos + ci * chunk_size
                    e = min(s + chunk_size, eos)
                    gc = g_cs[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    g_total = gc[-1]  # [K]
                    kc = k[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    uc = u[0, s:e, h, :].to(self.dtype)  # [c_len, V]
                    wc = w[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    s_snapshots[ci_base + ci, h] = S.clone()
                    v_corr = uc - wc @ S  # [c_len, V]
                    v_corr_out[0, s:e, h, :] = v_corr
                    k_rest = kc * torch.exp(g_total.unsqueeze(0) - gc)  # [c_len, K]
                    S = torch.exp(g_total).unsqueeze(-1) * S + k_rest.T @ v_corr
            ci_base += nc
        return s_snapshots, v_corr_out

    # ---------------------------------------------------------------------------
    # Stage 6 – chunk_o_kda: output pass (no sequential state dependency)
    # ---------------------------------------------------------------------------

    def chunk_o_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v_corr: torch.Tensor,
        s_snapshots: torch.Tensor,
        g_cs: torch.Tensor,
        chunk_size: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        """Output pass: compute o from pre-computed state snapshots and v_corr.

        Mirrors GDN's chunk_o.  Each chunk's output depends only on
        s_snapshots[ci] (the state *entering* that chunk) and v_corr — no
        sequential dependency between chunks within this pass.

        Args:
            q:           [B, T, HV, K]         queries (scale already applied)
            k:           [B, T, HV, K]         keys (GQA-expanded)
            v_corr:      [B, T, HV, V]         from chunk_h_kda
            s_snapshots: [total_chunks, HV, K, V]  from chunk_h_kda
            g_cs:        [B, T, HV, K]         within-chunk cumulative gate sum
            chunk_size, cu_seqlens: as in other stages

        Returns:
            o: [B, T, HV, V]  float32
        """
        B, T, HV, Kd = q.shape
        Vd = v_corr.shape[-1]
        o = torch.zeros(B, T, HV, Vd, dtype=self.dtype)
        ci_base = 0
        for bos, eos in _seq_ranges(T, cu_seqlens):
            nc = (eos - bos + chunk_size - 1) // chunk_size
            for h in range(HV):
                for ci in range(nc):
                    s = bos + ci * chunk_size
                    e = min(s + chunk_size, eos)
                    gc = g_cs[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    qc = q[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    kc = k[0, s:e, h, :].to(self.dtype)  # [c_len, K]
                    vc = v_corr[0, s:e, h, :].to(self.dtype)  # [c_len, V]
                    S = s_snapshots[ci_base + ci, h].to(self.dtype)  # [K, V]
                    q_eff = qc * torch.exp(gc)  # [c_len, K]
                    k_eff = kc * torch.exp(-gc)  # [c_len, K]
                    inter = q_eff @ S  # [c_len, V]
                    Aqk = torch.tril(q_eff @ k_eff.T, diagonal=0)  # [c_len, c_len]
                    o[0, s:e, h, :] = inter + Aqk @ vc
            ci_base += nc
        return o

    # ---------------------------------------------------------------------------
    # Full CPU pipeline
    # ---------------------------------------------------------------------------

    def full_pipeline(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_log: torch.Tensor,
        beta_sig: torch.Tensor,
        cu_seqlens_list,
        scale: float,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """Complete CPU float32 KDA reference.  Matches naive_chunk_kda exactly.

        Args:
            q:            [B, T, H,  K]  (NOT L2-normalised; naive doesn't)
            k:            [B, T, H,  K]
            v:            [B, T, HV, V]
            g_log:        [B, T, HV, K]  log-space per-dim gates (exp used inside)
            beta_sig:     [B, T, HV]     post-sigmoid beta in (0, 1)
            cu_seqlens_list: list[int] | cumulative sequence lengths
            scale:        query scale (typically K**-0.5)
            chunk_size:   must divide every sequence length

        Returns:
            o: [B, T, HV, V]  self.dtype
        """
        B, T, H, Kd = q.shape
        HV = v.shape[2]
        G = HV // H

        # GQA expansion + scale (mirrors naive's repeat_interleave * scale)
        qf = q.to(self.dtype).repeat_interleave(G, dim=2) * scale  # [B, T, HV, K]
        kf = k.to(self.dtype).repeat_interleave(G, dim=2)  # [B, T, HV, K]
        vf = v.to(self.dtype)
        bf = beta_sig.to(self.dtype)

        g_cs = self.gate_cumsum(g_log.to(self.dtype), chunk_size, cu_seqlens_list)
        L = self.kkt_kda(kf, g_cs, bf, chunk_size, cu_seqlens_list)
        A_inv = self.inversion_kda(L, chunk_size, cu_seqlens_list)
        u, w = self.wy_kda(kf, vf, g_cs, bf, A_inv, chunk_size, cu_seqlens_list)
        s_snapshots, v_corr = self.chunk_h_kda(
            kf, u, w, g_cs, chunk_size, cu_seqlens_list
        )
        return self.chunk_o_kda(
            qf, kf, v_corr, s_snapshots, g_cs, chunk_size, cu_seqlens_list
        )
