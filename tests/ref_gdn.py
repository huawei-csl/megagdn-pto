import numpy as np
import torch
from megagdn_pto.kernel_libs import total_chunks

# ---------------------------------------------------------------------------
# CPU Reference implementations
# ---------------------------------------------------------------------------


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def _seq_ranges(T: int, cu_seqlens=None) -> list[tuple[int, int]]:
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


class RefGDN:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def cumsum(self, g: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        """Chunk-local cumulative sum of gates (fp32)."""
        B, T, H = g.shape
        gf = g.to(self.dtype)
        out = torch.zeros_like(gf, dtype=self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                out[:, s:e, :] = gf[:, s:e, :].cumsum(dim=1)
        return out

    def solve_tril(self, A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        """CPU reference for solve_tril: computes (I + A)^{-1} per chunk submatrix.

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

    def kkt(
        self,
        k: torch.Tensor,
        beta: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        """CPU reference for scaled_dot_kkt (GQA: k has Hg heads, beta/g have H heads)."""
        B, T, Hg, Dd = k.shape
        H = beta.shape[2]
        grp = H // Hg
        out = torch.zeros(B, T, H, cs, dtype=self.dtype)
        kf, bf, gf = k.to(self.dtype), beta.to(self.dtype), g_cumsum.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                for h in range(H):
                    hg = h // grp
                    kc, gc = kf[0, s:e, hg, :], gf[0, s:e, h]
                    blk = (
                        (kc @ kc.T)
                        * _safe_exp(gc[:, None] - gc[None, :])
                        * bf[0, s:e, h, None]
                    )
                    mask = torch.arange(v)[:, None] > torch.arange(v)[None, :]
                    out[0, s:e, h, :v] = blk * mask.to(self.dtype)
        return out

    def chunk_h(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ):
        """CPU reference for chunk_h: states S, v_new, final states."""
        B, T, Hg, Dd = k.shape
        H = w.shape[2]
        grp = H // Hg
        kf, wf, uf, gf = (
            k.to(self.dtype),
            w.to(self.dtype),
            u.to(self.dtype),
            g_cumsum.to(self.dtype),
        )
        ranges = _seq_ranges(T, cu_seqlens)
        tc = total_chunks(
            len(ranges), T, cs, torch.tensor(cu_seqlens) if cu_seqlens else None
        )
        h_out = torch.zeros(tc, H, Dd, Dd, dtype=self.dtype)
        v_new = torch.zeros_like(uf)
        final = torch.zeros(len(ranges), H, Dd, Dd, dtype=self.dtype)
        ci_base = 0
        for si, (bos, eos) in enumerate(ranges):
            nc = (eos - bos + cs - 1) // cs
            for h in range(H):
                hg = h // grp
                S = torch.zeros(Dd, Dd, dtype=self.dtype)
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

    def wy_fast(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A_inv: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ):
        """CPU reference for wy_fast."""
        B, T, Hg, Kd = k.shape
        H = v.shape[2]
        grp = H // Hg
        w = torch.zeros(B, T, H, Kd, dtype=self.dtype)
        u = torch.zeros(B, T, H, v.shape[-1], dtype=self.dtype)
        kf, vf, bf, Af, gf = (
            k.to(dtype=self.dtype),
            v.to(dtype=self.dtype),
            beta.to(dtype=self.dtype),
            A_inv.to(dtype=self.dtype),
            g_cumsum.to(dtype=self.dtype),
        )
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                valid = e - s
                for h in range(H):
                    hg = h // grp
                    Ab = Af[0, s:e, h, :valid]
                    gc = gf[0, s:e, h]
                    vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                    kb = (
                        kf[0, s:e, hg, :] * bf[0, s:e, h, None] * torch.exp(gc)[:, None]
                    )
                    u[0, s:e, h, :] = Ab @ vb
                    w[0, s:e, h, :] = Ab @ kb
        return w, u

    def chunk_o(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v_new: torch.Tensor,
        h_states: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        """CPU reference for chunk_o (PTO gating convention: min(Δg, 0))."""
        B, T, Hg, Dd = q.shape
        H = v_new.shape[2]
        grp = H // Hg
        qf, kf, vf, hf, gf = (
            q.to(self.dtype),
            k.to(self.dtype),
            v_new.to(self.dtype),
            h_states.to(self.dtype),
            g_cumsum.to(self.dtype),
        )
        o = torch.zeros(B, T, H, Dd, dtype=self.dtype)
        ranges = _seq_ranges(T, cu_seqlens)
        ci_base = 0
        for bos, eos in ranges:
            nc = (eos - bos + cs - 1) // cs
            for h in range(H):
                hg = h // grp
                for ci in range(nc):
                    s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                    vlen = e - s
                    qc, kc, vc, gc = (
                        qf[0, s:e, hg, :],
                        kf[0, s:e, hg, :],
                        vf[0, s:e, h, :],
                        gf[0, s:e, h],
                    )
                    inter = (qc @ hf[ci_base + ci, h]) * torch.exp(gc)[:, None]
                    qk = qc @ kc.T
                    causal = torch.arange(vlen)[:, None] >= torch.arange(vlen)[None, :]
                    gate = torch.exp(
                        torch.minimum(
                            gc[:, None] - gc[None, :], torch.zeros(vlen, vlen)
                        )
                    )
                    o[0, s:e, h, :] = inter + (qk * gate * causal.to(self.dtype)) @ vc
            ci_base += nc
        return o
