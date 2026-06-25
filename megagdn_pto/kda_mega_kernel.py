"""Fused KDA mega-kernel: all six KDA stages in a single NPU launch.

Fuses gate_cumsum → (head-major transpose) → kkt → solve_tril → wy → chunk_h →
chunk_o into one ``call_kernel`` invocation, eliminating Python-level dispatch overhead
between stages.  Analogous to ``megagdn_pto.mega_kernel`` for GDN.

For step-by-step execution (useful for debugging or profiling individual stages), use the
standalone wrappers in ``megagdn_pto.kda_kernel_libs`` instead.

Input convention (mirrors the prepared tensors the staged wrappers receive):
  q, k:  ``[1, T, HV, K]`` fp16, already GQA-expanded; ``q`` already scaled.
  v:     ``[1, T, HV, V]`` fp16.
  g:     ``[1, T, HV, K]`` fp16, raw per-dimension gate logits (NOT cumulative).
  beta:  ``[1, T, HV]``    fp16, post-sigmoid beta in (0, 1).
Assumes packed varlen with ``B = 1`` and ``K == V``.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from megagdn_pto.compile import BLOCK_DIM, _KERNELS_PTO, compile_mega_kernel_kda
from megagdn_pto.kernel_libs import (
    chunk_gdn_causal_masks,
    precomputed_minus_identity,
    total_chunks,
    _vp,
)


@lru_cache(maxsize=None)
def _load_mega_kernel_kda(
    *,
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    # Head count (HV) is a runtime kernel argument now, so the fused .so is
    # head-count-agnostic; ``num_heads`` here is only the lru_cache key.
    mtime = os.stat(os.path.join(_KERNELS_PTO, "mega_kernel_kda.cpp")).st_mtime_ns
    lib_path = compile_mega_kernel_kda(
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        cpp_mtime_ns=mtime,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 24
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
           ctypes.c_uint32, ctypes.c_uint32]  # ..., num_matrices, num_heads
    )
    lib.call_kernel.restype = None
    return lib


def run_mega_kernel_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    *,
    stream,
    chunk_size: int = 128,
    block_dim: int | None = None,
    batch_size_override: int | None = None,
) -> torch.Tensor:
    """Run all six KDA stages in a single fused NPU kernel launch.

    Args:
        q, k:       ``[1, T, HV, K]`` fp16, GQA-expanded; ``q`` pre-scaled.
        v:          ``[1, T, HV, V]`` fp16.
        g:          ``[1, T, HV, K]`` fp16, raw per-dimension gate logits.
        beta:       ``[1, T, HV]``    fp16, post-sigmoid beta.
        cu_seqlens: ``int32`` cumulative sequence lengths, or ``None`` for one
                    sequence of length ``T`` (a ``[0, T]`` boundary is synthesised).
        stream:     NPU stream handle.
        chunk_size: Chunk size C (default 128).
        block_dim:  AI-Core block count (auto-detected if None).
        batch_size_override: Number of sequences ``N_seq`` (use with ``cu_seqlens``).

    Returns:
        ``O`` of shape ``[1, T, HV, V]`` fp16.
    """
    assert q.dtype == torch.float16 and v.dtype == torch.float16
    dev = q.device
    HV, K = q.shape[2], q.shape[3]
    V = v.shape[3]
    C = chunk_size
    T = q.shape[1]
    bd = block_dim or BLOCK_DIM

    # tri_inverse (solve_tril) needs cu_seqlens; synthesise [0, T] for the single-seq case
    # so every fused stage shares one boundary tensor.
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=dev)
    elif cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    N_seq = batch_size_override if batch_size_override is not None else (
        int(cu_seqlens.numel()) - 1
    )

    dt, di = dev.type, dev.index if dev.index is not None else -1
    mask_strict, mask_incl = chunk_gdn_causal_masks(dt, di, C)   # (rows>cols), (rows>=cols)
    minus_id = precomputed_minus_identity(dt, di, C)

    tc = total_chunks(N_seq, T, C, cu_seqlens)
    num_matrices = tc * HV

    # Head-major permutes (match the staged kkt/wy/chunk_h/chunk_o wrappers).
    q_hm = q.permute(0, 2, 1, 3).contiguous()      # [1, HV, T, K]
    k_hm = k.permute(0, 2, 1, 3).contiguous()      # [1, HV, T, K]
    beta_hm = beta.permute(0, 2, 1).contiguous()   # [1, HV, T]
    v = v.contiguous()
    g = g.contiguous()

    # Intermediates.
    g_sum   = torch.empty(1, T, HV, K, device=dev, dtype=torch.float16)
    g_cs_hm = torch.empty(1, HV, T, K, device=dev, dtype=torch.float16)
    L       = torch.zeros(1, T, HV, C, device=dev, dtype=torch.float16)
    A_inv   = torch.zeros(1, T, HV, C, device=dev, dtype=torch.float16)
    u       = torch.zeros(1, T, HV, V, device=dev, dtype=torch.float16)
    w       = torch.zeros(1, T, HV, K, device=dev, dtype=torch.float16)
    s       = torch.zeros(tc, HV, K, V, device=dev, dtype=torch.float16)
    v_corr  = torch.zeros(1, T, HV, V, device=dev, dtype=torch.float16)
    o_out   = torch.zeros(1, T, HV, V, device=dev, dtype=torch.float16)

    # Per-AI-core workspaces (shapes match the staged run_*_kda allocations).
    kkt_ws_in  = torch.zeros(bd * 2, 2 * C, K, device=dev, dtype=torch.float16)
    kkt_ws_out = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    wy_ws_a2   = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    wy_ws_keff = torch.zeros(bd, C, K, device=dev, dtype=torch.float16)
    h_ws       = torch.zeros(bd * 5, K, K, device=dev, dtype=torch.float16)
    o_ws       = torch.zeros(bd * 7, K, K, device=dev, dtype=torch.float16)

    # Cold-start FFTS handshake needs all pending work (incl. workspace zero-fill) drained.
    torch.npu.synchronize()

    lib = _load_mega_kernel_kda(num_heads=HV, hidden_size=K, chunk_size=C)
    lib.call_kernel(
        bd, stream,
        _vp(q_hm), _vp(k_hm), _vp(v), _vp(g), _vp(beta_hm),
        _vp(mask_strict), _vp(mask_incl), _vp(minus_id), _vp(cu_seqlens),
        _vp(o_out),
        _vp(g_sum), _vp(g_cs_hm), _vp(L), _vp(A_inv),
        _vp(u), _vp(w), _vp(s), _vp(v_corr),
        _vp(kkt_ws_in), _vp(kkt_ws_out), _vp(wy_ws_a2), _vp(wy_ws_keff),
        _vp(h_ws), _vp(o_ws),
        N_seq, T, T, num_matrices, HV,
    )
    return o_out
