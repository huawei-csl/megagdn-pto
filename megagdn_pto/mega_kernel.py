"""Fused mega-kernel: all seven GDN stages in a single NPU launch.

Fuses cumsum → transpose → scaled_dot_kkt → solve_tril → wy_fast → chunk_h → chunk_o
into one ``call_kernel`` invocation, eliminating Python-level dispatch overhead between
stages. Use this for maximum throughput in production inference.

For step-by-step execution (useful for debugging or profiling individual stages),
use ``megagdn_pto.kernel_libs`` + ``megagdn_pto.fast_inverse`` instead.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from megagdn_pto.compile import BLOCK_DIM, _KERNELS_PTO, compile_mega_kernel
from megagdn_pto.kernel_libs import total_chunks, _vp


@lru_cache(maxsize=None)
def _load_mega_kernel(
    *,
    num_heads: int,
    key_heads: int | None = None,
    hidden_size: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    mtime = os.stat(os.path.join(_KERNELS_PTO, "mega_kernel.cpp")).st_mtime_ns
    lib_path = compile_mega_kernel(
        num_heads=num_heads,
        key_heads=key_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        cpp_mtime_ns=mtime,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 28
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32]
    )
    lib.call_kernel.restype = None
    return lib


def run_mega_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_in: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    stream,
    chunk_size: int = 128,
    scale: float = 1.0,
    block_dim: int | None = None,
    key_heads: int | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run all seven GDN stages in a single fused NPU kernel launch.

    Args:
        q, k:             ``[B, T, Hg, D]`` fp16 query and key tensors.
        v:                ``[B, T, H, D]`` fp16 value tensor (H ≥ Hg, H % Hg == 0).
        g_in:             ``[B, T, H]`` float32 pre-cumsum gate logits.
        beta:             ``[B, T, H]`` fp16 gate bias.
        cu_seqlens:       ``int32`` cumulative sequence lengths ``[0, ..., T]``.
        stream:           NPU stream handle.
        chunk_size:       Chunk size C (default 128).
        scale:            Output scale factor (typically ``head_dim ** -0.5``).
        block_dim:        AI-Core block count (auto-detected if None).
        key_heads:        Number of Q/K heads Hg (inferred from ``q`` if None).
        return_final_state: If True, also return ``[N_seq, H, D, D]`` final states.

    Returns:
        ``O * scale`` of shape ``[B, T, H, D]`` fp16, and optionally the final
        recurrent state ``[N_seq, H, D, D]`` fp16.
    """
    dev = q.device
    kh = key_heads if key_heads is not None else q.shape[2]
    H, D = v.shape[2], q.shape[3]
    C = chunk_size
    T = q.shape[1]
    N_seq = int(cu_seqlens.numel()) - 1
    bd = block_dim or BLOCK_DIM

    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    # Pre-allocated buffers (intermediates passed directly into the fused kernel)
    msk_lower = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).float()
    msk_full  = torch.tril(torch.ones(C, C, device=dev), diagonal=0).float()
    minus_identity = torch.zeros(C, C, device=dev, dtype=torch.float16)
    minus_identity.fill_diagonal_(-1)

    tc = total_chunks(N_seq, T, C, cu_seqlens)
    num_matrices = tc * H

    g_sum    = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    g_t      = torch.empty(H, T, device=dev, dtype=torch.float32)
    beta_t   = torch.empty(H, T, device=dev, dtype=torch.float16)
    A        = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    A_inv_f32 = torch.zeros(1, T, H, C, device=dev, dtype=torch.float32)
    A_inv    = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    w        = torch.empty_like(v)
    u        = torch.empty_like(v)
    s        = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)
    v_new    = torch.empty_like(v)
    fs       = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)

    kkt_ws    = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    wy_ws_a1  = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    wy_ws_a2  = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    h_ws      = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)
    o_ws_qk   = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_ws_qs   = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
    o_ws_gated = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_out     = torch.empty_like(v)

    lib = _load_mega_kernel(num_heads=H, key_heads=kh, hidden_size=D, chunk_size=C)
    lib.call_kernel(
        bd, stream,
        _vp(q), _vp(k), _vp(v), _vp(g_in), _vp(beta),
        _vp(msk_lower), _vp(msk_full), _vp(minus_identity), _vp(cu_seqlens),
        _vp(o_out),
        _vp(g_sum), _vp(g_t), _vp(beta_t),
        _vp(A), _vp(A_inv_f32), _vp(A_inv),
        _vp(w), _vp(u), _vp(s), _vp(v_new), _vp(fs),
        _vp(kkt_ws), _vp(wy_ws_a1), _vp(wy_ws_a2), _vp(h_ws),
        _vp(o_ws_qk), _vp(o_ws_qs), _vp(o_ws_gated),
        N_seq, T, T, num_matrices,
    )

    o_scaled = o_out * scale
    if return_final_state:
        return o_scaled, fs.view(N_seq, H, D, D)
    return o_scaled
