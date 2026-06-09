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
from megagdn_pto.kernel_libs import (
    chunk_gdn_causal_masks,
    precomputed_minus_identity,
    total_chunks,
    _vp,
)


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
    # Register call_kernel_bf16 argtypes:
    #   (block_dim, stream,
    #    q_bf16, k_bf16, v_bf16, beta_bf16,   <- 4 bf16 inputs
    #    g_in,                                 <- float32 gate (Python does g.float())
    #    msk_lower, msk_full, minus_id, cu_seqlens, o,
    #    g_sum, g_t, beta_t, A, A_inv_f32, A_inv,
    #    w, u, s, v_new, fs,
    #    kkt_ws, wy_ws_a1, wy_ws_a2, h_ws,
    #    o_ws_qk, o_ws_qs, o_ws_gated,        <- 24 existing workspace void* args
    #    q_fp16, k_fp16, v_fp16, beta_fp16,    <- 4 fp16 workspace void*
    #    batch_size, seq_len, total_tokens,    <- int64 × 3
    #    num_matrices,                         <- uint32
    #    hg_elems, hv_elems, beta_elems)       <- int64 × 3
    lib.call_kernel_bf16.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * (4 + 1 + 23 + 4)  # bf16_in(4) + g(1) + workspace(23) + fp16_ws(4)
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32,
           ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    )
    lib.call_kernel_bf16.restype = None
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

    dt, di = dev.type, dev.index if dev.index is not None else -1
    msk_lower, msk_full = chunk_gdn_causal_masks(dt, di, C)
    minus_identity = precomputed_minus_identity(dt, di, C)

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


def run_mega_kernel_bf16(
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
    """Run all seven GDN stages in a single fused NPU kernel launch, with BF16 inputs.

    This is the BF16 variant of :func:`run_mega_kernel`. It accepts BF16 tensors
    for ``q``, ``k``, ``v``, ``beta`` and converts them to FP16 inside the kernel,
    eliminating the need for separate ``torch.Tensor.to(torch.float16)`` calls in Python.

    The backward-compatible FP16 path (``run_mega_kernel``) is unchanged.

    Args:
        q, k:             ``[B, T, Hg, D]`` bfloat16 query and key tensors.
        v:                ``[B, T, H, D]`` bfloat16 value tensor (H ≥ Hg, H % Hg == 0).
        g_in:             ``[B, T, H]`` float32 pre-cumsum gate logits.
                          (Caller does ``g.float()`` in Python — single-hop BF16→FP32
                          is cheap and torch-efficient for the smaller gate tensor.)
        beta:             ``[B, T, H]`` bfloat16 gate bias.
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
    assert q.dtype == k.dtype == v.dtype == beta.dtype == torch.bfloat16, \
        "run_mega_kernel_bf16: q, k, v, beta must be bfloat16"
    assert g_in.dtype == torch.float32, \
        "run_mega_kernel_bf16: g_in must be float32 (pass g.float() from Python)"

    dev = q.device
    kh = key_heads if key_heads is not None else q.shape[2]
    H, D = v.shape[2], q.shape[3]
    C = chunk_size
    T = q.shape[1]
    N_seq = int(cu_seqlens.numel()) - 1
    bd = block_dim or BLOCK_DIM

    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    dt, di = dev.type, dev.index if dev.index is not None else -1
    msk_lower, msk_full = chunk_gdn_causal_masks(dt, di, C)
    minus_identity = precomputed_minus_identity(dt, di, C)

    tc = total_chunks(N_seq, T, C, cu_seqlens)
    num_matrices = tc * H

    # ---- existing fp16 workspace (same as run_mega_kernel) ----
    g_sum    = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    g_t      = torch.empty(H, T, device=dev, dtype=torch.float32)
    beta_t   = torch.empty(H, T, device=dev, dtype=torch.float16)
    A        = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    A_inv_f32 = torch.zeros(1, T, H, C, device=dev, dtype=torch.float32)
    A_inv    = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    w        = torch.empty_like(v, dtype=torch.float16)
    u        = torch.empty_like(v, dtype=torch.float16)
    s        = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)
    v_new    = torch.empty_like(v, dtype=torch.float16)
    fs       = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)

    kkt_ws    = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    wy_ws_a1  = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    wy_ws_a2  = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    h_ws      = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)
    o_ws_qk   = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_ws_qs   = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
    o_ws_gated = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_out     = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)

    # ---- NEW: fp16 workspace for bf16→fp16 cast inside kernel ----
    q_fp16    = torch.empty_like(q, dtype=torch.float16)
    k_fp16    = torch.empty_like(k, dtype=torch.float16)
    v_fp16    = torch.empty_like(v, dtype=torch.float16)
    beta_fp16 = torch.empty_like(beta, dtype=torch.float16)

    # ---- element counts for cast dimensions ----
    hg_elems   = int(T * kh * D)   # q/k: total_tokens × Hg × D
    hv_elems   = int(T * H * D)    # v:   total_tokens × H × D
    beta_elems = int(T * H)        # beta: total_tokens × H

    lib = _load_mega_kernel(num_heads=H, key_heads=kh, hidden_size=D, chunk_size=C)
    lib.call_kernel_bf16(
        bd, stream,
        # BF16 inputs
        _vp(q), _vp(k), _vp(v), _vp(beta),
        # float32 gate
        _vp(g_in),
        # existing workspace (same order as call_kernel minus the first 5 ptrs)
        _vp(msk_lower), _vp(msk_full), _vp(minus_identity), _vp(cu_seqlens),
        _vp(o_out),
        _vp(g_sum), _vp(g_t), _vp(beta_t),
        _vp(A), _vp(A_inv_f32), _vp(A_inv),
        _vp(w), _vp(u), _vp(s), _vp(v_new), _vp(fs),
        _vp(kkt_ws), _vp(wy_ws_a1), _vp(wy_ws_a2), _vp(h_ws),
        _vp(o_ws_qk), _vp(o_ws_qs), _vp(o_ws_gated),
        # fp16 workspace for cast
        _vp(q_fp16), _vp(k_fp16), _vp(v_fp16), _vp(beta_fp16),
        # dimensions
        N_seq, T, T, num_matrices,
        hg_elems, hv_elems, beta_elems,
    )

    o_scaled = o_out * scale
    if return_final_state:
        return o_scaled, fs.view(N_seq, H, D, D)
    return o_scaled
