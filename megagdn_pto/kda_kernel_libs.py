"""Loaders and runners for individual KDA PTO kernels.

KDA (Kimi Linear Attention / GDN variant) pipeline:
  gate_cumsum_kda → kkt_kda → inversion_kda → wy_kda → chunk_h_kda → chunk_o_kda

This module provides standalone wrappers for each stage, analogous to the GDN
wrappers in ``megagdn_pto.kernel_libs``.  The full fused pipeline is in
``megagdn_pto.kda_mega_kernel``.

Key shape difference from GDN:
  - GDN gates: ``[B, T, H]``     (scalar per token/head)
  - KDA gates: ``[B, T, HV, K]`` (K-vector per token/head)
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from megagdn_pto.compile import (
    BLOCK_DIM,
    _KERNELS_PTO,
    compile_chunk_kernel,
)
from megagdn_pto.kernel_libs import _ensure_int32, _vp


def _mtime(name: str) -> int:
    return os.stat(os.path.join(_KERNELS_PTO, name)).st_mtime_ns


# ---------------------------------------------------------------------------
# gate_cumsum_kda — within-chunk prefix sum of g [B, T, HV, K]
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_gate_cumsum_kda(
    num_heads: int,
    k_dim: int = 128,
    chunk_size: int = 16,
) -> ctypes.CDLL:
    """Compile + load the KDA gate cumsum kernel.

    Template parameters injected at compile time:
        GDN_H = num_heads  (HV, number of value/gate heads)
        GDN_D = k_dim      (K, key/gate vector dimension)
        GDN_C = chunk_size (C, tokens per chunk)

    C signature::
        void call_kernel(uint32_t block_dim, void *stream,
                         uint8_t *g, uint8_t *g_sum, uint8_t *cu_seqlens,
                         int64_t batch_size, int64_t seq_len)
    """
    lib_path = compile_chunk_kernel(
        "gate_cumsum_kda.cpp",
        "gate_cumsum_kda",
        num_heads=num_heads,
        hidden_size=k_dim,
        chunk_size=chunk_size,
        key_heads=None,
        cpp_mtime_ns=_mtime("gate_cumsum_kda.cpp"),
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 3
        + [ctypes.c_int64, ctypes.c_int64]
    )
    lib.call_kernel.restype = None
    return lib


def run_gate_cumsum_kda(
    g: torch.Tensor,
    g_sum: torch.Tensor,
    *,
    stream,
    chunk_size: int = 16,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Compute within-chunk cumulative sum of KDA gate logits into ``g_sum``.

    Args:
        g:       ``[B, T, HV, K]`` float16, raw per-dimension gate values.
        g_sum:   ``[B, T, HV, K]`` float32, output (cumulative sums).
        stream:  NPU stream handle (``torch.npu.current_stream()._as_parameter_``).
        chunk_size: Tokens per chunk C.  Must match the compiled kernel.
        cu_seqlens: ``int32`` cumulative sequence lengths for packed varlen input.
                    If ``None``, assumes a single sequence of length ``T``.
        batch_size_override: Number of sequences (use with ``cu_seqlens``).
        block_dim: AI-Core count; auto-detected if ``None``.
    """
    assert g.dtype == torch.float16 and g_sum.dtype == torch.float32
    assert g.shape == g_sum.shape

    HV = g.shape[2]
    K  = g.shape[3]
    T  = g.shape[1]
    bd = block_dim or BLOCK_DIM
    batch = g.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)

    lib = load_gate_cumsum_kda(HV, K, chunk_size)
    lib.call_kernel(bd, stream, _vp(g), _vp(g_sum), _vp(cu32), batch, T)


# ---------------------------------------------------------------------------
# kkt_kda — within-chunk gated attention matrix L [B, T, HV, C]
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_kkt_kda(
    num_heads: int,
    k_dim: int = 128,
    chunk_size: int = 16,
) -> ctypes.CDLL:
    """Compile + load the KDA kkt kernel.

    Template parameters injected at compile time:
        GDN_H = num_heads  (HV, number of value/gate heads)
        GDN_D = k_dim      (K, key/gate vector dimension)
        GDN_C = chunk_size (C, tokens per chunk)

    C signature::
        void call_kernel(uint32_t block_dim, void *stream,
                         uint8_t *k, uint8_t *g_cs, uint8_t *beta,
                         uint8_t *mask, uint8_t *ws_in, uint8_t *ws_out,
                         uint8_t *L_out, uint8_t *cu_seqlens,
                         int64_t batch_size, int64_t seq_len, int64_t total_tokens)
    """
    lib_path = compile_chunk_kernel(
        "kkt_kda.cpp",
        "kkt_kda",
        num_heads=num_heads,
        hidden_size=k_dim,
        chunk_size=chunk_size,
        key_heads=None,
        cpp_mtime_ns=_mtime("kkt_kda.cpp"),
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 8   # k, g_cs, beta, mask, ws_in, ws_out, L_out, cu_seqlens
        + [ctypes.c_int64] * 3    # batch_size, seq_len, total_tokens
    )
    lib.call_kernel.restype = None
    return lib


def run_kkt_kda(
    k: torch.Tensor,
    g_cs: torch.Tensor,
    beta_sig: torch.Tensor,
    L_out: torch.Tensor,
    *,
    stream,
    chunk_size: int = 16,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Compute within-chunk gated attention matrix into ``L_out``.

    Args:
        k:        ``[B, T, HV, K]`` float16, key vectors.
        g_cs:     ``[B, T, HV, K]`` float32, within-chunk cumulative gate sums.
        beta_sig: ``[B, T, HV]`` float16, per-token sigmoid beta in (0, 1).
        L_out:    ``[B, T, HV, C]`` float16, output (pre-allocated, will be overwritten).
        stream:   NPU stream handle (``torch.npu.current_stream()._as_parameter_``).
        chunk_size: Tokens per chunk C.  Must match the compiled kernel.
        cu_seqlens: ``int32`` cumulative sequence lengths for packed varlen input.
        batch_size_override: Number of sequences (use with ``cu_seqlens``).
        block_dim: AI-Core count; auto-detected if ``None``.
    """
    assert k.dtype == torch.float16
    assert g_cs.dtype == torch.float32
    assert beta_sig.dtype == torch.float16
    assert L_out.dtype == torch.float16

    HV = k.shape[2]
    K  = k.shape[3]
    T  = k.shape[1]
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    total_tokens = T  # B=1 packed format

    # Transpose to head-major [B, HV, T, K] so that per-head token rows are
    # contiguous in memory, satisfying the MTE2 TLOAD row-stride == column-count
    # requirement (row stride K == column count K).
    k_t    = k.permute(0, 2, 1, 3).contiguous()
    g_cs_t = g_cs.permute(0, 2, 1, 3).contiguous()
    beta_t = beta_sig.permute(0, 2, 1).contiguous()

    # Strictly-lower-tri mask [C, C]: 1 below diagonal, 0 on/above diagonal.
    dev  = k.device
    rows = torch.arange(chunk_size, device=dev).unsqueeze(1)
    cols = torch.arange(chunk_size, device=dev).unsqueeze(0)
    mask = (rows > cols).to(torch.float32)

    # fp32: A_eff=k*exp(g_cs)/B_eff=k*exp(-g_cs) and the unmasked K·K^T reach
    # ~e^64 (per-128-chunk |g_cs|≈64), which overflows fp16.
    ws_in  = torch.zeros(bd * 2, 2 * chunk_size, K,          device=dev, dtype=torch.float32)
    ws_out = torch.zeros(bd * 2, chunk_size,      chunk_size, device=dev, dtype=torch.float32)
    # Force the workspace zero-fill (and any pending stream work) to fully
    # complete before launching the kkt_kda kernel.  Without this barrier,
    # cold-start runs can race the FFTS V↔C handshake against the zero-fill,
    # producing all-zero output for the first kkt_kda launch that follows a
    # different launch shape.
    torch.npu.synchronize()

    cu32 = _ensure_int32(cu_seqlens)

    lib = load_kkt_kda(HV, K, chunk_size)
    lib.call_kernel(
        bd, stream,
        _vp(k_t), _vp(g_cs_t), _vp(beta_t),
        _vp(mask), _vp(ws_in), _vp(ws_out), _vp(L_out),
        _vp(cu32),
        batch, T, total_tokens,
    )


# ---------------------------------------------------------------------------
# wy_kda — WY decomposition (u, w) for KDA, per-dim gate
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_wy_kda(
    num_heads: int,
    k_dim: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    """Compile + load the KDA wy kernel.

    Template parameters injected at compile time:
        GDN_H = num_heads  (HV)
        GDN_D = k_dim      (K, also used as V_DIM)
        GDN_C = chunk_size (C)

    C signature::
        void call_kernel(uint32_t block_dim, void *stream,
                         uint8_t *k, uint8_t *v, uint8_t *beta, uint8_t *g_cs, uint8_t *A,
                         uint8_t *ws_a2, uint8_t *ws_keff,
                         uint8_t *u, uint8_t *w,
                         uint8_t *cu_seqlens,
                         int64_t batch_size, int64_t seq_len, int64_t total_tokens)
    """
    lib_path = compile_chunk_kernel(
        "wy_kda.cpp",
        "wy_kda",
        num_heads=num_heads,
        hidden_size=k_dim,
        chunk_size=chunk_size,
        key_heads=None,
        cpp_mtime_ns=_mtime("wy_kda.cpp"),
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 10   # k, v, beta, g_cs, A, ws_a2, ws_keff, u, w, cu_seqlens
        + [ctypes.c_int64] * 3     # batch_size, seq_len, total_tokens
    )
    lib.call_kernel.restype = None
    return lib


def run_wy_kda(
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    beta_sig: torch.Tensor,
    INV: torch.Tensor,
    u_out: torch.Tensor,
    w_out: torch.Tensor,
    *,
    stream,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Compute the WY auxiliary tensors u, w for KDA.

    Math (per chunk, matches ``ref_wy_kda``):
        u = INV @ (beta * v)
        w = INV @ (beta * exp(g_cs) * k)

    Args:
        k:        ``[B, T, HV, K]`` float16 (BSND; GQA-expanded).
        v:        ``[B, T, HV, V]`` float16.
        g_cs:     ``[B, T, HV, K]`` float32, within-chunk cumulative gate sum (per-dim).
        beta_sig: ``[B, T, HV]``    float16, post-sigmoid beta in (0, 1).
        INV:      ``[B, T, HV, C]`` float16, full lower-tri inverse (I+L)^{-1}.
        u_out:    ``[B, T, HV, V]`` float16 (overwritten).
        w_out:    ``[B, T, HV, K]`` float16 (overwritten).
        stream:   NPU stream handle.
        chunk_size: Tokens per chunk C; must match the compiled kernel.
        cu_seqlens: ``int32`` cumulative sequence lengths for packed varlen input.
        batch_size_override: Number of sequences (use with ``cu_seqlens``).
        block_dim: AI-Core count; auto-detected if ``None``.
    """
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16
    assert g_cs.dtype == torch.float32
    assert beta_sig.dtype == torch.float16
    assert INV.dtype == torch.float16
    assert u_out.dtype == torch.float16
    assert w_out.dtype == torch.float16

    HV = k.shape[2]
    K  = k.shape[3]
    T  = k.shape[1]
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override

    # Head-major permutes (match kkt_kda convention, kkt_kda lines 195-197).
    k_t    = k.permute(0, 2, 1, 3).contiguous()        # [B, HV, T, K]
    g_cs_t = g_cs.permute(0, 2, 1, 3).contiguous()
    beta_t = beta_sig.permute(0, 2, 1).contiguous()    # [B, HV, T]
    ws_a2   = torch.zeros(bd, chunk_size, chunk_size, device=k.device, dtype=torch.float16)
    ws_keff = torch.zeros(bd, chunk_size, K,          device=k.device, dtype=torch.float16)

    torch.npu.synchronize()

    cu32 = _ensure_int32(cu_seqlens)

    lib = load_wy_kda(HV, K, chunk_size)
    lib.call_kernel(
        bd, stream,
        _vp(k_t), _vp(v.contiguous()), _vp(beta_t), _vp(g_cs_t), _vp(INV),
        _vp(ws_a2), _vp(ws_keff),
        _vp(u_out), _vp(w_out),
        _vp(cu32),
        batch, T, T,
    )


# ---------------------------------------------------------------------------
# chunk_h_kda — sequential state recurrence (snapshots + v_corr)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_chunk_h_kda(
    num_heads: int,
    k_dim: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    """Compile + load the KDA chunk_h kernel.

    Template parameters injected at compile time:
        GDN_H = num_heads  (HV)
        GDN_D = k_dim      (K, also used as V_DIM)
        GDN_C = chunk_size (C)

    C signature::
        void call_kernel(uint32_t block_dim, void *stream,
                         uint8_t *K, uint8_t *W, uint8_t *U, uint8_t *G,
                         uint8_t *S, uint8_t *V_corr,
                         uint8_t *workspace, uint8_t *cu_seqlens,
                         int64_t batch_size, int64_t seq_len, int64_t total_tokens)
    """
    lib_path = compile_chunk_kernel(
        "chunk_h_kda.cpp",
        "chunk_h_kda",
        num_heads=num_heads,
        hidden_size=k_dim,
        chunk_size=chunk_size,
        key_heads=None,
        cpp_mtime_ns=_mtime("chunk_h_kda.cpp"),
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 8   # K, W, U, G, S, V_corr, workspace, cu_seqlens
        + [ctypes.c_int64] * 3    # batch_size, seq_len, total_tokens
    )
    lib.call_kernel.restype = None
    return lib


def run_chunk_h_kda(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_cs: torch.Tensor,
    s_snapshots_out: torch.Tensor,
    v_corr_out: torch.Tensor,
    *,
    stream,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Sequential recurrent state pass for KDA.

    Math per chunk (matches ``ref_chunk_h_kda``):
        v_corr  = u - w @ S                              # [c_len, V]
        k_rest  = k * exp(g_total - g_cs)                # [c_len, K]
        S_new   = exp(g_total).unsqueeze(-1) * S + k_rest^T @ v_corr

    Args:
        k:               ``[B, T, HV, K]`` float16, keys (GQA-expanded).
        w:               ``[B, T, HV, K]`` float16, from wy_kda.
        u:               ``[B, T, HV, V]`` float16, from wy_kda.
        g_cs:            ``[B, T, HV, K]`` float32, within-chunk cumulative gate sum.
        s_snapshots_out: ``[total_chunks, HV, K, V]`` float16 (output).
        v_corr_out:      ``[B, T, HV, V]`` float16 (output).
        stream:          NPU stream handle.
        chunk_size:      Tokens per chunk C; must match the compiled kernel.
        cu_seqlens:      ``int32`` cumulative sequence lengths for packed varlen input.
        batch_size_override: Number of sequences (use with ``cu_seqlens``).
        block_dim:       AI-Core count; auto-detected if ``None``.
    """
    assert k.dtype == torch.float16
    assert w.dtype == torch.float16
    assert u.dtype == torch.float16
    assert g_cs.dtype == torch.float32
    assert s_snapshots_out.dtype == torch.float16
    assert v_corr_out.dtype == torch.float16

    HV = k.shape[2]
    K  = k.shape[3]
    T  = k.shape[1]
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override

    # Head-major permutes for K and g_cs (matches wy_kda's MTE2 row-stride
    # requirement).  W and U stay in BSND so Cube's K^T @ V_corr and Vec's
    # U load can use the BSND stride.
    k_t    = k.permute(0, 2, 1, 3).contiguous()         # [B, HV, T, K]
    g_cs_t = g_cs.permute(0, 2, 1, 3).contiguous()

    # Per-AI-core workspace: 5 slots × K*V half-elements.
    ws = torch.zeros(bd * 5, K, K, device=k.device, dtype=torch.float16)
    cu32 = _ensure_int32(cu_seqlens)

    torch.npu.synchronize()

    lib = load_chunk_h_kda(HV, K, chunk_size)
    lib.call_kernel(
        bd, stream,
        _vp(k_t), _vp(w.contiguous()), _vp(u), _vp(g_cs_t),
        _vp(s_snapshots_out), _vp(v_corr_out), _vp(ws), _vp(cu32),
        batch, T, T,
    )


# ---------------------------------------------------------------------------
# chunk_o_kda — output stage (q_eff @ S + tril(q_eff @ k_eff^T) @ v_corr)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_chunk_o_kda(
    num_heads: int,
    k_dim: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    """Compile + load the KDA chunk_o kernel.

    Template parameters injected at compile time:
        GDN_H = num_heads  (HV)
        GDN_D = k_dim      (K, also used as V_DIM)
        GDN_C = chunk_size (C)

    C signature::
        void call_kernel(uint32_t block_dim, void *stream,
                         uint8_t *Q, uint8_t *K, uint8_t *V_corr, uint8_t *S,
                         uint8_t *G, uint8_t *Mask,
                         uint8_t *workspace, uint8_t *O,
                         uint8_t *cu_seqlens,
                         int64_t batch_size, int64_t seq_len, int64_t total_tokens)
    """
    lib_path = compile_chunk_kernel(
        "chunk_o_kda.cpp",
        "chunk_o_kda",
        num_heads=num_heads,
        hidden_size=k_dim,
        chunk_size=chunk_size,
        key_heads=None,
        cpp_mtime_ns=_mtime("chunk_o_kda.cpp"),
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 9   # Q, K, V_corr, S, G, Mask, workspace, O, cu_seqlens
        + [ctypes.c_int64] * 3    # batch_size, seq_len, total_tokens
    )
    lib.call_kernel.restype = None
    return lib


def run_chunk_o_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v_corr: torch.Tensor,
    s_snapshots: torch.Tensor,
    g_cs: torch.Tensor,
    o_out: torch.Tensor,
    *,
    stream,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Output stage of the KDA pipeline.

    Math per chunk (matches ``ref_chunk_o_kda``):
        q_eff = q * exp(g_cs)              # [c_len, K]
        k_eff = k * exp(-g_cs)             # [c_len, K]
        Aqk   = tril(q_eff @ k_eff^T,      # inclusive diagonal
                     diagonal=0)
        o     = q_eff @ S + Aqk @ v_corr   # [c_len, V]

    Args:
        q:            ``[B, T, HV, K]`` float16 (BSND; scale and GQA already applied).
        k:            ``[B, T, HV, K]`` float16, keys.
        v_corr:       ``[B, T, HV, V]`` float16, from chunk_h_kda.
        s_snapshots:  ``[total_chunks, HV, K, V]`` float16, from chunk_h_kda.
        g_cs:         ``[B, T, HV, K]`` float32, within-chunk cumulative gate.
        o_out:        ``[B, T, HV, V]`` float16 (output, overwritten).
        stream:       NPU stream handle.
        chunk_size:   Tokens per chunk C; must match the compiled kernel.
        cu_seqlens:   ``int32`` cumulative sequence lengths for packed varlen input.
        batch_size_override: Number of sequences (use with ``cu_seqlens``).
        block_dim:    AI-Core count; auto-detected if ``None``.
    """
    assert q.dtype == torch.float16
    assert k.dtype == torch.float16
    assert v_corr.dtype == torch.float16
    assert s_snapshots.dtype == torch.float16
    assert g_cs.dtype == torch.float32
    assert o_out.dtype == torch.float16

    HV = q.shape[2]
    K  = q.shape[3]
    T  = q.shape[1]
    bd = block_dim or BLOCK_DIM
    batch = q.shape[0] if batch_size_override is None else batch_size_override

    # Head-major permutes for Q, K, g_cs (matches chunk_h_kda's MTE2 row-stride
    # requirement: row stride K == column count K).
    q_t    = q.permute(0, 2, 1, 3).contiguous()         # [B, HV, T, K]
    k_t    = k.permute(0, 2, 1, 3).contiguous()
    g_cs_t = g_cs.permute(0, 2, 1, 3).contiguous()

    # Causal mask [C, C] fp32, INCLUSIVE diagonal (matches torch.tril(..., diagonal=0)
    # used by ref_chunk_o_kda — differs from kkt_kda's strict-lower mask).
    dev  = q.device
    rows = torch.arange(chunk_size, device=dev).unsqueeze(1)
    cols = torch.arange(chunk_size, device=dev).unsqueeze(0)
    mask = (rows >= cols).to(torch.float32)

    # Per-AI-core workspace: 7 slots × K*V elements (fp32).
    #   WS_Q, WS_K [C, K], WS_V [C, V], WS_S [K, V], WS_QK [C, C],
    #   WS_QS, WS_QKV [C, V].  fp32 because k_eff=k*exp(-g_cs) and the unmasked
    #   QK = q_eff @ k_eff^T reach ~e^64, which overflows fp16.
    ws = torch.zeros(bd * 7, K, K, device=dev, dtype=torch.float32)
    cu32 = _ensure_int32(cu_seqlens)

    torch.npu.synchronize()

    lib = load_chunk_o_kda(HV, K, chunk_size)
    lib.call_kernel(
        bd, stream,
        _vp(q_t), _vp(k_t), _vp(v_corr), _vp(s_snapshots),
        _vp(g_cs_t), _vp(mask),
        _vp(ws), _vp(o_out), _vp(cu32),
        batch, T, T,
    )
