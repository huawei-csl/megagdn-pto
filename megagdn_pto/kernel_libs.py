"""Loaders and runners for the four chunk-GDN PTO kernels.

All kernels operate on packed-varlen BSND tensors with ``B=1`` and
``cu_seqlens`` encoding sequence boundaries. The GQA (group-value) layout
is supported: Q/K use ``Hg`` heads while V/gates use ``H ≥ Hg`` heads with
``H % Hg == 0``.

Kernel pipeline (matching ``dynamic_bsnd_groupvalue`` stage order):

    scaled_dot_kkt  →  (solve_tril via fast_inverse)  →  wy_fast  →  chunk_h  →  chunk_o

``run_chunk_cumsum`` is provided separately; for the full pipeline see
``megagdn_pto.mega_kernel.run_mega_kernel``.
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

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _vp(t: torch.Tensor | None) -> ctypes.c_void_p:
    if t is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(t.data_ptr())


def _ensure_int32(cu: torch.Tensor | None) -> torch.Tensor | None:
    if cu is None:
        return None
    return cu if cu.dtype == torch.int32 else cu.to(torch.int32)


_SUPPORTED_HEADS = frozenset({16, 24, 32, 48, 64})


def _check_supported_heads(value_heads: int, key_heads: int | None = None) -> None:
    if value_heads not in _SUPPORTED_HEADS:
        raise ValueError(
            f"H={value_heads} is not supported by the compiled dispatch binary; "
            f"choose one of {_SUPPORTED_HEADS}"
        )
    if key_heads is not None and value_heads % key_heads != 0:
        raise ValueError(f"H={value_heads} must be divisible by key_heads={key_heads}")


def transpose_gates(g_sum: torch.Tensor) -> torch.Tensor:
    """``[1, T, H]`` → ``[H, T]`` contiguous (per-head gate layout for kernels)."""
    return g_sum.squeeze(0).t().contiguous()


def transpose_beta(beta: torch.Tensor) -> torch.Tensor:
    """``[1, T, H]`` → ``[H, T]`` contiguous."""
    return beta.squeeze(0).t().contiguous()


def total_chunks(
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> int:
    """Number of chunks across all sequences in the batch."""
    if cu_seqlens is None:
        return batch_size * ((seq_len + chunk_size - 1) // chunk_size)
    cu = cu_seqlens.cpu().tolist()
    return sum(
        (cu[i + 1] - cu[i] + chunk_size - 1) // chunk_size
        for i in range(len(cu) - 1)
    )


@lru_cache(maxsize=48)
def precomputed_minus_identity(device_ty: str, device_index: int, chunk_size: int) -> torch.Tensor:
    """Shared ``[C,C] fp16`` buffer with diagonal ``-1`` for ``tri_inverse`` / mega-kernel."""
    idx = device_index if device_index >= 0 else 0
    dev = torch.device(device_ty, idx) if device_ty != "cpu" else torch.device("cpu")
    t = torch.zeros(chunk_size, chunk_size, device=dev, dtype=torch.float16)
    t.fill_diagonal_(-1)
    return t


@lru_cache(maxsize=48)
def chunk_gdn_causal_masks(device_ty: str, device_index: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower-triangle masks for intra-chunk KKT attention (reuse across forwards)."""
    idx = device_index if device_index >= 0 else 0
    dev = torch.device(device_ty, idx) if device_ty != "cpu" else torch.device("cpu")
    m_lower = torch.tril(torch.ones(chunk_size, chunk_size, device=dev), diagonal=-1).float()
    m_full = torch.tril(torch.ones(chunk_size, chunk_size, device=dev), diagonal=0).float()
    return m_lower, m_full


def _mtime(name: str) -> int:
    return os.stat(os.path.join(_KERNELS_PTO, name)).st_mtime_ns


# ---------------------------------------------------------------------------
# Kernel loading
# ---------------------------------------------------------------------------

def _load(
    cpp_name: str,
    so_stem: str,
    *,
    hidden_size: int = 128,
    chunk_size: int = 128,
    cpp_mtime_ns: int,
) -> ctypes.CDLL:
    lib_path = compile_chunk_kernel(
        cpp_name,
        so_stem,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        cpp_mtime_ns=cpp_mtime_ns,
    )
    return ctypes.CDLL(os.path.abspath(lib_path))


# ---------------------------------------------------------------------------
# chunk_cumsum  — chunk-local prefix sum of log-gate values G
# ---------------------------------------------------------------------------

def load_chunk_cumsum(chunk_size: int = 128) -> ctypes.CDLL:
    """Compile + load the standalone chunk_cumsum kernel.

    Signature::
        void call_kernel(uint32_t block_dim, void *stream,
                         uint8_t *g, uint8_t *g_sum,
                         uint8_t *cu_seqlens,
                         int64_t batch_size, int64_t seq_len,
                         uint32_t num_heads)
    """
    lib = _load(
        "chunk_cumsum.cpp", "chunk_cumsum",
        chunk_size=chunk_size,
        cpp_mtime_ns=_mtime("chunk_cumsum.cpp"),
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 3
        + [ctypes.c_int64, ctypes.c_int64]
        + [ctypes.c_uint32]
    )
    lib.call_kernel.restype = None
    return lib


def run_chunk_cumsum(
    g: torch.Tensor,
    g_sum: torch.Tensor,
    *,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Compute chunk-local cumulative sum of gate logits in-place into ``g_sum``.

    ``g``, ``g_sum``: ``[B, T, H]`` float32.
    """
    H = g.shape[2]
    _check_supported_heads(H)
    bd = block_dim or BLOCK_DIM
    batch = g.shape[0] if batch_size_override is None else batch_size_override
    T = g.shape[1]
    cu32 = _ensure_int32(cu_seqlens)
    lib = load_chunk_cumsum(chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(bd, stream, _vp(g), _vp(g_sum), _vp(cu32), batch, T, H)


# ---------------------------------------------------------------------------
# scaled_dot_kkt  — K @ K^T with gated causal mask → A  [B,T,H,C]
# ---------------------------------------------------------------------------

def load_scaled_dot_kkt(
    hidden_size: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    lib = _load(
        "scaled_dot_kkt.cpp", "scaled_dot_kkt",
        hidden_size=hidden_size, chunk_size=chunk_size,
        cpp_mtime_ns=_mtime("scaled_dot_kkt.cpp"),
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 7
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        + [ctypes.c_uint32, ctypes.c_uint32]
    )
    lib.call_kernel.restype = None
    return lib


def run_scaled_dot_kkt(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_sum: torch.Tensor,
    mask: torch.Tensor,
    A_out: torch.Tensor,
    *,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
    key_heads: int | None = None,
) -> None:
    """Compute gated intra-chunk attention matrix A ``[B,T,H,C]``.

    ``k``: ``[B, T, Hg, D]``; ``beta``, ``g_sum``: ``[B, T, H]``; ``A_out``: ``[B, T, H, C]``.
    """
    H = beta.shape[2]
    kh = key_heads if key_heads is not None else k.shape[2]
    _check_supported_heads(H, kh)
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)
    T = g_sum.shape[1]
    ws = torch.zeros(bd * 2, chunk_size, chunk_size, device=k.device, dtype=torch.float16)
    lib = load_scaled_dot_kkt(k.shape[3], chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        bd, stream,
        _vp(k), _vp(beta_t), _vp(g_t), _vp(mask), _vp(ws), _vp(A_out), _vp(cu32),
        batch, k.shape[1], T, H, kh,
    )


# ---------------------------------------------------------------------------
# wy_fast  —  A @ V → u,  A @ (K·β·exp(g)) → w
# ---------------------------------------------------------------------------

def load_wy_fast(
    hidden_size: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    lib = _load(
        "wy_fast.cpp", "wy_fast",
        hidden_size=hidden_size, chunk_size=chunk_size,
        cpp_mtime_ns=_mtime("wy_fast.cpp"),
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 10
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        + [ctypes.c_uint32, ctypes.c_uint32]
    )
    lib.call_kernel.restype = None
    return lib


def run_wy_fast(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_sum: torch.Tensor,
    A: torch.Tensor,
    w_out: torch.Tensor,
    u_out: torch.Tensor,
    *,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
    key_heads: int | None = None,
) -> None:
    """Compute W-Y decomposition vectors w, u.

    ``k``: ``[B, T, Hg, D]``; ``v``, ``w_out``, ``u_out``: ``[B, T, H, D]``; ``A``: ``[B, T, H, C]``.
    """
    H = v.shape[2]
    kh = key_heads if key_heads is not None else k.shape[2]
    _check_supported_heads(H, kh)
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)
    T = g_sum.shape[1]
    ws_a1 = torch.zeros(bd, chunk_size, chunk_size, device=k.device, dtype=torch.float16)
    ws_a2 = torch.zeros_like(ws_a1)
    lib = load_wy_fast(k.shape[3], chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        bd, stream,
        _vp(k), _vp(v), _vp(beta_t), _vp(g_t), _vp(A),
        _vp(ws_a1), _vp(ws_a2), _vp(w_out), _vp(u_out), _vp(cu32),
        batch, k.shape[1], T, H, kh,
    )


# ---------------------------------------------------------------------------
# chunk_h  —  Recurrent state update S and new-value v_new
# ---------------------------------------------------------------------------

def load_chunk_h(
    hidden_size: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    lib = _load(
        "chunk_h.cpp", "chunk_h",
        hidden_size=hidden_size, chunk_size=chunk_size,
        cpp_mtime_ns=_mtime("chunk_h.cpp"),
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 7
        + [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        + [ctypes.c_void_p] * 2
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        + [ctypes.c_uint32, ctypes.c_uint32]
    )
    lib.call_kernel.restype = None
    return lib


def run_chunk_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_sum: torch.Tensor,
    s_out: torch.Tensor,
    v_new_out: torch.Tensor,
    final_state_out: torch.Tensor | None,
    *,
    g_t: torch.Tensor,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
    key_heads: int | None = None,
    initial_state: torch.Tensor | None = None,
) -> None:
    """Compute chunk states S and per-token v_new (de-interfered values).

    ``k``: ``[B, T, Hg, D]``; ``w``, ``u``: ``[B, T, H, D]``.
    ``s_out``: ``[total_chunks*H, D, D]``; ``final_state_out`` may be
    ``[N_seq*H, D, D]`` or ``[N_seq, H, D, D]``.
    """
    H = w.shape[2]
    D = k.shape[3]
    kh = key_heads if key_heads is not None else k.shape[2]
    _check_supported_heads(H, kh)
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)
    T = g_sum.shape[1]
    ws = torch.zeros(bd * 4, D, D, device=k.device, dtype=torch.float16)

    if final_state_out is None:
        final_state_buf = None
        output_final_state = 0
    elif final_state_out.shape == (batch, H, D, D):
        final_state_buf = final_state_out.view(batch * H, D, D)
        output_final_state = 1
    elif final_state_out.shape == (batch * H, D, D):
        final_state_buf = final_state_out
        output_final_state = 1
    else:
        raise ValueError(
            "final_state_out must have shape "
            f"({batch * H}, {D}, {D}) or ({batch}, {H}, {D}, {D})"
        )
    if final_state_buf is not None and not final_state_buf.is_contiguous():
        raise ValueError("final_state_out must be contiguous")

    if initial_state is None:
        h0 = None
        has_initial_state = 0
    else:
        if initial_state.shape != (batch, H, D, D):
            raise ValueError(
                "initial_state must have shape "
                f"({batch}, {H}, {D}, {D})"
            )
        h0 = initial_state.to(device=k.device, dtype=torch.float16).contiguous()
        has_initial_state = 1
    lib = load_chunk_h(D, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        bd, stream,
        _vp(k), _vp(w), _vp(u), _vp(g_t),
        _vp(s_out), _vp(v_new_out), _vp(final_state_buf),
        _vp(h0), has_initial_state, output_final_state, _vp(ws), _vp(cu32),
        batch, k.shape[1], T, H, kh,
    )


# ---------------------------------------------------------------------------
# chunk_o  —  Gated intra-chunk + cross-chunk output
# ---------------------------------------------------------------------------

def load_chunk_o(
    hidden_size: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    lib = _load(
        "chunk_o.cpp", "chunk_o",
        hidden_size=hidden_size, chunk_size=chunk_size,
        cpp_mtime_ns=_mtime("chunk_o.cpp"),
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 11
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        + [ctypes.c_uint32, ctypes.c_uint32]
    )
    lib.call_kernel.restype = None
    return lib


def run_chunk_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    s: torch.Tensor,
    g_sum: torch.Tensor,
    mask: torch.Tensor,
    o_out: torch.Tensor,
    *,
    g_t: torch.Tensor,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
    key_heads: int | None = None,
) -> None:
    """Compute output O = intra-chunk gated attention + cross-chunk state contribution.

    ``q``, ``k``: ``[B, T, Hg, D]``; ``v_new``, ``o_out``: ``[B, T, H, D]``.
    """
    H = v_new.shape[2]
    D = q.shape[3]
    kh = key_heads if key_heads is not None else q.shape[2]
    _check_supported_heads(H, kh)
    bd = block_dim or BLOCK_DIM
    batch = q.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)
    T = g_sum.shape[1]
    ws_qk = torch.zeros(bd, chunk_size, chunk_size, device=q.device, dtype=torch.float16)
    ws_qs = torch.zeros(bd, chunk_size, D, device=q.device, dtype=torch.float16)
    ws_gated = torch.zeros(bd, chunk_size, chunk_size, device=q.device, dtype=torch.float16)
    lib = load_chunk_o(D, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        bd, stream,
        _vp(q), _vp(k), _vp(v_new), _vp(s), _vp(g_t), _vp(mask),
        _vp(ws_qk), _vp(ws_qs), _vp(ws_gated), _vp(o_out), _vp(cu32),
        batch, q.shape[1], T, H, kh,
    )
