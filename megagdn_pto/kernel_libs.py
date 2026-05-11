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


def _prepare_initial_state(
    initial_state: torch.Tensor | None,
    *,
    batch: int,
    num_heads: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return initial state as contiguous ``[batch*num_heads, D, D]`` fp16."""
    if initial_state is None:
        return None

    flat_shape = (batch * num_heads, hidden_size, hidden_size)
    view_shape = (batch, num_heads, hidden_size, hidden_size)
    if tuple(initial_state.shape) == view_shape:
        initial_state = initial_state.reshape(flat_shape)
    elif tuple(initial_state.shape) != flat_shape:
        raise ValueError(
            "initial_state must have shape "
            f"{view_shape} or {flat_shape}, got {tuple(initial_state.shape)}"
        )
    if initial_state.device != device:
        raise ValueError(
            f"initial_state must be on {device}, got {initial_state.device}"
        )
    return initial_state.to(dtype=torch.float16).contiguous()


def _seed_initial_state_snapshots(
    s_out: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    batch: int,
    num_heads: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> None:
    if initial_state is None:
        return

    chunks = s_out.view(-1, num_heads, initial_state.shape[-2], initial_state.shape[-1])
    init = initial_state.view(batch, num_heads, initial_state.shape[-2], initial_state.shape[-1])
    if cu_seqlens is None:
        chunks_per_seq = (seq_len + chunk_size - 1) // chunk_size
        for seq_idx in range(batch):
            chunks[seq_idx * chunks_per_seq].copy_(init[seq_idx])
        return

    cu = cu_seqlens.cpu().tolist()
    chunk_offset = 0
    for seq_idx in range(batch):
        chunks[chunk_offset].copy_(init[seq_idx])
        seqlen = cu[seq_idx + 1] - cu[seq_idx]
        chunk_offset += (seqlen + chunk_size - 1) // chunk_size


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

@lru_cache(maxsize=None)
def _load(
    cpp_name: str,
    so_stem: str,
    *,
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    key_heads: int | None = None,
) -> ctypes.CDLL:
    lib_path = compile_chunk_kernel(
        cpp_name,
        so_stem,
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        key_heads=key_heads,
        cpp_mtime_ns=_mtime(cpp_name),
    )
    return ctypes.CDLL(os.path.abspath(lib_path))


# ---------------------------------------------------------------------------
# chunk_cumsum  — chunk-local prefix sum of log-gate values G
# ---------------------------------------------------------------------------

def load_chunk_cumsum(
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
) -> ctypes.CDLL:
    """Compile + load the standalone chunk_cumsum kernel.

    Signature::
        void call_kernel(uint32_t block_dim, void *stream,
                         uint8_t *g, uint8_t *g_sum,
                         uint8_t *cu_seqlens,
                         int64_t batch_size, int64_t seq_len)
    """
    lib = _load(
        "chunk_cumsum.cpp", "chunk_cumsum",
        num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size,
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 3
        + [ctypes.c_int64, ctypes.c_int64]
    )
    lib.call_kernel.restype = None
    return lib


def run_chunk_cumsum(
    g: torch.Tensor,
    g_sum: torch.Tensor,
    *,
    stream,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Compute chunk-local cumulative sum of gate logits in-place into ``g_sum``.

    ``g``, ``g_sum``: ``[B, T, H]`` float32.
    """
    H = g.shape[2]
    bd = block_dim or BLOCK_DIM
    batch = g.shape[0] if batch_size_override is None else batch_size_override
    T = g.shape[1]
    cu32 = _ensure_int32(cu_seqlens)
    lib = load_chunk_cumsum(H, g.shape[2], chunk_size)
    lib.call_kernel(bd, stream, _vp(g), _vp(g_sum), _vp(cu32), batch, T)


# ---------------------------------------------------------------------------
# scaled_dot_kkt  — K @ K^T with gated causal mask → A  [B,T,H,C]
# ---------------------------------------------------------------------------

def load_scaled_dot_kkt(
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    *,
    key_heads: int | None = None,
) -> ctypes.CDLL:
    kh = key_heads if key_heads is not None else num_heads
    lib = _load(
        "scaled_dot_kkt.cpp", "scaled_dot_kkt",
        num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size,
        key_heads=key_heads,
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 7
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
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
    stream,
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
    hg = k.shape[2]
    kh = key_heads if key_heads is not None else hg
    H = beta.shape[2]
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)
    T = g_sum.shape[1]
    ws = torch.zeros(bd * 2, chunk_size, chunk_size, device=k.device, dtype=torch.float16)
    lib = load_scaled_dot_kkt(H, k.shape[3], chunk_size, key_heads=kh)
    lib.call_kernel(
        bd, stream,
        _vp(k), _vp(beta_t), _vp(g_t), _vp(mask), _vp(ws), _vp(A_out), _vp(cu32),
        batch, k.shape[1], T,
    )


# ---------------------------------------------------------------------------
# wy_fast  —  A @ V → u,  A @ (K·β·exp(g)) → w
# ---------------------------------------------------------------------------

def load_wy_fast(
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    *,
    key_heads: int | None = None,
) -> ctypes.CDLL:
    lib = _load(
        "wy_fast.cpp", "wy_fast",
        num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size,
        key_heads=key_heads,
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 10
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
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
    stream,
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
    hg = k.shape[2]
    kh = key_heads if key_heads is not None else hg
    H = v.shape[2]
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)
    T = g_sum.shape[1]
    ws_a1 = torch.zeros(bd, chunk_size, chunk_size, device=k.device, dtype=torch.float16)
    ws_a2 = torch.zeros_like(ws_a1)
    lib = load_wy_fast(H, k.shape[3], chunk_size, key_heads=kh)
    lib.call_kernel(
        bd, stream,
        _vp(k), _vp(v), _vp(beta_t), _vp(g_t), _vp(A),
        _vp(ws_a1), _vp(ws_a2), _vp(w_out), _vp(u_out), _vp(cu32),
        batch, k.shape[1], T,
    )


# ---------------------------------------------------------------------------
# chunk_h  —  Recurrent state update S and new-value v_new
# ---------------------------------------------------------------------------

def load_chunk_h(
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    *,
    key_heads: int | None = None,
) -> ctypes.CDLL:
    lib = _load(
        "chunk_h.cpp", "chunk_h",
        num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size,
        key_heads=key_heads,
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 10
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
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
    final_state_out: torch.Tensor,
    *,
    stream,
    g_t: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
    key_heads: int | None = None,
) -> None:
    """Compute chunk states S and per-token v_new (de-interfered values).

    ``k``: ``[B, T, Hg, D]``; ``w``, ``u``: ``[B, T, H, D]``.
    ``s_out``: ``[total_chunks*H, D, D]``; ``final_state_out``: ``[N_seq*H, D, D]``.
    ``initial_state``: optional ``[N_seq, H, D, D]`` or ``[N_seq*H, D, D]``.
    """
    hg = k.shape[2]
    kh = key_heads if key_heads is not None else hg
    H = w.shape[2]
    D = k.shape[3]
    bd = block_dim or BLOCK_DIM
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)
    T = g_sum.shape[1]
    initial_state_arg = _prepare_initial_state(
        initial_state,
        batch=batch,
        num_heads=H,
        hidden_size=D,
        device=k.device,
    )
    _seed_initial_state_snapshots(
        s_out,
        initial_state_arg,
        batch=batch,
        num_heads=H,
        seq_len=k.shape[1],
        chunk_size=chunk_size,
        cu_seqlens=cu32,
    )
    ws = torch.zeros(bd * 4, D, D, device=k.device, dtype=torch.float16)
    lib = load_chunk_h(H, D, chunk_size, key_heads=kh)
    lib.call_kernel(
        bd, stream,
        _vp(k), _vp(w), _vp(u), _vp(g_t),
        _vp(s_out), _vp(v_new_out), _vp(final_state_out),
        _vp(initial_state_arg), _vp(ws), _vp(cu32),
        batch, k.shape[1], T,
    )


# ---------------------------------------------------------------------------
# chunk_o  —  Gated intra-chunk + cross-chunk output
# ---------------------------------------------------------------------------

def load_chunk_o(
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    *,
    key_heads: int | None = None,
) -> ctypes.CDLL:
    lib = _load(
        "chunk_o.cpp", "chunk_o",
        num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size,
        key_heads=key_heads,
    )
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 11
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
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
    stream,
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
    hg = q.shape[2]
    kh = key_heads if key_heads is not None else hg
    H = v_new.shape[2]
    D = q.shape[3]
    bd = block_dim or BLOCK_DIM
    batch = q.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)
    T = g_sum.shape[1]
    ws_qk = torch.zeros(bd, chunk_size, chunk_size, device=q.device, dtype=torch.float16)
    ws_qs = torch.zeros(bd, chunk_size, D, device=q.device, dtype=torch.float16)
    ws_gated = torch.zeros(bd, chunk_size, chunk_size, device=q.device, dtype=torch.float16)
    lib = load_chunk_o(H, D, chunk_size, key_heads=kh)
    lib.call_kernel(
        bd, stream,
        _vp(q), _vp(k), _vp(v_new), _vp(s), _vp(g_t), _vp(mask),
        _vp(ws_qk), _vp(ws_qs), _vp(ws_gated), _vp(o_out), _vp(cu32),
        batch, q.shape[1], T,
    )
