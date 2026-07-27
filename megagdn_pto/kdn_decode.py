"""JIT launcher for the vector-only PTO KDN recurrent decode kernel."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from megagdn_pto.compile import BLOCK_DIM, _KERNELS_PTO, compile_kdn_decode


def _vp(tensor: torch.Tensor | None) -> ctypes.c_void_p:
    return ctypes.c_void_p() if tensor is None else ctypes.c_void_p(tensor.data_ptr())


@lru_cache(maxsize=None)
def load_kdn_decode(k_dim: int = 128, v_dim: int = 128, v_tile: int = 64) -> ctypes.CDLL:
    cpp_path = os.path.join(_KERNELS_PTO, "kdn_decode.cpp")
    lib_path = compile_kdn_decode(
        k_dim=k_dim, v_dim=v_dim, v_tile=v_tile,
        cpp_mtime_ns=os.stat(cpp_path).st_mtime_ns,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 9
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
           ctypes.c_float, ctypes.c_int32]
    )
    lib.call_kernel.restype = None
    return lib


def run_kdn_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, initial_state: torch.Tensor | None = None, *,
    state_indices: torch.Tensor | None = None, scale: float | None = None,
    out: torch.Tensor | None = None, stream=None, v_tile: int = 64,
    block_dim: int | None = None, cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the default ``fused_recurrent_kda`` recurrence.

    q/k/g are [B,T,H,K], v is [B,T,H,V], beta is [B,T,H], in bf16/fp16.
    State is fp32 V-first [slots,H,V,K] and is updated in place.

    ``cu_seqlens`` is an int32 ``[N+1]`` prefix sum of sequence lengths; it
    requires ``B == 1`` and turns the token axis into ``N`` variable-length
    sequences, one state slot each.  ``use_qk_l2norm`` normalizes q and k over
    K inside the kernel, matching the fused sglang reference.
    """
    if q.ndim != 4:
        raise ValueError("q must be [B,T,H,K]")
    b, t, h, kdim = q.shape
    if tuple(k.shape) != tuple(q.shape) or tuple(g.shape) != tuple(q.shape):
        raise ValueError("q, k, and g must have identical [B,T,H,K] shapes")
    if v.ndim != 4 or tuple(v.shape[:3]) != (b, t, h):
        raise ValueError("v must be [B,T,H,V]")
    vdim = v.shape[-1]
    if tuple(beta.shape) != (b, t, h):
        raise ValueError("beta must be [B,T,H]")
    if t <= 0 or kdim <= 0 or vdim <= 0 or kdim % 8 or v_tile % 8:
        raise ValueError("T/K/V and v_tile must be positive; K and v_tile must be multiples of 8")
    tensors = (q, k, v, g, beta)
    if any(x.dtype not in (torch.bfloat16, torch.float16) for x in tensors):
        raise TypeError("q, k, v, g, and beta must be bf16 or fp16")
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("q, k, v, g, and beta must be contiguous")
    if any(x.device != q.device for x in tensors[1:]):
        raise ValueError("all inputs must share a device")
    if cu_seqlens is None:
        n_seq = b
    else:
        if b != 1:
            raise ValueError("cu_seqlens requires B == 1 (flatten sequences onto the token axis)")
        if (cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2
                or cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous()
                or cu_seqlens.device != q.device):
            raise ValueError("cu_seqlens must be contiguous int32 [N+1] on q's device")
        n_seq = cu_seqlens.numel() - 1
    if initial_state is None:
        initial_state = torch.zeros((n_seq, h, vdim, kdim), dtype=torch.float32, device=q.device)
    if (initial_state.ndim != 4 or tuple(initial_state.shape[1:]) != (h, vdim, kdim)
            or initial_state.dtype != torch.float32 or not initial_state.is_contiguous()):
        raise ValueError("initial_state must be contiguous fp32 [slots,H,V,K]")
    if initial_state.device != q.device:
        raise ValueError("initial_state must share a device with q")
    if state_indices is None:
        if initial_state.shape[0] < n_seq:
            raise ValueError("initial_state needs one slot per sequence without state_indices")
    else:
        if (tuple(state_indices.shape) != (n_seq,) or state_indices.dtype != torch.int32
                or not state_indices.is_contiguous() or state_indices.device != q.device):
            raise ValueError("state_indices must be contiguous int32 [N] on q's device")
    if out is None:
        out = torch.zeros_like(v)
    elif (tuple(out.shape) != tuple(v.shape) or out.dtype != v.dtype
          or not out.is_contiguous() or out.device != q.device):
        raise ValueError("out must be contiguous with v's shape/dtype/device")
    else:
        out.zero_()
    if scale is None:
        scale = kdim ** -0.5
    tasks = n_seq * h * ((vdim + v_tile - 1) // v_tile)
    if block_dim is None:
        block_dim = min(BLOCK_DIM, max(1, (tasks + 1) // 2))
    # CANN's C220 vector compiler does not implement bf16->fp32 TCVT.  Keep
    # the kernel's wire format fp16 and perform one explicit model-dtype cast
    # at the launcher boundary.
    work_inputs = tuple(x if x.dtype == torch.float16 else x.to(torch.float16)
                        for x in tensors)
    work_out = out if out.dtype == torch.float16 else torch.empty_like(out, dtype=torch.float16)
    if work_out is not out:
        work_out.zero_()
    lib = load_kdn_decode(kdim, vdim, v_tile)
    stream = torch.npu.current_stream(q.device)._as_parameter_
    lib.call_kernel(
        ctypes.c_uint32(block_dim), stream, *(_vp(x) for x in work_inputs),
        _vp(initial_state), _vp(work_out), _vp(state_indices), _vp(cu_seqlens),
        ctypes.c_int64(n_seq), ctypes.c_int64(t), ctypes.c_int32(h),
        ctypes.c_int32(initial_state.shape[0]), ctypes.c_float(scale),
        ctypes.c_int32(1 if use_qk_l2norm else 0),
    )
    if work_out is not out:
        out.copy_(work_out)
    return out, initial_state
