"""Launcher for the state-sized KDN GM/UB copy bandwidth ceiling."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from megagdn_pto.compile import BLOCK_DIM, _KERNELS_PTO, compile_kdn_memcpy_bound


def _vp(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


@lru_cache(maxsize=None)
def load_kdn_memcpy_bound(k_dim: int = 128, v_dim: int = 128, v_tile: int = 32) -> ctypes.CDLL:
    cpp_path = os.path.join(_KERNELS_PTO, "kdn_memcpy_bound.cpp")
    lib_path = compile_kdn_memcpy_bound(
        k_dim=k_dim, v_dim=v_dim, v_tile=v_tile,
        cpp_mtime_ns=os.stat(cpp_path).st_mtime_ns,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int32,
    ]
    lib.call_kernel.restype = None
    return lib


def run_kdn_memcpy_bound(
    state: torch.Tensor, *, v_tile: int = 32, block_dim: int | None = None,
) -> None:
    """Round-trip every fp32 state element through UB in-place.

    ``state`` is contiguous ``[B, H, V, K]`` fp32.  The tile mapping and block
    sizing match ``run_kdn_decode``; this function intentionally performs no
    vector arithmetic, providing a state-traffic upper bound.
    """
    if state.ndim != 4 or state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError("state must be contiguous fp32 [B,H,V,K]")
    if state.device.type != "npu":
        raise ValueError("state must be on an NPU")
    batch, heads, v_dim, k_dim = state.shape
    if k_dim <= 0 or v_dim <= 0 or k_dim % 8 or v_dim % v_tile:
        raise ValueError("K must be a positive multiple of 8 and v_tile must divide V")
    tasks = batch * heads * (v_dim // v_tile)
    if block_dim is None:
        block_dim = min(BLOCK_DIM, max(1, (tasks + 1) // 2))
    lib = load_kdn_memcpy_bound(k_dim, v_dim, v_tile)
    stream = torch.npu.current_stream(state.device)._as_parameter_
    lib.call_kernel(
        ctypes.c_uint32(block_dim), stream, _vp(state),
        ctypes.c_int64(batch), ctypes.c_int32(heads),
    )
