"""MegaGDN-PTO: fast PTO kernels for chunk GatedDeltaNet on Ascend NPU."""

from megagdn_pto.compile import BLOCK_DIM, PTO_LIB_PATH
from megagdn_pto.kernel_libs import (
    run_chunk_h,
    run_chunk_o,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
)
from megagdn_pto.fast_inverse import load_tri_inverse
from megagdn_pto.mega_kernel import run_mega_kernel

__all__ = [
    "BLOCK_DIM",
    "PTO_LIB_PATH",
    "run_scaled_dot_kkt",
    "run_wy_fast",
    "run_chunk_h",
    "run_chunk_o",
    "run_mega_kernel",
    "load_tri_inverse",
    "total_chunks",
]
