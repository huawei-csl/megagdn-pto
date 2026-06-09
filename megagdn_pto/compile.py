"""Bisheng JIT compilation for PTO GDN kernels on Ascend NPU.

All kernels compile to shared libraries (.so) cached under ``kernels/pto/compiled_lib/``.
Re-compilation is triggered automatically when the C++ source is modified (mtime key in lru_cache).

Environment variables:
    PTO_LIB_PATH            Path to pto-isa header directory (contains ``include/``).
                            Auto-detected from ``third_party/pto-isa`` submodule or
                            ``/sources/pto-isa`` if not set.
    ASCEND_TOOLKIT_HOME     Ascend toolkit root (required).
    GDN_NPU_DEVICE          NPU device used to query ``cube_core_num`` (default ``npu:0``).
    VERBOSE_COMPILE         Set to ``1`` to print the full bisheng command.
    PTO_DYNAMIC_EXTRA_FLAGS Extra flags appended to every bisheng invocation.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
_KERNELS_PTO = os.path.join(_REPO_ROOT, "kernels", "pto")
_KERNEL_INCLUDE = os.path.join(_KERNELS_PTO, "include")
_COMPILED_DIR = os.path.join(_KERNELS_PTO, "compiled_lib")
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"

ASCEND_TOOLKIT_HOME: str = (
    os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH", "")
)
if not ASCEND_TOOLKIT_HOME:
    raise RuntimeError(
        "ASCEND_TOOLKIT_HOME (or ASCEND_HOME_PATH) must be set to the Ascend toolkit root."
    )


def _resolve_pto_lib_path() -> str:
    """Return the pto-isa header root, with fallback priority order."""
    if "PTO_LIB_PATH" in os.environ:
        return os.environ["PTO_LIB_PATH"]
    # Third-party git submodule inside this repo
    submodule = os.path.join(_REPO_ROOT, "third_party", "pto-isa")
    if os.path.isdir(os.path.join(submodule, "include")):
        os.environ["PTO_LIB_PATH"] = submodule
        return submodule
    # Pre-installed path used inside the reference Docker image
    fallback = "/sources/pto-isa"
    if os.path.isdir(os.path.join(fallback, "include")):
        os.environ["PTO_LIB_PATH"] = fallback
        return fallback
    return ASCEND_TOOLKIT_HOME


PTO_LIB_PATH: str = _resolve_pto_lib_path()

# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------
_npu_dev = os.environ.get("GDN_NPU_DEVICE", "npu:0")
try:
    BLOCK_DIM: int = int(
        getattr(torch.npu.get_device_properties(_npu_dev), "cube_core_num", 20)
    )
except (RuntimeError, AssertionError):
    BLOCK_DIM = 24


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def _common_flags(
    *,
    num_heads: int,
    key_heads: int,
    hidden_size: int,
    chunk_size: int,
) -> list[str]:
    """Return bisheng flags shared by all chunk-GDN kernels."""
    flags = [
        "-fPIC", "-shared", "-xcce", "-DMEMORY_BASE", "-O2", "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined", "-Wno-ignored-attributes",
        f"-I{_KERNEL_INCLUDE}",
        f"-I{os.path.join(PTO_LIB_PATH, 'include')}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
        f"-DGDN_H={num_heads}",
        f"-DGDN_HG={key_heads}",
        f"-DGDN_D={hidden_size}",
        f"-DGDN_C={chunk_size}",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")
    extra = os.environ.get("PTO_DYNAMIC_EXTRA_FLAGS", "").split()
    flags.extend(extra)
    return flags


def _run_bisheng(cmd: list[str], timeout: int) -> None:
    if os.environ.get("VERBOSE_COMPILE"):
        print("compile:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout)


@lru_cache(maxsize=None)
def compile_chunk_kernel(
    cpp_basename: str,
    so_stem: str,
    *,
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    key_heads: int | None = None,
    cpp_mtime_ns: int = 0,
) -> str:
    """Compile a chunk-GDN kernel and return the path to the resulting ``.so``."""
    kh = key_heads if key_heads is not None else num_heads
    os.makedirs(_COMPILED_DIR, exist_ok=True)
    cpp_path = os.path.join(_KERNELS_PTO, cpp_basename)
    lib_path = os.path.join(
        _COMPILED_DIR,
        f"{so_stem}_H{num_heads}_Hg{kh}_D{hidden_size}_C{chunk_size}.so",
    )
    flags = _common_flags(
        num_heads=num_heads, key_heads=kh, hidden_size=hidden_size, chunk_size=chunk_size
    )
    _run_bisheng(["bisheng", *flags, cpp_path, "-o", lib_path], timeout=300)
    return lib_path


@lru_cache(maxsize=None)
def compile_mega_kernel(
    *,
    num_heads: int = 16,
    key_heads: int | None = None,
    hidden_size: int = 128,
    chunk_size: int = 128,
    cpp_mtime_ns: int = 0,
) -> str:
    """Compile the fused mega-kernel and return the path to the resulting ``.so``.

    The compiled ``.so`` exports BOTH entry points:
      - ``call_kernel``:      fp16 input path (existing, unchanged)
      - ``call_kernel_bf16``: bf16 input path (new; casts bf16→fp16 inside kernel)
    """
    kh = key_heads if key_heads is not None else num_heads
    os.makedirs(_COMPILED_DIR, exist_ok=True)
    cpp_path = os.path.join(_KERNELS_PTO, "mega_kernel.cpp")
    lib_path = os.path.join(
        _COMPILED_DIR,
        f"mega_kernel_H{num_heads}_Hg{kh}_D{hidden_size}_C{chunk_size}.so",
    )
    flags = _common_flags(
        num_heads=num_heads, key_heads=kh, hidden_size=hidden_size, chunk_size=chunk_size
    )
    print(f"[megagdn_pto] Compiling mega_kernel (H={num_heads} Hg={kh}) …")
    _run_bisheng(["bisheng", *flags, cpp_path, "-o", lib_path], timeout=600)
    print(f"[megagdn_pto] Compiled → {lib_path}")
    return lib_path


@lru_cache(maxsize=None)
def compile_tri_inverse(cpp_mtime_ns: int = 0) -> str:
    """Compile the triangular-inverse CubeCore kernel and return the ``.so`` path."""
    os.makedirs(_COMPILED_DIR, exist_ok=True)
    cpp_path = os.path.join(_KERNELS_PTO, "tri_inverse.cpp")
    lib_path = os.path.join(_COMPILED_DIR, "tri_inverse_jit.so")
    flags = [
        "-fPIC", "-shared", "-xcce", "-DMEMORY_BASE", "-O2", "-std=c++17",
        f"-I{_KERNEL_INCLUDE}",
        f"-I{os.path.join(PTO_LIB_PATH, 'include')}",
        "--cce-soc-version=Ascend910B4",
        "--cce-soc-core-type=CubeCore",
    ]
    _run_bisheng(["bisheng", *flags, cpp_path, "-o", lib_path], timeout=180)
    return lib_path
