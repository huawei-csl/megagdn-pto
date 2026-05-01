"""Activate the PTO ``chunk_gated_delta_rule`` override in the current process.

Call this **after** ``vllm_ascend.utils.adapt_patch()`` (which installs the
Triton backend) so that the PTO wrapper replaces Triton's implementation.

This file is loaded by the vllm-ascend worker hook (injected by
``install_hook.py``) when ``VLLM_PTO_PATCH_DIR`` points to this directory.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger(__name__)
_PATCH_ACTIVE = False


def _ensure_pto_lib_path() -> None:
    """Set PTO_LIB_PATH if not already configured."""
    if "PTO_LIB_PATH" in os.environ:
        return
    # Fallback to the pre-installed path used in the reference Docker image
    fallback = "/sources/pto-isa"
    if os.path.isdir(os.path.join(fallback, "include")):
        os.environ["PTO_LIB_PATH"] = fallback


def apply_pto_patch() -> None:
    """Replace ``vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule`` with PTO.

    Must be called after the Triton implementation is installed by
    ``vllm_ascend.utils.adapt_patch()``.
    """
    global _PATCH_ACTIVE
    _ensure_pto_lib_path()

    import vllm.model_executor.layers.fla.ops as fla_ops
    import vllm_ascend.ops.triton.fla.chunk as _ascend_chunk_mod

    triton_impl = _ascend_chunk_mod.chunk_gated_delta_rule

    # Ensure this directory is on sys.path so ``chunk_gated_delta_rule`` imports work
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    from chunk_gated_delta_rule import bind_triton  # type: ignore[import]

    wrapped = bind_triton(triton_impl)
    fla_ops.chunk_gated_delta_rule = wrapped
    # vLLM-Ascend 0.19+ prefill uses ``vllm_ascend.ops.gdn`` which imports from this module
    # before ``vllm.model_executor.layers.fla.ops``; patch the defining module so new imports
    # and v0.18-style ``fla.ops`` routing both see PTO.
    _ascend_chunk_mod.chunk_gated_delta_rule = wrapped
    _gdn_mod = sys.modules.get("vllm_ascend.ops.gdn")
    if _gdn_mod is not None and hasattr(_gdn_mod, "chunk_gated_delta_rule"):
        _gdn_mod.chunk_gated_delta_rule = wrapped
    _PATCH_ACTIVE = True

    megakernel = os.environ.get("VLLM_PTO_MEGAKERNEL", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    _log.warning(
        "PTO patch active: %s (C=128).",
        "fused megakernel" if megakernel else "6-stage JIT kernels",
    )


def is_pto_patch_active() -> bool:
    return _PATCH_ACTIVE
