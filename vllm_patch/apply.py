"""Activate the PTO ``chunk_gated_delta_rule`` override in the current process.

Call this **after** ``vllm_ascend.utils.adapt_patch()`` (which installs the
Triton backend) so that the PTO wrapper replaces Triton's implementation.

This file is loaded by the vllm-ascend worker hook (injected by
``install_hook.py``) when ``VLLM_PTO_PATCH_DIR`` points to this directory.

Verified against vllm-ascend v0.18 / v0.19 / v0.23. The defining module
``vllm_ascend.ops.triton.fla.chunk`` is patched in all of them, plus every
module known to re-export the symbol via ``from ... import`` (those bind at
import time, so a late import order would otherwise keep the Triton reference).
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


# Modules that do ``from ... import chunk_gated_delta_rule`` at import time. Any
# of these already in ``sys.modules`` when the patch runs holds a stale Triton
# reference and must be rebound; the rest pick up the patched defining modules
# when they are imported later.
_REBIND_TARGETS = (
    # vllm-ascend 0.19+ GDN prefill path
    ("vllm_ascend.ops.gdn", "chunk_gated_delta_rule"),
    # vLLM 0.23 base layer, imported as ``fla_chunk_gated_delta_rule``
    ("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn", "fla_chunk_gated_delta_rule"),
)


def _rebind_already_imported(wrapped) -> None:
    for mod_name, attr in _REBIND_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, attr):
            setattr(mod, attr, wrapped)
            _log.debug("PTO rebound %s.%s", mod_name, attr)


def apply_pto_patch() -> None:
    """Replace ``vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule`` with PTO.

    Must be called after the Triton implementation is installed by
    ``vllm_ascend.utils.adapt_patch()``.
    """
    global _PATCH_ACTIVE
    _ensure_pto_lib_path()

    # Ensure this directory is on sys.path so ``chunk_gated_delta_rule`` imports work
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    try:
        import vllm.model_executor.layers.fla.ops as fla_ops
        import vllm_ascend.ops.triton.fla.chunk as _ascend_chunk_mod

        triton_impl = _ascend_chunk_mod.chunk_gated_delta_rule

        from chunk_gated_delta_rule import bind_triton  # type: ignore[import]

        wrapped = bind_triton(triton_impl)
        fla_ops.chunk_gated_delta_rule = wrapped
        # vLLM-Ascend 0.19+ prefill uses ``vllm_ascend.ops.gdn`` which imports from this module
        # before ``vllm.model_executor.layers.fla.ops``; patch the defining module so new imports
        # and v0.18-style ``fla.ops`` routing both see PTO.
        _ascend_chunk_mod.chunk_gated_delta_rule = wrapped
        _rebind_already_imported(wrapped)
        _PATCH_ACTIVE = True

        megakernel = os.environ.get("VLLM_PTO_MEGAKERNEL", "").strip().lower() in (
            "1", "true", "yes", "on"
        )
        _log.warning(
            "PTO patch active: %s (C=128).",
            "fused megakernel" if megakernel else "6-stage JIT kernels",
        )
    except Exception as _gdn_exc:
        _log.warning("GDN PTO patch skipped: %s", _gdn_exc)

    if os.environ.get("VLLM_PTO_KDA_MEGAKERNEL", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from apply_kda import apply_kda_patch  # type: ignore[import]
            apply_kda_patch()
        except Exception as _kda_exc:
            _log.warning("KDA patch skipped: %s", _kda_exc)


def is_pto_patch_active() -> bool:
    return _PATCH_ACTIVE
