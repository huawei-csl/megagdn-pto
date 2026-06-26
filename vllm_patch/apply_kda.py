"""Activate the PTO ``chunk_kda`` override and fix KimiLinear page alignment.

Two patches applied here:

1. **chunk_kda patch** — replaces the Triton ``chunk_kda`` used by
   ``KimiDeltaAttention`` with the PTO megakernel for prefill, falling back
   to Triton for decode.

2. **Page alignment bypass** — vllm-ascend v0.18.0rc1 requires the KDA
   recurrent-state page size (mamba_page_size_padded) to be a multiple of
   the MLA KV-cache page size.  For KimiLinear-48B (SSM state ≈524 KiB,
   MLA page ≈576 KiB with block_size=512) the original formula produces a
   non-multiple.  We patch verify_and_update_config to round the padded SSM
   size up to the next multiple of the true MLA page size.

Call this **after** ``vllm_ascend.utils.adapt_patch()``.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

_log = logging.getLogger(__name__)
_PATCH_ACTIVE = False


def _ensure_pto_lib_path() -> None:
    if "PTO_LIB_PATH" in os.environ:
        return
    fallback = "/sources/pto-isa"
    if os.path.isdir(os.path.join(fallback, "include")):
        os.environ["PTO_LIB_PATH"] = fallback


def _patch_mla_rope_cache() -> None:
    """Record cos/sin caches for plain ``RotaryEmbedding`` (KimiLinear MLA).

    vllm-ascend's ``get_cos_and_sin_mla`` reads module-level ``_cos_cache`` /
    ``_sin_cache`` which are only populated by the Ascend rope subclasses'
    ``__init__``.  KimiLinear MLA layers use vLLM's plain ``RotaryEmbedding``
    (no rope scaling), so the caches stay ``None`` and the first prefill
    crashes.  Mirror ``AscendRotaryEmbedding.__init__`` recording here.
    """
    try:
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        from vllm_ascend.ops import rotary_embedding as _ascend_rope
    except ImportError as e:
        _log.warning("MLA rope-cache patch skipped (import error): %s", e)
        return
    if getattr(RotaryEmbedding, "_pto_rope_record", False):
        return
    orig_init = RotaryEmbedding.__init__
    rope_holder: dict = {}

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        rope_holder.setdefault("rope", self)
        cache = getattr(self, "cos_sin_cache", None)
        if cache is not None:
            _ascend_rope._record_cos_sin_cache(cache)
            _ascend_rope._record_cos_and_sin_cache_interleaved(cache)

    RotaryEmbedding.__init__ = patched_init
    RotaryEmbedding._pto_rope_record = True

    # The caches may still be unset at init time (init_cache=False).  Wrap
    # get_cos_and_sin_mla to populate them lazily from the recorded rope.
    orig_get = _ascend_rope.get_cos_and_sin_mla

    def lazy_get_cos_and_sin_mla(positions, use_cache=False):
        if _ascend_rope._cos_cache is None or _ascend_rope._sin_cache is None:
            rope = rope_holder.get("rope")
            if rope is not None:
                cache = getattr(rope, "cos_sin_cache", None)
                if cache is None:
                    cache = rope._compute_cos_sin_cache()
                cache = cache.to(device=positions.device)
                _ascend_rope._record_cos_sin_cache(cache)
                _ascend_rope._record_cos_and_sin_cache_interleaved(cache)
                _log.warning(
                    "PTO rope patch: lazily recorded cos/sin cache shape=%s dtype=%s",
                    tuple(cache.shape), cache.dtype,
                )
            else:
                # KimiLinear MLA layers are NoPE (rotary_emb=None): identity
                # rotation (cos=1, sin=0) leaves q/k unchanged.
                import torch
                rope_dim = 64  # qk_rope_head_dim for KimiLinear-48B
                max_pos = 32768
                _ascend_rope._cos_cache = torch.ones(
                    max_pos, rope_dim, dtype=torch.bfloat16, device=positions.device)
                _ascend_rope._sin_cache = torch.zeros(
                    max_pos, rope_dim, dtype=torch.bfloat16, device=positions.device)
                _log.warning("PTO rope patch: NoPE identity cos/sin caches installed.")
        return orig_get(positions, use_cache)

    _ascend_rope.get_cos_and_sin_mla = lazy_get_cos_and_sin_mla
    # mla_v1 binds the symbol at import time; patch it there too if present.
    try:
        import importlib
        _mla_mod = importlib.import_module("vllm_ascend.attention.mla_v1")
    except Exception:
        _mla_mod = sys.modules.get("vllm_ascend.attention.mla_v1")
    if _mla_mod is not None and hasattr(_mla_mod, "get_cos_and_sin_mla"):
        _mla_mod.get_cos_and_sin_mla = lazy_get_cos_and_sin_mla
    _log.warning("KimiLinear MLA rope-cache patch applied.")


def _patch_page_alignment() -> None:
    """Replace the strict page-size assertion with a warning + padding."""
    try:
        # patch_mamba_config replaces HybridAttentionMambaModelConfig.verify_and_update_config
        # on import; import it first so our override below wins.
        import vllm_ascend.patch.platform.patch_mamba_config  # noqa: F401
        import vllm.model_executor.models.config as _mc
        from vllm.model_executor.models import ModelRegistry
        from vllm.model_executor.models.config import MambaModelConfig
        from vllm.utils.math_utils import cdiv
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
    except ImportError as e:
        _log.warning("Page-alignment patch skipped (import error): %s", e)
        return

    @classmethod  # type: ignore[misc]
    def _patched_verify(cls, vllm_config) -> None:
        MambaModelConfig.verify_and_update_config(vllm_config)

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        kernel_block_size = 128
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        attn_head_size = model_config.get_head_size()
        attn_single_token = (
            attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        )

        model_cls, _ = ModelRegistry.resolve_model_cls(
            model_config.architecture, model_config=model_config
        )
        mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
        mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
        mamba_sizes = [
            math.prod(s) * get_dtype_size(d)
            for s, d in zip(mamba_shapes, mamba_dtypes)
        ]
        ssm_page = max(mamba_sizes)
        conv_page = min(mamba_sizes)

        attn_block_size = kernel_block_size * cdiv(
            ssm_page, kernel_block_size * attn_single_token
        )

        if attn_single_token * attn_block_size != ssm_page:
            aligned = attn_single_token * attn_block_size
            _log.warning(
                "KimiLinear page-size mismatch: ssm_page=%d, "
                "attn_single_token=%d — not exactly divisible. "
                "Using attn_block_size=%d (%d B attn page, +%d B padding). "
                "This is a known vllm-ascend v0.18 limitation.",
                ssm_page, attn_single_token, attn_block_size,
                aligned, aligned - ssm_page,
            )

        if cache_config.block_size is None or cache_config.block_size < attn_block_size:
            cache_config.block_size = attn_block_size
            _log.info(
                "Setting attention block_size to %d for KimiLinear.", attn_block_size
            )

        # MLA uses a single latent KV tensor: page_size = block_size * num_kv_heads * head_size * dtype
        # (no ×2 K+V factor unlike FullAttentionSpec).
        attn_page_size = (
            cache_config.block_size
            * attn_num_kv_heads
            * attn_head_size
            * get_dtype_size(kv_cache_dtype)
        )

        # mamba_page_size_padded must be a multiple of attn_page_size so that
        # unify_kv_cache_spec_page_size can align the two spec types.
        total_state = ssm_page + conv_page
        padded = math.ceil(total_state / attn_page_size) * attn_page_size

        if (
            cache_config.mamba_page_size_padded is None
            or cache_config.mamba_page_size_padded != padded
        ):
            cache_config.mamba_page_size_padded = padded

        if (
            cache_config.enable_prefix_caching
            and cache_config.mamba_cache_mode == "align"
        ):
            cache_config.mamba_block_size = cache_config.block_size
        else:
            cache_config.mamba_block_size = model_config.max_model_len

    try:
        _mc.HybridAttentionMambaModelConfig.verify_and_update_config = _patched_verify
        _log.warning("KimiLinear page-alignment patch applied.")
    except AttributeError as e:
        _log.warning("Page-alignment patch skipped (attr error): %s", e)


def apply_kda_patch() -> None:
    """Replace ``chunk_kda`` with the PTO megakernel and fix page alignment.

    Idempotent: safe to call multiple times (e.g. from both the main process
    and the spawned worker hook).
    """
    global _PATCH_ACTIVE
    if _PATCH_ACTIVE:
        return
    _ensure_pto_lib_path()

    # Ensure vllm_patch dir is on sys.path for local imports
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    # 1. Bypass the strict page-size assertion for KimiLinear.
    _patch_page_alignment()

    # 1b. Populate Ascend cos/sin caches for the plain RotaryEmbedding used
    # by KimiLinear MLA layers.
    _patch_mla_rope_cache()

    # 2. Patch chunk_kda with the PTO megakernel wrapper.
    try:
        import vllm.model_executor.layers.fla.ops.kda as _kda_mod
    except ImportError as e:
        raise ImportError(
            "vllm.model_executor.layers.fla.ops.kda not found — "
            "is vllm-ascend installed and does it support KimiLinear?"
        ) from e

    triton_impl = _kda_mod.chunk_kda

    from chunk_kda_pto import bind_triton  # type: ignore[import]

    wrapped = bind_triton(triton_impl)
    _kda_mod.chunk_kda = wrapped

    # Also patch the importing module (vllm.model_executor.layers.kda) which
    # imports chunk_kda at the top level.
    _kda_layer_mod = sys.modules.get("vllm.model_executor.layers.kda")
    if _kda_layer_mod is not None and hasattr(_kda_layer_mod, "chunk_kda"):
        _kda_layer_mod.chunk_kda = wrapped

    _PATCH_ACTIVE = True
    _log.warning("KDA PTO patch active: fused megakernel (C=%d).", 128)


def is_kda_patch_active() -> bool:
    return _PATCH_ACTIVE
