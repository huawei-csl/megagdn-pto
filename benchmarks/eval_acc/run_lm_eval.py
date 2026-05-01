#!/usr/bin/env python3
"""Run lm-eval accuracy benchmarks via vLLM-Ascend with PTO or Triton backend.

Tasks (default): six representative MMLU subjects + wikitext perplexity.
(GPQA Diamond is omitted by default as it requires a Hugging Face token.)

Wikitext is evaluated on ``--wikitext-limit`` documents (default: 256).
Set to 0 to disable the limit (full dataset, ~40 min per run).

Preset models (resolved to local weight paths):
  qwen35_0_8b, qwen35_9b, qwen36_27b_w8a8, qwen36_35b_a3b_w8a8

Output: JSON results file at ``--output-json`` (or stdout summary).

Usage::

    export ASCEND_RT_VISIBLE_DEVICES=0
    python benchmarks/eval_acc/run_lm_eval.py --preset qwen35_0_8b \\
        --backend pto_mega --output-json outputs/data/eval/qwen35_0_8b_pto.json

    # Full wikitext (slow)
    python benchmarks/eval_acc/run_lm_eval.py --preset qwen36_35b_a3b_w8a8 \\
        --backend triton --wikitext-limit 0 --output-json outputs/data/eval/35b_triton.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VLLM_PATCH = _REPO_ROOT / "vllm_patch"

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MMLU_SUBSET = (
    "mmlu_astronomy,"
    "mmlu_high_school_mathematics,"
    "mmlu_college_biology,"
    "mmlu_high_school_world_history,"
    "mmlu_professional_law,"
    "mmlu_philosophy"
)
DEFAULT_TASKS = f"{MMLU_SUBSET},wikitext"


@dataclass(frozen=True)
class ModelPreset:
    path: str
    quantization: str | None
    expert_parallel: bool


_PRESETS: dict[str, ModelPreset] = {
    "qwen35_0_8b": ModelPreset(
        "/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
        None, False,
    ),
    "qwen35_9b": ModelPreset(
        "/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        None, False,
    ),
    "qwen36_27b_w8a8": ModelPreset(
        "/scratch/model_weights/Qwen3.6-27B-w8a8",
        "ascend", False,
    ),
    "qwen36_35b_a3b_w8a8": ModelPreset(
        "/scratch/model_weights/Qwen3.6-35B-A3B-w8a8",
        "ascend", True,
    ),
}


def _autoconfig_patch(model_path: str):
    """Workaround for local Qwen3.6 checkpoints with unregistered model types."""
    import json as _json
    from pathlib import Path as _Path
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3_moe import Qwen3MoeConfig

    cfg_file = _Path(model_path) / "config.json"
    if not cfg_file.is_file():
        return lambda: None
    meta = _json.loads(cfg_file.read_text())
    mt = meta.get("model_type")
    if mt == "qwen3_5_moe":
        def _build():
            tc = dict(meta.get("text_config") or {}); tc["model_type"] = "qwen3_moe"
            return Qwen3MoeConfig(**tc)
    elif mt == "qwen3_5":
        def _build():
            tc = dict(meta.get("text_config") or {}); tc["model_type"] = "qwen3"
            return Qwen3Config(**tc)
    else:
        return lambda: None

    orig = AutoConfig.from_pretrained.__func__
    resolved = _Path(model_path).resolve()

    @classmethod
    def _patched(cls, name, *a, **kw):
        try:
            if _Path(name).resolve() == resolved:
                return _build()
        except TypeError:
            pass
        return orig(cls, name, *a, **kw)

    AutoConfig.from_pretrained = _patched
    return lambda: setattr(AutoConfig, "from_pretrained", classmethod(orig))


def _apply_pto_patch(backend: str) -> None:
    for k in list(os.environ):
        if k.startswith("VLLM_PTO"):
            del os.environ[k]
    if backend in ("pto", "pto_mega"):
        os.environ["VLLM_PTO_PATCH_DIR"] = str(_VLLM_PATCH)
        if backend == "pto_mega":
            os.environ["VLLM_PTO_MEGAKERNEL"] = "1"
    if backend in ("pto", "pto_mega"):
        patch_dir = str(_VLLM_PATCH)
        if patch_dir not in sys.path:
            sys.path.insert(0, patch_dir)
        from apply import apply_pto_patch  # type: ignore[import]
        apply_pto_patch()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=list(_PRESETS), required=True)
    ap.add_argument("--backend", choices=("triton", "pto", "pto_mega"), default="pto_mega")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--device", default=None, help="ASCEND_RT_VISIBLE_DEVICES (e.g. '0')")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--full-mmlu", action="store_true", help="Use full MMLU task group.")
    ap.add_argument("--wikitext-limit", type=int, default=256,
                    help="Max wikitext documents to evaluate (0 = full dataset).")
    args = ap.parse_args()

    if args.device:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.device

    preset = _PRESETS[args.preset]
    if preset.quantization:
        os.environ.setdefault("VLLM_QUANTIZATION", preset.quantization)

    import vllm_ascend.utils as vua
    vua.adapt_patch(is_global_patch=False)
    _apply_pto_patch(args.backend)
    _restore = _autoconfig_patch(preset.path)

    import lm_eval
    from lm_eval.utils import handle_non_serializable, make_table

    tasks = "mmlu,gpqa_diamond_zeroshot,wikitext" if args.full_mmlu else args.tasks

    # Sample limit applied to all tasks.  256 docs keeps wikitext fast (~2 min
    # vs ~40 min for the full dataset) and still covers most MMLU subjects
    # entirely (professional_law has ~1534 questions, but 256 is representative).
    limit: int | None = args.wikitext_limit if args.wikitext_limit > 0 else None

    lm_kwargs: dict = dict(
        pretrained=preset.path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    if preset.quantization:
        lm_kwargs["quantization"] = preset.quantization
    # Note: additional_config (expert_parallel_size) is omitted here because
    # lm_eval serializes model kwargs as a flat string and pydantic cannot
    # coerce a string back to a dict.  expert_parallel_size defaults to 1
    # for single-NPU inference, so this is safe to omit.

    t0 = time.time()
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=",".join(f"{k}={v}" for k, v in lm_kwargs.items()),
        tasks=tasks.split(","),
        batch_size=args.max_batch_size,
        limit=limit,
        log_samples=False,
    )
    elapsed = time.time() - t0

    _restore()

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        results_with_meta = dict(results)
        results_with_meta["_meta"] = {
            "preset": args.preset,
            "backend": args.backend,
            "elapsed_s": elapsed,
        }
        with out.open("w") as f:
            json.dump(results_with_meta, f, indent=2, default=handle_non_serializable)
        print(f"\nResults written to: {out}")

    print(make_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
