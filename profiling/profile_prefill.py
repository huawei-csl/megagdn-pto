#!/usr/bin/env python3
"""Profile Ascend NPU prefill with torch_npu profiler.

Captures a Chrome-trace profiler run for one prefill batch under the
Triton, PTO staged, or PTO megakernel backend. Useful for understanding
the per-op breakdown and motivating the megakernel design.

The profiler trace is written under ``--profile-dir``. Open it in
``chrome://tracing`` or Perfetto UI to inspect the kernel timeline.

Usage::

    export ASCEND_RT_VISIBLE_DEVICES=4
    python profiling/profile_prefill.py --backend pto_mega \\
        --model qwen36_27b_w8a8 --seq-len 4096
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VLLM_PATCH = _REPO_ROOT / "vllm_patch"

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_MODEL_PRESETS = {
    "qwen35_0_8b":      "/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
    "qwen35_9b":        "/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "qwen36_27b_w8a8":  "/scratch/model_weights/Qwen3.6-27B-w8a8",
    "qwen36_35b_a3b_w8a8": "/scratch/model_weights/Qwen3.6-35B-A3B-w8a8",
}


def _configure_backend(backend: str, device: str) -> None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
    for k in list(os.environ):
        if k.startswith("VLLM_PTO"):
            del os.environ[k]
    if backend in ("pto", "pto_mega"):
        os.environ["VLLM_PTO_PATCH_DIR"] = str(_VLLM_PATCH)
    if backend == "pto_mega":
        os.environ["VLLM_PTO_MEGAKERNEL"] = "1"


def _apply_pto_in_process(backend: str) -> None:
    if backend not in ("pto", "pto_mega"):
        return
    patch_dir = str(_VLLM_PATCH)
    if patch_dir not in sys.path:
        sys.path.insert(0, patch_dir)
    from apply import apply_pto_patch  # type: ignore[import]
    apply_pto_patch()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("triton", "pto", "pto_mega"), default="pto_mega")
    ap.add_argument("--model", default="qwen36_27b_w8a8",
                    help="Model preset name or full path.")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--device", default=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0"))
    ap.add_argument("--profile-dir", type=Path,
                    default=Path(__file__).resolve().parent / "traces")
    ap.add_argument("--quantization", default=None)
    args = ap.parse_args()

    model_path = _MODEL_PRESETS.get(args.model, args.model)
    _configure_backend(args.backend, args.device)

    import vllm_ascend.utils as vua
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig

    vua.adapt_patch(is_global_patch=False)
    _apply_pto_in_process(args.backend)

    profile_dir = args.profile_dir / f"{args.backend}_B{args.batch_size}_T{args.seq_len}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profiler_cfg = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(profile_dir),
        torch_profiler_with_stack=True,
        delay_iterations=0,
        max_iterations=1,
        ignore_frontend=True,
    )

    llm_kwargs: dict = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max(args.seq_len + args.max_tokens + 32, 4096),
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=args.batch_size * args.seq_len + 256,
        max_num_seqs=args.batch_size + 2,
        profiler_config=profiler_cfg,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    llm = LLM(**llm_kwargs)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    seed = "The quick brown fox jumps over the lazy dog. "
    ids: list[int] = []
    while len(ids) < args.seq_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    prompt = tok.decode(ids[:args.seq_len])

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        min_tokens=args.max_tokens, ignore_eos=True)

    print(f"[profile] backend={args.backend}  seq_len={args.seq_len}  batch={args.batch_size}")
    print(f"[profile] trace dir: {profile_dir}")
    llm.generate([prompt] * args.batch_size, sp, use_tqdm=True)
    print("[profile] done. Open trace in chrome://tracing or Perfetto UI.")


if __name__ == "__main__":
    main()
