#!/usr/bin/env python3
"""Measure time-to-first-token (TTFT) prefill latency on Ascend NPU.

Compares three backends for chunk GatedDeltaNet (GDN):
  - ``triton``   : default vLLM-Ascend Triton kernels (chunk_size=64, bf16)
  - ``pto``      : PTO staged kernels (chunk_size=128, fp16, 6 launches)
  - ``pto_mega`` : PTO fused mega-kernel (chunk_size=128, fp16, 1 launch)

Sweeps multiple sequence lengths in a single vLLM session (one model load per
backend). Writes one JSONL line per (backend, seq_len) to ``--output-jsonl``.

Usage::

    export ASCEND_RT_VISIBLE_DEVICES=0
    python benchmarks/vllm_prefill/benchmark_prefill.py --case pto_mega \\
        --model /scratch/model_weights/Qwen3.6-35B-A3B-w8a8 \\
        --quantization ascend --seq-len 512 1024 4096 8192
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VLLM_PATCH = _REPO_ROOT / "vllm_patch"

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

_MODEL_PRESETS = {
    "qwen35_0_8b":      "/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/",
    "qwen35_9b":        "/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "qwen36_27b_w8a8":  "/scratch/model_weights/Qwen3.6-27B-w8a8",
    "qwen36_35b_a3b_w8a8": "/scratch/model_weights/Qwen3.6-35B-A3B-w8a8",
}


def _infer_label(model_path: str) -> str:
    m = re.search(r"Qwen3\.5-(\d+(?:\.\d+)?)B", model_path, re.IGNORECASE)
    if m:
        return f"Qwen3.5-{m.group(1)}B"
    m = re.search(r"Qwen3\.6-(\d+(?:\.\d+)?)B-?(\w+)?", model_path, re.IGNORECASE)
    if m:
        suffix = f"-{m.group(2)}" if m.group(2) else ""
        return f"Qwen3.6-{m.group(1)}B{suffix}"
    m = re.search(r"-(\d+(?:\.\d+)?)B(?:/|$)", model_path)
    return f"{m.group(1)}B" if m else "unknown"


# ---------------------------------------------------------------------------
# Backend env setup
# ---------------------------------------------------------------------------

def _configure_backend(case: str, device: str) -> None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device
    for k in list(os.environ):
        if k.startswith("VLLM_PTO"):
            del os.environ[k]
    if case == "pto":
        os.environ["VLLM_PTO_PATCH_DIR"] = str(_VLLM_PATCH)
    elif case == "pto_mega":
        os.environ["VLLM_PTO_PATCH_DIR"] = str(_VLLM_PATCH)
        os.environ["VLLM_PTO_MEGAKERNEL"] = "1"
    elif case != "triton":
        raise ValueError(f"case must be triton / pto / pto_mega, got {case!r}")


def _apply_pto_patch_in_process() -> None:
    patch_dir = os.environ.get("VLLM_PTO_PATCH_DIR")
    if not patch_dir or not os.path.isdir(patch_dir):
        return
    if patch_dir not in sys.path:
        sys.path.insert(0, patch_dir)
    from apply import apply_pto_patch  # type: ignore[import]
    apply_pto_patch()


# ---------------------------------------------------------------------------
# TTFT measurement
# ---------------------------------------------------------------------------

def _last_logged_tps(llm) -> float | None:
    mgr = getattr(llm.llm_engine, "logger_manager", None)
    if mgr is None:
        return None
    for sl in mgr.stat_loggers:
        per_engine = getattr(sl, "per_engine_stat_loggers", None)
        if isinstance(per_engine, dict) and 0 in per_engine:
            inner = per_engine[0]
            if hasattr(inner, "last_prompt_throughput"):
                return float(inner.last_prompt_throughput)
        if hasattr(sl, "last_prompt_throughput") and not isinstance(
            getattr(sl, "per_engine_stat_loggers", None), dict
        ):
            return float(sl.last_prompt_throughput)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("triton", "pto", "pto_mega"), required=True)
    p.add_argument("--model", default=None, help="Model path or preset name.")
    p.add_argument("--model-label", default=None)
    p.add_argument("--seq-len", type=int, nargs="+", default=[4096])
    p.add_argument("--device", default="0")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--quantization", default=None)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--output-jsonl", default=None)
    args = p.parse_args()

    model_path = _MODEL_PRESETS.get(args.model, args.model)
    if model_path is None:
        p.error("--model is required (path or preset name).")
    model_label = (args.model_label or "").strip() or _infer_label(model_path)

    _configure_backend(args.case, args.device)

    import vllm_ascend.utils as vua
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

    vua.adapt_patch(is_global_patch=False)
    if args.case in ("pto", "pto_mega"):
        _apply_pto_patch_in_process()

    max_sl = max(args.seq_len)
    max_model_len = max(max_sl + args.max_tokens + 32, 4096)
    prefill_tokens = max_sl + args.max_tokens + 256

    llm_kwargs: dict = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=prefill_tokens,
        max_num_seqs=8,
        disable_log_stats=False,
        enable_prefix_caching=False,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    seed = "The quick brown fox jumps over the lazy dog. "

    def _prompt(seq_len: int) -> list[str]:
        ids: list[int] = []
        while len(ids) < seq_len:
            ids.extend(tok.encode(seed, add_special_tokens=False))
        return [tok.decode(ids[:seq_len])]

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        min_tokens=args.max_tokens, ignore_eos=True, seed=42)

    def _ttft(output: RequestOutput) -> float:
        m = output.metrics
        if m is None:
            raise RuntimeError("metrics is None; disable_log_stats must be False.")
        ttft = getattr(m, "first_token_latency", None)
        if ttft is None:
            raise RuntimeError("first_token_latency missing in metrics.")
        return float(ttft)

    out_path = Path(args.output_jsonl) if args.output_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for seq_len in args.seq_len:
        prompts = _prompt(seq_len)
        for _ in range(args.warmup):
            llm.generate(prompts, sp, use_tqdm=False)
        ttfts_s = [_ttft(llm.generate(prompts, sp, use_tqdm=False)[0]) for _ in range(args.repeats)]
        ttfts_ms = [t * 1000 for t in ttfts_s]
        mean_ms = statistics.mean(ttfts_ms)
        median_ms = statistics.median(ttfts_ms)
        std_ms = statistics.pstdev(ttfts_ms) if len(ttfts_ms) > 1 else 0.0
        input_tps = seq_len / (mean_ms / 1000) if mean_ms > 0 else 0.0

        row = {
            "case": args.case,
            "seq_len": seq_len,
            "model_label": model_label,
            "median_ttft_ms": median_ms,
            "mean_ttft_ms": mean_ms,
            "std_ttft_ms": std_ms,
            "input_tps": input_tps,
            "vllm_interval_prompt_throughput_tps": _last_logged_tps(llm),
            "warmup": args.warmup,
            "repeats": args.repeats,
        }
        line = json.dumps(row)
        print(line, flush=True)
        if out_path:
            with out_path.open("a") as f:
                f.write(line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
