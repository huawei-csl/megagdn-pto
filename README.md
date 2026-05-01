# Fast PTO Kernels for Chunk GatedDeltaNet

Custom NPU kernels (compiled via [Bisheng](https://gitee.com/ascend/bisheng) JIT)
for the chunk GatedDeltaNet (GDN) recurrent layer, replacing the Triton baseline
used in [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend).

**Key results** (Ascend NPU, N=16 seqs × L=16384 tokens, D=128, Hg=16):

| Stage | H=16 | H=32 | H=48 | H=64 | Baseline |
|-------|------|------|------|------|----------|
| chunk_h  | 1.53× | 1.58× | 1.49× | 1.46× | Triton BT=128 |
| wy_fast  | 1.71× | 1.79× | 1.69× | 1.77× | Triton BT=128 |
| chunk_o  | 1.44× | 1.43× | 1.38× | — ¹  | Triton BT=64  |
| **Megakernel** (6 stages fused) | 4.2× | 2.6× | 2.1× | 1.8× | staged PTO |

¹ Triton `chunk_o` with H=64, BT=64 fails (known aicore incompatibility); PTO works fine.

vLLM prefill TTFT speedup (megakernel vs Triton, all 4 models): **1.07–1.25×** depending on model and prompt length.

Accuracy: PTO outputs match CPU fp64 reference within 5e-3 RMSE and R²>0.999.

---

## Repository layout

```
megagdn-pto/
├── kernels/
│   ├── pto/                    # C++ PTO kernel sources (Bisheng JIT)
│   │   ├── include/            # Shared utility header (kernel_utils.h)
│   │   ├── compiled_lib/       # .so files cached here at runtime
│   │   ├── scaled_dot_kkt.cpp
│   │   ├── wy_fast.cpp
│   │   ├── chunk_h.cpp
│   │   ├── chunk_o.cpp
│   │   ├── chunk_cumsum.cpp
│   │   ├── tri_inverse.cpp     # Entry point; includes tri_inverse_impl.cpp
│   │   ├── tri_inverse_impl.cpp
│   │   └── mega_kernel.cpp     # Fused megakernel (all 6 stages)
│   └── triton_baseline/
│       └── fla_vendor/         # FLA Triton kernels (reference/baseline)
├── megagdn_pto/                # Python package (pip install -e .)
│   ├── compile.py              # Bisheng JIT compilation helpers
│   ├── kernel_libs.py          # Individual kernel wrappers
│   ├── fast_inverse.py         # Triangular inverse (solve_tril)
│   └── mega_kernel.py          # Fused megakernel wrapper
├── tests/
│   ├── test_single_kernels.py  # Unit tests: each stage vs CPU reference
│   └── test_e2e.py             # E2E pipeline test: PTO vs CPU fp64 reference
├── benchmarks/
│   ├── kernel/
│   │   └── bench_gdn_kernels.py    # Per-stage + megakernel latency benchmark
│   ├── vllm_prefill/
│   │   ├── benchmark_prefill.py    # TTFT measurement (single model/backend)
│   │   └── run_prefill_sweep.sh    # Orchestration across all models & backends
│   └── eval_acc/
│       ├── run_lm_eval.py          # lm-eval accuracy benchmark
│       └── run_eval_suite.sh       # Full sweep across 4 models
├── vllm_patch/
│   ├── install_hook.py         # One-time in-source patch to vllm-ascend
│   ├── apply.py                # Runtime activation of PTO patch
│   └── chunk_gated_delta_rule.py   # PTO drop-in for chunk_gated_delta_rule
├── profiling/
│   └── profile_prefill.py      # torch_npu profiler for prefill traces
├── scripts/
│   ├── download_weights.md     # Commands to download model weights
│   └── plot_results.py         # Generate figures from benchmark outputs
├── outputs/
│   ├── data/                   # Raw benchmark results (JSON / JSONL)
│   └── figure/                 # PDF plots
└── third_party/
    └── pto-isa/                # Git submodule: Ascend PTO ISA headers
```

---

## Environment

Use the pre-built Docker image that ships vLLM-Ascend and Triton:

```bash
docker pull quay.io/ascend/vllm-ascend:v0.18.0rc1
```

The C++ kernels depend on **pto-isa** header-only library, included here as a git
submodule:

```bash
git submodule update --init --recursive
```

Install this package in editable mode so all scripts can import `megagdn_pto`:

```bash
pip install -e .
```

Check available NPU devices before running benchmarks:

```bash
npu-smi info
```

---

## Model weights

Four models are evaluated. Download instructions are in
[`scripts/download_weights.md`](scripts/download_weights.md).

Expected weight paths (already present on this server):

| Model | Path |
|-------|------|
| Qwen3.5-0.8B | `/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/...` |
| Qwen3.5-9B   | `/scratch/model_weights/models--Qwen--Qwen3.5-9B/...` |
| Qwen3.6-27B-w8a8 | `/scratch/model_weights/Qwen3.6-27B-w8a8` |
| Qwen3.6-35B-A3B-w8a8 | `/scratch/model_weights/Qwen3.6-35B-A3B-w8a8` |

---

## Quick-start

### 1 – Unit tests (single kernels + E2E)

```bash
# Single-kernel accuracy (all 4 head counts, ~3 min)
GDN_NPU_DEVICE=npu:0 python tests/test_single_kernels.py --device npu:0 --H-list 16,32,48,64

# End-to-end pipeline (PTO vs CPU fp64 reference)
GDN_NPU_DEVICE=npu:0 python tests/test_e2e.py --device npu:0
```

Expected output: `ALL PASS` for every test case.

### 2 – Kernel benchmarks

```bash
GDN_NPU_DEVICE=npu:0 python benchmarks/kernel/bench_gdn_kernels.py \
    --device npu:0 --H-list 16,32,48,64 --mega \
    --output-json outputs/data/kernel_bench.json
```

> **Triton notes:**
> - `chunk_o` with H=64, BT=64 is a known Triton failure (aicore exception); the
>   script skips it and marks it "fail".
> - PTO always uses `chunk_size=128`; Triton defaults to `BT=64`.

### 3 – vLLM-Ascend patch (one-time)

Apply in-source hooks to the installed vllm-ascend package:

```bash
python vllm_patch/install_hook.py
```

This is idempotent; run again if vllm-ascend is re-installed.

### 4 – Prefill TTFT benchmark

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
# Smoke-test on the smallest model
python benchmarks/vllm_prefill/benchmark_prefill.py \
    --case pto_mega --model qwen35_0_8b \
    --seq-len 512 1024 4096 --warmup 1 --repeats 3

# Full sweep (all 4 models × 3 backends × 8 seq-lens, ~2 h)
bash benchmarks/vllm_prefill/run_prefill_sweep.sh
```

### 5 – Accuracy evaluation (lm-eval)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
# Smoke-test
python benchmarks/eval_acc/run_lm_eval.py \
    --model qwen35_0_8b --case pto_mega \
    --tasks wikitext --limit 64

# Full sweep (all 4 models, ~2 h)
bash benchmarks/eval_acc/run_eval_suite.sh
```

### 6 – Plot results

```bash
# After running run_prefill_sweep.sh
python scripts/plot_results.py --auto
```

Figures are saved under `outputs/figure/`.

---

## How the PTO patch works

`vllm_patch/install_hook.py` makes two in-source edits to the installed
`vllm_ascend` package (idempotent, safe to re-run):

1. Injects an early hook in `patch/worker/__init__.py` that calls
   `apply_pto_patch()` when `VLLM_PTO_PATCH_DIR` is set.
2. Patches `patch_qwen3_5.py` and `patch_qwen3_next.py` to look up
   `chunk_gated_delta_rule` dynamically via `fla_ops.*` rather than a cached
   static import, so the monkey-patch takes effect.

At runtime, setting `VLLM_PTO_PATCH_DIR=/path/to/vllm_patch` activates PTO.
Setting `VLLM_PTO_MEGAKERNEL=1` additionally enables the fused megakernel.

---

## Computation stages

The GDN layer decomposes into six sequential stages:

| # | Stage | Kernel | Description |
|---|-------|--------|-------------|
| 1 | `chunk_cumsum` | (inside mega_kernel) | Chunk-local cumulative sum of log-gates |
| 2 | `scaled_dot_kkt` | `scaled_dot_kkt.cpp` | K^T·K with gate weighting → A matrix |
| 3 | `solve_tril`  | `tri_inverse.cpp` | Triangular inverse: (A^T + I)^{-1} |
| 4 | `wy_fast`  | `wy_fast.cpp` | Compute W and U via Woodbury identity |
| 5 | `chunk_h`  | `chunk_h.cpp` | Recurrent state update H per chunk |
| 6 | `chunk_o`  | `chunk_o.cpp` | Output projection using H |

The megakernel (`mega_kernel.cpp`) fuses all six into a single NPU launch,
eliminating five kernel-launch round-trips and the intermediate tensor traffic
between stages.

---

## Citation / acknowledgements

PTO kernels use the
[pto-isa](https://gitcode.com/cann/pto-isa) low-level Ascend ISA header library.
The Triton baseline is adapted from the
[FLA](https://github.com/sustcsonglin/flash-linear-attention) library.
