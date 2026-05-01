# Fast PTO Kernels for Chunk GatedDeltaNet

Custom NPU kernels (compiled via [Bisheng](https://gitee.com/ascend/bisheng) JIT)
for the chunk GatedDeltaNet (GDN) recurrent layer, replacing the Triton baseline
used in [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend).

**Key results** (Ascend NPU, N=16 seqs × L=16384 tokens, D=128, Hg=16):

PTO always uses C=128. Speedup format: `BT=64 (BT=128)` — n/a means Triton fails or is not supported.
Raw data: [`outputs/data/kernel_bench.json`](outputs/data/kernel_bench.json)

| Stage | H=16 | H=32 | H=48 | H=64 |
|-------|------|------|------|------|
| chunk_cumsum   | 3.1× (3.4×) | 3.1× (n/a ²) | 2.4× (n/a ²) | 5.2× (n/a ²) |
| scaled_dot_kkt | 0.89× (n/a ³) | 0.81× (n/a ³) | 0.82× (n/a ³) | 0.80× (n/a ³) |
| solve_tril     | n/a ¹ | n/a ¹ | n/a ¹ | n/a ¹ |
| wy_fast        | 3.4× (1.7×) | 3.5× (1.8×) | 3.3× (1.7×) | 3.5× (1.8×) |
| chunk_h        | 3.2× (1.6×) | 3.0× (1.5×) | 3.0× (1.5×) | n/a ⁴ (1.5×) |
| chunk_o        | 1.4× (n/a ³) | 1.4× (n/a ³) | 1.4× (n/a ³) | n/a ⁴ |
| **Megakernel** (6 stages fused) | **4.5×** | **2.8×** | **2.2×** | **1.9×** |

Megakernel speedup is vs staged PTO (individual kernel calls with synchronization).

¹ Triton `solve_tril` requires Triton grid ≤ 65536; fails at T=262144 tokens. PTO has no such limit.  
² Triton `chunk_cumsum` BT=128 compilation fails for H ≥ 32.  
³ Triton BT=128 compilation fails for `scaled_dot_kkt` and `chunk_o` on this configuration.  
⁴ Triton BT=64 triggers an aicore exception for H=64 (known NPU incompatibility); PTO works fine.

vLLM prefill TTFT speedup (megakernel vs Triton, all 4 models): **1.07–1.25×** depending on model and prompt length.

Accuracy (lm-eval, wikitext 256-doc subset, MMLU 6-subject subset):

| Model | Metric | PTO megakernel | Triton |
|-------|--------|---------------|--------|
| Qwen3.5-0.8B | WikiText PPL ↓ | 19.89 | 19.87 |
| Qwen3.5-9B   | WikiText PPL ↓ |  9.26 |  9.26 |
| Qwen3.6-27B-w8a8 | WikiText PPL ↓ | 8.20 | 8.21 |
| Qwen3.6-35B-A3B-w8a8 | WikiText PPL ↓ | 8.17 | 8.17 |
| Qwen3.5-0.8B | MMLU acc ↑ | 49.4% | 49.4% |
| Qwen3.5-9B   | MMLU acc ↑ | 79.7% | 79.5% |
| Qwen3.6-27B-w8a8 | MMLU acc ↑ | 82.8% | 82.8% |
| Qwen3.6-35B-A3B-w8a8 | MMLU acc ↑ | 83.6% | 84.2% |

PTO and Triton outputs are numerically equivalent; differences are within lm-eval sampling noise.
Unit test: PTO outputs match CPU fp64 reference within 5e-3 RMSE and R²>0.999.

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
pip install -e .                  # core (torch, numpy)
pip install -e '.[eval]'          # add lm-eval (required for §5 accuracy benchmarks)
pip install -e '.[plot]'          # add matplotlib (required for §6 plotting)
pip install -e '.[eval,plot]'     # both at once
```

> **Dependency note:** `lm-eval` pulls in a broad set of NLP packages that may
> conflict with the pinned versions inside the `vllm-ascend` Docker image
> (e.g. `fsspec`, `opencv-python-headless`). These conflicts are harmless for
> the evaluation itself, but if `pip` reports errors, install `lm-eval` with
> `--no-deps` and add missing packages manually, or run evaluations in a
> separate virtual environment.

Check available NPU devices before running benchmarks:

```bash
npu-smi info
```

---

## Model weights

Four models are evaluated. See [`scripts/download_weights.md`](scripts/download_weights.md)
for download commands.

The benchmark scripts expect weights at these paths (edit `run_lm_eval.py` and
`benchmark_prefill.py` to point to your local copies):

| Model | Default path |
|-------|------|
| Qwen3.5-0.8B | `/scratch/model_weights/models--Qwen--Qwen3.5-0.8B/snapshots/<hash>/` |
| Qwen3.5-9B   | `/scratch/model_weights/models--Qwen--Qwen3.5-9B/snapshots/<hash>/` |
| Qwen3.6-27B-w8a8 | `/scratch/model_weights/Qwen3.6-27B-w8a8` |
| Qwen3.6-35B-A3B-w8a8 | `/scratch/model_weights/Qwen3.6-35B-A3B-w8a8` |

---

## Quick-start

### 1 – Unit tests (single kernels + E2E)

```bash
# Single-kernel accuracy — all 6 stages, all 4 head counts (~5 min)
GDN_NPU_DEVICE=npu:0 python tests/test_single_kernels.py --device npu:0 --H-list 16,32,48,64

# End-to-end pipeline — PTO vs CPU fp64 reference, optional Triton cross-check
GDN_NPU_DEVICE=npu:0 python tests/test_e2e.py --device npu:0
GDN_NPU_DEVICE=npu:0 python tests/test_e2e.py --device npu:0 --no-triton  # skip cross-check
```

Expected output: `ALL PASS` for every test case.

> **Triton cross-check in `test_e2e.py`:** when Triton is available, each shape
> is also run through the Triton pipeline and compared with PTO. If Triton
> raises an exception for a particular shape (e.g. grid-size limits at long
> sequences), that shape's cross-check is skipped with a `[Triton skipped: …]`
> message and the test still passes — PTO correctness is determined solely by
> the CPU fp64 reference.

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

Requires `lm-eval` — install with `pip install -e '.[eval]'` first.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

# Single model, wikitext 256-doc subset (default, ~10 min)
python benchmarks/eval_acc/run_lm_eval.py \
    --preset qwen35_0_8b --backend pto_mega \
    --output-json outputs/data/eval/qwen35_0_8b_pto.json

# Single model, full wikitext (~40 min)
python benchmarks/eval_acc/run_lm_eval.py \
    --preset qwen35_0_8b --backend pto_mega --wikitext-limit 0 \
    --output-json outputs/data/eval/qwen35_0_8b_pto_full.json

# Full sweep: all 4 models × 2 backends, wikitext 256-doc subset (~3 h total)
bash benchmarks/eval_acc/run_eval_suite.sh

# Full wikitext sweep
WIKITEXT_LIMIT=0 bash benchmarks/eval_acc/run_eval_suite.sh
```

### 6 – Plot results

```bash
# Auto-detect latest prefill + eval results
python scripts/plot_results.py --auto

# Or specify explicitly
python scripts/plot_results.py \
    --prefill-dir outputs/data/prefill_<stamp> \
    --eval-dir    outputs/data/eval_<stamp>
```

Figures are saved under `outputs/figure/`.

---

## Known warnings and log noise

The following messages are harmless; they appear on every run and can be ignored.

**Bisheng JIT — `MEMORY_BASE` macro redefined**

```
warning: 'MEMORY_BASE' macro redefined [-Wmacro-redefined]
```

Emitted by the `bisheng` C++ compiler when building `tri_inverse.cpp`. The
macro is redefined by a compiler command-line flag; the duplicate definition is
intentional in the kernel build system and does not affect correctness.

**vLLM — unknown environment variables**

```
Unknown key in environment variable: VLLM_PTO_PATCH_DIR
Unknown key in environment variable: VLLM_PTO_MEGAKERNEL
```

vLLM-Ascend logs unrecognised `VLLM_*` keys at startup. The variables
are read by the injected hook before vLLM's config validation runs, so the
warnings are safe to ignore.

**vLLM — "Using Triton/FLA GDN prefill kernel"**

```
Using Triton/FLA GDN prefill kernel
```

This line comes from the patched `patch_qwen3_5.py`/`patch_qwen3_next.py`
before the PTO override is applied. Even for `pto_mega`, the PTO megakernel
subsequently replaces the Triton function at runtime. The log line is an
artefact of the patch ordering and does not mean Triton is being used.

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
