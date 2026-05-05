<div align="center">

# Megakernel for Gated DeltaNet </br> optimized for NPU using [PTO-ISA](gitcode.com/cann/pto-isa)

</div>

**TL;DR** This repo provides **1.5~3x faster**<sup>[1]</sup> NPU kernels for chunk GDN<sup>[2]</sup>, with integration to [vllm-ascend](https://github.com/vllm-project/vllm-ascend) to enable **~15% faster prefill** for Qwen3.5/3.6 models<sup>[3]</sup>.

- [1] Compared to the triton kernels used [in vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/v0.19.1rc1/vllm_ascend/ops/triton/fla)/[sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu/tree/2026.05.01/python/sgl_kernel_npu/sgl_kernel_npu/fla). See [Performance highlights](#performance-highlights).
- [2] All 6 stages: `chunk_cumsum`, `scaled_dot_kkt`, `solve_tril`, `wy_fast`, `chunk_h`, `chunk_o`. Plus a merged metakernel to save kernel launch overhead.
- [3] Tested model sizes: `0.8B`, `2B`, `4B`, `9B`, `27B`, `35B-A3B`. For larger `122B-A10B` `397B-A17B`, need to adapt for multi-device parallelism.

See full report [in English](blog/mega_gdn_en.md) or [in Chinese](blog/mega_gdn_zh.md).

# To reproduce

## NPU Environment

Recommend using [vllm-ascend docker images](https://quay.io/repository/ascend/vllm-ascend?tab=tags) with pre-installed vllm-ascend and triton-ascend (as baseline). This repo provides "plug-in" style patch compatile with vllm 0.18 and 0.19, without needing to rebuild vllm sources.

```bash
docker pull quay.io/ascend/vllm-ascend:v0.18.0rc1 
docker pull quay.io/ascend/vllm-ascend:v0.19.1rc1
```

If only test PTO kernels (no vllm and triton), then a standard CANN installation plus torch-npu is sufficent. Recommend [CANN docker images](https://quay.io/repository/ascend/cann?tab=tags). CANN 8.5.0~9.0.0 are verified. A minimum dockerfile example:

```dockerfile
FROM quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11

# older torch version also works
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-npu==2.9.0 \
    && pip install --no-cache-dir numpy pyyaml
```

## Installation

PTO kernels depend on [PTO-ISA](gitcode.com/cann/pto-isa) library which is included as git submodule.

```bash
git clone --recursive https://github.com/huawei-csl/megagdn-pto.git

# or init explicitly
cd megagdn-pto
git submodule update --init --recursive

# install `megagdn_pto` utilities, mostly just torch interface to PTO kernel call
pip install -e '.[eval,plot]' 
```

## Kernel-only unit tests and benchmarks

```bash
# accuracy tests
python tests/test_single_kernels.py --H-list 16,32,48,64
python tests/test_e2e.py --no-triton

# performance benchmark
 python benchmarks/kernel/bench_gdn_kernels.py \
    --device npu:0 --n-seq 16 --l-seg 8192 --H-list 16,32,48,64 
```

## End-to-end evaluation in vLLM

To prepare Qwen3.5/3.6 model weights, see [download_weights.md](scripts/download_weights.md).

```bash
# important! patch installed vllm to enable PTO kernel
python vllm_patch/install_hook.py

# Pick a free device ID. vllm uses separate worker processes, so `torch.npu.set_device` in master script is not enough
export ASCEND_RT_VISIBLE_DEVICES=0  # 0 ~ 7 on Atlas A2
bash benchmarks/vllm_prefill/run_prefill_sweep.sh  # prefill speed-up
bash benchmarks/eval_acc/run_eval_suite.sh  # lm-eval accuracy check, takes 20 min for one model
```

## Generating figures

After running the benchmarks, use `scripts/plot_results.py` to generate all figures under `outputs/figure/`.

```bash
# Plot all available data (prefill/eval via --auto paths; omit kernel unless flags set)
python scripts/plot_results.py --auto
```

# Performance highlights

Kernel performance vs Triton, measured on 910B2.

<img src="outputs/figure/kernel_stages_N16_L8192_H16.png" alt="alt text" style="width: 80%;" />

No loss in lm-eval scores for vllm-ascend inference. Just better & faster kernels.

<img src="outputs/figure/eval_accuracy.png" alt="alt text" style="width: 70%;" />

Prefill speed-up for vllm-ascend 0.18, measured on Atlas A2 (uses single NPU)

<img src="outputs/figure/prefill_qwen36_35b_a3b_w8a8.png" alt="alt text" style="width: 50%;" />

Prefill speed-up for vllm-ascend 0.19 (both triton baseline and PTO become faster than 0.18, due to framework-side optimizations)

<img src="outputs/figure/prefill_qwen36_35b_a3b_w8a8_v019.png" alt="alt text" style="width: 50%;" />
