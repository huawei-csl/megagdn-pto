# MegaGDN: One fused kernel cuts Qwen3.5/3.6 prefill TTFT by 15% <br/> (enable high-performance PTO kernels in vLLM-Ascend)

- Author: Jiawei Zhuang
- Team: Aleksandros Sobczyk, Gioele Gottardo, Filip Skogh, Mirko De Vita, Christos Konstantinos Matzoros, Anastasios Zouzias, Jiawei Zhuang

**TL;DR**: We reimplemented every stage of chunk Gated DeltaNet (GDN) using PTO ISA for NPU. Each stage is **1.5~3x** faster than the corresponding Triton kernels in vLLM-Ascend, and all stages are further merged together to reduce kernel launch overhead. Integrated into vLLM, we measured **15% lower prefill TTFT** for Qwen3.5/3.6, with **no accuracy regression** on lm-eval benchmarks.

**To reproduce everything in this post**, see:
- Operator source, micro-benchmarks, accuracy tests, and the vLLM-Ascend integration: https://github.com/huawei-csl/megagdn-pto
- vLLM-Ascend pull request: https://github.com/vllm-project/vllm-ascend/pulls (**TODO**)

# Outline

- [Overall design: Chunk-128 GDN kernel tailored to the NPU](#overall-design-chunk-128-gdn-kernel-tailored-to-the-npu)
- [Requirements: dynamic batch axes and the full Qwen3.5/3.6 shape matrix](#requirements-dynamic-batch-axes-and-the-full-qwen3536-shape-matrix)
- [Six kernels benchmarked: PTO vs Triton](#six-kernels-benchmarked-pto-vs-triton)
- [Megakernel: single launch to reduce host overhead](#megakernel-single-launch-to-reduce-host-overhead)
  - [Host overhead in eager PyTorch](#host-overhead-in-eager-pytorch)
  - [Simpler way for megakernel fusion](#simpler-way-for-megakernel-fusion)
- [Full-model results: vLLM-Ascend prefill and lm-eval accuracy](#full-model-results-vllm-ascend-prefill-and-lm-eval-accuracy)

# Overall design: Chunk-128 GDN kernel tailored to the NPU

In the [chunkwise algorithm](https://sustcsonglin.github.io/blog/2024/deltanet-2/#a-chunkwise-algorithm-for-deltanet) for linear attention family, the central knob is the chunk size `C`, analogous to the sequence tile `S` in FlashAttention: it determines arithmetic intensity and thus how well the matrix engine (Cube / Tensor Core) can be fed. GPU kernels prefer small chunks: [FLA](https://github.com/fla-org/flash-linear-attention) defaults to 64 (the `BT` parameter in the Triton sources), while [FlashKDA](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md) goes even smaller, down to 16 -- partly because the triangular-inverse step costs `O(C^2)` FLOPs that are not "just cheap Matmul FLOPs", so an oversized chunk makes inversion the slow bottleneck.

Our previous post on [NPU-friendly inversion algorithm](https://github.com/huawei-csl/gdn-tri-inverse/blob/0.1.0/markdown/fast_inverse_blog/fast_inverse.md) removed that bottleneck for chunk size 128, so the full GDN pipeline can be reformulated in chunk 128. (Why not 256? Past the corner of the hardware roofline, extra FLOPs stop being “free”; the trade-off turns negative.)

## Compared with existing Triton and TileLang samples

vLLM-Ascend and sgl-kernel-npu already ship Triton implementations -- so why not set `BT = 128` and rebuild? In practice, some kernels such as `chunk_o` and `scaled_dot_kkt` fail to compile (tiles get large enough to blow SRAM budgets and need deeper kernel surgery). Others like `chunk_h` may compile and gain a little, but nowhere near what we get from a ground-up rewrite. Chunk GDN is one coherent pipeline: every stage must agree on the same chunk size; you cannot mix sizes ad hoc. We bypass the Triton abstraction layer, program at the PTO-ISA level, reaching up to **3x** faster than the Triton baseline. ([cuLA](https://github.com/inclusionAI/cuLA) on GPUs does something similar with CuteDSL and also sees large gains over FLA/Triton.)

We also studied tilelang-ascend’s [opt_gdn](https://github.com/tile-ai/tilelang-ascend/tree/67d6a4a818e864b8cfb84e310ec568bd18b879fe/examples/linear_attention_and_rnn/opt_gdn) (static shapes only) and [chunk_gated_delta_rule](https://github.com/tile-ai/tilelang-ascend/tree/67d6a4a818e864b8cfb84e310ec568bd18b879fe/examples/chunk_gated_delta_rule) (currently only the `chunk_h` stage). We completed all six stages, added support for variable-length dynamic-shape batches, and hand-tuned the PTO-ISA C++ for a large performance jump.

# Requirements: dynamic batch axes and the full Qwen3.5/3.6 shape matrix

To land inside a production inference stack, we must decide which tensor axes are true runtime-dynamic shapes and which can be compile-time constants (macros or C++ template parameters) to simplify tiling and squeeze more performance.

## Dynamic axes

Batch and sequence length are obviously dynamic for prefill. Following FLA naming, the framework passes `cu_seqlens` (“cumulative sequence lengths”) into the kernel: it is a 1-D `int` array that stores the starting index for each variable-length sample in the batch, useful for global-memory addressing. Example:
- Suppose `cu_seqlens = [0, 5, 8, 15]`.
- Then `batch_size = len(cu_seqlens) - 1 = 3`.
- Per-sequence lengths are `[5, 3, 7]` via `seqlen[i] = cu_seqlens[i+1] - cu_seqlens[i]`.
- `pto::TLOAD` / `pto::TSTORE` offsets need the accumulated `cu_seqlens`, not raw per-seq lengths alone.

Readers familiar with FLA’s Triton will notice extra arguments such as `chunk_indices` and `chunk_offset`. Our NPU kernels drop them. Why it differs comes down to how multicore `block_idx` is assigned; see our earlier note on [NPU kernel launch behavior](https://github.com/huawei-csl/pto-dsl/blob/0.1.2/examples/aot/matmul_optimization_guide/matmul_optim_guide.md#typical-kernel-launch-syntax). In Triton/CUDA the launch grid often scales with data size, for example:
- In [chunk_delta_h.py](https://github.com/fla-org/flash-linear-attention/blob/v0.4.2/fla/ops/common/chunk_delta_h.py#L691C5-L691C61), `grid = (triton.cdiv(V, meta['BV']), N*H)`.
- Because `block_idx` (Triton’s `program_id`) is unbounded, mapping a program id to a chunk index needs side metadata (`chunk_indices`, `chunk_offset`).
- NPU code typically fixes `block_dim = num_cores` with `block_idx` in `[0, num_cores - 1]` and just assigns workload across cores inside main loops.

## Static axes

Chunk size is fixed at 128. Head count and embedding width take only a small discrete set of values per model family, so they are also suitable for compile-time constants -- much like FlashAttention templating head dim and [recompiling per `hdim`](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3/csrc/flash_attn/src).

Enumerating all shapes for [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) and [Qwen3.6](https://huggingface.co/collections/Qwen/qwen35):
- `linear_num_value_heads` is one of `{16, 32, 48, 64}`:
    - 16: `0.8B`, `2B`
    - 32: `4B`, `9B`, `35B-A3B`
    - 48: `27B`
    - 64: `122B-A10B`, `397B-A17B`
- Every variant uses `linear_num_key_heads = 16`.
- Every variant uses `linear_key_head_dim = 128`, `linear_value_head_dim = 128`.

We therefore bake `key_head_dim = value_head_dim = 128` into macros and treat `num_value_heads` (`H`) as a template parameter, instantiating four specialized instances. Sketch:

```cpp
#define GDN_D 128 // head_dim
#define GDN_C 128 // chunk_size
#define GDN_H 16  // num_key_heads

template <int32_t NumValueHeads>
AICORE void chunk_gdn_kernel(...)
```

Tensor-parallel sharding changes the effective shapes; this article targets single-device kernels first, leaving multi-device fusion as future work.

# Six kernels benchmarked: PTO vs Triton

Compared to the Triton kernels used [in vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/v0.19.1rc1/vllm_ascend/ops/triton/fla)/[sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu/tree/2026.05.01/python/sgl_kernel_npu/sgl_kernel_npu/fla), our PTO kernels are **3x** faster for the costly stages such as `chunk_h`, `chunk_k`, `wy_fast`. The Triton baseline uses default chunk_size(BT) = 64. As said before, we also tried recompiling Triton kernels with chunk_size=128 -- `chunk_h` and `wy_fast` run faster, only being **1.5x** slower than our PTO kernels. But `chunk_o` and `scaled_dot_kkt` failed, and the Triton `solve_tril` is hard-coded to chunk 64.

<img src="fig/kernel_stages_N16_L8192_H16.png" alt="alt text" style="width: 80%;" />

(Reproduced by scripts in the `benchmarks/kernel/` directory. Raw performance data for more shape configs is in the `outputs/data/` directory.)

Results are from 910B2 devices. 910C can reuse the same kernel code. PTO-ISA abstraction is also portable to 950, and we will update its performance numbers later.

# Megakernel: single launch to reduce host overhead

## Host overhead in eager PyTorch

Kernel-only optimizations alone are not enough to speed up end-to-end inference on Atlas A2/A3 systems -- those machines are notoriously easy to become host-bound when running PyTorch eager mode. Profiling vLLM prefill (original Triton backend) shows large bubbles on the device side, due to host-side Python overhead and kernel launch overhead.

<img src="fig/torch_profiling.png" alt="alt text" style="width: 80%;" />

(Reproduced by `profiling/` directory)

The standard mitigation is [ACL Graph](https://docs.vllm.ai/projects/ascend/en/v0.18.0/developer_guide/Design_Documents/ACL_Graph.html), the NPU analogue of CUDA Graph. vLLM-Ascend’s graph path applies to **`decode_only`**; prefill still runs in eager Torch. See [ACL graph limitations](https://docs.vllm.ai/projects/ascend/en/v0.18.0/developer_guide/Design_Documents/ACL_Graph.html#limitations). Prefill tensor shapes change often, but capturing and replaying graphs prefer fixed static shapes. Supporting dynamic-shape graphs means padding, bucketing, and other workarounds. GPUs hit the same limitation -- see [CUDA Graph dynamic shapes](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html#dynamic-shapes) and the discussion [why vLLM prefill does not use CUDA-graph](https://www.zhihu.com/question/7987565201).

## Simpler way for megakernel fusion

To not struggle with graph mode workarounds, we take a simpler but effective approach: we have the source code for all 6 computation stages, each only a few hundred lines of C++. We just merged them into one NPU kernel function, launched once with the familiar `<<< >>>` syntax.

Between the stages, we must fence across all cores for memory ordering. We use `TSyncAll` that is [just added to PTO-ISA](https://gitcode.com/cann/pto-isa/pull/878).

The fused source code is at path `megagdn-pto/kernels/pto/mega_kernel.cpp`. The megakernel source stays short and readable. The `launch_mega_kernel` kernel entry is a few hundred lines of C++. Stages reuse the single-kernel sources without copy-paste duplication. This is legal because one AICORE function can call another AICORE function, similar to CUDA `__forceinline__ __device__`.

# Full-model results: vLLM-Ascend prefill and lm-eval accuracy

We verified Qwen 0.8B, 2B, 27B, and 35B models on a single 910B2 device. The 27B and 35B models are quantized to W8A8 to fit into the 64 GiB device memory.

Here we highlight the 35B results. Complete raw data for all models are in the `outputs/data/` directory, reproduced by the `benchmarks/eval_acc/` and `benchmarks/vllm_prefill/` directories.

No loss in lm-eval scores for vllm-ascend inference. Just better & faster kernels.

<img src="fig/eval_accuracy.png" alt="alt text" style="width: 70%;" />

To measure the prefill TTFT, we use the `first_token_latency` metric in the built-in `RequestStateStats` class defined by [vllm/v1/metrics/stats.py](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/metrics/stats.py#L216). Median and mean TTFT agree well; we report the median here.

We observed a 15~25% (average 20%) prefill speed-up for vllm-ascend 0.18, measured on Atlas A2 (uses a single NPU)

<img src="fig/prefill_qwen36_35b_a3b_w8a8.png" alt="alt text" style="width: 50%;" />

We observed an average 15% prefill speed-up for vllm-ascend 0.19 (both the Triton baseline and the new PTO backend are faster than in 0.18, due to framework-side optimizations)

<img src="fig/prefill_qwen36_35b_a3b_w8a8_v019.png" alt="alt text" style="width: 50%;" />
