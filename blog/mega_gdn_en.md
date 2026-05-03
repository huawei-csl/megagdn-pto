# MegaGDN: One fused kernel cuts Qwen3.5/3.6 prefill TTFT by 15% <br/> (integrate PTO-ISA to vLLM-Ascend)

- Author: Jiawei Zhuang
- Team: Aleksandros Sobczyk, Gioele Gottardo, Filip Skogh, Mirko De Vita, Christos Konstantinos Matzoros, Anastasios Zouzias, Jiawei Zhuang

**TL;DR**: We reimplemented every stage of chunk Gated Delta Net (GDN) using the PTO instruction set. Each stage is about **2× faster on average** than the six Triton reference kernels in vLLM-Ascend. Fusing those six kernels into one megakernel adds another **~1.5×**. Integrated into vLLM, we measured roughly **15% lower end-to-end prefill TTFT** for Qwen3.5/3.6, with **no accuracy regression** on lm-eval style benchmarks.

**To reproduce everything in this post**, see:
- Operator source, micro-benchmarks, accuracy tests, and the vLLM-Ascend integration: https://github.com/huawei-csl/megagdn-pto
- vLLM-Ascend pull request: https://github.com/vllm-project/vllm-ascend/pulls (**TODO**)

# Outline

- Overall design: a Chunk-128 GDN stack tailored to the NPU
- Requirements: dynamic batch axes and the full Qwen3.5/3.6 shape matrix
- Six kernels benchmarked: PTO vs Triton baselines (accuracy and performance)
- Megakernel: a single launch path to remove the host bottleneck
- Full-model results: vLLM-Ascend prefill and lm-eval accuracy

# Overall design: a Chunk-128 GDN stack tailored to the NPU

The central knob in the [chunkwise algorithm](https://sustcsonglin.github.io/blog/2024/deltanet-2/#a-chunkwise-algorithm-for-deltanet) for linear attention family is the chunk size `C`, analogous to the sequence tile `S` in FlashAttention: it sets arithmetic intensity and thus how well the matrix engine (Cube / Tensor Core) can be fed. GPUs typically use smaller chunks: [FLA](https://github.com/fla-org/flash-linear-attention) defaults to 64 (the `BT` parameter in the Triton sources), while [FlashKDA](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md) goes as small as 16 —- partly because the triangular-inverse step costs `O(C^2)` FLOPs that are not "just cheap Matmul FLOPs", so an oversized chunk makes inversion the bottleneck.

Our previous post on [NPU-friendly inversion via PTO](https://github.com/huawei-csl/gdn-tri-inverse/blob/0.1.0/markdown/fast_inverse_blog/fast_inverse.md) removed that bottleneck for chunk size 128, so the full GDN pipeline can be retiled to chunk 128. (Why not 256? Past the corner of the hardware roofline, extra FLOPs stop being “free”; the trade-off turns negative.)

## Compared with existing Triton and TileLang samples

vLLM-Ascend and sgl-kernel-npu already ship Triton implementations—so why not set `BT = 128` and rebuild? In practice, some kernels such as `chunk_o` and `scaled_dot_kkt` fail to compile (tiles get large enough to blow SRAM budgets and need deeper kernel surgery). Others like `chunk_h` may compile and gain a little, but nowhere near what we get from a ground-up rewrite. Chunk GDN is one coherent pipeline: every stage must agree on chunk size; you cannot mix sizes ad hoc. We bypass the Triton abstraction layer, program at PTO level, and recover roughly **2×** on the stages we care about. ([cuLA](https://github.com/inclusionAI/cuLA) on GPUs does something similar with CuteDSL over FLA and also sees large gains over stock Triton.)

We also studied tilelang-ascend’s [opt_gdn](https://github.com/tile-ai/tilelang-ascend/tree/67d6a4a818e864b8cfb84e310ec568bd18b879fe/examples/linear_attention_and_rnn/opt_gdn) (static shapes only) and [chunk_gated_delta_rule](https://github.com/tile-ai/tilelang-ascend/tree/67d6a4a818e864b8cfb84e310ec568bd18b879fe/examples/chunk_gated_delta_rule) (currently centered on the `chunk_h` stage). We completed all six stages, added support for variable-length batches with non-power-of-two dynamic shapes, and hand-tuned the PTO-ISA C++ for a large performance jump.

# Requirements: dynamic batch axes and the full Qwen3.5/3.6 shape matrix

To land inside a production inference stack, we must decide which tensor axes are true runtime-dynamic shapes and which can be compile-time constants (macros or C++ template parameters) to simplify tiling and squeeze performance.

## Dynamic axes

Batch and sequence length are obviously dynamic for prefill. Following FLA naming, the framework passes `cu_seqlens` (“cumulative sequence lengths”) into the kernel: a 1-D `int` array that stores, for each variable-length sample in the batch, the starting index in the token sequence used for global-memory addressing. Example:
- Suppose `cu_seqlens = [0, 5, 8, 15]`.
- Then `batch_size = len(cu_seqlens) - 1 = 3`.
- Per-sequence lengths are `[5, 3, 7]` via `seqlen[i] = cu_seqlens[i+1] - cu_seqlens[i]`.
- `TLOAD` / `TSTORE` offsets should accumulate `cu_seqlens`, not raw per-seq lengths alone.

Readers familiar with FLA’s Triton will notice extra arguments such as `chunk_indices` and `chunk_offset`. Our NPU kernels drop them—the launch model is simpler. Why differs comes down to how multicore `block_idx` is assigned; see our earlier note on [NPU kernel launch behavior](https://github.com/huawei-csl/pto-dsl/blob/0.1.2/examples/aot/matmul_optimization_guide/matmul_optim_guide_zh.md#typical-kernel-launch-syntax). On Triton/CUDA the grid often scales with data volume, e.g.
- In [chunk_delta_h.py](https://github.com/fla-org/flash-linear-attention/blob/v0.4.2/fla/ops/common/chunk_delta_h.py#L691C5-L691C61), `grid = (triton.cdiv(V, meta['BV']), N*H)`.
- Because `block_idx` (Triton’s `program_id`) is unbounded in the upper range, mapping a program id to a chunk index needs side metadata (`chunk_indices`, `chunk_offset`).
- NPU code typically fixes `block_dim = num_cores` with `block_idx` in `[0, num_cores - 1]` and stripes work across cores inside loops.

## Static axes

Chunk size is fixed at 128. Head count and embedding width take only a small discrete set of values per model family, so they are good candidates for compile-time constants—much like FlashAttention templating head dim and [recompiling per `hdim`](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3/csrc/flash_attn/src).

Enumerating all public shapes for [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) and [Qwen3.6](https://huggingface.co/collections/Qwen/qwen35):
- Every variant uses `linear_key_head_dim = 128`, `linear_value_head_dim = 128`.
- Every variant uses `linear_num_key_heads = 16`.
- `linear_num_value_heads` is one of `{16, 32, 48, 64}`:
    - 16: `0.8B`, `2B`
    - 32: `4B`, `9B`, `35B-A3B`
    - 48: `27B`
    - 64: `122B-A10B`, `397B-A17B`

We therefore bake `key_head_dim = value_head_dim = 128` into macros and treat `num_value_heads` (`H`) as a template parameter, instantiating four specialized binaries to cover the catalog. Sketch:

```cpp
#define GDN_D 128 // head_dim
#define GDN_C 128 // chunk_size
#define GDN_H 16  // num_key_heads

template <int32_t NumValueHeads>
AICORE void chunk_gdn_kernel(...)
```

Tensor-parallel sharding changes the effective shapes; this article targets single-device kernels first, with broader fusion as future work.

# Six kernels benchmarked: PTO vs Triton (accuracy and performance)

![alt text](fig/kernel_stages_N16_L8192_H16.png)

# Megakernel: one launch, fewer host round trips

## Host overhead in eager PyTorch

Kernel micro-optimizations alone are not enough on deployed Atlas A2/A3 systems—those machines are notoriously easy to drive into a host-bound regime.

![alt text](fig/torch_profiling.png)

The textbook mitigation is [ACL Graph](https://docs.vllm.ai/projects/ascend/en/v0.18.0/developer_guide/Design_Documents/ACL_Graph.html), the NPU analogue of CUDA Graph. vLLM-Ascend’s graph path defaults to **`decode_only`**; prefill still runs in eager Torch. See [ACL graph limitations](https://docs.vllm.ai/projects/ascend/en/v0.18.0/developer_guide/Design_Documents/ACL_Graph.html#limitations). Prefill tensors reshape often; capturing and replaying graphs wants fixed shapes, so dynamic-shape support means padding, bucketing, and other workarounds. GPUs hit the same wall—compare NVIDIA’s notes on [CUDA Graph dynamic shapes](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html#dynamic-shapes) and the discussion [why vLLM did not CUDA-graph prefill early on](https://www.zhihu.com/question/7987565201).

## A direct megakernel approach

We take a blunt path: six compact C++ stages, each only a few hundred lines, merged into one NPU kernel launched once with the familiar `<<< >>>` syntax.

Stages must fence across all cores for memory ordering. We use `SyncAll` (see https://gitcode.com/cann/pto-isa/pull/878).

The fused source stays short—the full megakernel lives at `megagdn-pto/kernels/pto/mega_kernel.cpp`. The `launch_mega_kernel` entry is on the order of a hundred lines. Stages reuse the single-kernel sources without copy-paste duplication.

That is legal because one AICORE function can call another, CUDA-`__device__`-style and mostly inlined, without a conventional stack or recursion—clean structure with the same semantics as inlined code.

# Full-model results: vLLM-Ascend prefill and lm-eval accuracy

![alt text](fig/prefill_qwen36_35b_a3b_w8a8.png)

![alt text](fig/eval_accuracy.png)
