# Prebuilt GDN metadata (vLLM-Ascend 0.19) and PTO integration

This note explains why the **Triton** chunk-GatedDeltaNet baseline often looks faster after moving from **vLLM-Ascend 0.18 → 0.19**, what **prebuilt metadata** actually is, and how **PTO (Bisheng / C++)** could consume analogous information—plus a concrete **implementation plan**.

---

## 1. What “prebuilt metadata” is in 0.19

In **vLLM-Ascend 0.19**, the worker patch [`patch_gdn_attn.py`](https://github.com/vllm-project/vllm-ascend) cooperates with [`vllm_ascend.ops.triton.gdn_chunk_meta`](https://github.com/vllm-project/vllm-ascend) to build a **`GDNChunkedPrefillMetadata`** object *before* the GDN Triton pipeline runs. That object is attached to attention metadata (e.g. `non_spec_prefill_fallback_meta.chunk`) and passed into [`chunk_gated_delta_rule_fwd(..., prebuilt_meta=...)`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/chunk.py).

Roughly, it holds **device tensors** (and buffer slots for reuse) such as:

| Field | Role (intuition) |
|--------|------------------|
| `chunk_indices_chunk64` | Per-chunk **`(seq_idx, chunk_idx)`** pairs for **BT = 64** tiling (varlen packing). |
| `chunk_offsets_chunk64` | **Prefix sum** of per-sequence **chunk counts** (length `N_chunks + 1`). Used so each Triton program can map *global chunk id → sequence + base token offset* without recomputing from `cu_seqlens`. |
| `update_chunk_offsets_chunk64` | Offsets for the **h-update** path (`chunk_delta_hupdate`). |
| `final_chunk_indices_chunk64` | Indices of **last chunk** per sequence (pipeline-parallel / state handoff). |
| `chunk_indices_large_block` | Indices for the **`solve_tril`** “large block” schedule (aligned with `solve_tril.LARGE_BLOCK_T`). |
| `block_indices_cumsum` | Feeds **`chunk_local_cumsum`** so cumsum over gates respects chunk boundaries **without** rebuilding block maps each call. |

The exact layouts mirror the helpers in [`ops/triton/fla/utils.py`](https://github.com/vllm-project/vllm-ascend) (`prepare_chunk_indices`, `prepare_chunk_offsets`, `prepare_update_chunk_offsets`, `prepare_final_chunk_indices`), except the **0.19** path **precomputes** them once (often with **CPU build + device copy + buffer reuse**) instead of recomputing inside every Triton wrapper invocation.

---

## 2. Why Triton TTFT / prefill improved from 0.18 to 0.19

The improvement is **not** primarily “new Triton kernel math”. For the classic six FLA stages, the per-stage Python modules (`chunk_delta_h.py`, `wy_fast.py`, `solve_tril.py`, …) are **largely the same** between **0.18.0rc1** and **0.19.1**; the **coordinator** [`chunk.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/chunk.py) changed in a **targeted** way.

### 2.1 Avoiding redundant `prepare_chunk_offsets` on the hot path

Triton kernels such as [`chunk_gated_delta_rule_fwd_h`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/chunk_delta_h.py) and [`chunk_fwd_o`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/chunk_o.py) accept optional **`chunk_offsets`**. When it is **`None`**, they call:

```python
chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)  # BT = 64
```

That builds a **new** `torch.LongTensor` from **`cu_seqlens`** using Python-visible tensor ops (`ceil`, `cat`, `cumsum`). Doing this **inside every launch wrapper** costs:

- **Host-side** work** and **allocation** per stage (or per entry to the autograd op),
- Potential **synchronization** if shapes are read back,
- Less opportunity to **reuse** the same device buffer across requests when the engine already knows the batch structure.

**v0.19** threads **`chunk_offsets_chunk64` from `prebuilt_meta`** through:

- `chunk_gated_delta_rule_fwd_h` (already true in many 0.18 builds),
- **`chunk_fwd_o_update`** and **`chunk_fwd_o`** — this is the **concrete diff** vs **0.18** in [`chunk.py`](https://github.com/vllm-project/vllm-ascend): **0.18** called `chunk_fwd_o(...)` **without** `chunk_offsets`, so **`chunk_o.py`** always recomputed offsets; **0.19** passes the **prebuilt** tensor.

That single wiring change removes repeated **`prepare_chunk_offsets`** work on the **final output** stage—often visible in end-to-end TTFT because **`chunk_o`** sits on the critical path after all prior math.

### 2.2 One-shot metadata build + buffer reuse

**0.19** enlarges **`patch_gdn_attn`** so chunk metadata can be **constructed once per batch** (with sizing / validation and optional **buffer pools**). Even when the raw formulas match `utils.py`, **amortizing** the build and keeping tensors **device-resident** reduces overhead compared to **per-kernel** Python setup.

### 2.3 Broader stack effects (outside the six Triton files)

**0.19** also refactors **where** GDN lives (`vllm_ascend.ops.gdn`, custom ops for causal conv, `gdn_attention_core`, etc.). Those changes affect **TTFT** independently of the FLA chunk kernels. This document focuses on **prebuilt chunk metadata** because that is the part most analogous to what PTO can reuse.

---

## 3. Why PTO cannot blindly reuse vLLM’s `prebuilt_meta` today

vLLM-Ascend builds chunk metadata for **Triton block tile `BT = 64`** (`_GDN_CHUNK_SIZE = 64` in `patch_gdn_attn`).

PTO uses **chunk size `C = 128`** (`C_PTO` in megagdn-pto).

Therefore:

- **`chunk_indices_chunk64` / `chunk_offsets_chunk64`** count and layout chunks at **64-token** granularity.
- PTO’s internal recursion groups tokens as **128-token** chunks.

You **cannot** pass the 64-wide index tensors directly into PTO C++ without either:

1. **Building a parallel `GDNChunkedPrefillMetadata` for C=128** (either extend `patch_gdn`/meta builder, or compute equivalent tensors in Python from `cu_seqlens` only for the PTO path), or  
2. **Teaching PTO** to consume **token-level** `cu_seqlens` only but move **offset/index generation** into **device-side** or **single-shot host** code paths (same *idea* as prebuilt meta, different **shapes**).

---

## 4. How PTO can use prebuilt-style metadata

Two layers: **Python / Torch** boundary and **C++** kernel internals.

### 4.1 Today’s PTO interface (simplified)

- **Wrapper**: [`vllm_patch/chunk_gated_delta_rule.py`](../vllm_patch/chunk_gated_delta_rule.py) exposes `chunk_gated_delta_rule_pto(..., prebuilt_meta=None, ...)`.
- **Fact today**: `prebuilt_meta` is **ignored** on the PTO fast path; it is only forwarded to **Triton fallback**.
- **Megakernel entry**: [`megagdn_pto/mega_kernel.py`](../megagdn_pto/mega_kernel.py) `run_mega_kernel(q, k, v, g, beta, cu_seqlens, ...)` passes **`cu_seqlens`** and tensors into a **`call_kernel`** with a **fixed** ctypes signature (many `void*` tensor pointers + scalar dims).

**Staged path**: [`megagdn_pto/kernel_libs.py`](../megagdn_pto/kernel_libs.py) + Python **`_chunk_cumsum`** iterate using **`cu_seqlens`** on host/Python—good correctness, unnecessary host loops at scale.

### 4.2 Torch calling interface (recommended evolution)

1. **Extend** `run_mega_kernel` / staged loaders to accept **optional**:
   - `chunk_offsets`: `torch.int32` or `torch.int64`, shape `[N_pto_chunks + 1]` or per-design,
   - `chunk_indices`: optional `[N_pto_chunks, 2]` if a stage needs `(seq, chunk_id)` like Triton,
   - optionally `block_indices_cumsum` for cumsum if moved from FLA-compatible layout to PTO layout.

2. **Plumb from** `chunk_gated_delta_rule_pto`:
   - If `prebuilt_meta is None`: **derive** PTO-C128 tensors once from `cu_seqlens` (same math as `prepare_chunk_offsets` / `prepare_chunk_indices` but **`chunk_size=128`**), **or**
   - If vLLM later provides **C=128** metadata: pass through directly.

3. **Contiguity / dtype**: match what the C++ launcher expects (`int32` `cu_seqlens` is already used); keep index tensors **NPU resident** and **contiguous**.

4. **ctypes / `compile.py`**: extend `call_kernel`’s **`argtypes`** with additional `c_void_p` for new GM buffers and `c_int32` lengths (or pack into a small **struct** if you prefer one pointer).

### 4.3 C++ kernel internals (mega-kernel and/or stages)

The fused [`kernels/pto/mega_kernel.cpp`](../kernels/pto/mega_kernel.cpp) currently implements **vectorized / blocked** work over **`T`** and **`H`** using **compile-time `GDN_C`** and implicit indexing derived from **global token order** + **`cu_seqlens`** (wired through the existing launcher).

To **consume prebuilt offsets** (conceptually mirroring `tl.load(chunk_offsets + i_n)` in Triton):

1. **Add GM-derived chunk bounds**  
   For each **PTO chunk program** `i_n`, **load** `boh = chunk_offsets[i_n]` (and `eoh = chunk_offsets[i_n + 1] - 1` or equivalent) from **`__gm__ int32_t* chunk_offsets`**.

2. **Replace implicit “scan cu_seqlens every time”**  
   Any place the kernel currently recomputes “which sequence / which chunk” from a **flat token id** can instead use **`chunk_indices`** `(seq_id, chunk_id)` **or** the **`chunk_offsets`** prefix layout, depending on which is simpler for **Vec/Cube** partitioners.

3. **Cumsum stage**  
   Today PTO may duplicate **chunk-boundary** logic. If you pass **`block_indices_cumsum`**-equivalent for **C=128**, the cumsum stage can **index** partial sums the same way **`chunk_local_cumsum(..., block_indices=...)`** does on the Triton side.

4. **`solve_tril` / large-block**  
   Triton uses **`chunk_indices_large_block`**. PTO’s **`tri_inverse`** path may need an analogous **schedule table** if you stop using only “regular” 128-wide grids—this is the **highest-risk** design area; profile before wide refactors.

5. **Backward compatibility**  
   If optional pointers are **`nullptr`**, fall back to current **cu_seqlens-only** indexing so older call sites and unit tests keep working.

---

## 5. Recommended implementation plan and TODO list

### Phase A — Correctness and scaffolding (no kernel logic change)

- [ ] **A1.** Add a small module (e.g. `megagdn_pto/chunk_meta.py`) that, given **`cu_seqlens`** (CPU or NPU) and **`chunk_size=128`**, builds:
  - `chunk_offsets` (same recurrence as `prepare_chunk_offsets` in FLA `utils.py` but with **128**),
  - optionally `chunk_indices` / `final_chunk_indices` if needed by staged kernels.
- [ ] **A2.** Unit-test tensor shapes vs a **reference** implementation (compare to FLA `utils.py` outputs on random varlen batches, **scaled** from 64→128 by recomputing with `chunk_size=128`).
- [ ] **A3.** In `chunk_gated_delta_rule_pto`, **when PTO path is taken**, compute (or cache) these tensors **once per forward** and stop relying on redundant Python loops where a single offset table suffices.

### Phase B — Staged PTO kernels (lower risk than mega-kernel)

- [ ] **B1.** Thread **`chunk_offsets`** into **`megagdn_pto.kernel_libs`** entry points that launch **separate** `.so` stages (`chunk_h`, `chunk_o`, …) if those launches currently re-derive chunk bounds.
- [ ] **B2.** Benchmark **`bench_gdn_kernels.py`** staged path: expect small CPU-side wins first; NPU-side win depends on whether launches were host-bound.

### Phase C — Fused mega-kernel

- [ ] **C1.** Extend **`mega_kernel.cpp`** launcher (`call_kernel`) and **`compile_mega_kernel`** templates to accept **optional GM pointers** + lengths for **`chunk_offsets`** / **`chunk_indices`**.
- [ ] **C2.** Refactor **device** indexing: for each **subblock / program**, load **chunk start/end** from **GM** instead of recomputing from **cu_seqlens** tables inside inner loops.
- [ ] **C3.** Bisheng compile / link validation on **NPU CI**; run **`tests/test_e2e.py`** and **`tests/test_single_kernels.py`**.

### Phase D — vLLM integration and prefill re-test

- [ ] **D1.** If metadata is built inside megagdn only: **no vLLM patch** beyond what you have.  
       If you want **true sharing** with the engine: propose upstream **`GDNChunkedPrefillMetadata`-C128** or a **parallel** slot in `patch_gdn_attn` (coordination with vLLM-Ascend maintainers).
- [ ] **D2.** Re-run **`benchmarks/vllm_prefill/benchmark_prefill.py`** for **triton vs pto_mega** on **9B / 27B / 35B** and the **512–32k** grid; compare **relative** `median_ttft_ms` ratio **before/after**.
- [ ] **D3.** Document observed gains in `README.md` or `REPRODUCTION_FINDINGS.md` (absolute TTFT and **Triton/PTO** ratio).

### Phase E — Optional alignment with vLLM `prebuilt_meta`

- [ ] **E1.** Map fields of **`GDNChunkedPrefillMetadata`** to PTO-C128 tensors (or explicitly **reject** mixed 64/128 and build fresh).
- [ ] **E2.** When `prebuilt_meta is not None`, prefer **engine-built** buffers **only if** chunking policy matches PTO (or convert—usually **cheaper** to rebuild for **128** than to fix up **64**-based indices).

---

## 6. Summary

| Question | Short answer |
|----------|--------------|
| Why is Triton faster on 0.19? | Mostly **stack + metadata**: **prebuilt** chunk maps and **passing `chunk_offsets` into `chunk_fwd_o` / update** so kernels stop calling **`prepare_chunk_offsets`** every time; plus broader **GDN** integration changes. |
| Did the six FLA `.py` kernels change a lot? | **Little**; **`chunk.py`** coordination and **`patch_gdn_attn`** are the main deltas relevant to **prebuilt** maps. |
| Can PTO use the same trick? | **Yes in principle**: pass **C=128** offset/index tensors into **Torch** and then into **C++ GM** arguments, mirroring Triton’s `chunk_offsets` loads. |
| Biggest gotcha? | vLLM prebuilt meta is for **64-wide** chunks; PTO needs **128-wide** (or generic) metadata—**do not reuse raw `chunk_offsets_chunk64`**. |

---

*File added for megagdn-pto. Source references: `vllm-ascend` **0.18.0rc1** vs **0.19.1rc1** trees under `vllm_source_ref/` and megagdn-pto `vllm_patch/`, `megagdn_pto/`, `kernels/pto/`.*
