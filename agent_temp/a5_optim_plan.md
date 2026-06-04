---
name: megagdn a5 optimization
overview: Optimize the already-compiling MegaGDN A5 PTO kernels for real Ascend950PR performance, focusing on replacing GM Cube-Vector handoffs in `chunk_h` and `chunk_o` with direct A5 paths and benchmarking each attempted variant against the saved A2 PTO baseline.
todos:
  - id: profile-current-hotspots
    content: Map current `chunk_h` and `chunk_o` GM workspace handoffs and estimate bytes/FLOPs per handoff.
    status: completed
  - id: explore-a5-manual-patterns
    content: "Explore all relevant A5 manual optimization patterns: flash attention direct C/V modes, gemm_ar L1 reuse, and SIMT/D-cache guidance."
    status: completed
  - id: analyze-buffer-core-util
    content: Analyze A5 UB/L1/L0C/core utilization for current `chunk_h` and `chunk_o` and identify safe double-buffer opportunities.
    status: completed
  - id: prototype-chunk-h-c2v
    content: Prototype direct `L0C -> UB` replacement for `chunk_h` Cube-to-Vec WS/KV handoff and validate correctness.
    status: completed
  - id: prototype-chunk-h-v2c
    content: Prototype direct `UB -> L1` replacement for `chunk_h` Vec-to-Cube K/S handoff and validate correctness.
    status: completed
  - id: prototype-chunk-o-c2v
    content: Prototype direct `L0C -> UB` replacement for `chunk_o` QK/QS/QKV handoffs and validate correctness.
    status: completed
  - id: prototype-chunk-o-v2c
    content: Prototype direct `UB -> L1` replacement for `chunk_o` QK_gated handoff and validate correctness.
    status: completed
  - id: optimize-kkt
    content: Explore A5-specific optimization candidates for `scaled_dot_kkt`, including direct C/V paths and L1/UB reuse where applicable.
    status: completed
  - id: optimize-wy-fast
    content: Explore A5-specific optimization candidates for `wy_fast`, including direct handoff, L1 panel reuse, and H=64 timeout mitigation.
    status: completed
  - id: explore-simt-memory-bound
    content: Evaluate whether SIMT/D-cache techniques are applicable to memory-bound scalar or elementwise sections after direct C/V experiments.
    status: completed
  - id: benchmark-variants
    content: Benchmark all passing variants on real `npu:0` with reduced and large shapes, avoiding unsafe H=64 paths until stable.
    status: completed
  - id: select-best-report
    content: Keep the best-performing correct variants and write the final A5 optimization report and lessons learned.
    status: completed
isProject: false
---

# MegaGDN A5 Performance Optimization Plan

## Goal
- Improve A5 PTO kernel performance beyond the current mechanical port in [`/home/jzhuang/megagdn-pto/kernels/pto_a5`](/home/jzhuang/megagdn-pto/kernels/pto_a5).
- Prioritize `chunk_h` and `chunk_o`, then check `scaled_dot_kkt` and `wy_fast` if time permits.
- Leave PTO `tri_inverse` / `solve_tril` for future work, except keeping the current torch fallback for correctness.
- Target: approach or exceed `3x` speedup versus saved A2 PTO timings in [`/home/jzhuang/megagdn-pto/outputs/data/kernel_bench.json`](/home/jzhuang/megagdn-pto/outputs/data/kernel_bench.json), while preserving numerical correctness on real `npu:0`.

## Baseline To Preserve
- Current A5 correctness port lives in [`/home/jzhuang/megagdn-pto/kernels/pto_a5`](/home/jzhuang/megagdn-pto/kernels/pto_a5).
- Current A5 benchmark/comparison artifacts:
  - [`outputs/data/kernel_bench_a5.json`](/home/jzhuang/megagdn-pto/outputs/data/kernel_bench_a5.json)
  - [`outputs/data/kernel_bench_a5_comparison.md`](/home/jzhuang/megagdn-pto/outputs/data/kernel_bench_a5_comparison.md)
- Known limitation: H=64 large-shape `wy_fast` timed out. Avoid unsafe reruns until smaller variants validate.

## Optimization Candidates
- Direct C2V in `chunk_h`:
  - Replace Cube `TSTORE WS = W @ S -> GM workspace` plus Vec `TLOAD WS <- GM` with A5 `L0C -> UB` using `TMOV` / `copy_matrix_cc_to_ub`.
  - Use explicit ready/free sync so Cube does not overwrite the UB handoff before Vec consumes it.
- Direct V2C in `chunk_h`:
  - Replace Vec `TSTORE K_scaled/S -> GM workspace` plus Cube `TLOAD -> L1` with `TINSERT` / `copy_ubuf_to_cbuf` into L1.
  - Convert Vec ND tiles to NZ before Cube consumption.
  - Use one-slot conservative sync first; only then try double-buffering.
- Direct C2V in `chunk_o`:
  - Replace Cube QK/QS/QKV GM workspace stores with direct `L0C -> UB` where Vec immediately gates/combines outputs.
  - Start with QKV or QS only if full replacement is too risky.
- Direct V2C in `chunk_o`:
  - Replace Vec QK_gated GM workspace store with direct `UB -> L1` via `TINSERT` before Cube GEMM3.
  - Apply the verified `stream_v2c` and `add_matmul_v2c` ownership pattern from [`/home/jzhuang/pto-kernels-fork/examples/jit_cpp/cv_sync_demo_a5`](/home/jzhuang/pto-kernels-fork/examples/jit_cpp/cv_sync_demo_a5).
- A5 buffer/core utilization:
  - Re-check UB/L1/L0C footprint against DAV_3510 capacities from [`/home/jzhuang/cannbot-skills/ops/npu-arch/SKILL.md`](/home/jzhuang/cannbot-skills/ops/npu-arch/SKILL.md): UB ~248KB, L0C 256KB, Cube cores 28/32 depending SKU.
  - Increase local double-buffering only where capacity allows.
  - Avoid GM scratch when direct local buffers fit.
- Advanced patterns to inspect and reuse:
  - Manual A5 flash attention direct C/V modes in [`/home/jzhuang/pto-isa/kernels/manual/a5/flash_atten`](/home/jzhuang/pto-isa/kernels/manual/a5/flash_atten).
  - A5 matmul/L1 reuse patterns in [`/home/jzhuang/pto-isa/kernels/manual/a5/gemm_ar`](/home/jzhuang/pto-isa/kernels/manual/a5/gemm_ar).
  - SIMT/D-cache ideas only for memory-bound scalar/elementwise sections if direct C/V is insufficient.

## Execution Strategy
- Work in small variants, one kernel at a time:
  - Create/modify one candidate path.
  - Run quick correctness for `H=16`.
  - Run quick correctness for `H=16,32,48,64` if safe.
  - Benchmark only that stage at reduced iterations.
  - Keep or revert based on correctness and measured speed.
- Prefer preserving working A5 code with compile-time switches for experiments, e.g. `GDN_A5_DIRECT_CV_CHUNK_H`, until best variant is selected.
- Avoid full H=64 large-shape benchmarks until H=16/32/48 are stable.

## Validation Commands
- Environment:
  - `conda activate torch_npu_dev`
  - `source /usr/local/Ascend/cann-9.0.0/set_env.sh`
  - `export GDN_NPU_DEVICE=npu:0`
  - `export MEGAGDN_PTO_ARCH=a5`
- Quick correctness per stage:
  - `python3 tests/test_single_kernels.py --device npu:0 --quick --H-list 16 --stage chunk_h`
  - `python3 tests/test_single_kernels.py --device npu:0 --quick --H-list 16 --stage chunk_o`
- Broader correctness after a variant passes:
  - `python3 tests/test_single_kernels.py --device npu:0 --quick --H-list 16,32,48,64 --stage chunk_h,chunk_o`
- Stage benchmark examples:
  - `GDN_BENCH_WARMUP=1 GDN_BENCH_ITERS=3 python3 benchmarks/kernel/bench_gdn_kernels.py --device npu:0 --n-seq 16 --l-seg 16384 --H-list 16,32,48 --stage chunk_h,chunk_o --output-json outputs/data/kernel_bench_a5_opt.json`

## Reporting
- Produce a final optimization report, for example [`outputs/data/kernel_bench_a5_opt_report.md`](/home/jzhuang/megagdn-pto/outputs/data/kernel_bench_a5_opt_report.md), containing:
  - Variant list and whether each passed correctness.
  - Best timings by stage and H.
  - Speedup vs A2 baseline.
  - Which direct C/V exchanges were successfully eliminated from GM.
  - Any failed attempts and why.
- If any kernel reaches or exceeds `3x` vs A2, add a short “A5 optimization practices learned” section to [`kernels/pto_a5/PORT_STATUS.md`](/home/jzhuang/megagdn-pto/kernels/pto_a5/PORT_STATUS.md) or a new notes file.

## Guardrails
- Do not work on PTO `tri_inverse` in this optimization pass.
- Do not count torch fallback solve time as PTO performance.
- Do not claim speedups from noisy tiny-stage timings like `chunk_cumsum` unless measurement is robust.
- If a variant triggers AICore timeout, stop that path, verify device health, and document it rather than repeatedly rerunning unsafe cases.