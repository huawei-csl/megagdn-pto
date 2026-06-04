# MegaGDN PTO A5 Port

This directory is the A5 / DAV_3510 port of the original PTO kernels in
`kernels/pto`.

## What Changed

- Build target changed from DAV_2201 (`dav-c220`) to DAV_3510 (`dav-c310`).
- Old `__DAV_C220_CUBE__` / `__DAV_C220_VEC__` guards were replaced with A5
  guards: `__DAV_CUBE__` / `__DAV_VEC__`.
- A5 header conflicts were fixed:
  - local `block_num` variables were renamed because `block_num` is a CANN macro
    on this stack.
  - unqualified `Stride<...>` was changed to `pto::Stride<...>`.
  - `pipe_barrier(PIPE_V)` was replaced with `pipe_barrier(PIPE_ALL)` because
    DAV_3510 rejects the old PIPE_V barrier form.
  - custom L0A/Left tile definitions now use A5's `BLayout::ColMajor`.
- Cross-core waits use the A5 two-argument `wait_flag_dev(PIPE_S, flag)` form.

## Current Status

Validated on real `npu:0` with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_npu_dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export GDN_NPU_DEVICE=npu:0
export MEGAGDN_PTO_ARCH=a5
```

Quick correctness (`T=128`) passed for `H=16,32,48,64`:

```bash
python3 tests/test_single_kernels.py --device npu:0 --quick --H-list 16,32,48,64
```

The following PTO stages compile and pass quick correctness:

- `chunk_cumsum`
- `scaled_dot_kkt`
- `wy_fast`
- `chunk_h`
- `chunk_o`

## Known Limitation

`tri_inverse` / `solve_tril` is not fully ported to A5 yet.

The A5 copy of `tri_inverse` compiles after tile-layout fixes, but produces NaNs
on real hardware. For now, `megagdn_pto.fast_inverse.solve_tril` uses a torch
reference fallback when `MEGAGDN_PTO_ARCH=a5`. This keeps staged correctness
tests runnable, but it is not a PTO performance result.

The fused `mega_kernel` is also not considered validated on A5 yet because it
depends on the PTO triangular inverse path.

## Benchmark Results

PTO-only A5 timing for completed large-shape stages is stored in:

- `outputs/data/kernel_bench_a5.json`
- `outputs/data/kernel_bench_a5_comparison.json`
- `outputs/data/kernel_bench_a5_comparison.md`

Command used:

```bash
GDN_BENCH_WARMUP=1 GDN_BENCH_ITERS=3 \
python3 benchmarks/kernel/bench_gdn_kernels.py \
  --device npu:0 \
  --n-seq 16 \
  --l-seg 16384 \
  --H-list 16,32,48,64 \
  --stage cumsum,kkt,wy_fast,chunk_h,chunk_o \
  --output-json outputs/data/kernel_bench_a5.json
```

The H=64 run timed out in `wy_fast`, so `kernel_bench_a5.json` contains complete
rows for H=16,32,48 only. H=64 `cumsum` and `kkt` completed during the failed
run but were not written to the JSON because the script saves one row after all
requested stages for that H finish.

