# MegaGDN PTO A5 Port Status

## Summary

The A2-oriented PTO kernels were copied to `kernels/pto_a5` and mechanically
ported to compile under A5 / DAV_3510 with:

```bash
export MEGAGDN_PTO_ARCH=a5
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

The Python build path now selects `kernels/pto_a5` and compiles with
`--cce-aicore-arch=dav-c310`.

## Correctness

Quick single-kernel tests passed on real `npu:0` for `H=16,32,48,64`:

```bash
python3 tests/test_single_kernels.py --device npu:0 --quick --H-list 16,32,48,64
```

Stages covered:

- `chunk_cumsum`
- `scaled_dot_kkt`
- `solve_tril` via A5 torch fallback
- `wy_fast`
- `chunk_h`
- `chunk_o`

## Performance Artifacts

Generated:

- `outputs/data/kernel_bench_a5.json`
- `outputs/data/kernel_bench_a5_comparison.json`
- `outputs/data/kernel_bench_a5_comparison.md`

Command used for the completed large-shape run:

```bash
GDN_BENCH_WARMUP=1 GDN_BENCH_ITERS=3 \
MEGAGDN_PTO_ARCH=a5 \
python3 benchmarks/kernel/bench_gdn_kernels.py \
  --device npu:0 \
  --n-seq 16 \
  --l-seg 16384 \
  --H-list 16,32,48,64 \
  --stage cumsum,kkt,wy_fast,chunk_h,chunk_o \
  --output-json outputs/data/kernel_bench_a5.json
```

The run completed H=16,32,48 and wrote those rows. H=64 timed out during
`wy_fast`, so H=64 is intentionally omitted from the comparison JSON.

## Known Limitations

- `tri_inverse` / PTO `solve_tril` is not fully ported. The A5 copy compiles
  after layout fixes but produces NaNs. `solve_tril` uses a torch fallback for
  A5 correctness only, and this fallback is not counted as a PTO performance
  result.
- `mega_kernel` is not validated on A5 yet because it depends on PTO
  `tri_inverse`.
- The current A5 kernels still mostly use the original GM workspace exchange
  patterns. A deeper optimization pass should replace the remaining Cube-Vector
  GM handoffs in `wy_fast`, `chunk_h`, `chunk_o`, and `mega_kernel` with direct
  A5 `TMOV` / `TINSERT` paths.
- After an H=64 timeout, the NPU runtime reported device reopen failures in a
  later e2e attempt. A runtime/device reset is recommended before additional
  long A5 benchmark runs.

