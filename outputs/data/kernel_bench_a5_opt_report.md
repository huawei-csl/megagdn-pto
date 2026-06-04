# MegaGDN A5 Optimization Report

## Context

The current A5 port in `kernels/pto_a5` is a correctness/compilation port with
some A5 API fixes, but most high-volume Cube-Vector exchanges in `chunk_h` and
`chunk_o` still use GM workspace round trips.

Baseline comparison is in:

- `outputs/data/kernel_bench.json` for the saved A2 PTO timings.
- `outputs/data/kernel_bench_a5_comparison.md` for the current A5 PTO-only
  timing comparison.

The current A5 measured kernels are not yet 3x faster than A2. The main reason
is that the expensive GM workspace handoffs remain in the hot loop.

## Current Hotspots

### `chunk_h`

Per chunk/head, the current `chunk_h` hot loop performs these GM workspace
handoffs:

- Cube writes `WS = W @ S` from L0C to GM workspace (`WS_WS`), then Vec loads it
  to compute `V_new = U - WS`.
- Vec writes `K_scaled` to GM workspace (`WS_K`), then Cube loads it for
  `K_scaled^T @ V_new`.
- Vec writes recurrent state `S` to GM workspace (`WS_S`), then Cube loads it for
  `W @ S`.
- Cube writes `KV = K_scaled^T @ V_new` to GM workspace (`WS_KV`), then Vec
  loads it to update `S`.

At `C=D=128`, each full CxD or DxD tile is 32 KiB in fp16. These handoffs add
multiple on-chip-to-GM-to-on-chip trips per chunk, even though A5 supports direct
L0C->UB and UB->L1 exchange.

### `chunk_o`

Per chunk/head, `chunk_o` has similar GM workspace traffic:

- Cube writes QK and QS from L0C to GM workspace, then Vec loads them.
- Vec writes QK_gated to GM workspace, then Cube loads it for GEMM3.
- Cube writes QKV to GM workspace, then Vec loads it for the final output add.

The expected A5 speedup requires replacing these GM round trips with direct
`TMOV` / `TINSERT` handoffs.

## Optimization Candidates Considered

### Candidate 1: `chunk_h` direct C2V for WS and KV

Prototype:

- Add opt-in macro `GDN_A5_DIRECT_CHUNK_H_C2V`.
- Replace Cube `TSTORE(WS_WS)` with direct `TMOV` from `ws_l0` to a Vec UB tile.
- Replace Cube `TSTORE(WS_KV)` with direct `TMOV` from `kv_l0` to a Vec UB tile.
- Keep original GM path as default until device validation passes.

Status:

- The variant compiles with:

```bash
PTO_DYNAMIC_EXTRA_FLAGS='-DGDN_A5_DIRECT_CHUNK_H_C2V=1'
```

- On-device validation could not be run because `npu:0` cannot currently be
  opened after the earlier H=64 `wy_fast` AICore timeout.
- The macro is disabled by default (`0`) so the checked-in path remains the
  previously validated A5 correctness port.

### Candidate 2: `chunk_h` direct V2C for K/S

Planned next:

- Convert `K_scaled` and `S` Vec tiles to L1-compatible layout.
- Use `TINSERT` / `copy_ubuf_to_cbuf` into L1.
- Use a conservative ready/free protocol:
  - Vec waits for Cube free before overwriting L1 handoff slot.
  - Cube waits for both Vec subblocks (`flag` and `flag + 16`) before `TMOV` to
    L0.
  - Cube frees the slot only after MTE1 has captured the L1 tile.

Not attempted because real-device validation is currently unavailable.

### Candidate 3: `chunk_o` direct C2V for QK/QS/QKV

Planned next:

- Start with the QKV handoff because it is consumed directly by Vec for final
  output assembly.
- Then try QK/QS direct handoffs.
- Use separate UB regions to avoid overlap with gating coefficient scratch.

Not attempted because real-device validation is currently unavailable.

### Candidate 4: `chunk_o` direct V2C for QK_gated

Planned next:

- Convert Vec QK_gated tile to NZ layout.
- Use `TINSERT` into L1 for Cube GEMM3.
- Follow the verified `add_matmul_v2c` single-slot ownership pattern.

Not attempted because real-device validation is currently unavailable.

### Candidate 5: A5 manual-pattern reuse

Relevant patterns inspected:

- `flash_atten` has explicit modes for all-GM, all-UB, and mixed direct C/V
  paths. It also uses ready/free flag spacing and FIFO depth tuning.
- `gemm_ar` uses L1 panel caching and L0 ping-ponging to reduce repeated GM->L1
  traffic.
- `engram_simt` documents when SIMT/D-cache paths are useful for memory-bound
  scalar/gather work. This is less immediately applicable to the heavy Cube
  matmul sections of `chunk_h`/`chunk_o`.

## Device Blocker

The earlier H=64 large-shape `wy_fast` benchmark triggered an AICore timeout.
After that, new torch-npu processes fail at `torch.npu.set_device("npu:0")` with
runtime error `507033` / `TsdOpen failed`.

There are no leftover user Python processes holding the device. A runtime or
device reset is needed before further on-device correctness/benchmark work.

## Current Best Performance

From `outputs/data/kernel_bench_a5_comparison.md`, completed A5 rows are:

| H | stage | A2 ms | A5 ms | speedup |
|---:|---|---:|---:|---:|
| 16 | kkt | 4.668 | 5.441 | 0.86x |
| 16 | wy_fast | 6.965 | 6.770 | 1.03x |
| 16 | chunk_h | 10.121 | 12.796 | 0.79x |
| 16 | chunk_o | 11.120 | 13.919 | 0.80x |
| 32 | kkt | 9.417 | 10.911 | 0.86x |
| 32 | wy_fast | 13.090 | 13.078 | 1.00x |
| 32 | chunk_h | 20.459 | 25.934 | 0.79x |
| 32 | chunk_o | 21.959 | 27.920 | 0.79x |
| 48 | kkt | 13.680 | 16.394 | 0.83x |
| 48 | wy_fast | 20.871 | 19.669 | 1.06x |
| 48 | chunk_h | 30.183 | 39.470 | 0.76x |
| 48 | chunk_o | 33.228 | 41.754 | 0.80x |

No optimized variant has been validated on device yet after the timeout.

## Next Steps After Device Reset

1. Run quick validation for the compiled `chunk_h` direct C2V candidate:

```bash
PTO_DYNAMIC_EXTRA_FLAGS='-DGDN_A5_DIRECT_CHUNK_H_C2V=1' \
MEGAGDN_PTO_ARCH=a5 \
python3 tests/test_single_kernels.py --device npu:0 --quick --H-list 16 --stage chunk_h
```

2. If correct, benchmark just `chunk_h`:

```bash
GDN_BENCH_WARMUP=1 GDN_BENCH_ITERS=3 \
PTO_DYNAMIC_EXTRA_FLAGS='-DGDN_A5_DIRECT_CHUNK_H_C2V=1' \
MEGAGDN_PTO_ARCH=a5 \
python3 benchmarks/kernel/bench_gdn_kernels.py \
  --device npu:0 --n-seq 16 --l-seg 16384 --H-list 16,32,48 \
  --stage chunk_h --output-json outputs/data/kernel_bench_a5_opt_chunk_h_c2v.json
```

3. Only after that, proceed to V2C direct handoffs and `chunk_o` candidates.

## Lessons So Far

- A5 direct C/V exchange is required for major speedups. A mechanical
  DAV_2201-to-DAV_3510 compile port is not enough.
- `chunk_h` and `chunk_o` are dominated by GM workspace ping-pong between Cube
  and Vec; these are precisely the paths A5 can eliminate.
- Experimental A5 paths must remain opt-in until validated. Multi-wave C/V bugs
  can pass compile and still corrupt or hang at runtime.
- Avoid H=64 large-shape stress tests until smaller H variants are stable.

