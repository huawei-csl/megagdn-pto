// ============================================================================
// gate_cumsum_kda.cpp — Within-chunk prefix sum of KDA gate vectors
//
// Mathematical operation (per chunk of C tokens, per head h, per key-dim d):
//   g_sum[t, h, d] = Σ_{i=0}^{t} g[i, h, d]    for t = 0 .. valid-1
//
// Input:  g     [total_tokens, HV, K]  half    — raw per-dim gate values
// Output: g_sum [total_tokens, HV, K]  float32 — cumulative sums
//
// The prefix sum is accumulated and stored in fp32 (like GDN chunk_cumsum's
// fp32 gate): the per-128-chunk cumulative sum reaches ~-64 and fp16's ~0.06
// step there corrupts exp(g_cs) downstream.  Input g stays fp16 (model dtype)
// and is cast up before accumulating.
//
// Difference from GDN chunk_cumsum (kernels/pto/chunk_cumsum.h):
//   - GDN: gate shape [T, H], row width = H (~16-64).
//   - KDA: gate shape [T, HV, K], re-viewed as [T, HV*K].  Row width = HV*K is
//          ~512-2048, an order of magnitude larger.  A single chunk no longer
//          fits in UB, so we tile along the column (HV*K) dimension.
//
// Why tile along columns:
//   The prefix sum is along the time/row axis; each of the HV*K columns is an
//   independent cumulative series, so we can process column slices in any
//   order and reuse the same UB region for each slice.  Strided 2D DMA
//   (row_stride > col_count) is supported — see chunk_h.cpp's BSND loads.
//
// UB memory budget (per column tile): 2*ChunkSize*CTC*2 + CTC*2
//   With ColTile=128: 66 KB for C=128, 33 KB for C=64 (fits 256 KB UB).
//   Number of column tiles per chunk = RowWidth / ColTile
//   (e.g. HV=4,K=128 → 4 tiles; HV=8 → 8 tiles).
//
// Compile-time template parameters (injected by bisheng):
//   GDN_D  = K  (key/gate vector dimension per head)
//   GDN_C  = C  (chunk size in tokens)
// Runtime argument:
//   num_heads = HV (number of value/gate heads) — only affects loop bounds and
//   GM strides, so it need not be a compile-time constant.
//
// ─── NPU / PTO recap (see chunk_cumsum.h for the full primer) ─────────────
//   GM  — off-chip DRAM shared by all AI cores.
//   UB  — on-chip SRAM (~256 KB per core); Vec engine operates here only.
//   Vec — SIMD ALU; processes UB tiles element-wise.
//   MTE2/MTE3 — async DMA engines for GM↔UB transfers.
//   set_flag / wait_flag — explicit pipe synchronisation.
// ============================================================================

#include "gate_cumsum_kda.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

// ── Device-side entry point ────────────────────────────────────────────────
extern "C" __global__ AICORE void launch_gate_cumsum_kda(
    __gm__ uint8_t *g_ptr, __gm__ uint8_t *g_sum_ptr,
    __gm__ uint8_t *cu_seqlens, int64_t batch_size, int64_t seq_len,
    int32_t num_heads, uint64_t ffts_addr) {
  gate_cumsum_kda_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(g_ptr),
      reinterpret_cast<__gm__ float *>(g_sum_ptr),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens), batch_size, seq_len,
      num_heads, ffts_addr);
}

// ── Host-side launcher (called from Python via ctypes) ────────────────────
extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *g_ptr,
                            uint8_t *g_sum_ptr, uint8_t *cu_seqlens,
                            int64_t batch_size, int64_t seq_len,
                            uint32_t num_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_gate_cumsum_kda<<<block_dim, nullptr, stream>>>(
      g_ptr, g_sum_ptr, cu_seqlens, batch_size, seq_len,
      static_cast<int32_t>(num_heads), fftsAddr);
}
