// ============================================================================
// gate_cumsum_kda.cpp — Within-chunk prefix sum of KDA gate vectors
//
// Mathematical operation (per chunk of C tokens, per head h, per key-dim d):
//   g_sum[t, h, d] = Σ_{i=0}^{t} g[i, h, d]    for t = 0 .. valid-1
//
// Input:  g     [total_tokens, HV, K]  float  — raw per-dim gate values
// Output: g_sum [total_tokens, HV, K]  float  — cumulative sums
//
// Difference from GDN chunk_cumsum (kernels/pto/chunk_cumsum.cpp):
//   - GDN: gate shape [T, H], row width = H.
//   - KDA: gate shape [T, HV, K], which is re-viewed as [T, HV*K].
//          Row width = HV*K instead of H.  Everything else is identical.
//
// Why view [T, HV, K] as [T, HV*K]:
//   The NPU MTE2 DMA (TLOAD) requires that the GM row stride equals the
//   column count (contiguous rows).  A per-head slice [T, K] with stride HV*K
//   per row violates this — the DMA ignores the gap and reads a contiguous
//   HV*K-element block instead.  Treating the tensor as [T, HV*K] makes
//   stride == column_count, so TLOAD reads the correct tokens.
//   The prefix sum operates on all HV*K elements of each token row in
//   parallel (SIMD), so the result is identical to an independent per-head,
//   per-K-dim cumsum.
//
// UB memory budget: 2 * ChunkSize * HTC * 4 + HTC * 4
//   where HTC = ((HV*K + 7) / 8) * 8  (Vec alignment)
//   For HV=4,  K=128, C=16: HTC=512,  total≈66 KB  (fits in 256 KB UB)
//   For HV=8,  K=128, C=16: HTC=1024, total≈132 KB (fits)
//   For HV=16, K=128, C=16: HTC=2048, total≈264 KB (tight — use C≤12 or K≤112)
//
// Template parameters (injected by bisheng at compile time):
//   GDN_H  = HV (number of value/gate heads)
//   GDN_D  = K  (key/gate vector dimension per head)
//   GDN_C  = C  (chunk size in tokens)
//
// ─── NPU / PTO recap (see chunk_cumsum.cpp for the full primer) ────────────
//   GM  — off-chip DRAM shared by all AI cores.
//   UB  — on-chip SRAM (~256 KB per core); Vec engine operates here only.
//   Vec — SIMD ALU; processes UB tiles element-wise.
//   MTE2/MTE3 — async DMA engines for GM↔UB transfers.
//   set_flag / wait_flag — explicit pipe synchronisation.
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

#ifndef GDN_H
#define GDN_H 4
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 16
#endif

// UbND alias — identical to chunk_cumsum.cpp.
#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;
#endif

template <int32_t NumHeads, int32_t KDim, int32_t ChunkSize>
AICORE void gate_cumsum_kda_kernel(
    __gm__ float *g_ptr, __gm__ float *g_sum_ptr,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  auto cid       = get_block_idx();
  auto block_num = get_block_num();
  auto vid       = get_subblockid();
  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
  if (vid != 0) return;

  set_mask_norm();
  set_vector_mask(-1, -1);

  // RowWidth: treat g [T, HV, K] as a flat [T, HV*K] 2D tensor.
  // This makes the GM row stride equal to the column count, satisfying the
  // MTE2 contiguity requirement for TLOAD (identical to GDN's NumHeads).
  constexpr int32_t RowWidth = NumHeads * KDim;

  // HTC: RowWidth rounded up to 8-element (32-byte for float) Vec alignment.
  // When KDim % 8 == 0 (always true for standard dims), HTC == RowWidth.
  constexpr int32_t HTC = ((RowWidth + 7) / 8) * 8;

  // UB layout (same structure as chunk_cumsum.cpp, scaled to RowWidth):
  //   [0          .. BlockBytes)           = g input  (ChunkSize × HTC)
  //   [BlockBytes .. 2*BlockBytes)         = g_sum output (ChunkSize × HTC)
  //   [2*BlockBytes .. 2*BlockBytes+HTC*4) = row accumulator (1 × HTC)
  constexpr int32_t BlockBytes = ChunkSize * HTC * static_cast<int32_t>(sizeof(float));
  constexpr int32_t RowBytes   = HTC * static_cast<int32_t>(sizeof(float));
  constexpr int32_t GUbAddr    = 0;
  constexpr int32_t SUbAddr    = BlockBytes;
  constexpr int32_t AccUbAddr  = BlockBytes * 2;

  // Contiguous 2D view of g: g[t, :] = RowWidth elements at t * RowWidth.
  // stride[3] = RowWidth = column count → rows are contiguous in GM.
  // This is the only difference from chunk_cumsum.cpp's
  //   Stride<1, 1, 1, NumHeads, 1>
  // — RowWidth replaces NumHeads.
  using GmShape  = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmStride = Stride<1, 1, 1, RowWidth, 1>;
  using GmFloat  = GlobalTensor<float, GmShape, GmStride>;

  // Row accumulator — pre-assigned, reused across all chunks.
  UbND<float, 1, HTC> acc_ub;
  TASSIGN(acc_ub, AccUbAddr);

  int64_t num_seqs = batch_size;

  // ── Fixed-length sequence path (cu_seqlens == nullptr) ────────────────────
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t total_chunks   = num_seqs * chunks_per_seq;

    for (int64_t gi = static_cast<int64_t>(cid); gi < total_chunks;
         gi += static_cast<int64_t>(block_num)) {
      int64_t seq_idx     = gi / chunks_per_seq;
      int64_t local_chunk = gi % chunks_per_seq;
      int64_t bos         = seq_idx * seq_len;
      int64_t chunk_start = bos + local_chunk * ChunkSize;
      int64_t remaining   = seq_len - local_chunk * ChunkSize;
      int32_t valid = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      // ── MTE2: load g[chunk_start .. +valid, :] from GM → UB ───────────
      // Base pointer: token[chunk_start], offset 0 in the HV*K row.
      // Shape [valid, RowWidth] with contiguous row stride RowWidth.
      {
        GmShape gs; gs.shape[3] = valid; gs.shape[4] = RowWidth;
        GmFloat g_gm(g_ptr + chunk_start * RowWidth, gs);
        UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC, PadValue::Zero>
            g_load(valid, RowWidth);
        TASSIGN(g_load, GUbAddr);
        TLOAD(g_load, g_gm);
        if (valid != ChunkSize || RowWidth != HTC) {
          UbND<float, ChunkSize, HTC, ChunkSize, HTC, PadValue::Zero> g_pad;
          TASSIGN(g_pad, GUbAddr);
          TFILLPAD_INPLACE(g_pad, g_load);
        }
      }
      // MTE2 → Vec sync.
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // ── Vec: prefix sum over token rows (all HV*K elements in parallel) ─
      // Row 0: acc[:] = g[0, :];  g_sum[0, :] = acc[:]
      UbND<float, 1, HTC> g_row_0;
      TASSIGN(g_row_0, GUbAddr);
      TMOV(acc_ub, g_row_0);
      pipe_barrier(PIPE_V);

      UbND<float, 1, HTC> s_row_0;
      TASSIGN(s_row_0, SUbAddr);
      TMOV(s_row_0, acc_ub);
      pipe_barrier(PIPE_V);

      // Rows 1..valid-1: acc[:] += g[i, :];  g_sum[i, :] = acc[:]
      for (int32_t i = 1; i < valid; ++i) {
        UbND<float, 1, HTC> g_row_i;
        TASSIGN(g_row_i, GUbAddr + i * RowBytes);
        TADD(acc_ub, acc_ub, g_row_i);
        pipe_barrier(PIPE_V);

        UbND<float, 1, HTC> s_row_i;
        TASSIGN(s_row_i, SUbAddr + i * RowBytes);
        TMOV(s_row_i, acc_ub);
        pipe_barrier(PIPE_V);
      }

      // Zero-fill rows beyond valid (tail padding for downstream kernels).
      TEXPANDS(acc_ub, 0.0f);
      pipe_barrier(PIPE_V);
      for (int32_t i = valid; i < ChunkSize; ++i) {
        UbND<float, 1, HTC> s_row_i;
        TASSIGN(s_row_i, SUbAddr + i * RowBytes);
        TMOV(s_row_i, acc_ub);
        pipe_barrier(PIPE_V);
      }

      // ── MTE3: store g_sum from UB → GM ───────────────────────────────────
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      {
        GmShape ss; ss.shape[3] = valid; ss.shape[4] = RowWidth;
        GmFloat gs_gm(g_sum_ptr + chunk_start * RowWidth, ss);
        UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC> s_store(valid, RowWidth);
        TASSIGN(s_store, SUbAddr);
        TSTORE(gs_gm, s_store);
      }
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
  }
  // ── Variable-length sequence path (cu_seqlens != nullptr) ─────────────────
  else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos  = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos  = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc   = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t c = 0; c < nc; ++c) {
        if (gi % static_cast<int64_t>(block_num) ==
            static_cast<int64_t>(cid)) {
          int64_t chunk_start = bos + c * ChunkSize;
          int64_t remaining   = slen - c * ChunkSize;
          int32_t valid = static_cast<int32_t>(
              remaining < ChunkSize ? remaining : ChunkSize);

          {
            GmShape gs; gs.shape[3] = valid; gs.shape[4] = RowWidth;
            GmFloat g_gm(g_ptr + chunk_start * RowWidth, gs);
            UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                g_load(valid, RowWidth);
            TASSIGN(g_load, GUbAddr);
            TLOAD(g_load, g_gm);
            if (valid != ChunkSize || RowWidth != HTC) {
              UbND<float, ChunkSize, HTC, ChunkSize, HTC, PadValue::Zero> g_pad;
              TASSIGN(g_pad, GUbAddr);
              TFILLPAD_INPLACE(g_pad, g_load);
            }
          }
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

          UbND<float, 1, HTC> g_row_0;
          TASSIGN(g_row_0, GUbAddr);
          TMOV(acc_ub, g_row_0);
          pipe_barrier(PIPE_V);

          UbND<float, 1, HTC> s_row_0;
          TASSIGN(s_row_0, SUbAddr);
          TMOV(s_row_0, acc_ub);
          pipe_barrier(PIPE_V);

          for (int32_t i = 1; i < valid; ++i) {
            UbND<float, 1, HTC> g_row_i;
            TASSIGN(g_row_i, GUbAddr + i * RowBytes);
            TADD(acc_ub, acc_ub, g_row_i);
            pipe_barrier(PIPE_V);

            UbND<float, 1, HTC> s_row_i;
            TASSIGN(s_row_i, SUbAddr + i * RowBytes);
            TMOV(s_row_i, acc_ub);
            pipe_barrier(PIPE_V);
          }

          TEXPANDS(acc_ub, 0.0f);
          pipe_barrier(PIPE_V);
          for (int32_t i = valid; i < ChunkSize; ++i) {
            UbND<float, 1, HTC> s_row_i;
            TASSIGN(s_row_i, SUbAddr + i * RowBytes);
            TMOV(s_row_i, acc_ub);
            pipe_barrier(PIPE_V);
          }

          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

          {
            GmShape ss; ss.shape[3] = valid; ss.shape[4] = RowWidth;
            GmFloat gs_gm(g_sum_ptr + chunk_start * RowWidth, ss);
            UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC> s_store(valid, RowWidth);
            TASSIGN(s_store, SUbAddr);
            TSTORE(gs_gm, s_store);
          }
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        gi++;
      }
    }
  }
#endif
}

// ── Device-side entry point ────────────────────────────────────────────────
extern "C" __global__ AICORE void launch_gate_cumsum_kda(
    __gm__ uint8_t *g_ptr, __gm__ uint8_t *g_sum_ptr,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  gate_cumsum_kda_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ float *>(g_ptr),
      reinterpret_cast<__gm__ float *>(g_sum_ptr),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, ffts_addr);
}

// ── Host-side launcher (called from Python via ctypes) ────────────────────
extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *g_ptr, uint8_t *g_sum_ptr, uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_gate_cumsum_kda<<<block_dim, nullptr, stream>>>(
      g_ptr, g_sum_ptr, cu_seqlens, batch_size, seq_len, fftsAddr);
}
