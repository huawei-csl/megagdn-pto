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
// Difference from GDN chunk_cumsum (kernels/pto/chunk_cumsum.cpp):
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

// Cast g fp16 → fp32 in UB (accumulate in fp32 to match the reference).
// Reads ChunkSize×CTC half values at halfAddr, writes fp32 at ubAddr.
template <int32_t ChunkSize, int32_t CTC>
AICORE void cast_g_fp16_to_fp32(int32_t halfAddr, int32_t ubAddr)
{
  UbND<half, ChunkSize, CTC> g_h;
  TASSIGN(g_h, halfAddr);
  UbND<float, ChunkSize, CTC> g_f;
  TASSIGN(g_f, ubAddr);
  TCVT(g_f, g_h, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
}

template <int32_t KDim, int32_t ChunkSize>
AICORE void gate_cumsum_kda_kernel(
    __gm__ half *g_ptr, __gm__ float *g_sum_ptr,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int32_t num_heads, uint64_t ffts_addr)
{
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
  if (vid != 0)
    return;

  set_mask_norm();
  set_vector_mask(-1, -1);

  // Flat 2D view of g: [total_tokens, HV*K] (row-major, contiguous).
  // Head count is a runtime argument, so RowWidth is runtime too.
  const int32_t RowWidth = num_heads * KDim;

  // ColTile: number of HV*K columns processed in one UB-resident slice.
  // It is derived from the *compile-time* per-head dim KDim (not RowWidth), so
  // the UB tile buffers below stay statically sized regardless of head count —
  // 128 is safe for ChunkSize up to 128 (per-tile UB ≈ 66 KB).  KDim is a
  // multiple of 128 in practice, so ColTile=128 divides KDim and therefore
  // divides RowWidth = num_heads*KDim for any head count.
  constexpr int32_t ColTileTarget = 128;
  constexpr int32_t ColTile = (KDim < ColTileTarget) ? KDim : ColTileTarget;
  constexpr int32_t CTC = ((ColTile + 7) / 8) * 8; // 8-elem alignment
  static_assert(KDim % ColTile == 0,
                "KDim must be a multiple of ColTile (128).");
  // Number of column tiles a row spans — runtime, since RowWidth is runtime.
  const int32_t NumColTiles = RowWidth / ColTile;

  // Per-tile UB layout:
  //   [0           .. HalfBlockBytes)        = g input,  fp16 staging (ChunkSize × CTC)
  //   [GUbAddr     .. +BlockBytes)           = g input,  fp32       (ChunkSize × CTC)
  //   [SUbAddr     .. +BlockBytes)           = g_sum out, fp32      (ChunkSize × CTC)
  //   [AccUbAddr   .. +CTC*4)                = row accumulator fp32 (1 × CTC)
  constexpr int32_t HalfBlockBytes = ChunkSize * CTC * static_cast<int32_t>(sizeof(half));
  constexpr int32_t BlockBytes = ChunkSize * CTC * static_cast<int32_t>(sizeof(float));
  constexpr int32_t RowBytes = CTC * static_cast<int32_t>(sizeof(float));
  constexpr int32_t GHalfAddr = 0;
  constexpr int32_t GUbAddr = HalfBlockBytes;
  constexpr int32_t SUbAddr = GUbAddr + BlockBytes;
  constexpr int32_t AccUbAddr = SUbAddr + BlockBytes;

  // GM views: rows stride RowWidth apart, ColTile-wide window per load.
  // Strided 2D loads (row_stride > col_count) are supported — same pattern as
  // chunk_h.cpp's per-head BSND loads at e.g. lines 599-604.
  using GmShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using GmHalf = GlobalTensor<half, GmShape, GmStride>;   // input g (fp16)
  using GmFloat = GlobalTensor<float, GmShape, GmStride>; // output g_sum (fp32)
  // Runtime row stride (RowWidth) shared by every g / g_sum GM view below.
  GmStride row_stride(RowWidth);

  // Row accumulator — pre-assigned, re-initialised at each column tile.
  UbND<float, 1, CTC> acc_ub;
  TASSIGN(acc_ub, AccUbAddr);

  int64_t num_seqs = batch_size;

  // ── Fixed-length sequence path (cu_seqlens == nullptr) ────────────────────
  if (cu_seqlens == nullptr)
  {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t total_chunks = num_seqs * chunks_per_seq;

    for (int64_t gi = static_cast<int64_t>(cid); gi < total_chunks;
         gi += static_cast<int64_t>(block_num))
    {
      int64_t seq_idx = gi / chunks_per_seq;
      int64_t local_chunk = gi % chunks_per_seq;
      int64_t bos = seq_idx * seq_len;
      int64_t chunk_start = bos + local_chunk * ChunkSize;
      int64_t remaining = seq_len - local_chunk * ChunkSize;
      int32_t valid = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      for (int32_t ct = 0; ct < NumColTiles; ++ct)
      {
        int32_t col_off = ct * ColTile;

        // ── MTE2: load g[chunk_start..+valid, col_off..+ColTile] (fp16) ───
        {
          GmShape gs;
          gs.shape[3] = valid;
          gs.shape[4] = ColTile;
          GmHalf g_gm(g_ptr + chunk_start * RowWidth + col_off, gs, row_stride);
          UbND<half, ChunkSize, CTC, DYNAMIC, DYNAMIC, PadValue::Zero>
              g_load(valid, ColTile);
          TASSIGN(g_load, GHalfAddr);
          TLOAD(g_load, g_gm);
          if (valid != ChunkSize || ColTile != CTC)
          {
            UbND<half, ChunkSize, CTC, ChunkSize, CTC, PadValue::Zero> g_pad;
            TASSIGN(g_pad, GHalfAddr);
            TFILLPAD_INPLACE(g_pad, g_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        cast_g_fp16_to_fp32<ChunkSize, CTC>(GHalfAddr, GUbAddr);

        // ── Vec: prefix sum (all ColTile cols in parallel) ───────────────
        // Row 0: acc = g[0]; g_sum[0] = acc
        UbND<float, 1, CTC> g_row_0;
        TASSIGN(g_row_0, GUbAddr);
        TMOV(acc_ub, g_row_0);
        pipe_barrier(PIPE_V);

        UbND<float, 1, CTC> s_row_0;
        TASSIGN(s_row_0, SUbAddr);
        TMOV(s_row_0, acc_ub);
        pipe_barrier(PIPE_V);

        // Rows 1..valid-1: acc += g[i]; g_sum[i] = acc
        for (int32_t i = 1; i < valid; ++i)
        {
          UbND<float, 1, CTC> g_row_i;
          TASSIGN(g_row_i, GUbAddr + i * RowBytes);
          TADD(acc_ub, acc_ub, g_row_i);
          pipe_barrier(PIPE_V);

          UbND<float, 1, CTC> s_row_i;
          TASSIGN(s_row_i, SUbAddr + i * RowBytes);
          TMOV(s_row_i, acc_ub);
          pipe_barrier(PIPE_V);
        }

        // ── V → MTE2 sync ───────────────────────────────────────────────
        // Ensures the next iteration's TLOAD into UB[GUbAddr] does not race
        // ahead of this iteration's Vec reads of UB[GUbAddr].  Without this,
        // MTE2 (which has no other wait in this loop) can overwrite the input
        // buffer while Vec is still computing — manifests as iter N's stored
        // slot containing iter N+1's cumsum.
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // ── MTE3: store g_sum (rows 0..valid-1 only) ─────────────────────
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        {
          GmShape ss;
          ss.shape[3] = valid;
          ss.shape[4] = ColTile;
          GmFloat gs_gm(g_sum_ptr + chunk_start * RowWidth + col_off, ss, row_stride);
          UbND<float, ChunkSize, CTC, DYNAMIC, DYNAMIC>
              s_store(valid, ColTile);
          TASSIGN(s_store, SUbAddr);
          TSTORE(gs_gm, s_store);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }
    }
  }
  // ── Variable-length sequence path (cu_seqlens != nullptr) ─────────────────
  else
  {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si)
    {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t c = 0; c < nc; ++c)
      {
        if (gi % static_cast<int64_t>(block_num) ==
            static_cast<int64_t>(cid))
        {
          int64_t chunk_start = bos + c * ChunkSize;
          int64_t remaining = slen - c * ChunkSize;
          int32_t valid = static_cast<int32_t>(
              remaining < ChunkSize ? remaining : ChunkSize);

          for (int32_t ct = 0; ct < NumColTiles; ++ct)
          {
            int32_t col_off = ct * ColTile;

            {
              GmShape gs;
              gs.shape[3] = valid;
              gs.shape[4] = ColTile;
              GmHalf g_gm(g_ptr + chunk_start * RowWidth + col_off, gs, row_stride);
              UbND<half, ChunkSize, CTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                  g_load(valid, ColTile);
              TASSIGN(g_load, GHalfAddr);
              TLOAD(g_load, g_gm);
              if (valid != ChunkSize || ColTile != CTC)
              {
                UbND<half, ChunkSize, CTC, ChunkSize, CTC, PadValue::Zero> g_pad;
                TASSIGN(g_pad, GHalfAddr);
                TFILLPAD_INPLACE(g_pad, g_load);
              }
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            cast_g_fp16_to_fp32<ChunkSize, CTC>(GHalfAddr, GUbAddr);

            UbND<float, 1, CTC> g_row_0;
            TASSIGN(g_row_0, GUbAddr);
            TMOV(acc_ub, g_row_0);
            pipe_barrier(PIPE_V);

            UbND<float, 1, CTC> s_row_0;
            TASSIGN(s_row_0, SUbAddr);
            TMOV(s_row_0, acc_ub);
            pipe_barrier(PIPE_V);

            for (int32_t i = 1; i < valid; ++i)
            {
              UbND<float, 1, CTC> g_row_i;
              TASSIGN(g_row_i, GUbAddr + i * RowBytes);
              TADD(acc_ub, acc_ub, g_row_i);
              pipe_barrier(PIPE_V);

              UbND<float, 1, CTC> s_row_i;
              TASSIGN(s_row_i, SUbAddr + i * RowBytes);
              TMOV(s_row_i, acc_ub);
              pipe_barrier(PIPE_V);
            }

            // V → MTE2: prevent next iter's TLOAD from clobbering UB[GUbAddr]
            // before Vec has finished reading it.  See fixed-len path above.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            {
              GmShape ss;
              ss.shape[3] = valid;
              ss.shape[4] = ColTile;
              GmFloat gs_gm(g_sum_ptr + chunk_start * RowWidth + col_off, ss, row_stride);
              UbND<float, ChunkSize, CTC, DYNAMIC, DYNAMIC>
                  s_store(valid, ColTile);
              TASSIGN(s_store, SUbAddr);
              TSTORE(gs_gm, s_store);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          }
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
    int32_t num_heads, uint64_t ffts_addr)
{
  gate_cumsum_kda_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(g_ptr),
      reinterpret_cast<__gm__ float *>(g_sum_ptr),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, num_heads, ffts_addr);
}

// ── Host-side launcher (called from Python via ctypes) ────────────────────
extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *g_ptr, uint8_t *g_sum_ptr, uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, uint32_t num_heads)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_gate_cumsum_kda<<<block_dim, nullptr, stream>>>(
      g_ptr, g_sum_ptr, cu_seqlens, batch_size, seq_len,
      static_cast<int32_t>(num_heads), fftsAddr);
}
