// ============================================================================
// wy_kda.cpp — WY representation for KDA chunk recurrence (per-dim gate)
//
// Computes the two auxiliary tensors U and W per chunk:
//   U[r, d_v] = sum_c INV[r, c] * beta[c] * v[c, d_v]
//   W[r, d_k] = sum_c INV[r, c] * beta[c] * exp(g_cs[c, d_k]) * k[c, d_k]
//
// where INV = (I + L)^{-1} is full lower-triangular, beta is post-sigmoid
// scalar per token, and g_cs is per-DIMENSION cumulative gate sum (the KDA
// difference from GDN). Math reference: tests/test_kda_single_kernels.py
// `ref_wy_kda` (lines 154-200).
//
// Key insight: both U and W reuse a single beta-scaled A2 matrix:
//   A2[r, c]    = INV[r, c] * beta[c]            (column-scale by beta)
//   K_eff[c, d] = k[c, d] * exp(g_cs[c, d])      (element-wise per-dim)
//   U = A2 @ V                                    (V from BSND, fp16)
//   W = A2 @ K_eff                                (K_eff from workspace, fp16)
//
// Cube loads A2 once into L1 and reuses it across both GEMMs — saves one L1
// load and one workspace pass compared to GDN wy_fast.cpp (which needs two
// distinct reweighted-A matrices because GDN's scalar g could be folded into
// a column scale of A). In KDA g is per-dim so we must precompute K_eff
// instead.
//
//  Vec core (both sub-blocks active; each owns one HalfChunk-row stripe):
//    For each (seq, chunk, head):
//      Phase 1: Load beta [1,C], INV [HC,C] BSND -> build A2 = INV * beta_2d
//               -> fp16 -> store stripe to ws_a2 [block_dim, C, C]
//      Phase 2: Load k [HC,K], g_cs [HC,K] head-major -> K_eff = k*exp(g_cs)
//               -> fp16 -> store stripe to ws_keff [block_dim, C, K]
//      Signal Cube via cross-core flags 1 and 2.
//
//  Cube core:
//    For each (seq, chunk, head):
//      Load V from BSND -> v_l1
//      Wait flag 1 (ws_a2 ready) -> TLOAD a2_l1
//      GEMM U = A2 @ V -> u_l0 -> store BSND -> signal flag 3 (ws_a2 free)
//      Wait flag 2 (ws_keff ready) -> TLOAD keff_l1
//      GEMM W = A2 @ K_eff (a2_l1 still in L1!) -> w_l0 -> store BSND
//      Signal flag 4 (ws_keff free)
//
// FFTS flags (single-buffered with explicit free signals):
//   10 : V→C reduce (both vids must signal) "ws_a2 ready"
//   11 : V→C reduce "ws_keff ready"
//   12 : C→V broadcast "ws_a2 free"
//   13 : C→V broadcast "ws_keff free"
// Sync_all uses 6-9 (matches kkt_kda).  We pick data-flow IDs ≥ 10 so they
// don't collide with kkt_kda / tri_inverse / gate_cumsum_kda (which all use
// IDs 0-5 for data flow + 6-9 for sync_all).
//
// Layout (matches Python kda_kernel_libs convention; v cast to fp16 wrap-side):
//   k       head-major [HV, T, K]    fp32
//   v       BSND       [B, T, HV, V] fp16   (V == K in current setup)
//   beta    head-major [HV, T]       fp32
//   g_cs    head-major [HV, T, K]    fp32
//   A (INV) BSND       [B, T, HV, C] fp32
//   ws_a2              [bd, C, C]    fp16
//   ws_keff            [bd, C, K]    fp16
//   u_out   BSND       [B, T, HV, V] fp32
//   w_out   BSND       [B, T, HV, K] fp32
//
// Template parameters: GDN_H = HV, GDN_D = K (= V_DIM here), GDN_C = C.
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include <type_traits>
using namespace pto;

#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

#ifdef __CCE_AICORE__

// Global barrier across ALL AI cores: drains any leftover FFTS state from
// prior kernel launches and balances the trailing data-flow signals on
// exit.  Uses four reserved FFTS flag IDs (6, 7, 8, 9) — distinct from the
// data-flow flags 1-4 used by this kernel.  Pattern matches kkt_kda.cpp.
AICORE inline void sync_all()
{
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_C220_CUBE__)
    ffts_cross_core_sync(PIPE_FIX, 1 | (0 << 4) | (7 << 8));
    wait_flag_dev(7);
    ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (8 << 8));
    wait_flag_dev(9);
#elif defined(__DAV_C220_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, 1 | (0 << 4) | (6 << 8));
    wait_flag_dev(6);
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (9 << 8));
    wait_flag_dev(8);
#endif
    pipe_barrier(PIPE_ALL);
}

namespace {

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512,
                              pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0B = pto::Tile<pto::TileType::Right, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::ColMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                           pto::BLayout::ColMajor, pto::DYNAMIC,
                           pto::DYNAMIC, pto::SLayout::RowMajor, 512,
                           pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                             pto::BLayout::RowMajor, pto::DYNAMIC,
                             pto::DYNAMIC, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols>
using DynAccTile = pto::TileAcc<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

// Local K-sliced matmul helper:  C = A @ B (clear=true) on the Cube engine.
// Copied verbatim from wy_fast.cpp gemm_v0 — handles the L1 -> L0 movement
// and accumulator init/tail logic.  See wy_fast.cpp lines 149-260 for the
// commented version.
template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail, bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void
gemm_v0(std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                           TileMatL1<T1, M, K, validM, validK>> &A,
        std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                           TileMatL1<T1, K, N, validK, validN>> &B,
        pto::TileAcc<T2, M, N, validM, validN> &C, bool clear)
{
  constexpr uint32_t kL0Size = 128;
  const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; ++kL0Idx) {
    const bool initflag = clear && (kL0Idx == 0);
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
      TileMatL0A<T1, M, K_tail, M, K_tail> l0a;
      TileMatL0B<T1, K_tail, N, K_tail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * K_tail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * K_tail);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * K_tail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * K_tail, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }
    } else {
      TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
      TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      set_flag(PIPE_FIX, PIPE_M, war_event_id);
      wait_flag(PIPE_FIX, PIPE_M, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * kL0Size);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * kL0Size, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    }
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);

  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

} // namespace

#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void wy_kda_kernel(
    __gm__ half *K_handle, __gm__ half *V_handle,
    __gm__ half *Beta_handle, __gm__ half *G_handle,
    __gm__ half *A_handle,
    __gm__ half *workspace_a2_handle, __gm__ half *workspace_keff_handle,
    __gm__ half *U_handle, __gm__ half *W_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  // Per-chunk math (matches ref_wy_kda):
  //   A2[r, c]    = INV[r, c] * beta[c]
  //   K_eff[c, d] = k[c, d] * exp(g_cs[c, d])
  //   U = A2 @ V,   W = A2 @ K_eff
  constexpr int32_t HalfChunk = ChunkSize / 2;
  // The matmul inner-dimension here is ChunkSize (not HiddenSize), so KTail
  // must be computed from ChunkSize.  This differs from wy_fast.cpp where
  // HiddenSize and ChunkSize happen to coincide (both 128) so the wrong
  // expression silently worked.
  constexpr uint32_t KTail =
      (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

  constexpr int32_t H = NumHeads;
  // In KDA the keys arrive pre-expanded to HV heads (no GQA at this stage),
  // so K, V, U, W all use stride H * HiddenSize.
  constexpr int32_t BSND_STRIDE = H * HiddenSize;
  constexpr int32_t BSND_C_STRIDE = H * ChunkSize;

  // ── UB address plan ───────────────────────────────────────────────────────
  // Phase 1 (A2 = INV * beta_2d) and Phase 2 (K_eff = k * exp(g_cs)) run
  // sequentially, so their tiles overlap in UB.  Peak usage ≈ 116 KB
  // per Vec sub-block (well under the 192 KB budget).
  //
  // Phase 1:
  constexpr int32_t BetaUbAddr     = 0;
  constexpr int32_t BetaRUbAddr    = 512;
  constexpr int32_t Beta2dUbAddr   = 1024;
  constexpr int32_t InvUbAddr      = Beta2dUbAddr + HalfChunk * ChunkSize * 4;
  constexpr int32_t A2UbAddr       = InvUbAddr + HalfChunk * ChunkSize * 4;
  constexpr int32_t A2HalfUbAddr   = A2UbAddr + HalfChunk * ChunkSize * 4;
  // Phase 2 (overlaps with Phase 1 — addresses reused):
  constexpr int32_t KUbAddr        = 0;
  constexpr int32_t GcsUbAddr      = KUbAddr + HalfChunk * HiddenSize * 4;
  constexpr int32_t KeffUbAddr     = GcsUbAddr + HalfChunk * HiddenSize * 4;
  constexpr int32_t KeffHalfUbAddr = KeffUbAddr + HalfChunk * HiddenSize * 4;

  // ── Workspace element counts (fp16) ───────────────────────────────────────
  constexpr int32_t WsA2Size   = ChunkSize * ChunkSize;
  constexpr int32_t WsKeffSize = ChunkSize * HiddenSize;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  // ── UB tile declarations (TASSIGNed to fixed addresses) ──────────────────
  TileUbDataND<half, 1, ChunkSize, 1, ChunkSize,
               pto::PadValue::Zero> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> beta_r_ub;
  TASSIGN(beta_r_ub, BetaRUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> beta_2d_ub;
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize, pto::PadValue::Zero> inv_ub;
  TASSIGN(inv_ub, InvUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> a2_ub;
  TASSIGN(a2_ub, A2UbAddr);
  TileUbDataND<half, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> a2_ub_half;
  TASSIGN(a2_ub_half, A2HalfUbAddr);

  TileUbDataND<float, HalfChunk, HiddenSize,
               HalfChunk, HiddenSize, pto::PadValue::Zero> k_ub;
  TASSIGN(k_ub, KUbAddr);
  TileUbDataND<float, HalfChunk, HiddenSize,
               HalfChunk, HiddenSize, pto::PadValue::Zero> gcs_ub;
  TASSIGN(gcs_ub, GcsUbAddr);
  TileUbDataND<float, HalfChunk, HiddenSize,
               HalfChunk, HiddenSize> keff_ub;
  TASSIGN(keff_ub, KeffUbAddr);
  TileUbDataND<half, HalfChunk, HiddenSize,
               HalfChunk, HiddenSize> keff_ub_half;
  TASSIGN(keff_ub_half, KeffHalfUbAddr);

  // ── L1 tiles (Cube only) ──────────────────────────────────────────────────
  // v_l1 at 0, a2_l1 at 32K, keff_l1 at 64K — fits comfortably in 256 KB L1.
  TileMatL1<half, ChunkSize, HiddenSize,
            ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 0);
  TileMatL1<half, ChunkSize, ChunkSize,
            ChunkSize, ChunkSize> a2_l1;
  TASSIGN(a2_l1, 32768);
  TileMatL1<half, ChunkSize, HiddenSize,
            ChunkSize, HiddenSize> keff_l1;
  TASSIGN(keff_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> u_l0;
  TASSIGN(u_l0, 0);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> w_l0;
  TASSIGN(w_l0, 65536);

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Global all-core barrier at kernel start: drains any stale FFTS counters
  // left by previous launches (this kernel's Cube emits flags 3/4 N times
  // per block while Vec only waits N-1 times, so without this entry/exit
  // pair the trailing signal would leak into the next launch).
  sync_all();

  // Vec prepares the two workspaces (ws_a2 holding A2 = INV*beta_2d, and
  // ws_keff holding K_eff = k*exp(g_cs)) that the Cube phase then consumes.
  // `first_iter` is shared across the two branches and across the trailing
  // drain below — set to false once this block has processed any work item.
  bool first_iter = true;
  if (cu_seqlens == nullptr) {
    int64_t gi = 0;
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t head_idx = 0; head_idx < NumHeads; ++head_idx) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t local_rows = valid_rows -
                static_cast<int32_t>(vid) * HalfChunk;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;

            // ─── Phase 1: build A2 = INV * beta_2d ─────────────────────────
            // beta is pre-transposed to [HV, total_tokens] for contiguous loads.
            {
              GmShape2D beta_shape(1, valid_rows);
              GmStride2D beta_stride(1);
              GmTensor2D<half> beta_global(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  beta_shape, beta_stride);
              DynVecTile<half, 1, ChunkSize, pto::PadValue::Zero> beta_load(
                  1, valid_rows);
              TASSIGN(beta_load, BetaUbAddr);
              TLOAD(beta_load, beta_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(beta_ub, beta_load);
              }
            }

            // INV comes in BSND [B, T, HV, C] (output of solve_tril.float()).
            // Each Vec sub-block loads its HalfChunk-row stripe; tails get
            // zero-padded so the Cube GEMM always sees a clean [C, C] tile.
            if (local_rows > 0) {
              int64_t a_gm_offset =
                  ((chunk_token_start +
                    static_cast<int64_t>(vid) * HalfChunk) *
                   NumHeads + head_idx) *
                  static_cast<int64_t>(ChunkSize);
              GmShape2D a_shape(local_rows, ChunkSize);
              GmStride2D a_stride(NumHeads * ChunkSize);
              GmTensor2D<half> a_global(A_handle + a_gm_offset, a_shape,
                                        a_stride);
              TileUbDataND<half, HalfChunk, ChunkSize,
                           HalfChunk, ChunkSize,
                           pto::PadValue::Zero> a_stg_full;
              TASSIGN(a_stg_full, A2HalfUbAddr);
              DynVecTile<half, HalfChunk, ChunkSize,
                         pto::PadValue::Zero> a_load(local_rows, ChunkSize);
              TASSIGN(a_load, A2HalfUbAddr);
              TLOAD(a_load, a_global);
              if (local_rows != HalfChunk) {
                TFILLPAD_INPLACE(a_stg_full, a_load);
              }
            } else {
              // Fully empty lower-half stripe: emit zeros so the workspace tile
              // for this sub-block stays well-defined.
              TEXPANDS(a2_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if (local_rows > 0) {
              TCVT(beta_r_ub, beta_ub, pto::RoundMode::CAST_NONE);
              pipe_barrier(PIPE_V);
              TileUbDataND<half, HalfChunk, ChunkSize,
                           HalfChunk, ChunkSize> a_stg_cvt;
              TASSIGN(a_stg_cvt, A2HalfUbAddr);
              TCVT(inv_ub, a_stg_cvt, pto::RoundMode::CAST_NONE);
              pipe_barrier(PIPE_V);
              // Replicate beta_j across rows: beta_2d[i,j] = beta[j].
              TCOLEXPAND(beta_2d_ub, beta_r_ub);
              // A2 = INV * beta_2d (column-scale).
              TMUL(a2_ub, inv_ub, beta_2d_ub);
              TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);
            }

            if (!first_iter) wait_flag_dev(12);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a2_shape(HalfChunk, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<half> workspace_a2_global(
                  workspace_a2_handle +
                      static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a2_shape, a2_stride);
              TSTORE(workspace_a2_global, a2_ub_half);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (10 << 8));

            // ─── Phase 2: build K_eff = k * exp(g_cs) ──────────────────────
            // k and g_cs are head-major [HV, total_tokens, K] fp16; per-head
            // contiguous K-major rows mean stride K between successive tokens.
            if (local_rows > 0) {
              int64_t hk_base =
                  static_cast<int64_t>(head_idx) * total_tokens *
                      static_cast<int64_t>(HiddenSize) +
                  (chunk_token_start +
                   static_cast<int64_t>(vid) * HalfChunk) *
                      static_cast<int64_t>(HiddenSize);
              {
                GmShape2D k_shape(local_rows, HiddenSize);
                GmStride2D k_stride(HiddenSize);
                GmTensor2D<half> k_global(K_handle + hk_base, k_shape,
                                          k_stride);
                TileUbDataND<half, HalfChunk, HiddenSize,
                             HalfChunk, HiddenSize,
                             pto::PadValue::Zero> k_stg_full;
                TASSIGN(k_stg_full, KeffHalfUbAddr);
                DynVecTile<half, HalfChunk, HiddenSize,
                           pto::PadValue::Zero> k_load(local_rows, HiddenSize);
                TASSIGN(k_load, KeffHalfUbAddr);
                TLOAD(k_load, k_global);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(k_stg_full, k_load);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              {
                TileUbDataND<half, HalfChunk, HiddenSize,
                             HalfChunk, HiddenSize> k_stg_cvt;
                TASSIGN(k_stg_cvt, KeffHalfUbAddr);
                TCVT(k_ub, k_stg_cvt, pto::RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
              }
              {
                GmShape2D g_shape(local_rows, HiddenSize);
                GmStride2D g_stride(HiddenSize);
                GmTensor2D<half> g_global(G_handle + hk_base, g_shape,
                                          g_stride);
                TileUbDataND<half, HalfChunk, HiddenSize,
                             HalfChunk, HiddenSize,
                             pto::PadValue::Zero> g_stg_full;
                TASSIGN(g_stg_full, KeffHalfUbAddr);
                DynVecTile<half, HalfChunk, HiddenSize,
                           pto::PadValue::Zero> g_load(local_rows, HiddenSize);
                TASSIGN(g_load, KeffHalfUbAddr);
                TLOAD(g_load, g_global);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(g_stg_full, g_load);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              {
                TileUbDataND<half, HalfChunk, HiddenSize,
                             HalfChunk, HiddenSize> g_stg_cvt;
                TASSIGN(g_stg_cvt, KeffHalfUbAddr);
                TCVT(gcs_ub, g_stg_cvt, pto::RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
              }
              // exp(g_cs) in-place over gcs_ub, then K_eff = k * exp(g_cs).
              TEXP(gcs_ub, gcs_ub);
              pipe_barrier(PIPE_V);
              TMUL(keff_ub, k_ub, gcs_ub);
              pipe_barrier(PIPE_V);
              TCVT(keff_ub_half, keff_ub, pto::RoundMode::CAST_NONE);
            } else {
              TEXPANDS(keff_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TCVT(keff_ub_half, keff_ub, pto::RoundMode::CAST_NONE);
            }

            if (!first_iter) wait_flag_dev(13);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D keff_shape(HalfChunk, HiddenSize);
              GmStride2D keff_stride(HiddenSize);
              GmTensor2D<half> workspace_keff_global(
                  workspace_keff_handle +
                      static_cast<int64_t>(cid) * WsKeffSize +
                      static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                  keff_shape, keff_stride);
              TSTORE(workspace_keff_global, keff_ub_half);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (11 << 8));
            first_iter = false;
          }
          gi++;
        }
      }
    }
  } else {
    // Varlen branch: identical body to the fixed-length path; only the
    // sequence enumeration differs.
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t local_rows = valid_rows -
                static_cast<int32_t>(vid) * HalfChunk;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;
            int32_t head_idx = h;

            // ─── Phase 1: build A2 = INV * beta_2d ─────────────────────────
            {
              GmShape2D beta_shape(1, valid_rows);
              GmStride2D beta_stride(1);
              GmTensor2D<half> beta_global(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  beta_shape, beta_stride);
              DynVecTile<half, 1, ChunkSize, pto::PadValue::Zero> beta_load(
                  1, valid_rows);
              TASSIGN(beta_load, BetaUbAddr);
              TLOAD(beta_load, beta_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(beta_ub, beta_load);
              }
            }

            if (local_rows > 0) {
              int64_t a_gm_offset =
                  ((chunk_token_start +
                    static_cast<int64_t>(vid) * HalfChunk) *
                   NumHeads + head_idx) *
                  static_cast<int64_t>(ChunkSize);
              GmShape2D a_shape(local_rows, ChunkSize);
              GmStride2D a_stride(NumHeads * ChunkSize);
              GmTensor2D<half> a_global(A_handle + a_gm_offset, a_shape,
                                        a_stride);
              TileUbDataND<half, HalfChunk, ChunkSize,
                           HalfChunk, ChunkSize,
                           pto::PadValue::Zero> a_stg_full;
              TASSIGN(a_stg_full, A2HalfUbAddr);
              DynVecTile<half, HalfChunk, ChunkSize,
                         pto::PadValue::Zero> a_load(local_rows, ChunkSize);
              TASSIGN(a_load, A2HalfUbAddr);
              TLOAD(a_load, a_global);
              if (local_rows != HalfChunk) {
                TFILLPAD_INPLACE(a_stg_full, a_load);
              }
            } else {
              TEXPANDS(a2_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if (local_rows > 0) {
              TCVT(beta_r_ub, beta_ub, pto::RoundMode::CAST_NONE);
              pipe_barrier(PIPE_V);
              TileUbDataND<half, HalfChunk, ChunkSize,
                           HalfChunk, ChunkSize> a_stg_cvt;
              TASSIGN(a_stg_cvt, A2HalfUbAddr);
              TCVT(inv_ub, a_stg_cvt, pto::RoundMode::CAST_NONE);
              pipe_barrier(PIPE_V);
              TCOLEXPAND(beta_2d_ub, beta_r_ub);
              TMUL(a2_ub, inv_ub, beta_2d_ub);
              TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);
            }

            if (!first_iter) wait_flag_dev(12);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a2_shape(HalfChunk, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<half> workspace_a2_global(
                  workspace_a2_handle +
                      static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a2_shape, a2_stride);
              TSTORE(workspace_a2_global, a2_ub_half);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (10 << 8));

            // ─── Phase 2: build K_eff = k * exp(g_cs) ──────────────────────
            if (local_rows > 0) {
              int64_t hk_base =
                  static_cast<int64_t>(head_idx) * total_tokens *
                      static_cast<int64_t>(HiddenSize) +
                  (chunk_token_start +
                   static_cast<int64_t>(vid) * HalfChunk) *
                      static_cast<int64_t>(HiddenSize);
              {
                GmShape2D k_shape(local_rows, HiddenSize);
                GmStride2D k_stride(HiddenSize);
                GmTensor2D<half> k_global(K_handle + hk_base, k_shape,
                                          k_stride);
                TileUbDataND<half, HalfChunk, HiddenSize,
                             HalfChunk, HiddenSize,
                             pto::PadValue::Zero> k_stg_full;
                TASSIGN(k_stg_full, KeffHalfUbAddr);
                DynVecTile<half, HalfChunk, HiddenSize,
                           pto::PadValue::Zero> k_load(local_rows, HiddenSize);
                TASSIGN(k_load, KeffHalfUbAddr);
                TLOAD(k_load, k_global);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(k_stg_full, k_load);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              {
                TileUbDataND<half, HalfChunk, HiddenSize,
                             HalfChunk, HiddenSize> k_stg_cvt;
                TASSIGN(k_stg_cvt, KeffHalfUbAddr);
                TCVT(k_ub, k_stg_cvt, pto::RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
              }
              {
                GmShape2D g_shape(local_rows, HiddenSize);
                GmStride2D g_stride(HiddenSize);
                GmTensor2D<half> g_global(G_handle + hk_base, g_shape,
                                          g_stride);
                TileUbDataND<half, HalfChunk, HiddenSize,
                             HalfChunk, HiddenSize,
                             pto::PadValue::Zero> g_stg_full;
                TASSIGN(g_stg_full, KeffHalfUbAddr);
                DynVecTile<half, HalfChunk, HiddenSize,
                           pto::PadValue::Zero> g_load(local_rows, HiddenSize);
                TASSIGN(g_load, KeffHalfUbAddr);
                TLOAD(g_load, g_global);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(g_stg_full, g_load);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              {
                TileUbDataND<half, HalfChunk, HiddenSize,
                             HalfChunk, HiddenSize> g_stg_cvt;
                TASSIGN(g_stg_cvt, KeffHalfUbAddr);
                TCVT(gcs_ub, g_stg_cvt, pto::RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
              }
              TEXP(gcs_ub, gcs_ub);
              pipe_barrier(PIPE_V);
              TMUL(keff_ub, k_ub, gcs_ub);
              pipe_barrier(PIPE_V);
              TCVT(keff_ub_half, keff_ub, pto::RoundMode::CAST_NONE);
            } else {
              TEXPANDS(keff_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TCVT(keff_ub_half, keff_ub, pto::RoundMode::CAST_NONE);
            }

            if (!first_iter) wait_flag_dev(13);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D keff_shape(HalfChunk, HiddenSize);
              GmStride2D keff_stride(HiddenSize);
              GmTensor2D<half> workspace_keff_global(
                  workspace_keff_handle +
                      static_cast<int64_t>(cid) * WsKeffSize +
                      static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                  keff_shape, keff_stride);
              TSTORE(workspace_keff_global, keff_ub_half);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (11 << 8));
            first_iter = false;
          }
          gi++;
        }
      }
    }
  }

  // Drain the trailing "workspace free" signals emitted by Cube on the last
  // iteration of each active block.  Without this, every block leaves +1 on
  // flags 3 and 4 (because the in-loop wait is gated by `!first_iter`), and
  // that stale count corrupts the next launch's first non-first-iter wait.
  if (!first_iter) {
    wait_flag_dev(12);
    wait_flag_dev(13);
  }

  // Global all-core barrier at kernel exit: matches the entry sync_all so
  // the next launch starts with clean FFTS state.
  sync_all();
#endif

#if defined(__DAV_C220_CUBE__)
  // Global all-core barrier at kernel start (matches Vec side).
  sync_all();

  // Cube reads V from BSND and the two Vec-produced workspaces, then issues
  // U = A2 @ V and W = A2 @ K_eff.  a2_l1 stays resident in L1 across both
  // GEMMs so the second matmul is just a different B-side load away.
  if (cu_seqlens == nullptr) {
    int64_t gi = 0;
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t head_idx = 0; head_idx < NumHeads; ++head_idx) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int64_t v_off =
                (chunk_token_start * static_cast<int64_t>(H) +
                 static_cast<int64_t>(head_idx)) *
                static_cast<int64_t>(HiddenSize);

            // Load V from BSND [B, T, HV, V] as fp16 (cast in Python wrapper).
            {
              GmShape2D v_shape(valid_rows, HiddenSize);
              GmStride2D v_stride(BSND_STRIDE);
              GmTensor2D<half> v_global(V_handle + v_off, v_shape, v_stride);
              DynMatL1<half, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                              HiddenSize);
              TASSIGN(v_l1_load, 0);
              TLOAD(v_l1_load, v_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(v_l1_load, v_l1_load);
              }
            }

            wait_flag_dev(10);
            {
              GmShape2D a2_shape(ChunkSize, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<half> workspace_a2_global(
                  workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size,
                  a2_shape, a2_stride);
              TLOAD(a2_l1, workspace_a2_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // U = A2 @ V.
            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a2_l1, v_l1, u_l0, true);

            {
              GmShape2D u_shape(valid_rows, HiddenSize);
              GmStride2D u_stride(BSND_STRIDE);
              GmTensor2D<half> u_global(U_handle + v_off, u_shape, u_stride);
              DynAccTile<float, ChunkSize, HiddenSize> u_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(u_store, 0);
              TSTORE(u_global, u_store);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (12 << 8));

            wait_flag_dev(11);
            {
              GmShape2D keff_shape(ChunkSize, HiddenSize);
              GmStride2D keff_stride(HiddenSize);
              GmTensor2D<half> workspace_keff_global(
                  workspace_keff_handle +
                      static_cast<int64_t>(cid) * WsKeffSize,
                  keff_shape, keff_stride);
              TLOAD(keff_l1, workspace_keff_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // W = A2 @ K_eff — a2_l1 still resident in L1 from the U GEMM.
            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a2_l1, keff_l1, w_l0, true);

            {
              GmShape2D w_shape(valid_rows, HiddenSize);
              GmStride2D w_stride(BSND_STRIDE);
              GmTensor2D<half> w_global(W_handle + v_off, w_shape, w_stride);
              DynAccTile<float, ChunkSize, HiddenSize> w_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(w_store, 65536);
              TSTORE(w_global, w_store);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (13 << 8));
          }
          gi++;
        }
      }
    }
  } else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;
            int64_t v_off =
                (chunk_token_start * static_cast<int64_t>(H) +
                 static_cast<int64_t>(head_idx)) *
                static_cast<int64_t>(HiddenSize);

            {
              GmShape2D v_shape(valid_rows, HiddenSize);
              GmStride2D v_stride(BSND_STRIDE);
              GmTensor2D<half> v_global(V_handle + v_off, v_shape, v_stride);
              DynMatL1<half, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                              HiddenSize);
              TASSIGN(v_l1_load, 0);
              TLOAD(v_l1_load, v_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(v_l1_load, v_l1_load);
              }
            }

            wait_flag_dev(10);
            {
              GmShape2D a2_shape(ChunkSize, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<half> workspace_a2_global(
                  workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size,
                  a2_shape, a2_stride);
              TLOAD(a2_l1, workspace_a2_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a2_l1, v_l1, u_l0, true);

            {
              GmShape2D u_shape(valid_rows, HiddenSize);
              GmStride2D u_stride(BSND_STRIDE);
              GmTensor2D<half> u_global(U_handle + v_off, u_shape, u_stride);
              DynAccTile<float, ChunkSize, HiddenSize> u_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(u_store, 0);
              TSTORE(u_global, u_store);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (12 << 8));

            wait_flag_dev(11);
            {
              GmShape2D keff_shape(ChunkSize, HiddenSize);
              GmStride2D keff_stride(HiddenSize);
              GmTensor2D<half> workspace_keff_global(
                  workspace_keff_handle +
                      static_cast<int64_t>(cid) * WsKeffSize,
                  keff_shape, keff_stride);
              TLOAD(keff_l1, workspace_keff_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a2_l1, keff_l1, w_l0, true);

            {
              GmShape2D w_shape(valid_rows, HiddenSize);
              GmStride2D w_stride(BSND_STRIDE);
              GmTensor2D<half> w_global(W_handle + v_off, w_shape, w_stride);
              DynAccTile<float, ChunkSize, HiddenSize> w_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(w_store, 65536);
              TSTORE(w_global, w_store);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (13 << 8));
          }
          gi++;
        }
      }
    }
  }

  // Global all-core barrier at kernel exit (matches Vec side).
  sync_all();
#endif
}

extern "C" __global__ AICORE void launch_wy_kda(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle,
    __gm__ uint8_t *A_handle,
    __gm__ uint8_t *workspace_a2_handle, __gm__ uint8_t *workspace_keff_handle,
    __gm__ uint8_t *U_handle, __gm__ uint8_t *W_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  wy_kda_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ half *>(G_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ half *>(workspace_a2_handle),
      reinterpret_cast<__gm__ half *>(workspace_keff_handle),
      reinterpret_cast<__gm__ half *>(U_handle),
      reinterpret_cast<__gm__ half *>(W_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *k, uint8_t *v, uint8_t *beta, uint8_t *g_cs, uint8_t *A,
    uint8_t *workspace_a2, uint8_t *workspace_keff,
    uint8_t *u, uint8_t *w,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_wy_kda<<<block_dim, nullptr, stream>>>(
      k, v, beta, g_cs, A,
      workspace_a2, workspace_keff,
      u, w,
      cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
