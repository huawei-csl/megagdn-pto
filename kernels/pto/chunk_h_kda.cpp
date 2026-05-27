// ============================================================================
// chunk_h_kda.cpp — Recurrent hidden state update for KDA (per-dim gate)
//
// Math (per chunk, matches ref_chunk_h_kda in
//   tests/test_kda_single_kernels.py:276-326):
//   v_corr  = u - w @ S                              # [c_len, V]
//   k_rest  = k * exp(g_total - g_cs)                # [c_len, K]
//   S_new   = exp(g_total).unsqueeze(-1) * S + k_rest^T @ v_corr   # [K, V]
//
// where g_total = g_cs[valid-1, :] is the chunk's per-K-dim cumulative gate
// at the last valid token, and S is the [K, V] state. Snapshots produced:
//   s_snapshots[ci_base + ci, head, :, :] = S entering chunk ci.
//
// Differences from GDN chunk_h.cpp:
//   - g is per-DIMENSION here: g_cs has shape [HV, T, K] (head-major).
//   - State decay factor is a K-vector exp(g_total[k]), not a scalar.
//   - K rescaling coeff_2d[c, k] = exp(g_total[k] - g_cs[c, k]) is element-wise
//     per (token, k-dim), not a row-broadcast scalar.
//   - No GQA — K, W, U all use HV heads with the same BSND stride.
//   - Inputs U, K, G are fp32; W arrives as fp16 (cast in the Python wrapper)
//     so Cube can use it directly.  Outputs v_corr and snapshots are fp32.
//   - v_corr (fp16 copy) lives in a dedicated workspace slot (WS_V) so the
//     Cube K_rest^T @ V_corr GEMM has an fp16 source — the BSND output is fp32.
//
// Cross-core sync: same data-flow flags as GDN chunk_h (0-3), plus sync_all
// on entry/exit using reserved IDs 6-9 (matches wy_kda.cpp:89-104).
//
// Inputs:
//   K   [HV, T, K]              fp32  — keys (head-major)
//   W   [B, T, HV, K]           fp16  — wy_kda output, cast to fp16 in wrapper
//   U   [B, T, HV, V]           fp32  — wy_kda output (BSND)
//   G   [HV, T, K]              fp32  — per-dim cumulative gate sum (head-major)
//   S   [total_chunks, HV, K, V] fp32 — snapshots (output)
//   V_corr [B, T, HV, V]        fp32  — corrected values (BSND, output)
//   workspace [per-core scratch] fp16  — 5 slots × K*V halves
//
// Workspace per AI core (5 slots, fp16; assumes K == V == HiddenSize):
//   WS_WS [C, V]   Cube writes WS = W @ S          → Vec reads
//   WS_K  [C, K]   Vec writes K_rest               → Cube reads (^T view)
//   WS_V  [C, V]   Vec writes V_corr (fp16 copy)   → Cube reads
//   WS_S  [K, V]   Vec writes fp16(S)              → Cube reads (next chunk)
//   WS_KV [K, V]   Cube writes K_rest^T @ V_corr   → Vec reads
// ============================================================================

#include <pto/pto-inst.hpp>
#include <type_traits>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
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

// Global all-core barrier — drains stale FFTS counters from prior launches.
// Mirrors wy_kda.cpp:89-104.
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

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512,
                              pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0B = pto::Tile<pto::TileType::Right, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::ColMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                               pto::BLayout::RowMajor, RowValid, ColValid,
                               pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                               pto::BLayout::ColMajor, RowValid, ColValid,
                               pto::SLayout::NoneBox, 512, PadVal>;

// K-sliced matmul helper — verbatim copy from chunk_h.cpp::gemm_v0.
template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail = K, bool transpose_A = false,
          bool transpose_B = false>
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
AICORE void chunk_h_kda_kernel(
    __gm__ half *K_handle, __gm__ half *W_handle, __gm__ half *U_handle,
    __gm__ half *G_handle,
    __gm__ half *S_handle, __gm__ half *V_handle,
    __gm__ half  *workspace_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  set_ffts_base_addr(ffts_addr);

  // With wy_kda's convention K == V == HiddenSize.  Keeping two aliases so the
  // math reads correctly.
  constexpr int32_t K_DIM = HiddenSize;
  constexpr int32_t V_DIM = HiddenSize;
  constexpr int32_t C     = ChunkSize;
  constexpr int32_t H     = NumHeads;       // HV in KDA terminology
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t BSND_STRIDE = H * HiddenSize;
  constexpr int32_t HM_STRIDE   = HiddenSize;        // head-major K, G stride
  constexpr int32_t KV = K_DIM * V_DIM;

  // ── Workspace slots (fp16 half-elements, per AI core) ────────────────────
  constexpr int32_t WS_WS = 0;
  constexpr int32_t WS_K  = WS_WS + C * V_DIM;
  constexpr int32_t WS_V  = WS_K  + C * K_DIM;
  constexpr int32_t WS_S  = WS_V  + C * V_DIM;
  constexpr int32_t WS_KV = WS_S  + KV;
  constexpr int32_t WS_PER_CORE = WS_KV + KV;

  // ── Cube L1 tiles ────────────────────────────────────────────────────────
  TileMatL1<half, K_DIM, V_DIM, K_DIM, V_DIM> s_l1;
  TASSIGN(s_l1, 0);
  TileMatL1<half, C, K_DIM, C, K_DIM> w_l1;
  TASSIGN(w_l1, KV * sizeof(half));
  TileAcc<float, C, V_DIM, C, V_DIM> ws_l0;
  TASSIGN(ws_l0, 0);
  TileMatL1<half, K_DIM, C, K_DIM, C> k_l1;
  TASSIGN(k_l1, (KV + C * K_DIM) * sizeof(half));
  TileMatL1<half, C, V_DIM, C, V_DIM> v_l1;
  TASSIGN(v_l1, (KV + C * K_DIM + K_DIM * C) * sizeof(half));
  TileAcc<float, K_DIM, V_DIM, K_DIM, V_DIM> kv_l0;
  TASSIGN(kv_l0, C * V_DIM * sizeof(float));

  // ── Vec UB plan ──────────────────────────────────────────────────────────
  // Layout designed for KDA's larger per-K-dim gate buffers.  Buffer reuse:
  //   GCS_UB == U_UB == KV_FP32_UB        (sequential lifetimes B-C, B2-F, J)
  //   COEFF_UB == WS_UB == EXP_GT_2D_UB   (sequential C-D, E, H-I)
  //   U_UB_HALF == KV_LOAD_UB             (sequential E-F, J)
  // Total at peak ≈ 176 KB out of 192 KB budget.
  constexpr int32_t ZERO_UB     = 0;
  constexpr int32_t S_UB        = ZERO_UB + 64 * sizeof(float);
  constexpr int32_t GTOTAL_UB   = S_UB + HalfC * V_DIM * sizeof(float);
  constexpr int32_t K_UB        = GTOTAL_UB + K_DIM * sizeof(float);
  constexpr int32_t GCS_UB      = K_UB + HalfC * K_DIM * sizeof(float);
  constexpr int32_t COEFF_UB    = GCS_UB + HalfC * K_DIM * sizeof(float);
  constexpr int32_t K_UB_HALF   = COEFF_UB + HalfC * K_DIM * sizeof(float);
  constexpr int32_t U_UB_HALF   = K_UB_HALF + HalfC * K_DIM * sizeof(half);
  constexpr int32_t S_UB_HALF   = U_UB_HALF + HalfC * V_DIM * sizeof(half);
  // Aliases:
  constexpr int32_t U_UB        = GCS_UB;     // load U after g_cs is consumed
  constexpr int32_t KV_FP32_UB  = GCS_UB;     // cast KV → fp32 into freed buffer
  constexpr int32_t WS_UB       = COEFF_UB;   // load WS into freed coeff buffer
  constexpr int32_t EXP_GT_2D_UB = COEFF_UB;  // broadcast exp(g_total) similarly
  constexpr int32_t KV_LOAD_UB  = U_UB_HALF;  // load KV fp16 into freed U_HALF

  TileUbDataND<float, 1, 64, 1, 64> zero_ub;
  TASSIGN(zero_ub, ZERO_UB);
  TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> s_ub;
  TASSIGN(s_ub, S_UB);
  TileUbDataND<float, 1, K_DIM, 1, K_DIM, pto::PadValue::Zero> gtotal_ub;
  TASSIGN(gtotal_ub, GTOTAL_UB);
  TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero> k_ub;
  TASSIGN(k_ub, K_UB);
  TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero> gcs_ub;
  TASSIGN(gcs_ub, GCS_UB);
  TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> coeff_2d_ub;
  TASSIGN(coeff_2d_ub, COEFF_UB);
  TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM> k_ub_half;
  TASSIGN(k_ub_half, K_UB_HALF);
  TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM, pto::PadValue::Zero> u_ub_half;
  TASSIGN(u_ub_half, U_UB_HALF);
  TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM> s_ub_half;
  TASSIGN(s_ub_half, S_UB_HALF);
  TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM, pto::PadValue::Zero> u_ub;
  TASSIGN(u_ub, U_UB);
  TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> ws_ub;
  TASSIGN(ws_ub, WS_UB);
  TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> exp_gt_2d_ub;
  TASSIGN(exp_gt_2d_ub, EXP_GT_2D_UB);
  TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> kv_ub;
  TASSIGN(kv_ub, KV_FP32_UB);

  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * H;

#if defined(__DAV_C220_CUBE__)
  sync_all();

  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      // Wait for Vec to publish S entering this chunk (S₀=0 for ci=0).
      wait_flag_dev(3);

      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;

      // Load S [K, V] from workspace.
      {
        GmShape2D s_shape(K_DIM, V_DIM);
        GmStride2D s_stride(V_DIM);
        GmTensor2D<half> s_global(workspace_handle + ws_base + WS_S, s_shape,
                                  s_stride);
        DynMatL1<half, K_DIM, V_DIM> s_l1_load(K_DIM, V_DIM);
        TASSIGN(s_l1_load, 0);
        TLOAD(s_l1_load, s_global);
      }

      // Load W [valid, K] from BSND (already cast to fp16 by Python wrapper).
      int64_t w_offset = (chunk_start * H + head) * K_DIM;
      {
        GmShape2D w_shape(static_cast<int32_t>(valid), K_DIM);
        GmStride2D w_stride(BSND_STRIDE);
        GmTensor2D<half> w_global(W_handle + w_offset, w_shape, w_stride);
        DynMatL1<half, C, K_DIM> w_l1_load(static_cast<int32_t>(valid), K_DIM);
        TASSIGN(w_l1_load, KV * static_cast<int32_t>(sizeof(half)));
        TLOAD(w_l1_load, w_global);
        if (valid != C) {
          TFILLPAD(w_l1_load, w_l1_load);
        }
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      // WS = W @ S — [C, K] @ [K, V] → [C, V].
      gemm_v0<half, float, C, V_DIM, K_DIM, C, V_DIM, K_DIM, K_DIM, false, false>(
          w_l1, s_l1, ws_l0, (bool)1);

      {
        GmShape2D ws_shape(C, V_DIM);
        GmStride2D ws_stride(V_DIM);
        GmTensor2D<half> ws_global(workspace_handle + ws_base + WS_WS,
                                   ws_shape, ws_stride);
        DynAccTile<float, C, V_DIM> ws_store(C, V_DIM);
        TASSIGN(ws_store, 0);
        TSTORE(ws_global, ws_store);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

      // Wait for Vec to publish K_rest in WS_K and V_corr (fp16) in WS_V.
      wait_flag_dev(1);

      {
        GmShape2D k_shape(K_DIM, C);
        GmStride2D k_stride(C);
        GmTensor2D<half> k_global(workspace_handle + ws_base + WS_K, k_shape,
                                  k_stride);
        DynMatL1<half, K_DIM, C> k_l1_load(K_DIM, C);
        TASSIGN(k_l1_load,
                (KV + C * K_DIM) * static_cast<int32_t>(sizeof(half)));
        TLOAD(k_l1_load, k_global);
      }

      {
        GmShape2D v_shape(C, V_DIM);
        GmStride2D v_stride(V_DIM);
        GmTensor2D<half> v_global(workspace_handle + ws_base + WS_V, v_shape,
                                  v_stride);
        DynMatL1<half, C, V_DIM> v_l1_load(C, V_DIM);
        TASSIGN(v_l1_load,
                (KV + C * K_DIM + K_DIM * C) *
                static_cast<int32_t>(sizeof(half)));
        TLOAD(v_l1_load, v_global);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      // KV = K_rest^T @ V_corr — [K, C] @ [C, V] → [K, V].
      gemm_v0<half, float, K_DIM, V_DIM, C, K_DIM, V_DIM, C, C, true, false>(
          k_l1, v_l1, kv_l0, (bool)1);

      {
        GmShape2D kv_shape(K_DIM, V_DIM);
        GmStride2D kv_stride(V_DIM);
        GmTensor2D<half> kv_global(workspace_handle + ws_base + WS_KV,
                                   kv_shape, kv_stride);
        DynAccTile<float, K_DIM, V_DIM> kv_store(K_DIM, V_DIM);
        TASSIGN(kv_store, C * V_DIM * static_cast<int32_t>(sizeof(float)));
        TSTORE(kv_global, kv_store);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
    }
  }

  sync_all();
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  sync_all();

  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    // Initialise running state S₀ = 0.
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(zero_ub, 0.0f);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(s_ub, 0.0f);
    TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      GmShape2D s_shape(HalfC, V_DIM);
      GmStride2D s_stride(V_DIM);
      GmTensor2D<half> s_global(
          workspace_handle + ws_base + WS_S +
              static_cast<int64_t>(vid) * HalfC * V_DIM,
          s_shape, s_stride);
      DynVecTile<half, HalfC, V_DIM> s_store(HalfC, V_DIM);
      TASSIGN(s_store, S_UB_HALF);
      TSTORE(s_global, s_store);
    }
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));

    for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;
      int32_t valid_rows =
          static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
      if (valid_rows < 0) valid_rows = 0;
      if (valid_rows > HalfC) valid_rows = HalfC;

      // ── 1. Snapshot S entering this chunk (s_ub fp32 → S_handle) ───────
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        int64_t s_out_offset =
            (chunk_offset + static_cast<int64_t>(ci)) * H * KV +
            static_cast<int64_t>(head) * KV +
            static_cast<int64_t>(vid) * HalfC * V_DIM;
        GmShape2D s_out_shape(HalfC, V_DIM);
        GmStride2D s_out_stride(V_DIM);
        GmTensor2D<half> s_out_global(S_handle + s_out_offset, s_out_shape,
                                      s_out_stride);
        DynVecTile<half, HalfC, V_DIM> s_out_store(HalfC, V_DIM);
        TASSIGN(s_out_store, S_UB_HALF);
        TSTORE(s_out_global, s_out_store);
      }

      // ── 2. Load K, g_cs (head-major fp16) and g_total ──────────────────
      int64_t hk_base = static_cast<int64_t>(head) * total_tokens * K_DIM +
                        (chunk_start + static_cast<int64_t>(vid) * HalfC) *
                            K_DIM;
      if (valid_rows > 0) {
        {
          GmShape2D k_shape(valid_rows, K_DIM);
          GmStride2D k_stride(HM_STRIDE);
          GmTensor2D<half> k_global(K_handle + hk_base, k_shape, k_stride);
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM,
                       pto::PadValue::Zero> k_stg_full;
          TASSIGN(k_stg_full, K_UB_HALF);
          DynVecTile<half, HalfC, K_DIM, pto::PadValue::Zero> k_load(
              valid_rows, K_DIM);
          TASSIGN(k_load, K_UB_HALF);
          TLOAD(k_load, k_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(k_stg_full, k_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        {
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM> k_stg_cvt;
          TASSIGN(k_stg_cvt, K_UB_HALF);
          TCVT(k_ub, k_stg_cvt, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        }
        {
          GmShape2D g_shape(valid_rows, K_DIM);
          GmStride2D g_stride(HM_STRIDE);
          GmTensor2D<half> g_global(G_handle + hk_base, g_shape, g_stride);
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM,
                       pto::PadValue::Zero> g_stg_full;
          TASSIGN(g_stg_full, K_UB_HALF);
          DynVecTile<half, HalfC, K_DIM, pto::PadValue::Zero> g_load(
              valid_rows, K_DIM);
          TASSIGN(g_load, K_UB_HALF);
          TLOAD(g_load, g_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(g_stg_full, g_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        {
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM> g_stg_cvt;
          TASSIGN(g_stg_cvt, K_UB_HALF);
          TCVT(gcs_ub, g_stg_cvt, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        }
      } else {
        TEXPANDS(k_ub, 0.0f);
        TEXPANDS(gcs_ub, 0.0f);
      }

      // g_total = g_cs[valid-1, :] — both sub-blocks load independently.
      {
        int64_t gt_offset = static_cast<int64_t>(head) * total_tokens * K_DIM +
                            (chunk_start + (valid - 1)) * K_DIM;
        GmShape2D gt_shape(1, K_DIM);
        GmStride2D gt_stride(1);
        GmTensor2D<half> gt_global(G_handle + gt_offset, gt_shape, gt_stride);
        DynVecTile<half, 1, K_DIM, pto::PadValue::Zero> gt_load(1, K_DIM);
        TASSIGN(gt_load, K_UB_HALF);
        TLOAD(gt_load, gt_global);
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      {
        TileUbDataND<half, 1, K_DIM, 1, K_DIM> gt_stg_cvt;
        TASSIGN(gt_stg_cvt, K_UB_HALF);
        TCVT(gtotal_ub, gt_stg_cvt, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
      }

      // ── 3. coeff_2d = exp(g_total - g_cs)  [HalfC, K] ─────────────────
      // Broadcast g_total [1, K] across HalfC rows: 2d[i,j] = row[j] (TCOLEXPAND).
      TCOLEXPAND(coeff_2d_ub, gtotal_ub);
      pipe_barrier(PIPE_V);
      TSUB(coeff_2d_ub, coeff_2d_ub, gcs_ub);
      pipe_barrier(PIPE_V);
      TEXP(coeff_2d_ub, coeff_2d_ub);
      pipe_barrier(PIPE_V);

      // K_rest = K * coeff_2d (element-wise).
      TMUL(k_ub, k_ub, coeff_2d_ub);
      pipe_barrier(PIPE_V);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);

      // ── 4. Reuse GCS_UB region to load U (g_cs is now dead) ────────────
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      int64_t u_offset = (chunk_start * H + head) * V_DIM +
                         static_cast<int64_t>(vid) * HalfC * BSND_STRIDE;
      if (valid_rows > 0) {
        GmShape2D u_shape(valid_rows, V_DIM);
        GmStride2D u_stride(BSND_STRIDE);
        GmTensor2D<half> u_global(U_handle + u_offset, u_shape, u_stride);
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM,
                     pto::PadValue::Zero> u_stg_full;
        TASSIGN(u_stg_full, U_UB_HALF);
        DynVecTile<half, HalfC, V_DIM, pto::PadValue::Zero> u_load(
            valid_rows, V_DIM);
        TASSIGN(u_load, U_UB_HALF);
        TLOAD(u_load, u_global);
        if (valid_rows != HalfC) {
          TFILLPAD_INPLACE(u_stg_full, u_load);
        }
      }
      // Sync and TCVT before WS load overwrites U_UB_HALF.
      if (valid_rows > 0) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        {
          TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM> u_stg_cvt;
          TASSIGN(u_stg_cvt, U_UB_HALF);
          TCVT(u_ub, u_stg_cvt, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        }
      } else {
        TEXPANDS(u_ub, 0.0f);
      }

      // ── 5. Wait Cube WS ready, load WS (fp16) into U_UB_HALF buffer ───
      wait_flag_dev(0);
      {
        GmShape2D ws_shape(HalfC, V_DIM);
        GmStride2D ws_stride(V_DIM);
        GmTensor2D<half> ws_global(
            workspace_handle + ws_base + WS_WS +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            ws_shape, ws_stride);
        DynVecTile<half, HalfC, V_DIM, pto::PadValue::Zero> ws_load(HalfC,
                                                                    V_DIM);
        TASSIGN(ws_load, U_UB_HALF);
        TLOAD(ws_load, ws_global);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // Cast WS to fp32 into the broadcast/scratch buffer at WS_UB.
      TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TSUB(u_ub, u_ub, ws_ub);
      pipe_barrier(PIPE_V);
      // V_corr fp16 → U_UB_HALF (overwrites WS bytes).
      TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);

      // ── 6. Store V_corr (fp32 to output, fp16 to workspace WS_V) ───────
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      if (valid_rows > 0) {
        int64_t v_offset = (chunk_start * H + head) * V_DIM +
                           static_cast<int64_t>(vid) * HalfC * BSND_STRIDE;
        GmShape2D v_shape(valid_rows, V_DIM);
        GmStride2D v_stride(BSND_STRIDE);
        GmTensor2D<half> v_global(V_handle + v_offset, v_shape, v_stride);
        DynVecTile<half, HalfC, V_DIM> v_store(valid_rows, V_DIM);
        TASSIGN(v_store, U_UB_HALF);
        TSTORE(v_global, v_store);
      }
      {
        GmShape2D wsv_shape(HalfC, V_DIM);
        GmStride2D wsv_stride(V_DIM);
        GmTensor2D<half> wsv_global(
            workspace_handle + ws_base + WS_V +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            wsv_shape, wsv_stride);
        DynVecTile<half, HalfC, V_DIM> v_store(HalfC, V_DIM);
        TASSIGN(v_store, U_UB_HALF);
        TSTORE(wsv_global, v_store);
      }
      // Store K_rest fp16 to workspace WS_K.
      {
        GmShape2D k_shape(HalfC, K_DIM);
        GmStride2D k_stride(K_DIM);
        GmTensor2D<half> k_global(
            workspace_handle + ws_base + WS_K +
                static_cast<int64_t>(vid) * HalfC * K_DIM,
            k_shape, k_stride);
        DynVecTile<half, HalfC, K_DIM> k_store(HalfC, K_DIM);
        TASSIGN(k_store, K_UB_HALF);
        TSTORE(k_global, k_store);
      }
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

      // ── 7. State decay: S *= exp(g_total) (per-K row, broadcast over V) ─
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TEXP(gtotal_ub, gtotal_ub);
      pipe_barrier(PIPE_V);

      // This sub-block owns K-rows [vid*HalfC, (vid+1)*HalfC) of the state.
      // Reinterpret that slice of gtotal_ub as a [HalfC, 1] DN-column vector,
      // broadcast across V columns via TROWEXPAND, then row-scale s_ub.
      {
        TileUbDataDN<float, HalfC, 1, HalfC, 1> exp_gt_col;
        TASSIGN(exp_gt_col,
                GTOTAL_UB + static_cast<int32_t>(vid) * HalfC *
                    static_cast<int32_t>(sizeof(float)));
        TROWEXPAND(exp_gt_2d_ub, exp_gt_col);
      }
      pipe_barrier(PIPE_V);
      TMUL(s_ub, s_ub, exp_gt_2d_ub);
      pipe_barrier(PIPE_V);

      // ── 8. Wait Cube KV ready, S += KV ─────────────────────────────────
      wait_flag_dev(2);
      {
        GmShape2D kv_shape(HalfC, V_DIM);
        GmStride2D kv_stride(V_DIM);
        GmTensor2D<half> kv_global(
            workspace_handle + ws_base + WS_KV +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            kv_shape, kv_stride);
        DynVecTile<half, HalfC, V_DIM, pto::PadValue::Zero> kv_load(HalfC,
                                                                    V_DIM);
        TASSIGN(kv_load, KV_LOAD_UB);
        TLOAD(kv_load, kv_global);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // Cast fp16 KV (loaded at KV_LOAD_UB) to fp32 kv_ub.
      {
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM> kv_load_view;
        TASSIGN(kv_load_view, KV_LOAD_UB);
        TCVT(kv_ub, kv_load_view, pto::RoundMode::CAST_NONE);
      }
      pipe_barrier(PIPE_V);
      TADD(s_ub, s_ub, kv_ub);

      // ── 9. Write fp16(S) to workspace WS_S for next chunk's W @ S ──────
      // Only emit flag 3 when a next chunk exists — keeps the flag balance
      // (Cube waits flag 3 exactly N times, Vec emits 1× at init + N-1× here).
      // Same pattern as GDN chunk_h.cpp:826-855.
      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        {
          GmShape2D s_shape(HalfC, V_DIM);
          GmStride2D s_stride(V_DIM);
          GmTensor2D<half> s_global(
              workspace_handle + ws_base + WS_S +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              s_shape, s_stride);
          DynVecTile<half, HalfC, V_DIM> s_store(HalfC, V_DIM);
          TASSIGN(s_store, S_UB_HALF);
          TSTORE(s_global, s_store);
        }
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
      }
    }
  }

  sync_all();
#endif
}

extern "C" __global__ AICORE void launch_chunk_h_kda(
    __gm__ uint8_t *K, __gm__ uint8_t *W, __gm__ uint8_t *U,
    __gm__ uint8_t *G,
    __gm__ uint8_t *S, __gm__ uint8_t *V_corr,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  chunk_h_kda_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K),
      reinterpret_cast<__gm__ half *>(W),
      reinterpret_cast<__gm__ half *>(U),
      reinterpret_cast<__gm__ half *>(G),
      reinterpret_cast<__gm__ half *>(S),
      reinterpret_cast<__gm__ half *>(V_corr),
      reinterpret_cast<__gm__ half *>(workspace),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *K, uint8_t *W, uint8_t *U, uint8_t *G,
    uint8_t *S, uint8_t *V_corr,
    uint8_t *workspace,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_h_kda<<<block_dim, nullptr, stream>>>(
      K, W, U, G, S, V_corr, workspace, cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
