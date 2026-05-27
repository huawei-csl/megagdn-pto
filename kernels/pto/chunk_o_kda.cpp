// ============================================================================
// chunk_o_kda.cpp — Output stage for KDA (per-dim gate)
//
// Math (per chunk, matches ref_chunk_o_kda in
//   tests/test_kda_single_kernels.py:333-380):
//   q_eff = q * exp(g_cs)              # [c_len, K]
//   k_eff = k * exp(-g_cs)             # [c_len, K]
//   inter = q_eff @ S                  # [c_len, V]
//   Aqk   = tril(q_eff @ k_eff^T,      # [c_len, c_len], INCLUSIVE diagonal
//                diagonal=0)
//   o     = inter + Aqk @ v_corr       # [c_len, V]
//
// where S = s_snapshots[ci_base + ci, head] is the [K, V] state *entering*
// this chunk (already computed by chunk_h_kda), and v_corr = u - w @ S is
// the corrected values (also from chunk_h_kda).
//
// Differences from GDN chunk_o.cpp:
//   - Gate is per-DIMENSION (g_cs has shape [HV, T, K] head-major).
//   - Vec pre-scales Q and K element-wise (q*exp(g_cs), k*exp(-g_cs)) BEFORE
//     Cube sees them.  Cube does pure matmuls; there is no per-element gating
//     coefficient applied to QK on the Vec side.
//   - Causal mask is INCLUSIVE of the diagonal (rows >= cols), so the mask
//     tensor passed from Python differs from kkt_kda's strict-lower mask.
//   - No GQA: Q, K, V_corr, O all use HV heads.
//   - S is fp32 in GM (from chunk_h_kda's output) — Vec casts to fp16 into
//     workspace so Cube has fp16 sources for all three GEMMs.
//
// Chunks within a (seq, head) work item are fully independent (each reads
// its own s_snapshots entry).  Cube/Vec still process them sequentially per
// work item to keep the per-core 4-flag protocol simple.
//
// Cross-core sync: same data-flow flags as chunk_h_kda (0-3), plus sync_all
// on entry/exit using reserved IDs 6-9.
//
// Inputs:
//   Q       [HV, T, K]               fp32  — queries (head-major), scale pre-applied
//   K       [HV, T, K]               fp32  — keys    (head-major)
//   V_corr  [B, T, HV, V]            fp32  — corrected values from chunk_h_kda (BSND)
//   S       [total_chunks, HV, K, V] fp32  — snapshots from chunk_h_kda
//   G_cs    [HV, T, K]               fp32  — per-dim cumulative gate (head-major)
//   Msk     [C, C]                   fp32  — inclusive lower-tri mask (rows >= cols)
//   workspace [per-core scratch]     fp16  — 7 slots × K*V halves
//   O       [B, T, HV, V]            fp32  — output (BSND)
//
// Workspace per AI core (7 slots, fp16; assumes K == V == HiddenSize):
//   WS_Q   [C, K]   Vec writes q*exp(g_cs)  → Cube reads (GEMM1 A, GEMM2 A)
//   WS_K   [C, K]   Vec writes k*exp(-g_cs) → Cube reads (GEMM1 B, transposed)
//   WS_V   [C, V]   Vec writes V_corr fp16  → Cube reads (GEMM3 B)
//   WS_S   [K, V]   Vec writes S fp16       → Cube reads (GEMM2 B)
//   WS_QK  [C, C]   Cube writes QK fp16     → Vec masks → Cube reads (GEMM3 A)
//   WS_QS  [C, V]   Cube writes QS fp16     → Vec reads (final combine)
//   WS_QKV [C, V]   Cube writes QKV fp16    → Vec reads (final combine)
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
// Mirrors chunk_h_kda.cpp:67-82.
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

using GmShape2D  = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
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

// Single-shot dense GEMM via L0A/L0B — used when the K-dim is one L0 tile.
// All three of our GEMMs have inner-dim == 128 == L0 tile size, so a one-shot
// matmul is sufficient (no K-slicing needed, unlike chunk_h_kda's gemm_v0).
template <typename T1, typename T2, int32_t M, int32_t N, int32_t K,
          bool transpose_B = false>
AICORE PTO_INLINE void
gemm_oneshot(TileMatL1<T1, M, K, M, K> &A,
             std::conditional_t<transpose_B,
                                TileMatL1<T1, N, K, N, K>,
                                TileMatL1<T1, K, N, K, N>> &B,
             pto::TileAcc<T2, M, N, M, N> &C)
{
    TileMatL0A<T1, M, K, M, K> l0a;
    TileMatL0B<T1, K, N, K, N> l0b;
    pto::TASSIGN(l0a, 0x0);
    pto::TASSIGN(l0b, 0x0);

    auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
    set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
    wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
    set_flag(PIPE_M, PIPE_MTE1, war_event_id);
    wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

    pto::TEXTRACT(l0a, A, 0, 0);
    if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, 0, 0);
    } else {
        TileMatL1ZN<T1, K, N, K, N> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, 0, 0);
    }

    set_flag(PIPE_MTE1, PIPE_M, war_event_id);
    wait_flag(PIPE_MTE1, PIPE_M, war_event_id);
    pto::TMATMUL(C, l0a, l0b);

    set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    set_flag(PIPE_M, PIPE_FIX, war_event_id);
    wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

} // namespace

#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void chunk_o_kda_kernel(
    __gm__ half *Q_handle, __gm__ half *K_handle,
    __gm__ half *V_handle, __gm__ half *S_handle,
    __gm__ half *G_handle, __gm__ float *Mask_handle,
    __gm__ half  *workspace_handle,
    __gm__ half *O_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  set_ffts_base_addr(ffts_addr);

  constexpr int32_t K_DIM = HiddenSize;
  constexpr int32_t V_DIM = HiddenSize;
  constexpr int32_t C     = ChunkSize;
  constexpr int32_t H     = NumHeads;       // HV in KDA terminology
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t BSND_STRIDE = H * HiddenSize;
  constexpr int32_t HM_STRIDE   = HiddenSize;    // head-major Q, K, G stride
  constexpr int32_t KV = K_DIM * V_DIM;

  // ── Workspace slots (fp16 half-elements, per AI core) ────────────────────
  constexpr int32_t WS_Q   = 0;
  constexpr int32_t WS_K   = WS_Q   + C * K_DIM;
  constexpr int32_t WS_V   = WS_K   + C * K_DIM;
  constexpr int32_t WS_S   = WS_V   + C * V_DIM;
  constexpr int32_t WS_QK  = WS_S   + KV;
  constexpr int32_t WS_QS  = WS_QK  + C * C;
  constexpr int32_t WS_QKV = WS_QS  + C * V_DIM;
  constexpr int32_t WS_PER_CORE = WS_QKV + C * V_DIM;

#if defined(__DAV_C220_CUBE__)
  // ── Cube L1 tiles ────────────────────────────────────────────────────────
  // L1 layout (fp16, ~160 KB out of 1 MB budget):
  //   q_l1   @       0  : [C, K]     — input to GEMM1, GEMM2
  //   k_l1   @  32*1024 : [C, K]     — input to GEMM1 (transposed via TRESHAPE)
  //   s_l1   @  64*1024 : [K, V]     — input to GEMM2
  //   qkm_l1 @  96*1024 : [C, C]     — masked Aqk, input to GEMM3
  //   v_l1   @ 128*1024 : [C, V]     — V_corr, input to GEMM3
  TileMatL1<half, C, K_DIM, C, K_DIM> q_l1;
  TASSIGN(q_l1, 0);
  TileMatL1<half, C, K_DIM, C, K_DIM> k_l1;
  TASSIGN(k_l1, C * K_DIM * sizeof(half));
  TileMatL1<half, K_DIM, V_DIM, K_DIM, V_DIM> s_l1;
  TASSIGN(s_l1, (C * K_DIM + C * K_DIM) * sizeof(half));
  TileMatL1<half, C, C, C, C> qkm_l1;
  TASSIGN(qkm_l1, (C * K_DIM + C * K_DIM + KV) * sizeof(half));
  TileMatL1<half, C, V_DIM, C, V_DIM> v_l1;
  TASSIGN(v_l1, (C * K_DIM + C * K_DIM + KV + C * C) * sizeof(half));

  // L0C accumulators (separate physical L0C, not L1).
  //   qk_l0  @ 0 : [C, C]   — GEMM1 result; stored to GM, then space reused
  //   qs_l0  @ C*C*4 : [C, V] — GEMM2 result; stored to GM before GEMM3 starts
  //   qkv_l0 @ 0 : [C, V]   — GEMM3 result (reuses qk_l0's L0C bytes)
  TileAcc<float, C, C, C, C> qk_l0;
  TASSIGN(qk_l0, 0);
  TileAcc<float, C, V_DIM, C, V_DIM> qs_l0;
  TASSIGN(qs_l0, C * C * sizeof(float));
  TileAcc<float, C, V_DIM, C, V_DIM> qkv_l0;
  TASSIGN(qkv_l0, 0);
#endif

#if defined(__DAV_C220_VEC__)
  // ── Vec UB plan (192 KB budget) ──────────────────────────────────────────
  // Persistent (across entire kernel run):
  //   MASK_UB [HalfC, C] fp32 — loaded once, used in every chunk's Phase B.
  // Phase A (input load + scale + cast):
  //   Buffers Q_UB, K_UB, G_UB, EXP_UB, V_UB, S_UB, *H_UB (fp16) live with
  //   careful reuse so peak ≤ ~144 KB.
  // Phase B (mask): QK_UB fp32 + QKH_UB fp16 reuse Phase A addresses.
  // Phase C (combine): QSH/QKVH fp16 + QS/QKV fp32 + final O fp32, all reuse.
  constexpr int32_t MASK_UB_ADDR = 0;
  constexpr int32_t SLOT_A_ADDR  = MASK_UB_ADDR + HalfC * C * sizeof(float);
  constexpr int32_t SLOT_B_ADDR  = SLOT_A_ADDR  + HalfC * K_DIM * sizeof(float);
  constexpr int32_t SLOT_C_ADDR  = SLOT_B_ADDR  + HalfC * K_DIM * sizeof(float);
  constexpr int32_t SLOT_D_ADDR  = SLOT_C_ADDR  + HalfC * K_DIM * sizeof(float);
  // Aliases for clarity (all addresses overlap by phase; lifetimes never collide):
  // Phase A:
  //   SLOT_A: G_UB → V_UB → QSH_UB(fp16, lower-half) / QS_UB(fp32)
  //   SLOT_B: Q_UB / K_UB → QKVH_UB(fp16) / QKV_UB(fp32)
  //   SLOT_C: EXP_UB (scratch for exp(g) and exp(-g)) → QK_UB(fp32 mask) → O_UB
  //   SLOT_D: *H_UB (fp16 cast destination) → QKH_UB(fp16)
#endif

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
      // ── Wait Vec phase A: Q, K, V_corr, S all written to workspace ─────
      wait_flag_dev(0);

      // Load Q [C, K] fp16 from WS_Q.
      {
        GmShape2D q_shape(C, K_DIM);
        GmStride2D q_stride(K_DIM);
        GmTensor2D<half> q_global(workspace_handle + ws_base + WS_Q,
                                  q_shape, q_stride);
        DynMatL1<half, C, K_DIM> q_l1_load(C, K_DIM);
        TASSIGN(q_l1_load, 0);
        TLOAD(q_l1_load, q_global);
      }
      // Load K [C, K] fp16 from WS_K.
      {
        GmShape2D k_shape(C, K_DIM);
        GmStride2D k_stride(K_DIM);
        GmTensor2D<half> k_global(workspace_handle + ws_base + WS_K,
                                  k_shape, k_stride);
        DynMatL1<half, C, K_DIM> k_l1_load(C, K_DIM);
        TASSIGN(k_l1_load, C * K_DIM * sizeof(half));
        TLOAD(k_l1_load, k_global);
      }
      // Load S [K, V] fp16 from WS_S.
      {
        GmShape2D s_shape(K_DIM, V_DIM);
        GmStride2D s_stride(V_DIM);
        GmTensor2D<half> s_global(workspace_handle + ws_base + WS_S,
                                  s_shape, s_stride);
        DynMatL1<half, K_DIM, V_DIM> s_l1_load(K_DIM, V_DIM);
        TASSIGN(s_l1_load, (C * K_DIM + C * K_DIM) * sizeof(half));
        TLOAD(s_l1_load, s_global);
      }
      // Load V_corr [C, V] fp16 from WS_V.
      {
        GmShape2D v_shape(C, V_DIM);
        GmStride2D v_stride(V_DIM);
        GmTensor2D<half> v_global(workspace_handle + ws_base + WS_V,
                                  v_shape, v_stride);
        DynMatL1<half, C, V_DIM> v_l1_load(C, V_DIM);
        TASSIGN(v_l1_load,
                (C * K_DIM + C * K_DIM + KV + C * C) * sizeof(half));
        TLOAD(v_l1_load, v_global);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM1: QK = Q @ K^T — [C, K] @ [K, C] → [C, C].
      gemm_oneshot<half, float, C, C, K_DIM, /*transpose_B=*/true>(
          q_l1, k_l1, qk_l0);

      // Store QK fp16 → WS_QK.
      {
        GmShape2D qk_shape(C, C);
        GmStride2D qk_stride(C);
        GmTensor2D<half> qk_global(workspace_handle + ws_base + WS_QK,
                                   qk_shape, qk_stride);
        TileAcc<float, C, C, C, C> qk_store;
        TASSIGN(qk_store, 0);
        TSTORE(qk_global, qk_store);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM2: QS = Q @ S — [C, K] @ [K, V] → [C, V].
      gemm_oneshot<half, float, C, V_DIM, K_DIM, /*transpose_B=*/false>(
          q_l1, s_l1, qs_l0);

      // Store QS fp16 → WS_QS.
      {
        GmShape2D qs_shape(C, V_DIM);
        GmStride2D qs_stride(V_DIM);
        GmTensor2D<half> qs_global(workspace_handle + ws_base + WS_QS,
                                   qs_shape, qs_stride);
        TileAcc<float, C, V_DIM, C, V_DIM> qs_store;
        TASSIGN(qs_store, C * C * sizeof(float));
        TSTORE(qs_global, qs_store);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (1 << 8));

      // ── Wait Vec phase B: QK masked in WS_QK ──────────────────────────
      wait_flag_dev(2);

      // Load masked QK [C, C] fp16 → qkm_l1.
      {
        GmShape2D qkm_shape(C, C);
        GmStride2D qkm_stride(C);
        GmTensor2D<half> qkm_global(workspace_handle + ws_base + WS_QK,
                                    qkm_shape, qkm_stride);
        DynMatL1<half, C, C> qkm_l1_load(C, C);
        TASSIGN(qkm_l1_load,
                (C * K_DIM + C * K_DIM + KV) * sizeof(half));
        TLOAD(qkm_l1_load, qkm_global);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM3: QKV = QK_masked @ V_corr — [C, C] @ [C, V] → [C, V].
      gemm_oneshot<half, float, C, V_DIM, C, /*transpose_B=*/false>(
          qkm_l1, v_l1, qkv_l0);

      // Store QKV fp16 → WS_QKV.
      {
        GmShape2D qkv_shape(C, V_DIM);
        GmStride2D qkv_stride(V_DIM);
        GmTensor2D<half> qkv_global(workspace_handle + ws_base + WS_QKV,
                                    qkv_shape, qkv_stride);
        TileAcc<float, C, V_DIM, C, V_DIM> qkv_store;
        TASSIGN(qkv_store, 0);
        TSTORE(qkv_global, qkv_store);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));
    }
  }

  sync_all();
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  sync_all();

  auto vid = get_subblockid();
  int32_t my_row_offset = static_cast<int32_t>(vid) * HalfC;

  // ── Load this vid's HalfC rows of the causal mask once per launch ──────
  {
    TileUbDataND<float, HalfC, C, HalfC, C> mask_ub;
    TASSIGN(mask_ub, MASK_UB_ADDR);
    GmShape2D m_shape(HalfC, C);
    GmStride2D m_stride(C);
    GmTensor2D<float> m_global(
        Mask_handle + static_cast<int64_t>(my_row_offset) * C,
        m_shape, m_stride);
    TLOAD(mask_ub, m_global);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

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

    for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;
      int32_t valid_rows =
          static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
      if (valid_rows < 0) valid_rows = 0;
      if (valid_rows > HalfC) valid_rows = HalfC;

      // ====================================================================
      // PHASE A — load Q, K, G_cs; pre-scale q_eff/k_eff; cast V_corr, S.
      // ====================================================================
      int64_t hk_base = static_cast<int64_t>(head) * total_tokens * K_DIM +
                        (chunk_start + static_cast<int64_t>(vid) * HalfC) *
                            K_DIM;

      // Tile views into the UB slots (declared inside the loop so we can
      // re-bind them by phase without touching constexpr globals).
      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero> g_ub;
      TASSIGN(g_ub, SLOT_A_ADDR);
      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero> q_ub;
      TASSIGN(q_ub, SLOT_B_ADDR);
      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> exp_ub;
      TASSIGN(exp_ub, SLOT_C_ADDR);
      TileUbDataND<half,  HalfC, K_DIM, HalfC, K_DIM> qh_ub;
      TASSIGN(qh_ub, SLOT_D_ADDR);

      // ── (A.1) Load Q and G_cs (head-major fp16) ──────────────────────
      if (valid_rows > 0) {
        {
          GmShape2D q_shape(valid_rows, K_DIM);
          GmStride2D q_stride(HM_STRIDE);
          GmTensor2D<half> q_global(Q_handle + hk_base, q_shape, q_stride);
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM,
                       pto::PadValue::Zero> q_stg_full;
          TASSIGN(q_stg_full, SLOT_D_ADDR);
          DynVecTile<half, HalfC, K_DIM, pto::PadValue::Zero> q_load(
              valid_rows, K_DIM);
          TASSIGN(q_load, SLOT_D_ADDR);
          TLOAD(q_load, q_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(q_stg_full, q_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        {
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM> q_stg_cvt;
          TASSIGN(q_stg_cvt, SLOT_D_ADDR);
          TCVT(q_ub, q_stg_cvt, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        }
        {
          GmShape2D g_shape(valid_rows, K_DIM);
          GmStride2D g_stride(HM_STRIDE);
          GmTensor2D<half> g_global(G_handle + hk_base, g_shape, g_stride);
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM,
                       pto::PadValue::Zero> g_stg_full;
          TASSIGN(g_stg_full, SLOT_D_ADDR);
          DynVecTile<half, HalfC, K_DIM, pto::PadValue::Zero> g_load(
              valid_rows, K_DIM);
          TASSIGN(g_load, SLOT_D_ADDR);
          TLOAD(g_load, g_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(g_stg_full, g_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        {
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM> g_stg_cvt;
          TASSIGN(g_stg_cvt, SLOT_D_ADDR);
          TCVT(g_ub, g_stg_cvt, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        }
      } else {
        TEXPANDS(q_ub, 0.0f);
        TEXPANDS(g_ub, 0.0f);
      }

      // ── (A.2) q_eff = Q * exp(g_cs) ──────────────────────────────────
      TEXP(exp_ub, g_ub);
      pipe_barrier(PIPE_V);
      TMUL(q_ub, q_ub, exp_ub);
      pipe_barrier(PIPE_V);
      TCVT(qh_ub, q_ub, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);

      // Store q_eff fp16 → WS_Q (full HalfC rows; padded zeros for invalid).
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        GmShape2D q_shape(HalfC, K_DIM);
        GmStride2D q_stride(K_DIM);
        GmTensor2D<half> q_global(
            workspace_handle + ws_base + WS_Q +
                static_cast<int64_t>(vid) * HalfC * K_DIM,
            q_shape, q_stride);
        DynVecTile<half, HalfC, K_DIM> q_store(HalfC, K_DIM);
        TASSIGN(q_store, SLOT_D_ADDR);
        TSTORE(q_global, q_store);
      }

      // ── (A.3) Load K and compute k_eff = K * exp(-g_cs) ─────────────
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      if (valid_rows > 0) {
        GmShape2D k_shape(valid_rows, K_DIM);
        GmStride2D k_stride(HM_STRIDE);
        GmTensor2D<half> k_global(K_handle + hk_base, k_shape, k_stride);
        TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM,
                     pto::PadValue::Zero> k_stg_full;
        TASSIGN(k_stg_full, SLOT_D_ADDR);
        DynVecTile<half, HalfC, K_DIM, pto::PadValue::Zero> k_load(
            valid_rows, K_DIM);
        TASSIGN(k_load, SLOT_D_ADDR);
        TLOAD(k_load, k_global);
        if (valid_rows != HalfC) {
          TFILLPAD_INPLACE(k_stg_full, k_load);
        }
      } else {
        TEXPANDS(q_ub, 0.0f);
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      if (valid_rows > 0) {
        TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM> k_stg_cvt;
        TASSIGN(k_stg_cvt, SLOT_D_ADDR);
        TCVT(q_ub, k_stg_cvt, pto::RoundMode::CAST_NONE);  // q_ub aliases SLOT_B
        pipe_barrier(PIPE_V);
      }

      // exp_ub = exp(-g_cs):  exp_ub = 0; exp_ub -= g_ub; exp_ub = exp(exp_ub).
      TEXPANDS(exp_ub, 0.0f);
      pipe_barrier(PIPE_V);
      TSUB(exp_ub, exp_ub, g_ub);
      pipe_barrier(PIPE_V);
      TEXP(exp_ub, exp_ub);
      pipe_barrier(PIPE_V);
      // q_ub now holds K (we reused the buffer).
      TMUL(q_ub, q_ub, exp_ub);
      pipe_barrier(PIPE_V);
      TCVT(qh_ub, q_ub, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        GmShape2D k_shape(HalfC, K_DIM);
        GmStride2D k_stride(K_DIM);
        GmTensor2D<half> k_global(
            workspace_handle + ws_base + WS_K +
                static_cast<int64_t>(vid) * HalfC * K_DIM,
            k_shape, k_stride);
        DynVecTile<half, HalfC, K_DIM> k_store(HalfC, K_DIM);
        TASSIGN(k_store, SLOT_D_ADDR);
        TSTORE(k_global, k_store);
      }

      // ── (A.4) Load V_corr fp16 (BSND), store to WS_V ────────────────
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      {
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM,
                     pto::PadValue::Zero> vh_ub;
        TASSIGN(vh_ub, SLOT_D_ADDR);

        int64_t v_offset = (chunk_start * H + head) * V_DIM +
                           static_cast<int64_t>(vid) * HalfC * BSND_STRIDE;
        if (valid_rows > 0) {
          GmShape2D v_shape(valid_rows, V_DIM);
          GmStride2D v_stride(BSND_STRIDE);
          GmTensor2D<half> v_global(V_handle + v_offset, v_shape, v_stride);
          DynVecTile<half, HalfC, V_DIM, pto::PadValue::Zero> v_load(
              valid_rows, V_DIM);
          TASSIGN(v_load, SLOT_D_ADDR);
          TLOAD(v_load, v_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(vh_ub, v_load);
          }
        } else {
          TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> v_zero;
          TASSIGN(v_zero, SLOT_A_ADDR);
          TEXPANDS(v_zero, 0.0f);
          pipe_barrier(PIPE_V);
          TCVT(vh_ub, v_zero, pto::RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmShape2D vw_shape(HalfC, V_DIM);
        GmStride2D vw_stride(V_DIM);
        GmTensor2D<half> vw_global(
            workspace_handle + ws_base + WS_V +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            vw_shape, vw_stride);
        DynVecTile<half, HalfC, V_DIM> v_store(HalfC, V_DIM);
        TASSIGN(v_store, SLOT_D_ADDR);
        TSTORE(vw_global, v_store);
      }

      // ── (A.5) Load S fp16 from snapshots, store to WS_S ─────────────
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      {
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM> sh_ub;
        TASSIGN(sh_ub, SLOT_D_ADDR);

        int64_t s_in_offset =
            (chunk_offset + static_cast<int64_t>(ci)) * H * KV +
            static_cast<int64_t>(head) * KV +
            static_cast<int64_t>(vid) * HalfC * V_DIM;
        GmShape2D s_shape(HalfC, V_DIM);
        GmStride2D s_stride(V_DIM);
        GmTensor2D<half> s_global(S_handle + s_in_offset, s_shape, s_stride);
        DynVecTile<half, HalfC, V_DIM> s_load(HalfC, V_DIM);
        TASSIGN(s_load, SLOT_D_ADDR);
        TLOAD(s_load, s_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmShape2D sw_shape(HalfC, V_DIM);
        GmStride2D sw_stride(V_DIM);
        GmTensor2D<half> sw_global(
            workspace_handle + ws_base + WS_S +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            sw_shape, sw_stride);
        DynVecTile<half, HalfC, V_DIM> s_store(HalfC, V_DIM);
        TASSIGN(s_store, SLOT_D_ADDR);
        TSTORE(sw_global, s_store);
      }

      // ── (A.6) Signal Cube: phase A workspace ready ───────────────────
      pipe_barrier(PIPE_ALL);
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (0 << 8));

      // ====================================================================
      // PHASE B — wait QK from Cube; apply causal mask; write back.
      // ====================================================================
      wait_flag_dev(1);
      pipe_barrier(PIPE_ALL);

      {
        TileUbDataND<half,  HalfC, C, HalfC, C> qkh_ub;
        TASSIGN(qkh_ub, SLOT_D_ADDR);
        TileUbDataND<float, HalfC, C, HalfC, C> qk_ub;
        TASSIGN(qk_ub, SLOT_C_ADDR);
        TileUbDataND<float, HalfC, C, HalfC, C> mask_view;
        TASSIGN(mask_view, MASK_UB_ADDR);

        // Load QK [HalfC, C] fp16 → UB (only this vid's rows).
        GmShape2D qk_shape(HalfC, C);
        GmStride2D qk_stride(C);
        GmTensor2D<half> qk_global(
            workspace_handle + ws_base + WS_QK +
                static_cast<int64_t>(vid) * HalfC * C,
            qk_shape, qk_stride);
        DynVecTile<half, HalfC, C> qk_load(HalfC, C);
        TASSIGN(qk_load, SLOT_D_ADDR);
        TLOAD(qk_load, qk_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TCVT(qk_ub, qkh_ub, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TMUL(qk_ub, qk_ub, mask_view);
        pipe_barrier(PIPE_V);
        TCVT(qkh_ub, qk_ub, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DynVecTile<half, HalfC, C> qk_store(HalfC, C);
        TASSIGN(qk_store, SLOT_D_ADDR);
        TSTORE(qk_global, qk_store);
      }
      pipe_barrier(PIPE_ALL);
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));

      // ====================================================================
      // PHASE C — wait QKV; combine O = QS + QKV; write to GM.
      //
      // Layout: load QS fp16 → SLOT_D, cast to fp32 → SLOT_A.  Then reuse
      // SLOT_D to load QKV fp16, cast to fp32 → SLOT_B.  Add SLOT_A+=SLOT_B,
      // store SLOT_A to GM.  Only one fp16 buffer live at a time.
      // ====================================================================
      wait_flag_dev(3);
      pipe_barrier(PIPE_ALL);

      if (valid_rows > 0) {
        TileUbDataND<half,  HalfC, V_DIM, HalfC, V_DIM> qsh_ub;
        TASSIGN(qsh_ub, SLOT_D_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> qs_ub;
        TASSIGN(qs_ub, SLOT_A_ADDR);

        // Load QS fp16 → SLOT_D.
        {
          GmShape2D qs_shape(HalfC, V_DIM);
          GmStride2D qs_stride(V_DIM);
          GmTensor2D<half> qs_global(
              workspace_handle + ws_base + WS_QS +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              qs_shape, qs_stride);
          DynVecTile<half, HalfC, V_DIM> qs_load(HalfC, V_DIM);
          TASSIGN(qs_load, SLOT_D_ADDR);
          TLOAD(qs_load, qs_global);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Cast QS to fp32 → SLOT_A (frees SLOT_D for QKV load).
        TCVT(qs_ub, qsh_ub, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);

        // Load QKV fp16 → SLOT_D (reuse).
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TileUbDataND<half,  HalfC, V_DIM, HalfC, V_DIM> qkvh_ub;
        TASSIGN(qkvh_ub, SLOT_D_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> qkv_ub;
        TASSIGN(qkv_ub, SLOT_B_ADDR);
        {
          GmShape2D qkv_shape(HalfC, V_DIM);
          GmStride2D qkv_stride(V_DIM);
          GmTensor2D<half> qkv_global(
              workspace_handle + ws_base + WS_QKV +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              qkv_shape, qkv_stride);
          DynVecTile<half, HalfC, V_DIM> qkv_load(HalfC, V_DIM);
          TASSIGN(qkv_load, SLOT_D_ADDR);
          TLOAD(qkv_load, qkv_global);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TCVT(qkv_ub, qkvh_ub, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TADD(qs_ub, qs_ub, qkv_ub);
        pipe_barrier(PIPE_V);

        // Convert O fp32 → fp16, store to GM (BSND).
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM> oh_ub;
        TASSIGN(oh_ub, SLOT_D_ADDR);
        TCVT(oh_ub, qs_ub, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int64_t o_offset = (chunk_start * H + head) * V_DIM +
                           static_cast<int64_t>(vid) * HalfC * BSND_STRIDE;
        GmShape2D o_shape(valid_rows, V_DIM);
        GmStride2D o_stride(BSND_STRIDE);
        GmTensor2D<half> o_global(O_handle + o_offset, o_shape, o_stride);
        DynVecTile<half, HalfC, V_DIM> o_store(valid_rows, V_DIM);
        TASSIGN(o_store, SLOT_D_ADDR);
        TSTORE(o_global, o_store);
      }
      // Drain all pipes before next chunk iteration.  Without this, the next
      // iteration's Phase A.1 TLOAD (PIPE_MTE2 → SLOT_A/B) can race with the
      // in-flight Phase C TSTORE (PIPE_MTE3 reading SLOT_A) or TADD writes
      // (PIPE_V on SLOT_A/B) from this iteration.
      pipe_barrier(PIPE_ALL);
    }
  }

  sync_all();
#endif
}

extern "C" __global__ AICORE void launch_chunk_o_kda(
    __gm__ uint8_t *Q, __gm__ uint8_t *K, __gm__ uint8_t *V_corr,
    __gm__ uint8_t *S, __gm__ uint8_t *G, __gm__ uint8_t *Mask,
    __gm__ uint8_t *workspace, __gm__ uint8_t *O,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  chunk_o_kda_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(Q),
      reinterpret_cast<__gm__ half *>(K),
      reinterpret_cast<__gm__ half *>(V_corr),
      reinterpret_cast<__gm__ half *>(S),
      reinterpret_cast<__gm__ half *>(G),
      reinterpret_cast<__gm__ float *>(Mask),
      reinterpret_cast<__gm__ half  *>(workspace),
      reinterpret_cast<__gm__ half *>(O),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *Q, uint8_t *K, uint8_t *V_corr, uint8_t *S,
    uint8_t *G, uint8_t *Mask,
    uint8_t *workspace, uint8_t *O,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_o_kda<<<block_dim, nullptr, stream>>>(
      Q, K, V_corr, S, G, Mask, workspace, O, cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
