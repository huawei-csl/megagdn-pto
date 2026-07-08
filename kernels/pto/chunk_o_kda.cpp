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
//   workspace [per-core scratch]     float32 — 7 slots × K*V floats
//   O       [B, T, HV, V]            fp32  — output (BSND)
//
// Aqk on the Cube — per-token exp offset (mirrors kkt_kda.cpp):
//   The gate is per-DIMENSION, so exp(g_cs[r,d]-g_cs[c,d]) lives inside the sum
//   over d and cannot be pulled out of a plain matmul.  The naive Cube form
//   A=q*exp(g_cs), B=k*exp(-g_cs), M=A@B^T computes exp(-g_cs)=exp(+500) on Kimi
//   gates → inf → 0*inf = NaN.  Fix: pick one scalar per token b[t]=max_d g_cs[t,d]
//   and factor it out of BOTH legs:
//       A[r,d] = q[r,d]*exp(g_cs[r,d]-b[r])   (<= q, never overflows)
//       B[c,d] = k[c,d]*exp(b[c]-g_cs[c,d])   (clamped at exp(80) for safety)
//       M[r,c] = sum_d A[r,d]*B[c,d]          (= A @ B^T, on the Cube, fp32)
//       Aqk[r,c] = exp(min(b[r]-b[c],0)) * M[r,c]   for r >= c (inclusive mask)
//   b cancels exactly in the per-d product; the residual is the post-matmul
//   scalar correction exp(b[r]-b[c]) (<=1 for kept r>=c, min(.,0) keeps the
//   masked r<c entries finite before the mask zeroes them → no 0*inf=NaN).
//
// NOTE: the workspace (and all three GEMMs) are fp32, not fp16: B=k*exp(b-g_cs)
//   reaches ~e^64 (within-token cross-channel spread) which overflows fp16
//   (max 6.5e4).  fp32 (max 3.4e38) holds it; q/k/v/S arrive fp16 and are cast
//   up; O is cast back to fp16 on write.
//
// Workspace per AI core (9 slots, float32; assumes K == V == HiddenSize):
//   WS_Q   [C, K]   Vec writes q_eff=q*exp(g_cs)     → Cube reads (GEMM2 A)
//   WS_A   [C, K]   Vec writes A=q*exp(g_cs-b)        → Cube reads (GEMM1 A)
//   WS_B   [C, K]   Vec writes B=k*exp(b-g_cs)        → Cube reads (GEMM1 B, transposed)
//   WS_V   [C, V]   Vec writes V_corr fp32            → Cube reads (GEMM3 B)
//   WS_S   [K, V]   Vec writes S fp32                 → Cube reads (GEMM2 B)
//   WS_QK  [C, C]   Cube writes M → Vec corrects+masks → Aqk → Cube reads (GEMM3 A)
//   WS_QS  [C, V]   Cube writes QS fp32               → Vec reads (final combine)
//   WS_QKV [C, V]   Cube writes QKV fp32              → Vec reads (final combine)
//   WS_BOFF [C]     Vec writes b[t] (phase A)         → Vec reads (phase B)
//
// Cross-core FFTS flags (Cube↔Vec, 4-flag round trip; mirrors chunk_o.cpp):
//   flag 0: Vec→Cube — phase-A workspace ready (A, B, q_eff, S, V, b)
//   flag 1: Cube→Vec — M (GEMM1) and QS (GEMM2) written back
//   flag 2: Vec→Cube — Aqk (masked+corrected) ready for GEMM3
//   flag 3: Cube→Vec — QKV (GEMM3) written back
// ============================================================================

#include <pto/pto-inst.hpp>
#include <type_traits>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

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

// Column-vector ([R,1]) UB tiles must be ColMajor: RowMajor NoneBox needs the
// column byte-width 32-byte aligned, which width-1 tiles fail.  Used as the
// per-row offset b[r] source of TROWEXPAND* and the dest of TROWMAX.
template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileUbDataDN = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                               pto::BLayout::ColMajor, RowValid, ColValid,
                               pto::SLayout::NoneBox, 512>;

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

template <int32_t HiddenSize, int32_t ChunkSize>
AICORE void chunk_o_kda_kernel(
    __gm__ half *Q_handle, __gm__ half *K_handle,
    __gm__ half *V_handle, __gm__ half *S_handle,
    __gm__ float *G_handle, __gm__ float *Mask_handle,
    __gm__ float *workspace_handle,
    __gm__ half *O_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    int32_t num_heads, uint64_t ffts_addr)
{
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  set_ffts_base_addr(ffts_addr);

  constexpr int32_t K_DIM = HiddenSize;
  constexpr int32_t V_DIM = HiddenSize;
  constexpr int32_t C     = ChunkSize;
  // Head count (HV) is a runtime argument; it only drives the work-item decode
  // and the BSND GM stride, never a UB buffer size or tile shape.
  const int32_t H     = num_heads;          // HV in KDA terminology
  constexpr int32_t HalfC = C / 2;
  const int32_t BSND_STRIDE = H * HiddenSize;
  constexpr int32_t HM_STRIDE   = HiddenSize;    // head-major Q, K, G stride
  constexpr int32_t KV = K_DIM * V_DIM;

  // ── Workspace slots (fp32 elements, per AI core) ─────────────────────────
  // WS_A / WS_B are the per-token-offset factors of the intra-chunk attention
  // matrix:  A[r,d] = q[r,d]*exp(g_cs[r,d]-b[r]),  B[c,d] = k[c,d]*exp(b[c]-g_cs[c,d])
  // with b[t] = max_d g_cs[t,d].  Cube computes M = A @ B^T; Vec then applies
  // exp(min(b[r]-b[c],0))*mask to recover Aqk.  WS_BOFF ferries b[t] from Vec
  // phase A to Vec phase B (each vid computes half the rows; exchanged via GM).
  constexpr int32_t WS_Q    = 0;
  constexpr int32_t WS_A    = WS_Q    + C * K_DIM;
  constexpr int32_t WS_B    = WS_A    + C * K_DIM;
  constexpr int32_t WS_V    = WS_B    + C * K_DIM;
  constexpr int32_t WS_S    = WS_V    + C * V_DIM;
  constexpr int32_t WS_QK   = WS_S    + KV;
  constexpr int32_t WS_QS   = WS_QK   + C * C;
  constexpr int32_t WS_QKV  = WS_QS   + C * V_DIM;
  constexpr int32_t WS_BOFF = WS_QKV  + C * V_DIM;   // b[t], only [C] used
  constexpr int32_t WS_PER_CORE = WS_BOFF + C * V_DIM;

#if defined(__DAV_C220_CUBE__)
  // ── Cube L1 tiles (fp32, 256 KB high-water) ──────────────────────────────
  // Step 1 (GEMM1+GEMM2) holds A, B, q_eff, S (4×[C,K] = 256 KB).  Step 2
  // (GEMM3) runs after the flag1→phaseB→flag2 round trip, so A/B are consumed —
  // qkm_l1 and v_l1 REUSE the A/B L1 bytes rather than growing the footprint.
  //   a_l1   @ L1_A   : [C, K]  — GEMM1 A factor  (A = q*exp(g_cs-b))
  //   b_l1   @ L1_B   : [C, K]  — GEMM1 B factor  (B = k*exp(b-g_cs), transposed)
  //   q_l1   @ L1_Q   : [C, K]  — GEMM2 A         (q_eff)
  //   s_l1   @ L1_S   : [K, V]  — GEMM2 B         (S)
  //   qkm_l1 @ L1_A   : [C, C]  — GEMM3 A         (masked/corrected Aqk)
  //   v_l1   @ L1_B   : [C, V]  — GEMM3 B         (V_corr)
  constexpr int32_t L1_A   = 0;
  constexpr int32_t L1_B   = L1_A + C * K_DIM * static_cast<int32_t>(sizeof(float));
  constexpr int32_t L1_Q   = L1_B + C * K_DIM * static_cast<int32_t>(sizeof(float));
  constexpr int32_t L1_S   = L1_Q + C * K_DIM * static_cast<int32_t>(sizeof(float));
  constexpr int32_t L1_QKM = L1_A;   // step-2 reuse of A's L1
  constexpr int32_t L1_V   = L1_B;   // step-2 reuse of B's L1

  TileMatL1<float, C, K_DIM, C, K_DIM> a_l1;
  TASSIGN(a_l1, L1_A);
  TileMatL1<float, C, K_DIM, C, K_DIM> b_l1;
  TASSIGN(b_l1, L1_B);
  TileMatL1<float, C, K_DIM, C, K_DIM> q_l1;
  TASSIGN(q_l1, L1_Q);
  TileMatL1<float, K_DIM, V_DIM, K_DIM, V_DIM> s_l1;
  TASSIGN(s_l1, L1_S);
  TileMatL1<float, C, C, C, C> qkm_l1;
  TASSIGN(qkm_l1, L1_QKM);
  TileMatL1<float, C, V_DIM, C, V_DIM> v_l1;
  TASSIGN(v_l1, L1_V);

  // L0C accumulators (separate physical L0C, not L1).
  //   m_l0   @ 0     : [C, C]  — GEMM1 result; stored to WS_QK, then space reused
  //   qs_l0  @ C*C*4 : [C, V]  — GEMM2 result; stored to WS_QS
  //   qkv_l0 @ 0     : [C, V]  — GEMM3 result (reuses m_l0's L0C bytes)
  TileAcc<float, C, C, C, C> m_l0;
  TASSIGN(m_l0, 0);
  TileAcc<float, C, V_DIM, C, V_DIM> qs_l0;
  TASSIGN(qs_l0, C * C * sizeof(float));
  TileAcc<float, C, V_DIM, C, V_DIM> qkv_l0;
  TASSIGN(qkv_l0, 0);
#endif

#if defined(__DAV_C220_VEC__)
  // ── Vec UB plan (192 KB budget) ──────────────────────────────────────────
  // Persistent (across entire kernel run):
  //   MASK_UB [HalfC, C] fp32 — loaded once, used in every chunk's Phase B.
  // Four 32 KB slots (A/B/C/D) reused across phases; BCOL is a tiny tail tile.
  //   Phase A: SLOT_A=g_cs, SLOT_B=q→A factor, SLOT_C=exp/q_eff→k→B factor,
  //            SLOT_D=fp16 staging + exp temp, BCOL=b[r] (ColMajor).
  //   Phase B: SLOT_A=M→Aqk, SLOT_B=corr, SLOT_C=full b[0..C-1] row (+ b[r] view).
  //   Phase C: SLOT_A=QS, SLOT_B=QKV, SLOT_D=fp16 O.
  constexpr int32_t MASK_UB_ADDR = 0;
  constexpr int32_t SLOT_A_ADDR  = MASK_UB_ADDR + HalfC * C * sizeof(float);
  constexpr int32_t SLOT_B_ADDR  = SLOT_A_ADDR  + HalfC * K_DIM * sizeof(float);
  constexpr int32_t SLOT_C_ADDR  = SLOT_B_ADDR  + HalfC * K_DIM * sizeof(float);
  constexpr int32_t SLOT_D_ADDR  = SLOT_C_ADDR  + HalfC * K_DIM * sizeof(float);
  constexpr int32_t BCOL_ADDR    = SLOT_D_ADDR  + HalfC * K_DIM * sizeof(float);  // [HalfC,1] fp32
  constexpr int32_t BROW_ADDR    = SLOT_C_ADDR;  // Phase B: full b[0..C-1] ([1,C] fp32)
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
      // ── STEP 1: wait Vec phase A (A, B, q_eff, S in workspace) ──────────
      wait_flag_dev(0);

      // Load A [C, K] from WS_A → a_l1  (GEMM1 A factor).
      {
        GmShape2D a_shape(C, K_DIM);
        GmStride2D a_stride(K_DIM);
        GmTensor2D<float> a_global(workspace_handle + ws_base + WS_A,
                                   a_shape, a_stride);
        DynMatL1<float, C, K_DIM> a_l1_load(C, K_DIM);
        TASSIGN(a_l1_load, L1_A);
        TLOAD(a_l1_load, a_global);
      }
      // Load B [C, K] from WS_B → b_l1  (GEMM1 B factor, transposed in GEMM1).
      {
        GmShape2D b_shape(C, K_DIM);
        GmStride2D b_stride(K_DIM);
        GmTensor2D<float> b_global(workspace_handle + ws_base + WS_B,
                                   b_shape, b_stride);
        DynMatL1<float, C, K_DIM> b_l1_load(C, K_DIM);
        TASSIGN(b_l1_load, L1_B);
        TLOAD(b_l1_load, b_global);
      }
      // Load q_eff [C, K] from WS_Q → q_l1  (GEMM2 A).
      {
        GmShape2D q_shape(C, K_DIM);
        GmStride2D q_stride(K_DIM);
        GmTensor2D<float> q_global(workspace_handle + ws_base + WS_Q,
                                  q_shape, q_stride);
        DynMatL1<float, C, K_DIM> q_l1_load(C, K_DIM);
        TASSIGN(q_l1_load, L1_Q);
        TLOAD(q_l1_load, q_global);
      }
      // Load S [K, V] from WS_S → s_l1  (GEMM2 B).
      {
        GmShape2D s_shape(K_DIM, V_DIM);
        GmStride2D s_stride(V_DIM);
        GmTensor2D<float> s_global(workspace_handle + ws_base + WS_S,
                                  s_shape, s_stride);
        DynMatL1<float, K_DIM, V_DIM> s_l1_load(K_DIM, V_DIM);
        TASSIGN(s_l1_load, L1_S);
        TLOAD(s_l1_load, s_global);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM1: M = A @ B^T — [C, K] @ [K, C] → [C, C]  (intra-chunk attn, raw).
      gemm_oneshot<float, float, C, C, K_DIM, /*transpose_B=*/true>(
          a_l1, b_l1, m_l0);

      // Store M fp32 → WS_QK (Vec phase B corrects + masks it into Aqk).
      {
        GmShape2D m_shape(C, C);
        GmStride2D m_stride(C);
        GmTensor2D<float> m_global(workspace_handle + ws_base + WS_QK,
                                   m_shape, m_stride);
        TileAcc<float, C, C, C, C> m_store;
        TASSIGN(m_store, 0);
        TSTORE(m_global, m_store);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM2: QS = q_eff @ S — [C, K] @ [K, V] → [C, V]  (inter-chunk term).
      gemm_oneshot<float, float, C, V_DIM, K_DIM, /*transpose_B=*/false>(
          q_l1, s_l1, qs_l0);

      // Store QS fp32 → WS_QS.
      {
        GmShape2D qs_shape(C, V_DIM);
        GmStride2D qs_stride(V_DIM);
        GmTensor2D<float> qs_global(workspace_handle + ws_base + WS_QS,
                                   qs_shape, qs_stride);
        TileAcc<float, C, V_DIM, C, V_DIM> qs_store;
        TASSIGN(qs_store, C * C * sizeof(float));
        TSTORE(qs_global, qs_store);
      }

      // Signal Vec: M (GEMM1) and QS (GEMM2) written back (flag 1).
      pipe_barrier(PIPE_ALL);
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (1 << 8));

      // ── STEP 2: wait Vec phase B (Aqk masked+corrected in WS_QK) ────────
      wait_flag_dev(2);

      // Load Aqk [C, C] from WS_QK → qkm_l1  (GEMM3 A).
      {
        GmShape2D qkm_shape(C, C);
        GmStride2D qkm_stride(C);
        GmTensor2D<float> qkm_global(workspace_handle + ws_base + WS_QK,
                                    qkm_shape, qkm_stride);
        DynMatL1<float, C, C> qkm_l1_load(C, C);
        TASSIGN(qkm_l1_load, L1_QKM);
        TLOAD(qkm_l1_load, qkm_global);
      }
      // Load V_corr [C, V] from WS_V → v_l1  (GEMM3 B).
      {
        GmShape2D v_shape(C, V_DIM);
        GmStride2D v_stride(V_DIM);
        GmTensor2D<float> v_global(workspace_handle + ws_base + WS_V,
                                  v_shape, v_stride);
        DynMatL1<float, C, V_DIM> v_l1_load(C, V_DIM);
        TASSIGN(v_l1_load, L1_V);
        TLOAD(v_l1_load, v_global);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM3: QKV = Aqk @ V_corr — [C, C] @ [C, V] → [C, V].
      gemm_oneshot<float, float, C, V_DIM, C, /*transpose_B=*/false>(
          qkm_l1, v_l1, qkv_l0);

      // Store QKV fp32 → WS_QKV.
      {
        GmShape2D qkv_shape(C, V_DIM);
        GmStride2D qkv_stride(V_DIM);
        GmTensor2D<float> qkv_global(workspace_handle + ws_base + WS_QKV,
                                    qkv_shape, qkv_stride);
        TileAcc<float, C, V_DIM, C, V_DIM> qkv_store;
        TASSIGN(qkv_store, 0);
        TSTORE(qkv_global, qkv_store);
      }

      // Signal Vec: QKV (GEMM3) written back (flag 3).
      pipe_barrier(PIPE_ALL);
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
          GmTensor2D<float> g_global(G_handle + hk_base, g_shape, g_stride);
          TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM,
                       pto::PadValue::Zero> g_stg_full;
          TASSIGN(g_stg_full, SLOT_A_ADDR);
          DynVecTile<float, HalfC, K_DIM, pto::PadValue::Zero> g_load(
              valid_rows, K_DIM);
          TASSIGN(g_load, SLOT_A_ADDR);
          TLOAD(g_load, g_global);  // g_cs fp32 → g_ub directly
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(g_stg_full, g_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        TEXPANDS(q_ub, 0.0f);
        TEXPANDS(g_ub, 0.0f);
      }

      // ── (A.2) q_eff = Q * exp(g_cs) ──────────────────────────────────
      // exp(g_cs) ≤ 1 (g_cs ≤ 0) so q_eff is bounded; kept fp32 to match the
      // fp32 GEMM (k_eff below overflows fp16).
      TEXP(exp_ub, g_ub);
      pipe_barrier(PIPE_V);
      // q_eff into exp_ub (SLOT_C) so q_ub (SLOT_B) keeps the raw scaled Q,
      // which the A = q*exp(g_cs-b) factor below needs as its row factor.
      TMUL(exp_ub, q_ub, exp_ub);
      pipe_barrier(PIPE_V);

      // Store q_eff fp32 → WS_Q (full HalfC rows; padded zeros for invalid).
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        GmShape2D q_shape(HalfC, K_DIM);
        GmStride2D q_stride(K_DIM);
        GmTensor2D<float> q_global(
            workspace_handle + ws_base + WS_Q +
                static_cast<int64_t>(vid) * HalfC * K_DIM,
            q_shape, q_stride);
        DynVecTile<float, HalfC, K_DIM> q_store(HalfC, K_DIM);
        TASSIGN(q_store, SLOT_C_ADDR);
        TSTORE(q_global, q_store);
      }

      // ── (A.3) Per-token offset factors A, B, b for the Cube GEMM1 ────
      // b[r] = max_d g_cs[r,d];  A = q*exp(g_cs-b) (<=q);  B = k*exp(b-g_cs)
      // (exponent clamped at 80).  Cube forms M = A@B^T; Vec phase B applies
      // exp(min(b_r-b_c,0))*mask to recover Aqk.  q_ub (SLOT_B) still holds the
      // raw scaled Q → becomes A in place; k_ub (SLOT_C) → becomes B in place.
      pipe_barrier(PIPE_ALL);  // drain the q_eff store (read SLOT_C) before reuse
      {
        TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> k_ub;
        TASSIGN(k_ub, SLOT_C_ADDR);

        // Load K (head-major fp16) → SLOT_D staging → cvt fp32 → k_ub (SLOT_C).
        if (valid_rows > 0) {
          {
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
          }
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          {
            TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM> k_stg_cvt;
            TASSIGN(k_stg_cvt, SLOT_D_ADDR);
            TCVT(k_ub, k_stg_cvt, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
          }
        } else {
          TEXPANDS(k_ub, 0.0f);  // pad rows: g/q already zero (A.1 else branch)
          pipe_barrier(PIPE_V);
        }

        // b[r] = rowmax_d g_cs   (ColMajor [HalfC,1]; SLOT_D as the reduce tmp).
        TileUbDataDN<float, HalfC, 1> bcol;
        TASSIGN(bcol, BCOL_ADDR);
        {
          TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> rmax_tmp;
          TASSIGN(rmax_tmp, SLOT_D_ADDR);
          TROWMAX(bcol, g_ub, rmax_tmp);
          pipe_barrier(PIPE_V);
        }

        // A = q * exp(g_cs - b)   (exp(g-b) <= 1, bounded) → q_ub (SLOT_B).
        {
          TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> ex;
          TASSIGN(ex, SLOT_D_ADDR);
          TROWEXPANDEXPDIF(ex, g_ub, bcol);  // exp(g_cs - b)
          pipe_barrier(PIPE_V);
          TMUL(q_ub, q_ub, ex);              // A = q * exp(g_cs - b)
          pipe_barrier(PIPE_V);
        }

        // B = k * exp(b - g_cs), exponent saturated at 80 → k_ub (SLOT_C).
        {
          TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> ex;
          TASSIGN(ex, SLOT_D_ADDR);
          TROWEXPANDSUB(ex, g_ub, bcol);  // g_cs - b   (<= 0)
          pipe_barrier(PIPE_V);
          TNEG(ex, ex);                   // b - g_cs   (>= 0)
          pipe_barrier(PIPE_V);
          TMINS(ex, ex, 80.0f);           // saturating exp
          pipe_barrier(PIPE_V);
          TEXP(ex, ex);
          pipe_barrier(PIPE_V);
          TMUL(k_ub, k_ub, ex);           // B = k * exp(b - g_cs)
          pipe_barrier(PIPE_V);
        }

        // Store A (SLOT_B) → WS_A[my rows], B (SLOT_C) → WS_B[my rows].
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        {
          GmShape2D a_shape(HalfC, K_DIM);
          GmStride2D a_stride(K_DIM);
          GmTensor2D<float> a_global(
              workspace_handle + ws_base + WS_A +
                  static_cast<int64_t>(vid) * HalfC * K_DIM,
              a_shape, a_stride);
          DynVecTile<float, HalfC, K_DIM> a_store(HalfC, K_DIM);
          TASSIGN(a_store, SLOT_B_ADDR);
          TSTORE(a_global, a_store);
        }
        {
          GmShape2D b_shape(HalfC, K_DIM);
          GmStride2D b_stride(K_DIM);
          GmTensor2D<float> b_global(
              workspace_handle + ws_base + WS_B +
                  static_cast<int64_t>(vid) * HalfC * K_DIM,
              b_shape, b_stride);
          DynVecTile<float, HalfC, K_DIM> b_store(HalfC, K_DIM);
          TASSIGN(b_store, SLOT_C_ADDR);
          TSTORE(b_global, b_store);
        }
        // Store b[my rows] → WS_BOFF.  bcol is ColMajor [my_rows,1] but its bytes
        // are my_rows contiguous floats == RowMajor [1,my_rows]; alias & store as
        // a row (ND2ND).  Invalid (pad) tokens' b is left stale but harmless: their
        // A,B are zero so M's columns are zero, and corr is bounded (<=1).
        if (valid_rows > 0) {
          GmShape2D bo_shape(1, valid_rows);
          GmStride2D bo_stride(valid_rows);
          GmTensor2D<float> bo_global(
              workspace_handle + ws_base + WS_BOFF +
                  static_cast<int64_t>(my_row_offset),
              bo_shape, bo_stride);
          DynVecTile<float, 1, HalfC> bo_store(1, valid_rows);
          TASSIGN(bo_store, BCOL_ADDR);
          TSTORE(bo_global, bo_store);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
      }

      // ── (A.4) Load V_corr fp16 (BSND), store to WS_V ────────────────
      // WAR on SLOT_D: the V staging TLOAD (MTE2) must wait for the phase-A.3
      // stores (MTE3) that read SLOT_B/C and the SLOT_D exp temp.  MTE3→V also
      // covers the valid_rows==0 branch, which writes SLOT_D via the V pipe.
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      {
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM,
                     pto::PadValue::Zero> vh_ub;
        TASSIGN(vh_ub, SLOT_D_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> v_f_ub;
        TASSIGN(v_f_ub, SLOT_A_ADDR);

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
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TCVT(v_f_ub, vh_ub, pto::RoundMode::CAST_NONE);  // fp16 → fp32
          pipe_barrier(PIPE_V);
        } else {
          TEXPANDS(v_f_ub, 0.0f);
          pipe_barrier(PIPE_V);
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmShape2D vw_shape(HalfC, V_DIM);
        GmStride2D vw_stride(V_DIM);
        GmTensor2D<float> vw_global(
            workspace_handle + ws_base + WS_V +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            vw_shape, vw_stride);
        DynVecTile<float, HalfC, V_DIM> v_store(HalfC, V_DIM);
        TASSIGN(v_store, SLOT_A_ADDR);
        TSTORE(vw_global, v_store);
      }

      // ── (A.5) Load S fp16 from snapshots, store to WS_S ─────────────
      // WAR on SLOT_D: the S staging TLOAD (MTE2) must wait for the WS_V
      // store (MTE3) that just read SLOT_D.
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      {
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM> sh_ub;
        TASSIGN(sh_ub, SLOT_D_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> s_f_ub;
        TASSIGN(s_f_ub, SLOT_A_ADDR);

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
        TCVT(s_f_ub, sh_ub, pto::RoundMode::CAST_NONE);  // fp16 → fp32
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmShape2D sw_shape(HalfC, V_DIM);
        GmStride2D sw_stride(V_DIM);
        GmTensor2D<float> sw_global(
            workspace_handle + ws_base + WS_S +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            sw_shape, sw_stride);
        DynVecTile<float, HalfC, V_DIM> s_store(HalfC, V_DIM);
        TASSIGN(s_store, SLOT_A_ADDR);
        TSTORE(sw_global, s_store);
      }

      // ── (A.6) Signal Cube: phase A workspace ready (flag 0) ──────────
      pipe_barrier(PIPE_ALL);
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (0 << 8));

      // ====================================================================
      // PHASE B — wait M (GEMM1) from Cube; correct + mask M → Aqk → WS_QK.
      //   Aqk[r,c] = exp(min(b[r]-b[c],0)) * M[r,c] * mask[r,c]  (r >= c).
      // ====================================================================
      wait_flag_dev(1);
      pipe_barrier(PIPE_ALL);

      if (valid_rows > 0) {
        // Load M[my rows, :] fp32 from WS_QK → SLOT_A.
        {
          GmShape2D m_shape(HalfC, C);
          GmStride2D m_stride(C);
          GmTensor2D<float> m_global(
              workspace_handle + ws_base + WS_QK +
                  static_cast<int64_t>(my_row_offset) * C,
              m_shape, m_stride);
          DynVecTile<float, HalfC, C> m_load(HalfC, C);
          TASSIGN(m_load, SLOT_A_ADDR);
          TLOAD(m_load, m_global);
        }
        // Load full b[0..C-1] fp32 from WS_BOFF → BROW_ADDR ([1,C], ND2ND).
        {
          GmShape2D bo_shape(1, C);
          GmStride2D bo_stride(C);
          GmTensor2D<float> bo_global(workspace_handle + ws_base + WS_BOFF,
                                      bo_shape, bo_stride);
          TileUbDataND<float, 1, C, 1, C> brow_load;
          TASSIGN(brow_load, BROW_ADDR);
          TLOAD(brow_load, bo_global);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TileUbDataND<float, HalfC, C, HalfC, C> m_ub;
        TASSIGN(m_ub, SLOT_A_ADDR);
        TileUbDataND<float, HalfC, C, HalfC, C> corr;
        TASSIGN(corr, SLOT_B_ADDR);
        TileUbDataND<float, 1, C, 1, C> brow;
        TASSIGN(brow, BROW_ADDR);
        // b[my_off + r] sub-range of brow, viewed as a ColMajor [HalfC,1].
        TileUbDataDN<float, HalfC, 1> bcol;
        TASSIGN(bcol, BROW_ADDR + static_cast<int32_t>(my_row_offset) * 4);
        TileUbDataND<float, HalfC, C, HalfC, C> mask_ub;
        TASSIGN(mask_ub, MASK_UB_ADDR);

        // corr[r,c] = exp(min(b[r] - b[c], 0)).
        TEXPANDS(corr, 0.0f);
        pipe_barrier(PIPE_V);
        TROWEXPANDADD(corr, corr, bcol);  // corr[r,c] = b[r]
        pipe_barrier(PIPE_V);
        TCOLEXPANDSUB(corr, corr, brow);  // corr[r,c] = b[r] - b[c]
        pipe_barrier(PIPE_V);
        TMINS(corr, corr, 0.0f);
        pipe_barrier(PIPE_V);
        TEXP(corr, corr);
        pipe_barrier(PIPE_V);

        // Aqk = M * corr * mask.
        TMUL(m_ub, m_ub, corr);
        pipe_barrier(PIPE_V);
        TMUL(m_ub, m_ub, mask_ub);
        pipe_barrier(PIPE_V);

        // Store Aqk fp32 → WS_QK[my rows].
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        {
          GmShape2D aqk_shape(HalfC, C);
          GmStride2D aqk_stride(C);
          GmTensor2D<float> aqk_global(
              workspace_handle + ws_base + WS_QK +
                  static_cast<int64_t>(my_row_offset) * C,
              aqk_shape, aqk_stride);
          DynVecTile<float, HalfC, C> aqk_store(HalfC, C);
          TASSIGN(aqk_store, SLOT_A_ADDR);
          TSTORE(aqk_global, aqk_store);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }

      // Signal Cube: Aqk ready for GEMM3 (flag 2).  Both vids signal (mode-2
      // reduce) so the Cube waits for the full [C,C] Aqk.
      pipe_barrier(PIPE_ALL);
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));

      // ====================================================================
      // PHASE C — wait QKV (GEMM3, flag 3); O = QS + QKV → GM.
      //   QS was produced by the Cube alongside GEMM1 (step 1) and sits in WS_QS.
      // ====================================================================
      wait_flag_dev(3);
      pipe_barrier(PIPE_ALL);

      if (valid_rows > 0) {
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> qs_ub;
        TASSIGN(qs_ub, SLOT_A_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> qkv_ub;
        TASSIGN(qkv_ub, SLOT_B_ADDR);

        // Load QS fp32 → SLOT_A.
        {
          GmShape2D qs_shape(HalfC, V_DIM);
          GmStride2D qs_stride(V_DIM);
          GmTensor2D<float> qs_global(
              workspace_handle + ws_base + WS_QS +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              qs_shape, qs_stride);
          DynVecTile<float, HalfC, V_DIM> qs_load(HalfC, V_DIM);
          TASSIGN(qs_load, SLOT_A_ADDR);
          TLOAD(qs_load, qs_global);
        }
        // Load QKV fp32 → SLOT_B.
        {
          GmShape2D qkv_shape(HalfC, V_DIM);
          GmStride2D qkv_stride(V_DIM);
          GmTensor2D<float> qkv_global(
              workspace_handle + ws_base + WS_QKV +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              qkv_shape, qkv_stride);
          DynVecTile<float, HalfC, V_DIM> qkv_load(HalfC, V_DIM);
          TASSIGN(qkv_load, SLOT_B_ADDR);
          TLOAD(qkv_load, qkv_global);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // O = QS + QKV  (both bounded; fp32).
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
    int32_t num_heads, uint64_t ffts_addr)
{
  chunk_o_kda_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(Q),
      reinterpret_cast<__gm__ half *>(K),
      reinterpret_cast<__gm__ half *>(V_corr),
      reinterpret_cast<__gm__ half *>(S),
      reinterpret_cast<__gm__ float *>(G),
      reinterpret_cast<__gm__ float *>(Mask),
      reinterpret_cast<__gm__ float *>(workspace),
      reinterpret_cast<__gm__ half *>(O),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, num_heads, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *Q, uint8_t *K, uint8_t *V_corr, uint8_t *S,
    uint8_t *G, uint8_t *Mask,
    uint8_t *workspace, uint8_t *O,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint32_t num_heads)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_o_kda<<<block_dim, nullptr, stream>>>(
      Q, K, V_corr, S, G, Mask, workspace, O, cu_seqlens,
      batch_size, seq_len, total_tokens,
      static_cast<int32_t>(num_heads), fftsAddr);
}
