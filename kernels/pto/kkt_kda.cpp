// ============================================================================
// kkt_kda.cpp — Within-chunk gated attention matrix for KDA
//
// Mathematical operation (per chunk of C tokens, per head h):
//   A_eff[r,d] = k[r,d] * exp(g_cs[r,d])
//   B_eff[c,d] = k[c,d] * exp(-g_cs[c,d])
//   L_full[r,c] = sum_d A_eff[r,d] * B_eff[c,d]   (= A_eff @ B_eff^T via GEMM)
//   L[r,c]     = beta[r] * L_full[r,c]  for r > c (strictly lower-tri), else 0
//
// Inputs (all on GM):
//   k       [HV, total_tokens, K]  float32  — keys, head-major (pre-transposed)
//   g_cs    [HV, total_tokens, K]  float32  — within-chunk cumulative gate sum
//   beta    [HV, total_tokens]     float32  — post-sigmoid beta in (0, 1)
//   mask    [C, C]                 float32  — strictly-lower-tri mask (1 below diag, else 0)
//   ws_in   [block_dim*2, 2*C, K]  half     — workspace: slot A_eff (rows 0..C-1) + B_eff (C..2C-1)
//   ws_out  [block_dim*2, C, C]    half     — workspace: GEMM result L_full
//
// Output:
//   L_out   [total_tokens, HV, C]  float32  — strictly-lower-tri L matrix (BSND)
//
// Difference from GDN scaled_dot_kkt (kernels/pto/scaled_dot_kkt.cpp):
//   GDN: gate is scalar, applied after K@K^T as exp(g[r]-g[c])*beta[r].
//   KDA: gate is a K-vector applied inside the dot product.
//        Requires Vec pre-gating phase before Cube GEMM (3-phase pipeline).
//   GDN: K is fp16; KDA: k,g_cs are fp32, cast to fp16 after gating.
//   GDN: output fp16, causal mask inclusive of diagonal.
//   KDA: output fp32, strictly lower-tri mask (diagonal excluded).
//   GDN: 4 FFTS flags; KDA: 6 FFTS flags (ws_in ready/free pair added).
//
// Cross-core architecture:
//   Vec pre:  load k, g_cs → A_eff=k*exp(g), B_eff=k*exp(-g), cast fp16 → ws_in
//   Cube:     load A_eff, B_eff from ws_in → GEMM A_eff @ B_eff^T → ws_out
//   Vec post: load ws_out, cast fp32 → apply mask + beta row-scale → L_out
//
// FFTS flags (double-buffered, slot = ci & 1):
//   0, 1 : Vec → Cube  "ws_in[slot] ready"  (A_eff/B_eff stored)
//   2, 3 : Cube → Vec  "ws_out[slot] ready"  (GEMM done; ws_in[slot] also free)
//   4, 5 : Vec → Cube  "ws_out[slot] free"   (Vec done reading L_full)
//
// Vec uses only vid=0 (vid=1 exits immediately) — avoids inter-vid sync on
// ws_in writes.  Mirrors gate_cumsum_kda.cpp.
//
// UB budget (vid=0, full C rows, for H=4/K=128/C=16):
//   pre: 5×[C,KTC]fp32 + 2×[C,KTC]fp16 ≈ 48 KB
//   post: mask+L+beta+beta2d [C,C] each ≈ 4 KB
//   total ≈ 52 KB  (fits in 256 KB)
//
// Template parameters:
//   GDN_H = HV, GDN_D = K, GDN_C = C
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

#ifdef __CCE_AICORE__
// Row-major UB tile (Vec engine).
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

// Column-major UB tile (required as TROWEXPAND source — broadcasts column→rows).
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                       RV, CV, pto::SLayout::NoneBox, 512>;

// L1 tile in NZ fractal layout (Cube left/right operand).
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1 tile in ZN layout — logical transpose for right GEMM operand.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor,
                          RV, CV, pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;
#endif

template <int32_t NumHeads, int32_t KDim, int32_t ChunkSize>
AICORE void kkt_kda_kernel(
    __gm__ float *k_ptr,
    __gm__ float *g_cs_ptr,
    __gm__ float *beta_ptr,
    __gm__ float *mask_ptr,
    __gm__ half *ws_in_ptr,
    __gm__ half *ws_out_ptr,
    __gm__ float *L_out_ptr,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
    auto cid = get_block_idx();
    auto block_num = get_block_num();
    auto vid = get_subblockid();
    set_ffts_base_addr(ffts_addr);

    // KTC: KDim rounded up to Vec 8-element alignment (32-byte for fp32).
    constexpr int32_t KTC = ((KDim + 7) / 8) * 8;
    constexpr int32_t ChunkSq = ChunkSize * ChunkSize;

    // ── UB address map (bytes, vid=0 only) ──────────────────────────────────
    // Five [ChunkSize, KTC] float32 tiles for pre-compute:
    constexpr int32_t GUbAddr = 0;
    constexpr int32_t KUbAddr = GUbAddr + ChunkSize * KTC * 4;
    constexpr int32_t ExpGUbAddr = KUbAddr + ChunkSize * KTC * 4;
    constexpr int32_t AUbAddr = ExpGUbAddr + ChunkSize * KTC * 4;
    constexpr int32_t BUbAddr = AUbAddr + ChunkSize * KTC * 4;
    // Two [ChunkSize, KTC] fp16 tiles (cast before workspace store):
    constexpr int32_t AHalfUbAddr = BUbAddr + ChunkSize * KTC * 4;
    constexpr int32_t BHalfUbAddr = AHalfUbAddr + ChunkSize * KTC * 2;
    // Post-process tiles:
    constexpr int32_t MskUbAddr = BHalfUbAddr + ChunkSize * KTC * 2;
    constexpr int32_t LHalfUbAddr = MskUbAddr + ChunkSq * 4;
    constexpr int32_t LUbAddr = LHalfUbAddr + ChunkSq * 2;
    // Beta: [1, ChunkSize] row vector, then expanded to [ChunkSize, ChunkSize]:
    constexpr int32_t BetaUbAddr = LUbAddr + ChunkSq * 4;
    constexpr int32_t Beta2dUbAddr = BetaUbAddr + ChunkSize * 4;

    // Workspace element counts per slot (fp16 elements):
    constexpr int32_t WsInSlotElems = 2 * ChunkSize * KDim; // A_eff + B_eff
    constexpr int32_t WsOutSlotElems = ChunkSq;             // L_full

    int64_t num_seqs = batch_size;
    int64_t total_work = num_seqs * NumHeads;

    // ── GM type aliases ──────────────────────────────────────────────────────
    using GmShapeDyn = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    // k/g_cs: [HV, total_tokens, K] fp32, row stride = K between tokens.
    using GmFloat32K = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    // beta: [HV, total_tokens] fp32, stride 1 between tokens.
    using GmFloat32_1 = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, 1, 1>>;
    // L_out: [total_tokens, HV, C] fp32, row stride = HV*C between tokens (BSND).
    using GmFloat32Out = GlobalTensor<float, GmShapeDyn,
                                      Stride<1, 1, 1, NumHeads * ChunkSize, 1>>;
    // workspace_in: [block_dim*2, 2*C, K] fp16, contiguous rows.
    using GmHalfWsIn = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    // workspace_out: [block_dim*2, C, C] fp16, contiguous rows.
    using GmHalfWsOut = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, ChunkSize, 1>>;

    // =========================================================================
    // VEC PHASE
    // =========================================================================
#if defined(__DAV_C220_VEC__)
    if (vid == 1)
    {
        for (int64_t work_idx = 0;
             work_idx < (total_work + block_num - 1) / block_num; ++work_idx)
        {
            int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                          static_cast<int64_t>(cid);
            if (pid >= total_work)
                continue;
            int32_t head_idx = static_cast<int32_t>(pid % NumHeads);
            int64_t seq_idx = pid / NumHeads;

            int64_t bos, slen;
            if (cu_seqlens != nullptr)
            {
                bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
                slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
            }
            else
            {
                bos = seq_idx * seq_len;
                slen = seq_len;
            }
            int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

            for (int64_t ci = 0; ci < num_chunks; ++ci)
            {
                int32_t slot = static_cast<int32_t>(ci & 1);
                // Signal Cube : ws_in[slot] is ready
                ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (slot << 8));
                // Wait for Cube: ws_out[slot] ready (GEMM done)
                wait_flag_dev(2 + slot);
                pipe_barrier(PIPE_ALL);
                ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((4 + slot) << 8));
            }
        }

        return; // vid=1 exits; vid=0 processes all rows.
    }

    set_mask_norm();
    set_vector_mask(-1, -1);

    // ── Bind post-process UB tiles (reused across all chunks) ───────────────
    UbND<float, ChunkSize, ChunkSize> msk_ub;
    TASSIGN(msk_ub, MskUbAddr);
    UbND<float, ChunkSize, ChunkSize> L_ub;
    TASSIGN(L_ub, LUbAddr);

    // ── Load strictly-lower-tri mask once ────────────────────────────────────
    {
        GmShapeDyn gs;
        gs.shape[3] = ChunkSize;
        gs.shape[4] = ChunkSize;
        GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, ChunkSize, 1>>
            msk_gm(mask_ptr, gs);
        TLOAD(msk_ub, msk_gm);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ── Work item loop ───────────────────────────────────────────────────────
    for (int64_t work_idx = 0;
         work_idx < (total_work + block_num - 1) / block_num; ++work_idx)
    {
        int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                      static_cast<int64_t>(cid);
        if (pid >= total_work)
            continue;

        int32_t head_idx = static_cast<int32_t>(pid % NumHeads);
        int64_t seq_idx = pid / NumHeads;

        int64_t bos, slen;
        if (cu_seqlens != nullptr)
        {
            bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
            slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
        }
        else
        {
            bos = seq_idx * seq_len;
            slen = seq_len;
        }
        int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

        for (int64_t ci = 0; ci < num_chunks; ++ci)
        {
            int32_t slot = static_cast<int32_t>(ci & 1);

            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);

            // Head-major offset for k and g_cs: head_idx * T * K + token * K
            int64_t hk_base = static_cast<int64_t>(head_idx) * total_tokens * KDim + (bos + chunk_start) * KDim;

            // ── PRE-COMPUTE: load k, g_cs → A_eff, B_eff → ws_in ────────────

            // Load g_cs [valid_rows, KDim] → UB (zero-pad to [C, KTC])
            {
                GmShapeDyn gs;
                gs.shape[3] = valid_rows;
                gs.shape[4] = KDim;
                GmFloat32K gm(g_cs_ptr + hk_base, gs);
                UbND<float, ChunkSize, KTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                    g_ld(valid_rows, KDim);
                TASSIGN(g_ld, GUbAddr);
                TLOAD(g_ld, gm);
                if (valid_rows != ChunkSize || KDim != KTC)
                {
                    UbND<float, ChunkSize, KTC, ChunkSize, KTC, PadValue::Zero> g_pad;
                    TASSIGN(g_pad, GUbAddr);
                    TFILLPAD_INPLACE(g_pad, g_ld);
                }
            }
            // Load k [valid_rows, KDim] → UB (zero-pad to [C, KTC])
            {
                GmShapeDyn gs;
                gs.shape[3] = valid_rows;
                gs.shape[4] = KDim;
                GmFloat32K gm(k_ptr + hk_base, gs);
                UbND<float, ChunkSize, KTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                    k_ld(valid_rows, KDim);
                TASSIGN(k_ld, KUbAddr);
                TLOAD(k_ld, gm);
                if (valid_rows != ChunkSize || KDim != KTC)
                {
                    UbND<float, ChunkSize, KTC, ChunkSize, KTC, PadValue::Zero> k_pad;
                    TASSIGN(k_pad, KUbAddr);
                    TFILLPAD_INPLACE(k_pad, k_ld);
                }
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            UbND<float, ChunkSize, KTC> g_ub;
            TASSIGN(g_ub, GUbAddr);
            UbND<float, ChunkSize, KTC> k_ub;
            TASSIGN(k_ub, KUbAddr);
            UbND<float, ChunkSize, KTC> exp_g_ub;
            TASSIGN(exp_g_ub, ExpGUbAddr);
            UbND<float, ChunkSize, KTC> A_ub;
            TASSIGN(A_ub, AUbAddr);
            UbND<float, ChunkSize, KTC> B_ub;
            TASSIGN(B_ub, BUbAddr);
            UbND<half, ChunkSize, KTC> A_half_ub;
            TASSIGN(A_half_ub, AHalfUbAddr);
            UbND<half, ChunkSize, KTC> B_half_ub;
            TASSIGN(B_half_ub, BHalfUbAddr);

            // A_eff = k * exp(g_cs)
            TEXP(exp_g_ub, g_ub);
            pipe_barrier(PIPE_V);
            TMUL(A_ub, k_ub, exp_g_ub);
            pipe_barrier(PIPE_V);

            // B_eff = k * exp(-g_cs): reuse exp_g_ub for -g_cs then exp
            TEXPANDS(exp_g_ub, 0.0f);
            pipe_barrier(PIPE_V);
            TSUB(exp_g_ub, exp_g_ub, g_ub); // 0 - g_cs = -g_cs
            pipe_barrier(PIPE_V);
            TEXP(exp_g_ub, exp_g_ub); // exp(-g_cs)
            pipe_barrier(PIPE_V);
            TMUL(B_ub, k_ub, exp_g_ub);
            pipe_barrier(PIPE_V);

            // Cast A_eff, B_eff to fp16
            TCVT(A_half_ub, A_ub, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            TCVT(B_half_ub, B_ub, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);

            // Store A_half (rows 0..C-1) and B_half (rows C..2C-1) to ws_in[slot]
            int64_t ws_in_base = (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsInSlotElems);
            {
                GmShapeDyn gs;
                gs.shape[3] = ChunkSize;
                gs.shape[4] = KDim;
                GmHalfWsIn gm_a(ws_in_ptr + ws_in_base, gs);
                UbND<half, ChunkSize, KTC, ChunkSize, KTC> A_st;
                TASSIGN(A_st, AHalfUbAddr);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(gm_a, A_st);
            }
            {
                GmShapeDyn gs;
                gs.shape[3] = ChunkSize;
                gs.shape[4] = KDim;
                GmHalfWsIn gm_b(ws_in_ptr + ws_in_base + static_cast<int64_t>(ChunkSize) * KDim, gs);
                UbND<half, ChunkSize, KTC, ChunkSize, KTC> B_st;
                TASSIGN(B_st, BHalfUbAddr);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(gm_b, B_st);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

            // Signal Cube : ws_in[slot] is ready
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (slot << 8));
            // Wait for Cube: ws_out[slot] ready (GEMM done)
            wait_flag_dev(2 + slot);
            pipe_barrier(PIPE_ALL);

            // ── POST-PROCESS: load ws_out → mask + beta → L_out ─────────────

            // Load L_full_h [valid_rows, C] from ws_out[slot], zero-pad to [C, C]
            {
                int64_t wo_base = (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsOutSlotElems);
                GmShapeDyn gs;
                gs.shape[3] = valid_rows;
                gs.shape[4] = ChunkSize;
                GmHalfWsOut gm(ws_out_ptr + wo_base, gs);
                UbND<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero>
                    L_h_ld(valid_rows, ChunkSize);
                TASSIGN(L_h_ld, LHalfUbAddr);
                TLOAD(L_h_ld, gm);
                if (valid_rows != ChunkSize)
                {
                    UbND<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize,
                         PadValue::Zero>
                        L_h_pad;
                    TASSIGN(L_h_pad, LHalfUbAddr);
                    TFILLPAD_INPLACE(L_h_pad, L_h_ld);
                }
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Cast L_full fp16 → fp32
            UbND<half, ChunkSize, ChunkSize> L_half_ub;
            TASSIGN(L_half_ub, LHalfUbAddr);
            TCVT(L_ub, L_half_ub, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);

            // Apply strictly-lower-tri mask (zeros diagonal and upper triangle)
            TMUL(L_ub, L_ub, msk_ub);
            pipe_barrier(PIPE_V);

            // Load beta [valid_rows] for this chunk/head, zero-pad to [1, C]
            {
                int64_t beta_base = static_cast<int64_t>(head_idx) * total_tokens + (bos + chunk_start);
                GmShapeDyn gs;
                gs.shape[3] = 1;
                gs.shape[4] = valid_rows;
                GmFloat32_1 gm(beta_ptr + beta_base, gs);
                UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero>
                    beta_ld(1, valid_rows);
                TASSIGN(beta_ld, BetaUbAddr);
                TLOAD(beta_ld, gm);
                if (valid_rows != ChunkSize)
                {
                    UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> beta_pad;
                    TASSIGN(beta_pad, BetaUbAddr);
                    TFILLPAD_INPLACE(beta_pad, beta_ld);
                }
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Row-scale L by beta: L[r,:] *= beta[r].
            // beta is stored as [1, C] at BetaUbAddr.  Alias it as a UbDN [C, 1]
            // column vector (same bytes, column-major interpretation).
            // TROWEXPAND broadcasts the column to [C, C]: out[r,c] = col[r].
            {
                UbDN<float, ChunkSize, 1> beta_col;
                TASSIGN(beta_col, BetaUbAddr);
                UbND<float, ChunkSize, ChunkSize> beta_2d;
                TASSIGN(beta_2d, Beta2dUbAddr);
                TROWEXPAND(beta_2d, beta_col);
                pipe_barrier(PIPE_V);
                TMUL(L_ub, L_ub, beta_2d);
                pipe_barrier(PIPE_V);
            }

            // Store valid_rows rows to L_out in BSND layout [total_tokens, HV, C]
            {
                int64_t l_base = (bos + chunk_start) * static_cast<int64_t>(NumHeads) * ChunkSize + static_cast<int64_t>(head_idx) * ChunkSize;
                GmShapeDyn gs;
                gs.shape[3] = valid_rows;
                gs.shape[4] = ChunkSize;
                GmFloat32Out gm(L_out_ptr + l_base, gs);
                UbND<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC>
                    L_st(valid_rows, ChunkSize);
                TASSIGN(L_st, LUbAddr);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(gm, L_st);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

            // Signal Cube: ws_out[slot] is free for reuse
            if (ci < num_chunks - 2)
                ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((4 + slot) << 8));
        }
    }
#endif // __DAV_C220_VEC__

// =========================================================================
// CUBE PHASE: GEMM  A_eff @ B_eff^T  → L_full
// =========================================================================
#if defined(__DAV_C220_CUBE__)
    // Number of valid columns in the last 128-wide L1 fractal block.
    constexpr uint32_t KTail =
        (KDim % 128 == 0) ? 128u : static_cast<uint32_t>(KDim % 128);

    // L1 address for B_eff tile (placed immediately after A_eff).
    constexpr int32_t L1BAddr = ChunkSize * KDim * static_cast<int32_t>(sizeof(half));

    // Declare L1 tiles for A_eff and B_eff at fixed L1 addresses.
    L1Mat<half, ChunkSize, KDim, ChunkSize, KDim> a_l1;
    TASSIGN(a_l1, 0);
    L1Mat<half, ChunkSize, KDim, ChunkSize, KDim> b_l1;
    TASSIGN(b_l1, L1BAddr);
    TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a_l0;
    TASSIGN(a_l0, 0);

    for (int64_t work_idx = 0;
         work_idx < (total_work + block_num - 1) / block_num; ++work_idx)
    {
        int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                      static_cast<int64_t>(cid);
        if (pid >= total_work)
            continue;

        int64_t seq_idx = pid / NumHeads;

        int64_t bos, slen;
        if (cu_seqlens != nullptr)
        {
            bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
            slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
        }
        else
        {
            bos = seq_idx * seq_len;
            slen = seq_len;
        }
        int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

        for (int64_t ci = 0; ci < num_chunks; ++ci)
        {
            int32_t slot = static_cast<int32_t>(ci & 1);

            // For ci >= 2: wait until Vec finishes post-processing the previous
            // chunk with the same slot (so ws_out[slot] is free to overwrite).
            if (ci >= 2)
            {
                wait_flag_dev(4 + slot);
                pipe_barrier(PIPE_ALL);
            }
            // Wait for Vec pre-compute: ws_in[slot] is ready.
            wait_flag_dev(slot);
            pipe_barrier(PIPE_ALL);

            int64_t ws_in_base = (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsInSlotElems);

            // ── Load A_eff [C, K] from ws_in[slot, 0:C, :] into L1 ──────────
            {
                L1Mat<half, ChunkSize, KDim, DYNAMIC, DYNAMIC> _l1(ChunkSize, KDim);
                TASSIGN(_l1, 0);
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = ChunkSize;
                _gs.shape[4] = KDim;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, KDim, 1>>
                    _gm(ws_in_ptr + ws_in_base, _gs);
                TLOAD(_l1, _gm);
            }
            // ── Load B_eff [C, K] from ws_in[slot, C:2C, :] into L1 ─────────
            {
                L1Mat<half, ChunkSize, KDim, DYNAMIC, DYNAMIC> _l1(ChunkSize, KDim);
                TASSIGN(_l1, L1BAddr);
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = ChunkSize;
                _gs.shape[4] = KDim;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, KDim, 1>>
                    _gm(ws_in_ptr + ws_in_base + static_cast<int64_t>(ChunkSize) * KDim, _gs);
                TLOAD(_l1, _gm);
            }

            // ── GEMM: A_eff @ B_eff^T → L0C accumulator ─────────────────────
            // WAR sync: ensure MTE2 (load) and M (prior GEMM) are done before
            // MTE1 extracts to L0A/L0B.
            {
                TileLeft<half, ChunkSize, KDim, ChunkSize, KDim> _l0a;
                TileRight<half, KDim, ChunkSize, KDim, ChunkSize> _l0b;
                TASSIGN(_l0a, 0x0);
                TASSIGN(_l0b, 0x0);
                auto _we = EVENT_ID1;
                set_flag(PIPE_MTE2, PIPE_MTE1, _we);
                wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
                set_flag(PIPE_M, PIPE_MTE1, _we);
                wait_flag(PIPE_M, PIPE_MTE1, _we);

                // A_eff (NZ) → L0A
                TEXTRACT(_l0a, a_l1, 0, 0);
                // B_eff → ZN layout (logical transpose = B_eff^T) → L0B
                L1MatZN<half, KDim, ChunkSize> _b_zn;
                TASSIGN(_b_zn, L1BAddr);
                TRESHAPE(_b_zn, b_l1);
                TEXTRACT(_l0b, _b_zn, 0, 0);

                set_flag(PIPE_MTE1, PIPE_M, _we);
                wait_flag(PIPE_MTE1, PIPE_M, _we);
                TMATMUL(a_l0, _l0a, _l0b);
                set_flag(PIPE_MTE1, PIPE_MTE2, _we);
                wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
                set_flag(PIPE_M, PIPE_FIX, _we);
                wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // ── Store L_full [C, C] from L0C → ws_out[slot] (fp16) ──────────
            {
                int64_t wo_base = (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsOutSlotElems);
                TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC>
                    _l0(ChunkSize, ChunkSize);
                TASSIGN(_l0, 0);
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = ChunkSize;
                _gs.shape[4] = ChunkSize;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>>
                    _gm(ws_out_ptr + wo_base, _gs);
                TSTORE(_gm, _l0);
            }

            // Signal Vec: ws_out[slot] ready (GEMM done; ws_in[slot] is also free)
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | ((2 + slot) << 8));
        }
    }

#endif // __DAV_C220_CUBE__
}

// ── Device entry point ────────────────────────────────────────────────────────
extern "C" __global__ AICORE void launch_kkt_kda(
    __gm__ uint8_t *k_ptr,
    __gm__ uint8_t *g_cs_ptr,
    __gm__ uint8_t *beta_ptr,
    __gm__ uint8_t *mask_ptr,
    __gm__ uint8_t *ws_in_ptr,
    __gm__ uint8_t *ws_out_ptr,
    __gm__ uint8_t *L_out_ptr,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
    kkt_kda_kernel<GDN_H, GDN_D, GDN_C>(
        reinterpret_cast<__gm__ float *>(k_ptr),
        reinterpret_cast<__gm__ float *>(g_cs_ptr),
        reinterpret_cast<__gm__ float *>(beta_ptr),
        reinterpret_cast<__gm__ float *>(mask_ptr),
        reinterpret_cast<__gm__ half *>(ws_in_ptr),
        reinterpret_cast<__gm__ half *>(ws_out_ptr),
        reinterpret_cast<__gm__ float *>(L_out_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
        batch_size, seq_len, total_tokens, ffts_addr);
}

// ── Host entry point (called from Python via ctypes) ─────────────────────────
extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *k_ptr,
    uint8_t *g_cs_ptr,
    uint8_t *beta_ptr,
    uint8_t *mask_ptr,
    uint8_t *ws_in_ptr,
    uint8_t *ws_out_ptr,
    uint8_t *L_out_ptr,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kkt_kda<<<block_dim, nullptr, stream>>>(
        k_ptr, g_cs_ptr, beta_ptr, mask_ptr,
        ws_in_ptr, ws_out_ptr, L_out_ptr, cu_seqlens,
        batch_size, seq_len, total_tokens, fftsAddr);
}
