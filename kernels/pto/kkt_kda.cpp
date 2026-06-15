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
//   ws_in   [block_dim*2, 2*C, K]  float32  — workspace: slot A_eff (rows 0..C-1) + B_eff (C..2C-1)
//   ws_out  [block_dim*2, C, C]    float32  — workspace: GEMM result L_full
//
// NOTE: ws_in/ws_out are fp32 (not fp16): A_eff=k*exp(g_cs) and B_eff=k*exp(-g_cs)
//   stage exp(±g_cs) separately, and per-128-chunk |g_cs| can reach ~64 →
//   exp(-g_cs)≈e^64≈6e27 overflows fp16 (max 6.5e4) but fits fp32 (max 3.4e38).
//   The K·K^T GEMM is therefore (float,float,float). L_out stays fp16 (bounded
//   ≤ beta ≤ 1 after the strict-lower mask).
//
// Output:
//   L_out   [total_tokens, HV, C]  float32  — strictly-lower-tri L matrix (BSND)
//
// Cross-core architecture (mirrors GDN scaled_dot_kkt pattern):
//   Both Vec sub-blocks (vid=0,1) do real work: each handles HalfChunk rows.
//     vid=0 → rows [0, C/2),  vid=1 → rows [C/2, C)
//   Vec pre:  load k, g_cs (my rows) → A_eff = k*exp(g), B_eff = k*exp(-g),
//             cast fp16 → ws_in[my rows]
//   Cube:     load full A_eff, B_eff from ws_in → GEMM A @ B^T → ws_out
//   Vec post: load ws_out[my rows], cast fp32 → apply mask + beta row-scale → L_out
//
// FFTS flags (double-buffered, slot = ci & 1):
//   0, 1 : Vec → Cube  "ws_in[slot] ready"  (both vids must sig under mode-2 reduce)
//   2, 3 : Cube → Vec  "ws_out[slot] ready" (broadcast: each vid gets a signal)
//   4, 5 : Vec → Cube  "ws_out[slot] free"  (Vec done reading L_full; conditional)
//
// UB budget (per vid, HalfChunk=C/2 rows; UB ~192 KB per Vec sub-block):
//   mask fp32 [C/2, C] lives always at offset 0 (loaded once per launch).
//   The rest of UB is a shared pool reused between pre-compute and post-process
//   (they never run concurrently within a chunk).
//   Pre-compute pool (live simultaneously):
//     g_ub  fp32 [C/2, KTC],  k_ub fp32 [C/2, KTC],
//     ab_ub fp32 [C/2, KTC],  half_buf fp16 [C/2, KTC]    (scratch reused A → B)
//   Post-process pool (live simultaneously; overlaps pre-compute addresses):
//     L_half fp16 [C/2, C],  L_ub fp32 [C/2, C],
//     beta_2d fp32 [C/2, C], beta fp32 [1, C/2]
//   Peak @ C=128, K=128: mask 32 + pre 112 = 144 KB ✓ (under 192 KB)
//   Peak @ C=16,  K=128: mask 0.5 + pre 14 ≈ 15 KB ✓
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
// Global barrier across ALL AI cores: every Cube and every Vec sub-block must
// reach this point before any of them proceeds.  Uses four reserved FFTS flag
// IDs (6, 7, 8, 9) — must not collide with the data-flow flags (0-5) used by
// the kernel pipeline.
//
//   Flag 6 : V_ALL_CORE  — each Vec sub-block signals & waits  (mode 0)
//   Flag 7 : C_ALL_CORE  — each Cube core signals & waits       (mode 0)
//   Flag 8 : C→V CV sync — Cube signals, each Vec waits         (mode 2)
//   Flag 9 : V→C CV sync — both vids signal, Cube waits for two (mode 2)
//
// Steps:
//   1. Local pipe drain
//   2. All-cores-of-my-type barrier (mode 0)  → my "side" is in lockstep
//   3. Cross-core (mode 2) signal in both directions → C and V meet
//   4. Wait for the other side's signal
//   5. Local pipe drain
AICORE inline void sync_all()
{
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_C220_CUBE__)
    // (1) All-Cube barrier
    ffts_cross_core_sync(PIPE_FIX, 1 | (0 << 4) | (7 << 8));
    wait_flag_dev(7);
    // (2) C→V signal + wait V→C
    ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (8 << 8));
    wait_flag_dev(9);
#elif defined(__DAV_C220_VEC__)
    // (1) All-Vec barrier
    ffts_cross_core_sync(PIPE_MTE3, 1 | (0 << 4) | (6 << 8));
    wait_flag_dev(6);
    // (2) V→C signal + wait C→V
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (9 << 8));
    wait_flag_dev(8);
#endif
    pipe_barrier(PIPE_ALL);
}

template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                       RV, CV, pto::SLayout::NoneBox, 512>;

template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor,
                          RV, CV, pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;
#endif

template <int32_t NumHeads, int32_t KDim, int32_t ChunkSize>
AICORE void kkt_kda_kernel(
    __gm__ half *k_ptr,
    __gm__ float *g_cs_ptr,
    __gm__ half *beta_ptr,
    __gm__ float *mask_ptr,
    __gm__ float *ws_in_ptr,
    __gm__ float *ws_out_ptr,
    __gm__ half *L_out_ptr,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
    auto cid = get_block_idx();
    auto block_num = get_block_num();
    auto vid = get_subblockid();
    set_ffts_base_addr(ffts_addr);

    constexpr int32_t HalfChunk = ChunkSize / 2;
    constexpr int32_t KTC = ((KDim + 7) / 8) * 8;
    constexpr int32_t ChunkSq = ChunkSize * ChunkSize;

    // ── UB address map ─────────────────────────────────────────────────────
    // Mask always alive at offset 0. Pool starts after mask and is reused
    // between pre-compute (g/k/ab/half) and post-process (L_half/L/beta/beta_2d)
    // since those phases are sequential within each chunk iteration.
    constexpr int32_t MskUbAddr = 0;
    constexpr int32_t PoolBase = MskUbAddr + HalfChunk * ChunkSize * 4;
    // Pre-compute tiles (offsets within pool):
    constexpr int32_t GUbAddr = PoolBase;
    constexpr int32_t KUbAddr = GUbAddr + HalfChunk * KTC * 4;
    constexpr int32_t ABUbAddr = KUbAddr + HalfChunk * KTC * 4;
    constexpr int32_t HalfBufAddr = ABUbAddr + HalfChunk * KTC * 4;
    // Post-process tiles (overlap with pre-compute; reuse pool addresses):
    constexpr int32_t LHalfUbAddr = PoolBase;
    constexpr int32_t LUbAddr = LHalfUbAddr + HalfChunk * ChunkSize * 2;
    constexpr int32_t BetaUbAddr = LUbAddr + HalfChunk * ChunkSize * 4;
    constexpr int32_t Beta2dUbAddr = BetaUbAddr + HalfChunk * 4;

    // Workspace element counts per slot (fp16 elements):
    constexpr int32_t WsInSlotElems = 2 * ChunkSize * KDim; // A_eff + B_eff
    constexpr int32_t WsOutSlotElems = ChunkSq;             // L_full

    int64_t num_seqs = batch_size;
    int64_t total_work = num_seqs * NumHeads;

    // ── GM type aliases ──────────────────────────────────────────────────────
    using GmShapeDyn = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using GmHalfK = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmFloatK = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmHalf_1 = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, 1, 1>>;
    using GmHalfOut = GlobalTensor<half, GmShapeDyn,
                                   Stride<1, 1, 1, NumHeads * ChunkSize, 1>>;
    using GmFloatWsIn = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmFloatWsOut = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, ChunkSize, 1>>;

    // =========================================================================
    // VEC PHASE — both vids do real work on HalfChunk rows each
    // =========================================================================
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);

    // Global all-core barrier at kernel start: drains any leftover FFTS state
    // from prior kernel launches before we begin using the data-flow flags.
    sync_all();

    // Each vid handles a fixed row range within the chunk:
    //   vid=0 → rows [0, HalfChunk),  vid=1 → rows [HalfChunk, ChunkSize)
    int32_t my_row_offset = static_cast<int32_t>(vid) * HalfChunk;

    // ── Load this vid's HalfChunk rows of the strictly-lower-tri mask ──────
    {
        UbND<float, HalfChunk, ChunkSize> msk_ub;
        TASSIGN(msk_ub, MskUbAddr);
        GmShapeDyn gs;
        gs.shape[3] = HalfChunk;
        gs.shape[4] = ChunkSize;
        GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, ChunkSize, 1>>
            msk_gm(mask_ptr + static_cast<int64_t>(my_row_offset) * ChunkSize, gs);
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
            // local_valid: how many of this vid's HalfChunk rows are real (in [0, HalfChunk]).
            int32_t local_valid =
                valid_rows > my_row_offset
                    ? (valid_rows - my_row_offset < HalfChunk
                           ? valid_rows - my_row_offset
                           : HalfChunk)
                    : 0;

            // Head-major offset: head_idx * T * K + (bos + chunk_start + my_row_offset) * K
            int64_t hk_base = static_cast<int64_t>(head_idx) * total_tokens * KDim + (bos + chunk_start + my_row_offset) * KDim;

            // ws_in slot base (fp16 element offset)
            int64_t ws_in_base = (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsInSlotElems);

            // ── PRE-COMPUTE (only if this vid has valid rows) ───────────────
            if (local_valid > 0)
            {
                // Load g_cs fp32 directly → GUbAddr (gate kept in fp32)
                {
                    GmShapeDyn gs;
                    gs.shape[3] = local_valid;
                    gs.shape[4] = KDim;
                    GmFloatK gm(g_cs_ptr + hk_base, gs);
                    UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                        g_f_ld(local_valid, KDim);
                    TASSIGN(g_f_ld, GUbAddr);
                    TLOAD(g_f_ld, gm);
                    if (local_valid != HalfChunk || KDim != KTC)
                    {
                        UbND<float, HalfChunk, KTC, HalfChunk, KTC, PadValue::Zero> g_f_pad;
                        TASSIGN(g_f_pad, GUbAddr);
                        TFILLPAD_INPLACE(g_f_pad, g_f_ld);
                    }
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                // Load k fp16 → HalfBufAddr staging → TCVT to fp32 KUbAddr
                {
                    GmShapeDyn gs;
                    gs.shape[3] = local_valid;
                    gs.shape[4] = KDim;
                    GmHalfK gm(k_ptr + hk_base, gs);
                    UbND<half, HalfChunk, KTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                        k_h_ld(local_valid, KDim);
                    TASSIGN(k_h_ld, HalfBufAddr);
                    TLOAD(k_h_ld, gm);
                    if (local_valid != HalfChunk || KDim != KTC)
                    {
                        UbND<half, HalfChunk, KTC, HalfChunk, KTC, PadValue::Zero> k_h_pad;
                        TASSIGN(k_h_pad, HalfBufAddr);
                        TFILLPAD_INPLACE(k_h_pad, k_h_ld);
                    }
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                {
                    UbND<half, HalfChunk, KTC> k_h_cvt;
                    TASSIGN(k_h_cvt, HalfBufAddr);
                    UbND<float, HalfChunk, KTC> k_fp32_dst;
                    TASSIGN(k_fp32_dst, KUbAddr);
                    TCVT(k_fp32_dst, k_h_cvt, pto::RoundMode::CAST_NONE);
                    pipe_barrier(PIPE_V);
                }

                UbND<float, HalfChunk, KTC> g_ub;
                TASSIGN(g_ub, GUbAddr);
                UbND<float, HalfChunk, KTC> k_ub;
                TASSIGN(k_ub, KUbAddr);
                UbND<float, HalfChunk, KTC> ab_ub;
                TASSIGN(ab_ub, ABUbAddr);

                // A = k * exp(g)  (in-place scratch reuse: ab_ub = exp(g), then ab_ub *= k)
                // Kept in fp32: exp(g) is bounded ≤1 here, but B below stages
                // exp(-g)≈e^64 which overflows fp16, so the whole GEMM is fp32.
                TEXP(ab_ub, g_ub);
                pipe_barrier(PIPE_V);
                TMUL(ab_ub, k_ub, ab_ub);
                pipe_barrier(PIPE_V);

                // Store A_eff (fp32; this vid's HalfChunk rows of A) to ws_in[slot]
                //   ws_in layout: rows 0..C-1 = A, rows C..2C-1 = B
                //   This vid writes A at offset my_row_offset.
                {
                    GmShapeDyn gs;
                    gs.shape[3] = HalfChunk;
                    gs.shape[4] = KDim;
                    int64_t off = ws_in_base + static_cast<int64_t>(my_row_offset) * KDim;
                    GmFloatWsIn gm_a(ws_in_ptr + off, gs);
                    UbND<float, HalfChunk, KTC, HalfChunk, KTC> A_st;
                    TASSIGN(A_st, ABUbAddr);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(gm_a, A_st);
                }
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

                // B = k * exp(-g): compute -g in ab_ub, then exp, then multiply by k
                TEXPANDS(ab_ub, 0.0f);
                pipe_barrier(PIPE_V);
                TSUB(ab_ub, ab_ub, g_ub);
                pipe_barrier(PIPE_V);
                TEXP(ab_ub, ab_ub);
                pipe_barrier(PIPE_V);
                TMUL(ab_ub, k_ub, ab_ub);
                pipe_barrier(PIPE_V);

                // Store B_eff (fp32) to ws_in[slot] rows C..2C-1 at offset my_row_offset
                {
                    GmShapeDyn gs;
                    gs.shape[3] = HalfChunk;
                    gs.shape[4] = KDim;
                    int64_t off = ws_in_base + static_cast<int64_t>(ChunkSize) * KDim + static_cast<int64_t>(my_row_offset) * KDim;
                    GmFloatWsIn gm_b(ws_in_ptr + off, gs);
                    UbND<float, HalfChunk, KTC, HalfChunk, KTC> B_st;
                    TASSIGN(B_st, ABUbAddr);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                    TSTORE(gm_b, B_st);
                }
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }

            // Both vids must signal flag (slot) under V→C reduce mode 2.
            // pipe_barrier ensures all preceding Vec/MTE2/MTE3 ops complete
            // before the cross-core signal — required even for vid=1 idle path
            // so FFTS state is well-defined.
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (slot << 8));
            // Wait for Cube: ws_out[slot] ready
            wait_flag_dev(2 + slot);
            pipe_barrier(PIPE_ALL);

            // ── POST-PROCESS (only if this vid has valid rows) ──────────────
            if (local_valid > 0)
            {
                // Load L_full[my_row_offset..+local_valid, :] from ws_out[slot] (fp32)
                {
                    int64_t wo_base = (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsOutSlotElems);
                    int64_t off = wo_base + static_cast<int64_t>(my_row_offset) * ChunkSize;
                    GmShapeDyn gs;
                    gs.shape[3] = local_valid;
                    gs.shape[4] = ChunkSize;
                    GmFloatWsOut gm(ws_out_ptr + off, gs);
                    UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero>
                        L_f_ld(local_valid, ChunkSize);
                    TASSIGN(L_f_ld, LUbAddr);
                    TLOAD(L_f_ld, gm);
                    if (local_valid != HalfChunk)
                    {
                        UbND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize,
                             PadValue::Zero>
                            L_f_pad;
                        TASSIGN(L_f_pad, LUbAddr);
                        TFILLPAD_INPLACE(L_f_pad, L_f_ld);
                    }
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                UbND<float, HalfChunk, ChunkSize> L_ub;
                TASSIGN(L_ub, LUbAddr);
                UbND<float, HalfChunk, ChunkSize> msk_ub;
                TASSIGN(msk_ub, MskUbAddr);

                // Apply strict-lower mask.  The unmasked upper-tri holds
                // exp(g_r-g_c)>1 (up to ~e^64); in fp32 that is finite (~6e27),
                // so msk*L = 0 cleanly (fp16 would have been inf → inf*0 = NaN).
                TMUL(L_ub, L_ub, msk_ub);
                pipe_barrier(PIPE_V);

                // Load beta fp16 → HalfBufAddr staging → TCVT to fp32 BetaUbAddr
                {
                    int64_t beta_base = static_cast<int64_t>(head_idx) * total_tokens + (bos + chunk_start + my_row_offset);
                    GmShapeDyn gs;
                    gs.shape[3] = 1;
                    gs.shape[4] = local_valid;
                    GmHalf_1 gm(beta_ptr + beta_base, gs);
                    UbND<half, 1, HalfChunk, DYNAMIC, DYNAMIC, PadValue::Zero>
                        beta_h_ld(1, local_valid);
                    TASSIGN(beta_h_ld, HalfBufAddr);
                    TLOAD(beta_h_ld, gm);
                    if (local_valid != HalfChunk)
                    {
                        UbND<half, 1, HalfChunk, 1, HalfChunk, PadValue::Zero> beta_h_pad;
                        TASSIGN(beta_h_pad, HalfBufAddr);
                        TFILLPAD_INPLACE(beta_h_pad, beta_h_ld);
                    }
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                {
                    UbND<half, 1, HalfChunk> beta_h_cvt;
                    TASSIGN(beta_h_cvt, HalfBufAddr);
                    UbND<float, 1, HalfChunk, 1, HalfChunk, PadValue::Zero> beta_fp32_dst;
                    TASSIGN(beta_fp32_dst, BetaUbAddr);
                    TCVT(beta_fp32_dst, beta_h_cvt, pto::RoundMode::CAST_NONE);
                    pipe_barrier(PIPE_V);
                }

                // Row-scale L by beta:  L[r, :] *= beta[r].
                {
                    UbDN<float, HalfChunk, 1> beta_col;
                    TASSIGN(beta_col, BetaUbAddr);
                    UbND<float, HalfChunk, ChunkSize> beta_2d;
                    TASSIGN(beta_2d, Beta2dUbAddr);
                    TROWEXPAND(beta_2d, beta_col);
                    pipe_barrier(PIPE_V);
                    TMUL(L_ub, L_ub, beta_2d);
                    pipe_barrier(PIPE_V);
                }

                // TCVT fp32 L_ub → fp16 at LHalfUbAddr, then store to L_out half*
                {
                    UbND<float, HalfChunk, ChunkSize> L_fp32_src;
                    TASSIGN(L_fp32_src, LUbAddr);
                    UbND<half, HalfChunk, ChunkSize> L_half_dst;
                    TASSIGN(L_half_dst, LHalfUbAddr);
                    TCVT(L_half_dst, L_fp32_src, pto::RoundMode::CAST_NONE);
                    pipe_barrier(PIPE_V);
                }
                {
                    int64_t l_base = (bos + chunk_start + my_row_offset) * static_cast<int64_t>(NumHeads) * ChunkSize + static_cast<int64_t>(head_idx) * ChunkSize;
                    GmShapeDyn gs;
                    gs.shape[3] = local_valid;
                    gs.shape[4] = ChunkSize;
                    GmHalfOut gm(L_out_ptr + l_base, gs);
                    UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC>
                        L_h_st(local_valid, ChunkSize);
                    TASSIGN(L_h_st, LHalfUbAddr);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(gm, L_h_st);
                }
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }

            // Drain Vec pipes before moving to the next chunk so the L_out
            // TSTORE is guaranteed in HBM before pre-compute reuses the UB
            // pool addresses (and so any cross-core sig below sees a
            // well-defined state).
            pipe_barrier(PIPE_ALL);
            // Signal Cube: ws_out[slot] is free for reuse (only when needed).
            // Both vids must sig (mode-2 V→C reduce); both share same condition.
            if (ci < num_chunks - 2)
            {
                ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((4 + slot) << 8));
            }
        }
    }

    // Global all-core barrier at kernel end: ensures every Vec sub-block is
    // done before the kernel returns, leaving FFTS counters in a clean state
    // for subsequent launches.
    sync_all();
#endif // __DAV_C220_VEC__

// =========================================================================
// CUBE PHASE: GEMM  A_eff @ B_eff^T  → L_full
// =========================================================================
#if defined(__DAV_C220_CUBE__)
    constexpr int32_t L1BAddr = ChunkSize * KDim * static_cast<int32_t>(sizeof(float));

    L1Mat<float, ChunkSize, KDim, ChunkSize, KDim> a_l1;
    TASSIGN(a_l1, 0);
    L1Mat<float, ChunkSize, KDim, ChunkSize, KDim> b_l1;
    TASSIGN(b_l1, L1BAddr);
    TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a_l0;
    TASSIGN(a_l0, 0);

    // Global all-core barrier at kernel start (matches Vec side).
    sync_all();

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

            // For ci >= 2: wait until Vec finished post-processing the previous
            // chunk with the same slot (so ws_out[slot] is free to overwrite).
            if (ci >= 2)
            {
                wait_flag_dev(4 + slot);
                pipe_barrier(PIPE_ALL);
            }
            // Wait for Vec pre-compute: ws_in[slot] is ready (both vids sig'd).
            wait_flag_dev(slot);
            pipe_barrier(PIPE_ALL);

            int64_t ws_in_base = (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsInSlotElems);

            // ── Load A_eff [C, K] from ws_in[slot, 0:C, :] into L1 ──────────
            {
                L1Mat<float, ChunkSize, KDim, DYNAMIC, DYNAMIC> _l1(ChunkSize, KDim);
                TASSIGN(_l1, 0);
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = ChunkSize;
                _gs.shape[4] = KDim;
                GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, KDim, 1>>
                    _gm(ws_in_ptr + ws_in_base, _gs);
                TLOAD(_l1, _gm);
            }
            // ── Load B_eff [C, K] from ws_in[slot, C:2C, :] into L1 ─────────
            {
                L1Mat<float, ChunkSize, KDim, DYNAMIC, DYNAMIC> _l1(ChunkSize, KDim);
                TASSIGN(_l1, L1BAddr);
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = ChunkSize;
                _gs.shape[4] = KDim;
                GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, KDim, 1>>
                    _gm(ws_in_ptr + ws_in_base + static_cast<int64_t>(ChunkSize) * KDim, _gs);
                TLOAD(_l1, _gm);
            }

            // ── GEMM: A_eff @ B_eff^T → L0C accumulator ─────────────────────
            {
                TileLeft<float, ChunkSize, KDim, ChunkSize, KDim> _l0a;
                TileRight<float, KDim, ChunkSize, KDim, ChunkSize> _l0b;
                TASSIGN(_l0a, 0x0);
                TASSIGN(_l0b, 0x0);
                auto _we = EVENT_ID1;
                // FIX→M: ensure previous iteration's TSTORE has finished reading
                // L0C before this iteration's TMATMUL overwrites it.
                set_flag(PIPE_FIX, PIPE_M, _we);
                wait_flag(PIPE_FIX, PIPE_M, _we);
                set_flag(PIPE_MTE2, PIPE_MTE1, _we);
                wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
                set_flag(PIPE_M, PIPE_MTE1, _we);
                wait_flag(PIPE_M, PIPE_MTE1, _we);

                TEXTRACT(_l0a, a_l1, 0, 0);
                L1MatZN<float, KDim, ChunkSize> _b_zn;
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
                GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>>
                    _gm(ws_out_ptr + wo_base, _gs);
                TSTORE(_gm, _l0);
            }

            // Drain FIX so the cross-core sig is emitted only after the
            // TSTORE has actually committed ws_out[slot] to HBM.
            pipe_barrier(PIPE_ALL);
            // Signal Vec: ws_out[slot] ready (GEMM done; ws_in[slot] also free).
            // Mode-2 C→V broadcast: each vid increments its own counter.
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | ((2 + slot) << 8));
        }
    }

    // Global all-core barrier at kernel end: matches Vec side.
    sync_all();
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
        reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ float *>(g_cs_ptr),
        reinterpret_cast<__gm__ half *>(beta_ptr),
        reinterpret_cast<__gm__ float *>(mask_ptr),
        reinterpret_cast<__gm__ float *>(ws_in_ptr),
        reinterpret_cast<__gm__ float *>(ws_out_ptr),
        reinterpret_cast<__gm__ half *>(L_out_ptr),
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
