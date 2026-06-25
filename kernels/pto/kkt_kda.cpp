// ============================================================================
// kkt_kda.cpp — Within-chunk gated attention matrix for KDA (numerically stable)
//
// Mathematical operation (per chunk of C tokens, per head h):
//   L[r,c] = beta[r] * sum_d k[r,d] * k[c,d] * exp(g_cs[r,d] - g_cs[c,d])
//            for r > c (strictly lower-tri), else 0
//
// STABILITY: Kimi KDA gates (g = -exp(A_log)*softplus(...)) are unbounded; the
//   within-chunk cumulative gate g_cs can reach ~-500.  The previous factorized
//   form  A_eff=k*exp(g_cs), B_eff=k*exp(-g_cs), L=A_eff@B_eff^T  computes
//   exp(-g_cs)=exp(+500) which overflows fp32 (max e^88) -> inf -> inf*0 = NaN.
//
//   This kernel instead computes exp(g_cs[r]-g_cs[c]) as a DIFFERENCE, never the
//   product of two separate exponentials.  For the kept (lower-tri) entries r>c,
//   g_cs[r] <= g_cs[c] (g_cs monotone decreasing within a chunk) so the argument
//   is <= 0 and exp(.) <= 1 — always finite.  We clamp the argument with
//   min(., 0) so the masked (upper-tri) entries also stay finite (then discarded
//   by only storing the strict-lower part).  No pivot, no saturation, exact.
//
// IMPLEMENTATION: per (head, chunk) the work is split across the two Vec
//   sub-blocks by row range (vid=0 -> rows [0,C/2), vid=1 -> rows [C/2, C)),
//   mirroring the GDN scaled_dot_kkt row split.  Each vid loops over columns c
//   and computes its rows' column-c of L via a per-column elementwise reduction:
//
//     diff[r,d] = g_cs[my_row r, d] - g_cs[c, d]   (TCOLEXPANDSUB, per-dim c)
//     diff      = min(diff, 0)                       (TMINS)
//     t[r,d]    = exp(diff) * k[c,d] * k[my_row r,d] (TEXP, TCOLEXPANDMUL, TMUL)
//     L[my_row r, c] = beta[r] * sum_d t[r,d]        (TROWSUM, TMUL)
//
//   then stores the strict-lower rows (global_row > c) to L_out.  This is a
//   Vec-only kernel; the Cube pass only participates in the entry/exit barriers.
//   (A GEMM-accelerated off-diagonal path is a future optimization.)
//
// Inputs (all on GM, head-major [HV, total_tokens, K]):
//   k       [HV, total_tokens, K]  float16  — keys
//   g_cs    [HV, total_tokens, K]  float32  — within-chunk cumulative gate sum
//   beta    [HV, total_tokens]     float16  — post-sigmoid beta in (0, 1)
//   mask    [C, C]                 float32  — (unused; kept for ABI stability)
//   ws_in   [block_dim*2, 2*C, K]  float32  — (unused; kept for ABI stability)
//   ws_out  [block_dim*2, C, C]    float32  — (unused; kept for ABI stability)
//   L_out   [total_tokens, HV, C]  float16  — strictly-lower-tri L (BSND)
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
//   Compile-time: GDN_D = K, GDN_C = C.  Runtime: num_heads = HV.
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

#ifdef __CCE_AICORE__
// Global barrier across ALL AI cores: every Cube and every Vec sub-block must
// reach this point before any of them proceeds.  Uses four reserved FFTS flag
// IDs (6, 7, 8, 9).
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

template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

// Column-vector tiles ([R,1]) must be ColMajor: RowMajor NoneBox requires the
// column byte-width to be 32-byte aligned, which width-1 tiles fail.
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                       RV, CV, pto::SLayout::NoneBox, 512>;
#endif

template <int32_t KDim, int32_t ChunkSize>
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
    int32_t num_heads, uint64_t ffts_addr)
{
    auto cid = get_block_idx();
    auto block_num = get_block_num();
    auto vid = get_subblockid();
    set_ffts_base_addr(ffts_addr);

    // Head count is a runtime argument (HV).  Only loop bounds, the work-item
    // decode, and GM strides depend on it — no UB buffer or tile shape does —
    // so it never needs to be a compile-time constant.
    const int32_t NumHeads = num_heads;

    constexpr int32_t HalfChunk = ChunkSize / 2;
    constexpr int32_t KTC = ((KDim + 7) / 8) * 8;

    int64_t num_seqs = batch_size;
    int64_t total_work = num_seqs * NumHeads;

    // ── GM type aliases (head-major [HV, T, K]) ──────────────────────────────
    using GmShapeDyn = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using GmFloatK = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmHalfK = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmHalf_1 = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, 1, 1>>;
    // L output is BSND-interleaved [total_tokens, NumHeads, ChunkSize]; the
    // token stride NumHeads*ChunkSize is now runtime, so the GM stride is
    // DYNAMIC and supplied at construction (see the store site below).
    using GmHalfOut = GlobalTensor<half, GmShapeDyn,
                                   Stride<1, 1, 1, DYNAMIC, 1>>;
    using GmHalfWsIn = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmHalfWsOut = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, ChunkSize, 1>>;

#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    sync_all();

    // ── UB layout (per vid) ──────────────────────────────────────────────────
    constexpr int32_t MYG_ADDR = 0;                               // [HalfChunk, K] fp32
    constexpr int32_t MYK_ADDR = MYG_ADDR + HalfChunk * KTC * 4;  // [HalfChunk, K] fp32
    constexpr int32_t DIFF_ADDR = MYK_ADDR + HalfChunk * KTC * 4; // [HalfChunk, K] fp32 (diff/t)
    constexpr int32_t TMP_ADDR = DIFF_ADDR + HalfChunk * KTC * 4; // [HalfChunk, K] fp32 (rowsum tmp)
    constexpr int32_t MYKH_ADDR = TMP_ADDR + HalfChunk * KTC * 4; // [HalfChunk, K] fp16 (k staging)
    constexpr int32_t GC_ADDR = MYKH_ADDR + HalfChunk * KTC * 2;  // [1, K] fp32 (column g_c)
    constexpr int32_t KC_ADDR = GC_ADDR + KTC * 4;                // [1, K] fp32 (column k_c)
    constexpr int32_t KCH_ADDR = KC_ADDR + KTC * 4;               // [1, K] fp16 (k_c staging)
    constexpr int32_t COL_ADDR = KCH_ADDR + KTC * 2;              // [HalfChunk, 16] fp32 (colsum, RowMajor padded)
    constexpr int32_t COLH_ADDR = COL_ADDR + HalfChunk * 16 * 4;  // [HalfChunk, 16] fp16 (padded store, RowMajor)
    constexpr int32_t BETA_ADDR = COLH_ADDR + HalfChunk * 16 * 2; // [1, HalfChunk] fp32 (beta)
    constexpr int32_t BETAH_ADDR = BETA_ADDR + HalfChunk * 4;     // [1, HalfChunk] fp16 (beta staging)
    constexpr int32_t MSKC_ADDR = BETAH_ADDR + HalfChunk * 2;     // [1, HalfChunk] fp32 (mask col)

    int32_t my_off = static_cast<int32_t>(vid) * HalfChunk;

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
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);

            // This vid's row range within the chunk: [my_off, my_off + my_rows).
            int32_t my_rows = valid_rows - my_off;
            if (my_rows > HalfChunk)
                my_rows = HalfChunk;
            if (my_rows <= 0)
                continue; // no rows for this vid in this chunk

            // Columns this vid must cover: c in [0, col_end).  A row r is kept
            // for column c only if global_row(r) > c, so the largest column any
            // of my rows touches is (my_off + my_rows - 1).
            int32_t col_end = my_off + my_rows; // exclusive upper bound for c

            int64_t hbase = static_cast<int64_t>(head_idx) * total_tokens * KDim;
            int64_t my_first = bos + chunk_start + my_off; // global row index of my row 0

            // ── Load my rows' g_cs (fp32) and k (fp16 -> fp32) ───────────────
            {
                GmShapeDyn gs;
                gs.shape[3] = my_rows;
                gs.shape[4] = KDim;
                GmFloatK g_gm(g_cs_ptr + hbase + my_first * KDim, gs);
                UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                    g_ld(my_rows, KDim);
                TASSIGN(g_ld, MYG_ADDR);
                TLOAD(g_ld, g_gm);
            }
            {
                GmShapeDyn gs;
                gs.shape[3] = my_rows;
                gs.shape[4] = KDim;
                GmHalfK k_gm(k_ptr + hbase + my_first * KDim, gs);
                UbND<half, HalfChunk, KTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                    k_ld(my_rows, KDim);
                TASSIGN(k_ld, MYKH_ADDR);
                TLOAD(k_ld, k_gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            {
                UbND<half, HalfChunk, KTC, DYNAMIC, DYNAMIC> k_h(my_rows, KDim);
                TASSIGN(k_h, MYKH_ADDR);
                UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> k_f(my_rows, KDim);
                TASSIGN(k_f, MYK_ADDR);
                TCVT(k_f, k_h, pto::RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
            }
            // ── Load my rows' beta (fp16 -> fp32) as a [1, my_rows] row, then
            //    re-view as a [my_rows, 1] column for the per-row scale. ───────
            {
                GmShapeDyn gs;
                gs.shape[3] = 1;
                gs.shape[4] = my_rows;
                GmHalf1 b_gm(beta_ptr + static_cast<int64_t>(head_idx) * total_tokens +
                                 my_first,
                             gs);
                UbND<half, 1, HalfChunk, DYNAMIC, DYNAMIC> b_ld(1, my_rows);
                TASSIGN(b_ld, BETAH_ADDR);
                TLOAD(b_ld, b_gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            {
                UbND<half, 1, HalfChunk, DYNAMIC, DYNAMIC> b_h(1, my_rows);
                TASSIGN(b_h, BETAH_ADDR);
                UbND<float, 1, HalfChunk, DYNAMIC, DYNAMIC> b_f(1, my_rows);
                TASSIGN(b_f, BETA_ADDR);
                TCVT(b_f, b_h, pto::RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
            }

            UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> myg(my_rows, KDim);
            TASSIGN(myg, MYG_ADDR);
            UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> myk(my_rows, KDim);
            TASSIGN(myk, MYK_ADDR);
            UbDN<float, HalfChunk, 1, DYNAMIC, DYNAMIC> beta_col(my_rows, 1);
            TASSIGN(beta_col, BETA_ADDR);

            // ── Column loop ──────────────────────────────────────────────────
            for (int32_t c = 0; c < col_end; ++c)
            {
                // Load column c's g_cs (fp32) and k (fp16 -> fp32) — [1, K].
                int64_t col_off = hbase + (bos + chunk_start + c) * KDim;
                {
                    GmShapeDyn gs;
                    gs.shape[3] = 1;
                    gs.shape[4] = KDim;
                    GmFloatK gc_gm(g_cs_ptr + col_off, gs);
                    UbND<float, 1, KTC, 1, KTC> gc_ld;
                    TASSIGN(gc_ld, GC_ADDR);
                    TLOAD(gc_ld, gc_gm);
                    GmHalfK kc_gm(k_ptr + col_off, gs);
                    UbND<half, 1, KTC, 1, KTC> kc_ld;
                    TASSIGN(kc_ld, KCH_ADDR);
                    TLOAD(kc_ld, kc_gm);
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                {
                    UbND<half, 1, KTC, 1, KTC> kc_h;
                    TASSIGN(kc_h, KCH_ADDR);
                    UbND<float, 1, KTC, 1, KTC> kc_f;
                    TASSIGN(kc_f, KC_ADDR);
                    TCVT(kc_f, kc_h, pto::RoundMode::CAST_NONE);
                    pipe_barrier(PIPE_V);
                }

                UbND<float, 1, KTC, 1, KTC> gc;
                TASSIGN(gc, GC_ADDR);
                UbND<float, 1, KTC, 1, KTC> kc;
                TASSIGN(kc, KC_ADDR);
                UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> diff(my_rows, KDim);
                TASSIGN(diff, DIFF_ADDR);
                UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> tmp(my_rows, KDim);
                TASSIGN(tmp, TMP_ADDR);
                UbND<float, HalfChunk, 16, DYNAMIC, DYNAMIC> colsum(my_rows, 1);
                TASSIGN(colsum, COL_ADDR);

                // diff[r,d] = g_cs[my r, d] - g_cs[c, d]
                TCOLEXPANDSUB(diff, myg, gc);
                pipe_barrier(PIPE_V);
                // clamp to <= 0 so exp(.) is finite for masked (r<c) entries too
                TMINS(diff, diff, 0.0f);
                pipe_barrier(PIPE_V);
                TEXP(diff, diff);
                pipe_barrier(PIPE_V);
                // *= k[c,d]  (per-dim broadcast), then *= k[my r, d]
                TCOLEXPANDMUL(diff, diff, kc);
                pipe_barrier(PIPE_V);
                TMUL(diff, diff, myk);
                pipe_barrier(PIPE_V);
                // per-row beta scale (TMUL can't take a [R,1] column directly)
                TROWEXPANDMUL(diff, diff, beta_col);
                pipe_barrier(PIPE_V);
                // colsum[r] = beta[r] * sum_d diff[r,d]   (unmasked)
                TROWSUM(colsum, diff, tmp);
                pipe_barrier(PIPE_V);

                // Strict-lower mask: load mask[my_off+r, c] as a padded [my_rows,1]
                // strip (row-strided gather, innermost contiguous) and zero the
                // upper-tri rows (my_off+r <= c) via elementwise TMUL.
                {
                    UbND<float, HalfChunk, 16, DYNAMIC, DYNAMIC> mk(my_rows, 1);
                    TASSIGN(mk, MSKC_ADDR);
                    GmShapeDyn gs;
                    gs.shape[3] = my_rows;
                    gs.shape[4] = 1;
                    GmFloatMaskColRow mk_gm(
                        mask_ptr + static_cast<int64_t>(my_off) * ChunkSize + c, gs);
                    TLOAD(mk, mk_gm);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    TMUL(colsum, colsum, mk);
                    pipe_barrier(PIPE_V);
                }

                // cvt colsum -> fp16 into a padded RowMajor [my_rows, 1] tile and
                // store column c of L for the strided set of tokens (row dim steps
                // by NumHeads*ChunkSize, the single column is contiguous).
                UbND<half, HalfChunk, 16, DYNAMIC, DYNAMIC> col_h(my_rows, 1);
                TASSIGN(col_h, COLH_ADDR);
                TCVT(col_h, colsum, pto::RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                {
                    int64_t l_off = my_first * static_cast<int64_t>(NumHeads) * ChunkSize +
                                    static_cast<int64_t>(head_idx) * ChunkSize + c;
                    GmShapeDyn gs;
                    gs.shape[3] = my_rows;
                    gs.shape[4] = 1;
                    GmHalfLoutCol l_gm(L_out_ptr + l_off, gs);
                    TSTORE(l_gm, col_h);
                }
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                // Drain all pipes before the next column iteration so its gc/kc/
                // mask loads (MTE2) cannot race the current column's still-draining
                // Vec reads of the same UB slots.
                pipe_barrier(PIPE_ALL);
            }
        }
    }

    sync_all();
#endif // __DAV_C220_VEC__
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
    int32_t num_heads, uint64_t ffts_addr)
{
    kkt_kda_kernel<GDN_D, GDN_C>(
        reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ float *>(g_cs_ptr),
        reinterpret_cast<__gm__ half *>(beta_ptr),
        reinterpret_cast<__gm__ float *>(mask_ptr),
        reinterpret_cast<__gm__ float *>(ws_in_ptr),
        reinterpret_cast<__gm__ float *>(ws_out_ptr),
        reinterpret_cast<__gm__ half *>(L_out_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
        batch_size, seq_len, total_tokens, num_heads, ffts_addr);
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
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint32_t num_heads)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kkt_kda<<<block_dim, nullptr, stream>>>(
        k_ptr, g_cs_ptr, beta_ptr, mask_ptr,
        ws_in_ptr, ws_out_ptr, L_out_ptr, cu_seqlens,
        batch_size, seq_len, total_tokens,
        static_cast<int32_t>(num_heads), fftsAddr);
}
