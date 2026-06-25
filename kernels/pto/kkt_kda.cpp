// ============================================================================
// kkt_kda.cpp — Within-chunk gated attention matrix for KDA (Cube-accelerated,
//               numerically stable via per-token exp offset)
//
// Mathematical operation (per chunk of C tokens, per head h):
//   L[r,c] = beta[r] * sum_d k[r,d] * k[c,d] * exp(g_cs[r,d] - g_cs[c,d])
//            for r > c (strictly lower-tri), else 0
//
// THE STABILITY PROBLEM:
//   The Cube-friendly factorization  A=k*exp(g_cs), B=k*exp(-g_cs), L=A@B^T
//   computes exp(-g_cs)=exp(+500) on Kimi KDA gates (cumulative g_cs ~ -500),
//   which overflows even fp32 (max e^88) -> inf -> inf*0 = NaN.
//
// THE FIX — per-token offset (two exponentials that *almost* cancel):
//   Pick one scalar per token  b[t] = max_d g_cs[t,d]  and factor it out of
//   BOTH legs of the matmul:
//       A[t,d] = k[t,d] * exp(g_cs[t,d] - b[t])     (<= k  ⇒ never overflows)
//       B[t,d] = k[t,d] * exp(b[t] - g_cs[t,d])     (>= k, |exp| = channel spread)
//       M[r,c] = sum_d A[r,d]*B[c,d]                (= A @ B^T, on the Cube, fp32)
//       L[r,c] = beta[r] * exp(min(b[r]-b[c], 0)) * M[r,c]      for r > c
//   The b[t] inside A and B cancels exactly in the per-d product; the residual
//   is the post-matmul correction exp(b[r]-b[c]).  Because b is monotone
//   decreasing within a chunk (g_cs is), b[r]-b[c] <= 0 for kept entries (r>c),
//   so the correction is <= 1; min(.,0) keeps the masked (r<c) entries finite
//   too (they are then zeroed by the strict-lower mask, avoiding inf*0=NaN).
//
//   The cross-TOKEN cumulative blow-up (~-500) is fully absorbed into b[t]; the
//   only magnitude left in the matmul factors is the within-token cross-CHANNEL
//   spread  max_d g_cs[t,d] - min_d g_cs[t,d]  (carried by the B factor).  The
//   fp32 GEMM (e^88) is safe as long as that spread stays < ~88.
//
// Inputs (all on GM, head-major [HV, total_tokens, K]):
//   k       [HV, total_tokens, K]  float16  — keys
//   g_cs    [HV, total_tokens, K]  float32  — within-chunk cumulative gate sum
//   beta    [HV, total_tokens]     float16  — post-sigmoid beta in (0, 1)
//   mask    [C, C]                 float32  — strict-lower-tri (1 below diag, else 0)
//   ws_in   [block_dim*2, 2*C, K]  float32  — workspace: A (rows 0..C-1) + B (C..2C-1)
//   ws_out  [block_dim*2, C, C]    float32  — workspace: GEMM result M
//   b_ws    [block_dim*2, C]       float32  — workspace: per-token offset b[t]
//   L_out   [total_tokens, HV, C]  float16  — strictly-lower-tri L (BSND)
//
// Cross-core architecture (mirrors GDN scaled_dot_kkt / main kkt_kda pattern):
//   Both Vec sub-blocks (vid=0,1) do real work: each handles HalfChunk rows.
//     vid=0 → rows [0, C/2),  vid=1 → rows [C/2, C)
//   Vec pre:  load k, g_cs (my rows) → b=rowmax(g_cs) → A=k*exp(g-b),
//             B=k*exp(b-g) (fp32) → ws_in[my rows];  b[my rows] → b_ws.
//   Cube:     load full A, B from ws_in → fp32 GEMM A @ B^T → ws_out.
//   Vec post: load M[my rows] + full b → corr=exp(min(b[r]-b[c],0)) →
//             L = M*corr*beta*mask → cast fp16 → L_out.
//
// FFTS flags (double-buffered, slot = ci & 1):
//   0, 1 : Vec → Cube  "ws_in[slot] ready"  (both vids must sig under mode-2 reduce)
//   2, 3 : Cube → Vec  "ws_out[slot] ready" (broadcast: each vid gets a signal)
//   4, 5 : Vec → Cube  "ws_out[slot] free"  (Vec done reading M; conditional)
//   6-9  : reserved for sync_all() entry/exit barriers
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

// ── Cube tile aliases (fp32 GEMM), copied from chunk_o_kda.cpp ──────────────
template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0B = pto::Tile<pto::TileType::Right, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                           pto::BLayout::ColMajor, pto::DYNAMIC,
                           pto::DYNAMIC, pto::SLayout::RowMajor, 512,
                           pto::PadValue::Zero>;

// Single-shot dense GEMM  C = A @ B^T  via L0A/L0B — used when the K-dim is one
// L0 tile (inner dim KDim == 128 == L0 tile size, so no K-slicing needed).
// transpose-B only (avoids a <type_traits> dependency, which would break the
// mega kernel's namespaced #include ordering vs chunk_o_kda.cpp).
template <typename T1, typename T2, int32_t M, int32_t N, int32_t K>
AICORE PTO_INLINE void
gemm_abt(TileMatL1<T1, M, K, M, K> &A,
         TileMatL1<T1, N, K, N, K> &B,
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
    TileMatL1ZN<T1, K, N, K, N> B_t;
    pto::TRESHAPE(B_t, B);
    pto::TEXTRACT(l0b, B_t, 0, 0);

    set_flag(PIPE_MTE1, PIPE_M, war_event_id);
    wait_flag(PIPE_MTE1, PIPE_M, war_event_id);
    pto::TMATMUL(C, l0a, l0b);

    set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    set_flag(PIPE_M, PIPE_FIX, war_event_id);
    wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}
#endif

template <int32_t KDim, int32_t ChunkSize>
AICORE void kkt_kda_kernel(
    __gm__ half *k_ptr,
    __gm__ float *g_cs_ptr,
    __gm__ half *beta_ptr,
    __gm__ float *mask_ptr,
    __gm__ float *ws_in_ptr,
    __gm__ float *ws_out_ptr,
    __gm__ float *b_ws_ptr,
    __gm__ half *L_out_ptr,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    int32_t num_heads, uint64_t ffts_addr)
{
    auto cid = get_block_idx();
    auto block_num = get_block_num();
    set_ffts_base_addr(ffts_addr);

    const int32_t NumHeads = num_heads;

    constexpr int32_t HalfChunk = ChunkSize / 2;
    constexpr int32_t KTC = ((KDim + 7) / 8) * 8;

    int64_t num_seqs = batch_size;
    int64_t total_work = num_seqs * NumHeads;

    // Workspace element counts per slot (fp32 elements):
    constexpr int32_t WsInSlotElems = 2 * ChunkSize * KDim;   // A + B
    constexpr int32_t WsOutSlotElems = ChunkSize * ChunkSize; // M
    constexpr int32_t BWsSlotElems = ChunkSize;               // b[t]

    // ── GM type aliases (head-major [HV, T, K]) ──────────────────────────────
    using GmShapeDyn = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using GmFloatK = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmHalfK = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmHalf_1 = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, 1, 1>>;
    using GmFloat_1 = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, 1, 1>>;
    // L output is BSND-interleaved [total_tokens, NumHeads, ChunkSize].
    using GmHalfOut = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, DYNAMIC, 1>>;
    using GmFloatWsIn = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
    using GmFloatWsOut = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, ChunkSize, 1>>;

    // =========================================================================
    // VEC PHASE
    // =========================================================================
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    sync_all();

    auto vid = get_subblockid();
    int32_t my_off = static_cast<int32_t>(vid) * HalfChunk;

    // ── UB layout (per vid) ──────────────────────────────────────────────────
    // Mask persists at offset 0; the rest is a pool reused between pre-compute
    // and post-process (they never run concurrently within a chunk).
    constexpr int32_t MSK_ADDR = 0; // [HalfChunk, C] fp32
    constexpr int32_t POOL = MSK_ADDR + HalfChunk * ChunkSize * 4;
    // Pre-compute tiles (live simultaneously: g, k, A, B):
    constexpr int32_t G_ADDR = POOL;                            // [HalfChunk, KTC] fp32
    constexpr int32_t K_ADDR = G_ADDR + HalfChunk * KTC * 4;    // [HalfChunk, KTC] fp32
    constexpr int32_t A_ADDR = K_ADDR + HalfChunk * KTC * 4;    // [HalfChunk, KTC] fp32
    constexpr int32_t B_ADDR = A_ADDR + HalfChunk * KTC * 4;    // [HalfChunk, KTC] fp32
    constexpr int32_t BCOL_ADDR = B_ADDR + HalfChunk * KTC * 4; // [HalfChunk, 1] fp32 (b, ColMajor)
    // Post-process tiles (overlap pre-compute addresses):
    constexpr int32_t M_ADDR = POOL;                                   // [HalfChunk, C] fp32
    constexpr int32_t CORR_ADDR = M_ADDR + HalfChunk * ChunkSize * 4;  // [HalfChunk, C] fp32
    constexpr int32_t LH_ADDR = CORR_ADDR + HalfChunk * ChunkSize * 4; // [HalfChunk, C] fp16
    constexpr int32_t BROW_ADDR = LH_ADDR + HalfChunk * ChunkSize * 2; // [1, C] fp32 (b cols)
    constexpr int32_t BETA_ADDR = BROW_ADDR + ChunkSize * 4;           // [HalfChunk, 1] fp32 (beta, ColMajor)
    constexpr int32_t BETAH_ADDR = BETA_ADDR + HalfChunk * 4;          // [1, HalfChunk] fp16 (beta staging)

    // ── Load this vid's HalfChunk rows of the strict-lower mask (once) ──────
    {
        UbND<float, HalfChunk, ChunkSize> msk_ub;
        TASSIGN(msk_ub, MSK_ADDR);
        GmShapeDyn gs;
        gs.shape[3] = HalfChunk;
        gs.shape[4] = ChunkSize;
        GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, ChunkSize, 1>>
            msk_gm(mask_ptr + static_cast<int64_t>(my_off) * ChunkSize, gs);
        TLOAD(msk_ub, msk_gm);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

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
            int32_t my_rows = valid_rows - my_off;
            if (my_rows > HalfChunk)
                my_rows = HalfChunk;
            if (my_rows < 0)
                my_rows = 0;

            int64_t hbase = static_cast<int64_t>(head_idx) * total_tokens * KDim;
            int64_t my_first = bos + chunk_start + my_off;
            int64_t ws_in_base =
                (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsInSlotElems);
            int64_t b_ws_base =
                (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(BWsSlotElems);

            // ── PRE-COMPUTE ────────────────────────────────────────────────
            if (my_rows > 0)
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
                GmHalf_1 b_gm(beta_ptr + static_cast<int64_t>(head_idx) * total_tokens +
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
                    gs.shape[3] = my_rows;
                    gs.shape[4] = KDim;
                    GmFloatK g_gm(g_cs_ptr + hbase + my_first * KDim, gs);
                    UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                        g_ld(my_rows, KDim);
                    TASSIGN(g_ld, G_ADDR);
                    TLOAD(g_ld, g_gm);
                    if (my_rows != HalfChunk || KDim != KTC)
                    {
                        UbND<float, HalfChunk, KTC, HalfChunk, KTC, PadValue::Zero> g_pad;
                        TASSIGN(g_pad, G_ADDR);
                        TFILLPAD_INPLACE(g_pad, g_ld);
                    }
                }
                // Load k (fp16) → A_ADDR (fp16 staging), cvt → K_ADDR (fp32).
                {
                    GmShapeDyn gs;
                    gs.shape[3] = my_rows;
                    gs.shape[4] = KDim;
                    GmHalfK k_gm(k_ptr + hbase + my_first * KDim, gs);
                    UbND<half, HalfChunk, KTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                        k_ld(my_rows, KDim);
                    TASSIGN(k_ld, A_ADDR);
                    TLOAD(k_ld, k_gm);
                    if (my_rows != HalfChunk || KDim != KTC)
                    {
                        UbND<half, HalfChunk, KTC, HalfChunk, KTC, PadValue::Zero> k_pad;
                        TASSIGN(k_pad, A_ADDR);
                        TFILLPAD_INPLACE(k_pad, k_ld);
                    }
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                {
                    UbND<half, HalfChunk, KTC> k_h;
                    TASSIGN(k_h, A_ADDR);
                    UbND<float, HalfChunk, KTC> k_f;
                    TASSIGN(k_f, K_ADDR);
                    TCVT(k_f, k_h, pto::RoundMode::CAST_NONE);
                    pipe_barrier(PIPE_V);
                }

                UbND<float, HalfChunk, KTC> g_ub;
                TASSIGN(g_ub, G_ADDR);
                UbND<float, HalfChunk, KTC> k_ub;
                TASSIGN(k_ub, K_ADDR);
                UbND<float, HalfChunk, KTC> a_ub;
                TASSIGN(a_ub, A_ADDR);
                UbND<float, HalfChunk, KTC> b_ub;
                TASSIGN(b_ub, B_ADDR);
                UbDN<float, HalfChunk, 1> bcol;
                TASSIGN(bcol, BCOL_ADDR);

                // b[r] = max_d g_cs[r,d]   (use B_ADDR space as the rowmax tmp).
                {
                    UbND<float, HalfChunk, KTC> rmax_tmp;
                    TASSIGN(rmax_tmp, B_ADDR);
                    TROWMAX(bcol, g_ub, rmax_tmp);
                    pipe_barrier(PIPE_V);
                }
                // A = k * exp(g_cs - b)        (exp(g-b) <= 1, never overflows)
                TROWEXPANDEXPDIF(a_ub, g_ub, bcol);
                pipe_barrier(PIPE_V);
                TMUL(a_ub, a_ub, k_ub);
                pipe_barrier(PIPE_V);
                // B = k * exp(b - g_cs) = k * exp(-(g_cs - b))
                // The exponent (b - g_cs) >= 0 equals the within-token cross-channel
                // spread; for heavily-decayed channels (Kimi per-channel gate spikes
                // ~-160) it exceeds the fp32 exp limit.  Saturate it at 80 so B stays
                // finite (exp(80)~5.5e34; 128-term sum stays < fp32 max 3.4e38).  This
                // keeps the result exact wherever the spread < 80 and merely
                // under-counts the rare heavily-decayed channel (whose A factor has
                // already underflowed to 0) instead of producing inf -> 0*inf = NaN.
                TROWEXPANDSUB(b_ub, g_ub, bcol); // g - b   (<= 0)
                pipe_barrier(PIPE_V);
                TNEG(b_ub, b_ub); // b - g   (>= 0)
                pipe_barrier(PIPE_V);
                TMINS(b_ub, b_ub, 80.0f); // saturating exp
                pipe_barrier(PIPE_V);
                TEXP(b_ub, b_ub);
                pipe_barrier(PIPE_V);
                TMUL(b_ub, b_ub, k_ub);
                pipe_barrier(PIPE_V);

                // Store A → ws_in[slot] rows [my_off, …).
                {
                    GmShapeDyn gs;
                    gs.shape[3] = HalfChunk;
                    gs.shape[4] = KDim;
                    int64_t off = ws_in_base + static_cast<int64_t>(my_off) * KDim;
                    GmFloatWsIn gm_a(ws_in_ptr + off, gs);
                    UbND<float, HalfChunk, KTC, HalfChunk, KTC> a_st;
                    TASSIGN(a_st, A_ADDR);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(gm_a, a_st);
                }
                // Store B → ws_in[slot] rows [C+my_off, …).
                {
                    GmShapeDyn gs;
                    gs.shape[3] = HalfChunk;
                    gs.shape[4] = KDim;
                    int64_t off = ws_in_base +
                                  static_cast<int64_t>(ChunkSize) * KDim +
                                  static_cast<int64_t>(my_off) * KDim;
                    GmFloatWsIn gm_b(ws_in_ptr + off, gs);
                    UbND<float, HalfChunk, KTC, HalfChunk, KTC> b_st;
                    TASSIGN(b_st, B_ADDR);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                    TSTORE(gm_b, b_st);
                }
                // Store b[my rows] → b_ws[slot] (contiguous [my_rows] floats).
                // bcol is ColMajor [my_rows,1] but its bytes are my_rows contiguous
                // floats, identical to a RowMajor [1,my_rows] — alias and store as a
                // row so the GM (ND) store is a supported ND2ND copy.
                {
                    GmShapeDyn gs;
                    gs.shape[3] = 1;
                    gs.shape[4] = my_rows;
                    GmFloat_1 b_gm(b_ws_ptr + b_ws_base + my_off, gs);
                    UbND<float, 1, HalfChunk, DYNAMIC, DYNAMIC> b_st(1, my_rows);
                    TASSIGN(b_st, BCOL_ADDR);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                    TSTORE(b_gm, b_st);
                }
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }

            // Both vids signal flag(slot) under V→C reduce mode 2.
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (slot << 8));
            wait_flag_dev(2 + slot); // Cube: ws_out[slot] ready
            pipe_barrier(PIPE_ALL);

            // ── POST-PROCESS ───────────────────────────────────────────────
            if (my_rows > 0)
            {
                // Load M[my rows, :] (fp32) from ws_out[slot].
                {
                    int64_t wo_base = (static_cast<int64_t>(cid) * 2 + slot) *
                                      static_cast<int64_t>(WsOutSlotElems);
                    int64_t off = wo_base + static_cast<int64_t>(my_off) * ChunkSize;
                    GmShapeDyn gs;
                    gs.shape[3] = my_rows;
                    gs.shape[4] = ChunkSize;
                    GmFloatWsOut gm(ws_out_ptr + off, gs);
                    UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero>
                        m_ld(my_rows, ChunkSize);
                    TASSIGN(m_ld, M_ADDR);
                    TLOAD(m_ld, gm);
                }
                // Load full b[0..C-1] → BROW_ADDR ([1,C], RowMajor, ND2ND).  The
                // per-row offsets b[my_off + r] are the contiguous sub-range starting
                // at BROW_ADDR + my_off floats; alias it below as a ColMajor [my_rows,1]
                // (same bytes) for the per-row broadcast — no separate load needed.
                {
                    GmShapeDyn gs;
                    gs.shape[3] = 1;
                    gs.shape[4] = ChunkSize;
                    GmFloat_1 brow_gm(b_ws_ptr + b_ws_base, gs);
                    UbND<float, 1, ChunkSize> brow_ld;
                    TASSIGN(brow_ld, BROW_ADDR);
                    TLOAD(brow_ld, brow_gm);
                }
                // Load beta[my rows] (fp16) → BETAH_ADDR.
                {
                    GmShapeDyn gs;
                    gs.shape[3] = 1;
                    gs.shape[4] = my_rows;
                    GmHalf_1 b_gm(beta_ptr +
                                      static_cast<int64_t>(head_idx) * total_tokens +
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

                UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> m_ub(my_rows, ChunkSize);
                TASSIGN(m_ub, M_ADDR);
                UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> corr(my_rows, ChunkSize);
                TASSIGN(corr, CORR_ADDR);
                UbND<float, 1, ChunkSize> brow;
                TASSIGN(brow, BROW_ADDR);
                // b[my_off + r] sub-range of brow, viewed as a ColMajor [my_rows,1].
                UbDN<float, HalfChunk, 1, DYNAMIC, DYNAMIC> bcol(my_rows, 1);
                TASSIGN(bcol, BROW_ADDR + my_off * 4);
                UbDN<float, HalfChunk, 1, DYNAMIC, DYNAMIC> beta_col(my_rows, 1);
                TASSIGN(beta_col, BETA_ADDR);
                UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> msk_ub(my_rows, ChunkSize);
                TASSIGN(msk_ub, MSK_ADDR);

                // corr[r,c] = exp(min(b[r] - b[c], 0))
                //   fill with b[r] per row, subtract b[c] per col, clamp, exp.
                TEXPANDS(corr, 0.0f);
                pipe_barrier(PIPE_V);
                TROWEXPANDADD(corr, corr, bcol); // corr[r,c] = b[r]
                pipe_barrier(PIPE_V);
                TCOLEXPANDSUB(corr, corr, brow); // corr[r,c] = b[r] - b[c]
                pipe_barrier(PIPE_V);
                TMINS(corr, corr, 0.0f);
                pipe_barrier(PIPE_V);
                TEXP(corr, corr);
                pipe_barrier(PIPE_V);

                // L = M * corr * beta[r] * mask
                TMUL(m_ub, m_ub, corr);
                pipe_barrier(PIPE_V);
                TROWEXPANDMUL(m_ub, m_ub, beta_col);
                pipe_barrier(PIPE_V);
                TMUL(m_ub, m_ub, msk_ub);
                pipe_barrier(PIPE_V);

                // Cast fp32 → fp16 and store to L_out (BSND).
                {
                    UbND<half, HalfChunk, ChunkSize> l_h;
                    TASSIGN(l_h, LH_ADDR);
                    TCVT(l_h, m_ub, pto::RoundMode::CAST_NONE);
                    pipe_barrier(PIPE_V);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    int64_t l_off = my_first * static_cast<int64_t>(NumHeads) * ChunkSize +
                                    static_cast<int64_t>(head_idx) * ChunkSize;
                    GmShapeDyn gs;
                    gs.shape[3] = my_rows;
                    gs.shape[4] = ChunkSize;
                    Stride<1, 1, 1, DYNAMIC, 1> l_stride(NumHeads * ChunkSize);
                    GmHalfOut l_gm(L_out_ptr + l_off, gs, l_stride);
                    UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> l_st(my_rows, ChunkSize);
                    TASSIGN(l_st, LH_ADDR);
                    TSTORE(l_gm, l_st);
                }
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }

            // Drain Vec pipes before next chunk, then free ws_out[slot].
            pipe_barrier(PIPE_ALL);
            if (ci < num_chunks - 2)
            {
                ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((4 + slot) << 8));
            }
        }
    }

    sync_all();
#endif // __DAV_C220_VEC__

    // =========================================================================
    // CUBE PHASE: fp32 GEMM  A @ B^T → M
    // =========================================================================
#if defined(__DAV_C220_CUBE__)
    constexpr int32_t L1BAddr = ChunkSize * KDim * static_cast<int32_t>(sizeof(float));

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

            if (ci >= 2)
            {
                wait_flag_dev(4 + slot); // ws_out[slot] free
                pipe_barrier(PIPE_ALL);
            }
            wait_flag_dev(slot); // ws_in[slot] ready (both vids sig'd)
            pipe_barrier(PIPE_ALL);

            int64_t ws_in_base =
                (static_cast<int64_t>(cid) * 2 + slot) * static_cast<int64_t>(WsInSlotElems);

            TileMatL1<float, ChunkSize, KDim, ChunkSize, KDim> a_l1;
            TASSIGN(a_l1, 0);
            TileMatL1<float, ChunkSize, KDim, ChunkSize, KDim> b_l1;
            TASSIGN(b_l1, L1BAddr);
            TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> m_l0;
            TASSIGN(m_l0, 0);

            // Load A [C,K] from ws_in[slot, 0:C, :].
            {
                DynMatL1<float, ChunkSize, KDim> _l1(ChunkSize, KDim);
                TASSIGN(_l1, 0);
                GmShapeDyn _gs;
                _gs.shape[3] = ChunkSize;
                _gs.shape[4] = KDim;
                GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>
                    _gm(ws_in_ptr + ws_in_base, _gs);
                TLOAD(_l1, _gm);
            }
            // Load B [C,K] from ws_in[slot, C:2C, :].
            {
                DynMatL1<float, ChunkSize, KDim> _l1(ChunkSize, KDim);
                TASSIGN(_l1, L1BAddr);
                GmShapeDyn _gs;
                _gs.shape[3] = ChunkSize;
                _gs.shape[4] = KDim;
                GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>
                    _gm(ws_in_ptr + ws_in_base + static_cast<int64_t>(ChunkSize) * KDim, _gs);
                TLOAD(_l1, _gm);
            }

            set_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);

            // M = A @ B^T  (fp32).
            gemm_abt<float, float, ChunkSize, ChunkSize, KDim>(a_l1, b_l1, m_l0);

            // Store M [C,C] (fp32) → ws_out[slot].
            {
                int64_t wo_base = (static_cast<int64_t>(cid) * 2 + slot) *
                                  static_cast<int64_t>(WsOutSlotElems);
                GmShapeDyn _gs;
                _gs.shape[3] = ChunkSize;
                _gs.shape[4] = ChunkSize;
                GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, ChunkSize, 1>>
                    _gm(ws_out_ptr + wo_base, _gs);
                TSTORE(_gm, m_l0);
            }

            pipe_barrier(PIPE_ALL);
            // Signal Vec: ws_out[slot] ready (GEMM done; ws_in[slot] also free).
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | ((2 + slot) << 8));
        }
    }

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
    __gm__ uint8_t *b_ws_ptr,
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
        reinterpret_cast<__gm__ float *>(b_ws_ptr),
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
    uint8_t *b_ws_ptr,
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
        ws_in_ptr, ws_out_ptr, b_ws_ptr, L_out_ptr, cu_seqlens,
        batch_size, seq_len, total_tokens,
        static_cast<int32_t>(num_heads), fftsAddr);
}
