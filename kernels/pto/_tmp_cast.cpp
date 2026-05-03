// cast_bf16_fp16.cpp — Standalone BF16 <-> FP16 cast kernel for Ascend NPU.
//
// Root cause discovery: GlobalTensor<bfloat16_t,...> for TLOAD/TSTORE is
// unreliable on dav-c220 when multiple loop iterations run per AIcore
// (i.e., n_elem > BLOCK_DIM*C). Pointer arithmetic `bfloat16_t* + offset`
// expands to offset * sizeof(bfloat16_t) bytes; if sizeof(bfloat16_t) != 2
// (or if the GlobalTensor DMA controller has a bfloat16 bug), only the
// first chunk (offset=0) is read/written correctly.
//
// Fix: use GlobalTensor<half,...> for all DMA (TLOAD/TSTORE), passing the
// bf16 pointer cast to half*. Inside UB, a bfloat16_t tile alias at the
// same UB address lets TCVT correctly interpret the bits. A pipe_barrier(PIPE_V)
// is inserted between consecutive TCVT calls to flush any intra-pipe hazard.
//
// Conversion paths:
//   BF16 → FP16  (two-hop: half-DMA-load → bf16-alias TCVT → fp32 → fp16)
//   FP16 → BF16  (two-hop: half-DMA-load → fp16 TCVT → fp32 → bf16-alias → half-DMA-store)
//   FP32 → FP16  (one-hop: float-DMA-load → TCVT → fp16-DMA-store)
//   FP16 → FP32  (one-hop: half-DMA-load → TCVT → float-DMA-store)
//
// C wrapper:
//   call_kernel(block_dim, stream, src, dst, n_elem, direction)
//     direction == 0 : BF16 → FP16
//     direction == 1 : FP16 → BF16
//     direction == 2 : FP32 → FP16  (diagnostic / reference)
//     direction == 3 : FP16 → FP32  (diagnostic / reference)

#ifndef GDN_C
#define GDN_C 128
#endif
#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <type_traits>
using namespace pto;

// ===================================================================
// Device-side helpers
// ===================================================================
#ifdef __CCE_AICORE__

// ------------------------------------------------------------------
// Shared type aliases
// ------------------------------------------------------------------
template <int32_t C>
struct TileTypes {
    // Half tile for DMA-compatible loads/stores
    using HalfFull = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor,
                          1, C, SLayout::NoneBox, 512, PadValue::Zero>;
    using HalfDyn  = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor,
                          DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using HalfOut  = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor,
                          1, C, SLayout::NoneBox, 512>;
    using HalfOutDyn = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor,
                            DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;

    // BF16 UB-only alias (never used for GM DMA, only for TCVT)
    using BF16Alias = Tile<TileType::Vec, bfloat16_t, 1, C, BLayout::RowMajor,
                           1, C, SLayout::NoneBox, 512, PadValue::Zero>;

    // FP32 intermediate
    using F32Full = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor,
                         1, C, SLayout::NoneBox, 512, PadValue::Zero>;
    using F32Dyn  = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor,
                         DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;

    // FP32 for GM (direction=2/3)
    using F32GMFull = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor,
                           1, C, SLayout::NoneBox, 512, PadValue::Zero>;
};

// ------------------------------------------------------------------
// BF16 → FP16  (two-hop: BF16→FP32→FP16 via UB alias)
// DMA: load via half* pointer (same raw 16-bit layout as BF16).
// TCVT: interpret UB bits as bfloat16_t via alias; then fp32→fp16.
// ------------------------------------------------------------------
template <int32_t C>
AICORE void cast_bf16_to_fp16_1d(__gm__ half *src,  // bf16 data, read as half bits
                                  __gm__ half *dst,  // fp16 result
                                  int64_t n_elem)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    // UB layout (byte offsets):
    //   [0x000, 0x100)  raw16 / BF16 alias  : C * 2 = 256 B
    //   [0x100, 0x300)  FP32 temp            : C * 4 = 512 B
    //   [0x300, 0x400)  FP16 dest            : C * 2 = 256 B
    constexpr int32_t RAW16_UB = 0;
    constexpr int32_t F32_UB   = C * static_cast<int32_t>(sizeof(half));       // 256
    constexpr int32_t F16_UB   = F32_UB + C * static_cast<int32_t>(sizeof(float)); // 768

    using TT = TileTypes<C>;

    typename TT::HalfFull raw_ub;    TASSIGN(raw_ub,    RAW16_UB);
    typename TT::F32Full   f32_ub;   TASSIGN(f32_ub,    F32_UB);
    typename TT::HalfOut   f16_ub;   TASSIGN(f16_ub,    F16_UB);

    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    auto cid       = get_block_idx();
    auto block_num = get_block_num();
    int64_t n_chunks = (n_elem + C - 1) / C;

    for (int64_t ci = static_cast<int64_t>(cid); ci < n_chunks;
         ci += static_cast<int64_t>(block_num)) {
        int64_t off   = ci * C;
        int32_t valid = (off + C <= n_elem) ? C
                                             : static_cast<int32_t>(n_elem - off);

        // Load raw BF16 bits via half-typed GlobalTensor
        {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<half, Gm1D, GmS1> gm(src + off, gs);
            typename TT::HalfDyn ld(1, valid);
            TASSIGN(ld, RAW16_UB);
            TLOAD(ld, gm);
            if (valid != C) TFILLPAD_INPLACE(raw_ub, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Declare bf16_alias INSIDE the loop to ensure fresh metadata each iteration.
        // The UB bytes at RAW16_UB were written by the half-typed TLOAD above.
        // Aliasing as bfloat16_t lets TCVT generate vconv_bf162f32.
        typename TT::BF16Alias bf16_alias; TASSIGN(bf16_alias, RAW16_UB);

        // BF16 → FP32 (reads raw_ub bits as bfloat16_t → vconv_bf162f32)
        TCVT(f32_ub, bf16_alias, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        // FP32 → FP16
        TCVT(f16_ub, f32_ub, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<half, Gm1D, GmS1> gm(dst + off, gs);
            typename TT::HalfOutDyn st(1, valid);
            TASSIGN(st, F16_UB);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // Full pipeline fence to prevent any inter-iteration hazard with
        // bfloat16_t TCVT state.
        
    }
#endif
}

// ------------------------------------------------------------------
// FP16 → BF16  (two-hop: FP16→FP32→BF16 via UB alias)
// DMA: load/store via half* pointer for BF16 side (same raw 16 bits).
// ------------------------------------------------------------------
template <int32_t C>
AICORE void cast_fp16_to_bf16_1d(__gm__ half *src,  // fp16 source
                                  __gm__ half *dst,  // bf16 result, stored as half bits
                                  int64_t n_elem)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    // UB layout:
    //   [0x000, 0x100)  FP16 source          : C * 2 = 256 B
    //   [0x100, 0x300)  FP32 temp            : C * 4 = 512 B
    //   [0x300, 0x400)  BF16 alias / half out: C * 2 = 256 B
    constexpr int32_t F16_UB   = 0;
    constexpr int32_t F32_UB   = C * static_cast<int32_t>(sizeof(half));        // 256
    constexpr int32_t BF16_UB  = F32_UB + C * static_cast<int32_t>(sizeof(float)); // 768

    using TT = TileTypes<C>;

    typename TT::HalfFull  f16_ub;     TASSIGN(f16_ub,     F16_UB);
    typename TT::F32Full   f32_ub;     TASSIGN(f32_ub,     F32_UB);
    typename TT::HalfOut   raw_out;    TASSIGN(raw_out,    BF16_UB); // same bytes as bf16_alias

    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    auto cid       = get_block_idx();
    auto block_num = get_block_num();
    int64_t n_chunks = (n_elem + C - 1) / C;

    for (int64_t ci = static_cast<int64_t>(cid); ci < n_chunks;
         ci += static_cast<int64_t>(block_num)) {
        int64_t off   = ci * C;
        int32_t valid = (off + C <= n_elem) ? C
                                             : static_cast<int32_t>(n_elem - off);

        // Load FP16 via half-typed GlobalTensor
        {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<half, Gm1D, GmS1> gm(src + off, gs);
            typename TT::HalfDyn ld(1, valid);
            TASSIGN(ld, F16_UB);
            TLOAD(ld, gm);
            if (valid != C) TFILLPAD_INPLACE(f16_ub, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // FP16 → FP32
        TCVT(f32_ub, f16_ub, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        // FP32 → BF16 (writes to bf16_alias UB; read back via raw_out for TSTORE)
        // bf16_alias declared inside loop for fresh metadata each iteration.
        typename TT::BF16Alias bf16_alias; TASSIGN(bf16_alias, BF16_UB);
        TCVT(bf16_alias, f32_ub, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // Store BF16 bits via half-typed GlobalTensor
        {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<half, Gm1D, GmS1> gm(dst + off, gs);
            typename TT::HalfOutDyn st(1, valid);
            TASSIGN(st, BF16_UB);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        
    }
#endif
}

// ------------------------------------------------------------------
// FP32 → FP16
// ------------------------------------------------------------------
template <int32_t C>
AICORE void cast_f32_to_f16_1d(__gm__ float *src, __gm__ half *dst,
                                int64_t n_elem)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    constexpr int32_t F32_UB = 0;
    constexpr int32_t F16_UB = C * static_cast<int32_t>(sizeof(float));  // 512

    using TT = TileTypes<C>;
    using F32Full = typename TT::F32GMFull;
    using F32Dyn  = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor,
                         DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;

    F32Full f32_ub; TASSIGN(f32_ub, F32_UB);
    typename TT::HalfOut f16_ub; TASSIGN(f16_ub, F16_UB);

    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    auto cid       = get_block_idx();
    auto block_num = get_block_num();
    int64_t n_chunks = (n_elem + C - 1) / C;

    for (int64_t ci = static_cast<int64_t>(cid); ci < n_chunks;
         ci += static_cast<int64_t>(block_num)) {
        int64_t off   = ci * C;
        int32_t valid = (off + C <= n_elem) ? C
                                             : static_cast<int32_t>(n_elem - off);

        {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<float, Gm1D, GmS1> gm(src + off, gs);
            F32Dyn ld(1, valid);
            TASSIGN(ld, F32_UB);
            TLOAD(ld, gm);
            if (valid != C) TFILLPAD_INPLACE(f32_ub, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TCVT(f16_ub, f32_ub, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<half, Gm1D, GmS1> gm(dst + off, gs);
            typename TT::HalfOutDyn st(1, valid);
            TASSIGN(st, F16_UB);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
#endif
}

// ------------------------------------------------------------------
// FP16 → FP32
// ------------------------------------------------------------------
template <int32_t C>
AICORE void cast_f16_to_f32_1d(__gm__ half *src, __gm__ float *dst,
                                int64_t n_elem)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    constexpr int32_t F16_UB = 0;
    constexpr int32_t F32_UB = C * static_cast<int32_t>(sizeof(half));  // 256

    using TT = TileTypes<C>;

    typename TT::HalfFull f16_ub; TASSIGN(f16_ub, F16_UB);
    typename TT::F32Full  f32_ub; TASSIGN(f32_ub, F32_UB);

    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    auto cid       = get_block_idx();
    auto block_num = get_block_num();
    int64_t n_chunks = (n_elem + C - 1) / C;

    for (int64_t ci = static_cast<int64_t>(cid); ci < n_chunks;
         ci += static_cast<int64_t>(block_num)) {
        int64_t off   = ci * C;
        int32_t valid = (off + C <= n_elem) ? C
                                             : static_cast<int32_t>(n_elem - off);

        {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<half, Gm1D, GmS1> gm(src + off, gs);
            typename TT::HalfDyn ld(1, valid);
            TASSIGN(ld, F16_UB);
            TLOAD(ld, gm);
            if (valid != C) TFILLPAD_INPLACE(f16_ub, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TCVT(f32_ub, f16_ub, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        {
            Gm1D gs; gs.shape[4] = valid;
            GlobalTensor<float, Gm1D, GmS1> gm(dst + off, gs);
            typename TT::F32Dyn st(1, valid);
            TASSIGN(st, F32_UB);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
#endif
}

#endif  // __CCE_AICORE__

// ===================================================================
// Global kernel entry points
// ===================================================================

// BF16→FP16: pass src bf16 pointer as half* (same raw bits)
extern "C" __global__ AICORE void cast_bf16_to_fp16_kernel(
    __gm__ uint8_t *src, __gm__ uint8_t *dst, int64_t n_elem)
{
    cast_bf16_to_fp16_1d<GDN_C>(
        reinterpret_cast<__gm__ half *>(src),
        reinterpret_cast<__gm__ half *>(dst),
        n_elem);
}

// FP16→BF16: pass dst bf16 pointer as half* (same raw bits)
extern "C" __global__ AICORE void cast_fp16_to_bf16_kernel(
    __gm__ uint8_t *src, __gm__ uint8_t *dst, int64_t n_elem)
{
    cast_fp16_to_bf16_1d<GDN_C>(
        reinterpret_cast<__gm__ half *>(src),
        reinterpret_cast<__gm__ half *>(dst),
        n_elem);
}

extern "C" __global__ AICORE void cast_f32_to_f16_kernel(
    __gm__ uint8_t *src, __gm__ uint8_t *dst, int64_t n_elem)
{
    cast_f32_to_f16_1d<GDN_C>(
        reinterpret_cast<__gm__ float *>(src),
        reinterpret_cast<__gm__ half *>(dst),
        n_elem);
}

extern "C" __global__ AICORE void cast_f16_to_f32_kernel(
    __gm__ uint8_t *src, __gm__ uint8_t *dst, int64_t n_elem)
{
    cast_f16_to_f32_1d<GDN_C>(
        reinterpret_cast<__gm__ half *>(src),
        reinterpret_cast<__gm__ float *>(dst),
        n_elem);
}

// direction == 0 : BF16 → FP16
// direction == 1 : FP16 → BF16
// direction == 2 : FP32 → FP16
// direction == 3 : FP16 → FP32
extern "C" void call_kernel(uint32_t block_dim, void *stream,
                             uint8_t *src, uint8_t *dst,
                             int64_t n_elem, int32_t direction)
{
    if (direction == 0) {
        cast_bf16_to_fp16_kernel<<<block_dim, nullptr, stream>>>(src, dst, n_elem);
    } else if (direction == 1) {
        cast_fp16_to_bf16_kernel<<<block_dim, nullptr, stream>>>(src, dst, n_elem);
    } else if (direction == 2) {
        cast_f32_to_f16_kernel<<<block_dim, nullptr, stream>>>(src, dst, n_elem);
    } else {
        cast_f16_to_f32_kernel<<<block_dim, nullptr, stream>>>(src, dst, n_elem);
    }
}
