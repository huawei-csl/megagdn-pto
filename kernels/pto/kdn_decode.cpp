// ============================================================================
// kdn_decode.cpp -- naive_recurrent_kda single-step decode,
//                                fully compute-fused into ONE launch
//
// Stage role:
//   Collapses the three validated decode stages (state_decay_key_readout ->
//   delta_rank1_update -> query_readout) into a single kernel. Per work item
//   (a flat index over B*HV, i.e. one independent [K,V] fp32 state tile):
//     S_dec[k,v] = S_in[k,v] * exp(g[k])
//     kS[v]      = sum_k key[k] * S_dec[k,v]
//     delta[v]   = val[v] - kS[v]
//     S_out[k,v] = S_dec[k,v] + (beta*key[k]) * delta[v]
//     o[v]       = sum_k (scale*q[k]) * S_out[k,v],  scale = K ** -0.5
//   The [K,V] tile is TLOADed ONCE and TSTOREd ONCE. S_dec, kS, delta and the
//   beta-scaled key column NEVER round-trip through global memory: the state
//   tile is decayed in place, the rank-1 update is folded in place, and the
//   query readout consumes the in-place S_out before the single store.
//
// Architecture / dataflow:
//   vec_only. Grid-stride loop over total_work = B*HV work items. No Cube
//   engine is used, therefore no cross-core FFTS handshake exists in this
//   kernel; the only synchronisation is intra-core MTE<->Vec ordering.
//   Per item the UB residency is:
//     state  [K,V] fp32 64 KB  -- S_in -> S_dec -> S_out, all in place
//     scrat  [K,V] fp32 64 KB  -- key*S_dec product, then the rank-1 term,
//                                 then the q*S_out product (three lifetimes)
//     tmp    [K/2,V] fp32 32 KB -- TCOLSUM binary-tree scratch
//     small [1,*] rows: gate, key, query, value, kS, delta, out, beta window
//   The Mode 1 per-row-scalar operands (gate, key, query) are ColMajor [K,1]
//   tiles TASSIGNed to the SAME UB bytes as their RowMajor [1,K] row tiles: a
//   ColMajor tile with exactly one column stores its K values contiguously, so
//   the alias is byte-exact. C28(c) requires ColMajor [K,1] for Mode 1; a
//   RowMajor [K,8] operand would silently select Mode 2 and zero the result
//   outside column 0.
//
// Key PTO ops used:
//   TLOAD, TEXP, TMULS, TSUB, TROWEXPANDMUL, TCOLEXPAND, TCOLSUM, TADD,
//   TSTORE, set_flag, wait_flag, pipe_barrier
//
// Evidence gaps / conservative choices:
//   scale = 1/sqrt(K) is computed on device with a scalar Babylonian sqrt plus
//   a reciprocal because sqrtf() is a [host] function and is not callable from
//   [aicore] code under -xcce (C23/S2). It is evaluated once, outside the work
//   loop. K and V are compile-time capped at 128 (the contract-locked state
//   tile size) and clamped at runtime; all GM extents use the runtime dims.
// ============================================================================

// megagdn build convention (mirrors chunk_o_kda.cpp / chunk_cumsum.cpp):
// direct PTO + ACL + runtime includes, compiled device-side via bisheng -xcce
// (megagdn_pto/compile.py -> compile_kdn_decode). `runtime/rt_ffts.h` resolves
// through the toolkit's pkg_inc/runtime include dir.
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include <cmath>
#include <cstdint>
using namespace pto;

// Under `-xcce` the PTO headers define AICORE = [aicore]; keep a guarded
// fallback so the translation unit is self-contained.
#ifndef AICORE
#define AICORE [aicore]
#endif

#if defined(__CCE_AICORE__)

// Contract-locked tile geometry (C29: kernel-private, distinctive names).
constexpr int32_t kFdKCap = 128;   // state rows (qk head dim)
constexpr int32_t kFdVCap = 128;   // state cols (v head dim)
constexpr int32_t kFdSumRows = kFdKCap / 2;   // TCOLSUM binary-tree scratch
constexpr int32_t kFdBetaLanes = 8;           // 32-byte minimum fp32 width (C14)

// ---- UB address map (A2/A3: 192 KB; the PTO library reserves the top 8 KB
// from 184 KB for TROWEXPANDMUL's internal broadcast scratch) ----------------
constexpr int32_t kFdTileBytes = kFdKCap * kFdVCap * 4;          // 65536
constexpr int32_t kFdTmpBytes  = kFdSumRows * kFdVCap * 4;       // 32768
constexpr int32_t kFdRowBytes  = kFdVCap * 4;                    // 512

constexpr int32_t kFdStateUb = 0;                                //      0
constexpr int32_t kFdScratUb = kFdStateUb + kFdTileBytes;        //  65536
constexpr int32_t kFdTmpUb   = kFdScratUb + kFdTileBytes;        // 131072
constexpr int32_t kFdGateUb  = kFdTmpUb   + kFdTmpBytes;         // 163840
constexpr int32_t kFdKeyUb   = kFdGateUb  + kFdRowBytes;         // 164352
constexpr int32_t kFdQryUb   = kFdKeyUb   + kFdRowBytes;         // 164864
constexpr int32_t kFdValUb   = kFdQryUb   + kFdRowBytes;         // 165376
constexpr int32_t kFdKsUb    = kFdValUb   + kFdRowBytes;         // 165888
constexpr int32_t kFdDeltaUb = kFdKsUb    + kFdRowBytes;         // 166400
constexpr int32_t kFdOutUb   = kFdDeltaUb + kFdRowBytes;         // 166912
constexpr int32_t kFdBetaUb  = kFdOutUb   + kFdRowBytes;         // 167424
constexpr int32_t kFdMaxUb   = kFdBetaUb  + kFdBetaLanes * 4;    // 167456

static_assert(kFdMaxUb <= 184 * 1024,
              "UB footprint collides with the reserved PTO scratch window");
static_assert(kFdMaxUb <= 196608,
              "UB footprint exceeds A2/A3 capacity (192 KB)");

template <typename T, int R, int C, int RV, int CV>
using FdRow = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                        RV, CV, pto::SLayout::NoneBox, 512,
                        pto::PadValue::Null>;

template <typename T, int R, int C, int RV, int CV>
using FdCol = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::NoneBox, 512,
                        pto::PadValue::Null>;

using FdGmShape  = pto::Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using FdGmStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
using FdGmF      = pto::GlobalTensor<float, FdGmShape, FdGmStride>;

// 1/sqrt(x) with no [host] math function (C23). Babylonian iteration on sqrt
// followed by a reciprocal; deterministic and fp32-accurate to ~1 ulp.
AICORE inline float fd_rsqrt(float x)
{
    if (x <= 0.0f) {
        return 1.0f;
    }
    float s = x;
    if (s < 1.0f) {
        s = 1.0f;
    }
    for (int it = 0; it < 40; ++it) {
        s = 0.5f * (s + x / s);
    }
    return 1.0f / s;
}

#endif  // __CCE_AICORE__

#if defined(__CCE_AICORE__) || defined(__CPU_SIM)

AICORE void fused_kda_decode_kernel(__gm__ float *s_in, __gm__ float *q,
                                    __gm__ float *key, __gm__ float *val,
                                    __gm__ float *gate, __gm__ float *beta,
                                    __gm__ float *s_out, __gm__ float *o,
                                    int64_t total_work, int64_t k, int64_t v,
                                    uint64_t ffts_addr)
{
    set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
    auto vid = get_subblockid();
    if (vid != 0) {
        return;
    }
    set_mask_norm();
    set_vector_mask(-1, -1);

    if (k <= 0 || v <= 0 || total_work <= 0) {
        return;
    }
    if (k > kFdKCap) {
        k = kFdKCap;
    }
    if (v > kFdVCap) {
        v = kFdVCap;
    }

    const int32_t kv = static_cast<int32_t>(k);
    const int32_t vv = static_cast<int32_t>(v);
    const float scale = fd_rsqrt(static_cast<float>(kv));

    int64_t block_num = static_cast<int64_t>(get_block_num());
    if (block_num <= 0) {   // C17
        block_num = 1;
    }
    const int64_t core_idx = static_cast<int64_t>(get_block_idx());

    // ---- persistent UB tiles: shapes fixed, only the GM base moves per item
    FdRow<float, kFdKCap, kFdVCap, DYNAMIC, DYNAMIC> state_tile(kv, vv);
    FdRow<float, kFdKCap, kFdVCap, DYNAMIC, DYNAMIC> scrat_tile(kv, vv);
    FdRow<float, kFdSumRows, kFdVCap, DYNAMIC, DYNAMIC> tmp_tile(kFdSumRows, vv);

    FdRow<float, 1, kFdKCap, 1, DYNAMIC> gate_row(kv);
    FdRow<float, 1, kFdKCap, 1, DYNAMIC> key_row(kv);
    FdRow<float, 1, kFdKCap, 1, DYNAMIC> qry_row(kv);
    FdRow<float, 1, kFdVCap, 1, DYNAMIC> val_row(vv);
    FdRow<float, 1, kFdVCap, 1, DYNAMIC> ks_row(vv);
    FdRow<float, 1, kFdVCap, 1, DYNAMIC> delta_row(vv);
    FdRow<float, 1, kFdVCap, 1, DYNAMIC> o_row(vv);

    // ColMajor [K,1] aliases of the three per-row-scalar rows (C28c, Mode 1).
    FdCol<float, kFdKCap, 1, DYNAMIC, 1> gate_col(kv);
    FdCol<float, kFdKCap, 1, DYNAMIC, 1> key_col(kv);
    FdCol<float, kFdKCap, 1, DYNAMIC, 1> qry_col(kv);

    const int32_t beta_cnt = (total_work >= kFdBetaLanes)
                                 ? kFdBetaLanes
                                 : static_cast<int32_t>(total_work);
    FdRow<float, 1, kFdBetaLanes, 1, DYNAMIC> beta_tile(beta_cnt);

    TASSIGN(state_tile, kFdStateUb);
    TASSIGN(scrat_tile, kFdScratUb);
    TASSIGN(tmp_tile,   kFdTmpUb);
    TASSIGN(gate_row,   kFdGateUb);
    TASSIGN(gate_col,   kFdGateUb);
    TASSIGN(key_row,    kFdKeyUb);
    TASSIGN(key_col,    kFdKeyUb);
    TASSIGN(qry_row,    kFdQryUb);
    TASSIGN(qry_col,    kFdQryUb);
    TASSIGN(val_row,    kFdValUb);
    TASSIGN(ks_row,     kFdKsUb);
    TASSIGN(delta_row,  kFdDeltaUb);
    TASSIGN(o_row,      kFdOutUb);
    TASSIGN(beta_tile,  kFdBetaUb);

    const int64_t tile_elems = k * v;

    for (int64_t wi = core_idx; wi < total_work; wi += block_num) {
        pipe_barrier(PIPE_ALL);   // C20: flush the previous item's Vec state

        // beta is one fp32 at beta[wi]; stage an aligned 8-lane window that
        // never reads past the end of the tensor.
        int64_t beta_base = 0;
        if (total_work >= kFdBetaLanes) {
            beta_base = (wi / kFdBetaLanes) * kFdBetaLanes;
            if (beta_base + kFdBetaLanes > total_work) {
                beta_base = total_work - kFdBetaLanes;
            }
        }
        const uint32_t beta_idx = static_cast<uint32_t>(wi - beta_base);

        // ------------------------------------------------------------------
        // 1. ONE bulk load of the [K,V] state tile plus the small vectors.
        // ------------------------------------------------------------------
        {
            FdGmShape  ss(1, 1, 1, k, v);
            FdGmStride sst(1, 1, 1, v, 1);
            FdGmF s_gm(s_in + wi * tile_elems, ss, sst);
            TLOAD(state_tile, s_gm);

            FdGmShape  ks_shape(1, 1, 1, 1, k);
            FdGmStride ks_stride(1, 1, 1, k, 1);
            FdGmF g_gm(gate + wi * k, ks_shape, ks_stride);
            FdGmF k_gm(key  + wi * k, ks_shape, ks_stride);
            FdGmF q_gm(q    + wi * k, ks_shape, ks_stride);
            TLOAD(gate_row, g_gm);
            TLOAD(key_row, k_gm);
            TLOAD(qry_row, q_gm);

            FdGmShape  vs(1, 1, 1, 1, v);
            FdGmStride vst(1, 1, 1, v, 1);
            FdGmF v_gm(val + wi * v, vs, vst);
            TLOAD(val_row, v_gm);

            FdGmShape  bs(1, 1, 1, 1, static_cast<int64_t>(beta_cnt));
            FdGmStride bst(1, 1, 1, static_cast<int64_t>(beta_cnt), 1);
            FdGmF b_gm(beta + beta_base, bs, bst);
            TLOAD(beta_tile, b_gm);
        }
        pipe_barrier(PIPE_ALL);

        const float beta_val = beta_tile.GetValue(beta_idx);

        // ------------------------------------------------------------------
        // 2. S_dec = S_in * exp(g)[k], IN PLACE in the resident state tile.
        // ------------------------------------------------------------------
        TEXP(gate_row, gate_row);
        pipe_barrier(PIPE_V);

        TROWEXPANDMUL(state_tile, state_tile, gate_col);
        pipe_barrier(PIPE_V);

        // ------------------------------------------------------------------
        // 3. kS[v] = sum_k key[k] * S_dec[k,v]   -- stays on chip.
        // ------------------------------------------------------------------
        TROWEXPANDMUL(scrat_tile, state_tile, key_col);
        pipe_barrier(PIPE_V);

        TCOLSUM(ks_row, scrat_tile, tmp_tile, true);
        pipe_barrier(PIPE_V);   // OPT: Vec->Vec (was PIPE_ALL)

        // ------------------------------------------------------------------
        // 4. delta = val - kS, with the per-item beta folded into the cheap
        //    [1,V] row rather than the [K,V] tile.  Stays on chip.
        // ------------------------------------------------------------------
        TMULS(val_row, val_row, 1.0f);   // C27: push the loaded row to the pipe
        pipe_barrier(PIPE_V);   // OPT: Vec->Vec (was PIPE_ALL)

        TSUB(delta_row, val_row, ks_row);
        pipe_barrier(PIPE_V);   // OPT: Vec->Vec (was PIPE_ALL)

        TMULS(delta_row, delta_row, beta_val);
        pipe_barrier(PIPE_V);   // OPT: Vec->Vec (was PIPE_ALL)

        // ------------------------------------------------------------------
        // 5. S_out = S_dec + (beta*key)[k] outer delta[v], IN PLACE.
        // ------------------------------------------------------------------
        TCOLEXPAND(scrat_tile, delta_row);     // scrat[k,v] = beta*delta[v]
        pipe_barrier(PIPE_V);   // OPT: Vec->Vec (was PIPE_ALL)

        TROWEXPANDMUL(scrat_tile, scrat_tile, key_col);   // *= key[k]
        pipe_barrier(PIPE_V);   // OPT: Vec->Vec (was PIPE_ALL)

        TADD(state_tile, state_tile, scrat_tile);
        pipe_barrier(PIPE_V);

        // ------------------------------------------------------------------
        // 6. ONE bulk store of S_out.
        // ------------------------------------------------------------------
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        {
            FdGmShape  ss(1, 1, 1, k, v);
            FdGmStride sst(1, 1, 1, v, 1);
            FdGmF so_gm(s_out + wi * tile_elems, ss, sst);
            TSTORE(so_gm, state_tile);
        }
        // OPT: no PIPE_ALL here. The S_out store (MTE3) only READS state_tile;
        // the query readout below also only READS state_tile (writing scrat/o),
        // so they overlap safely. MTE3 completion before the next iteration's
        // reload of state_tile is guaranteed by the top-of-loop PIPE_ALL.

        // ------------------------------------------------------------------
        // 7. o[v] = sum_k (scale*q[k]) * S_out[k,v], reading the resident
        //    S_out tile -- no reload from GM.
        // ------------------------------------------------------------------
        TMULS(qry_row, qry_row, scale);
        pipe_barrier(PIPE_V);

        TROWEXPANDMUL(scrat_tile, state_tile, qry_col);
        pipe_barrier(PIPE_V);

        TCOLSUM(o_row, scrat_tile, tmp_tile, true);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        {
            FdGmShape  os(1, 1, 1, 1, v);
            FdGmStride ost(1, 1, 1, v, 1);
            FdGmF o_gm(o + wi * v, os, ost);
            TSTORE(o_gm, o_row);
        }
        pipe_barrier(PIPE_ALL);
    }
#else
    (void)s_in;
    (void)q;
    (void)key;
    (void)val;
    (void)gate;
    (void)beta;
    (void)s_out;
    (void)o;
    (void)total_work;
    (void)k;
    (void)v;
#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__ || __CPU_SIM

extern "C" __global__ AICORE void launch_fused_kda_decode(
    __gm__ uint8_t *s_in, __gm__ uint8_t *q, __gm__ uint8_t *key,
    __gm__ uint8_t *val, __gm__ uint8_t *gate, __gm__ uint8_t *beta,
    __gm__ uint8_t *s_out, __gm__ uint8_t *o, int64_t total_work, int64_t k,
    int64_t v, uint64_t ffts_addr)
{
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
    fused_kda_decode_kernel(reinterpret_cast<__gm__ float *>(s_in),
                            reinterpret_cast<__gm__ float *>(q),
                            reinterpret_cast<__gm__ float *>(key),
                            reinterpret_cast<__gm__ float *>(val),
                            reinterpret_cast<__gm__ float *>(gate),
                            reinterpret_cast<__gm__ float *>(beta),
                            reinterpret_cast<__gm__ float *>(s_out),
                            reinterpret_cast<__gm__ float *>(o),
                            total_work, k, v, ffts_addr);
#else
    (void)s_in;
    (void)q;
    (void)key;
    (void)val;
    (void)gate;
    (void)beta;
    (void)s_out;
    (void)o;
    (void)total_work;
    (void)k;
    (void)v;
    (void)ffts_addr;
#endif
}

extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *s_in,
                            uint8_t *q, uint8_t *key, uint8_t *val,
                            uint8_t *gate, uint8_t *beta, uint8_t *s_out,
                            uint8_t *o, int64_t total_work, int64_t k,
                            int64_t v)
{
    uint32_t ffts_len = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    uint32_t blocks = (block_dim > 0) ? block_dim : 1;
    launch_fused_kda_decode<<<blocks, nullptr, stream>>>(
        s_in, q, key, val, gate, beta, s_out, o, total_work, k, v, ffts_addr);
}
