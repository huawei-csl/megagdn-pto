// Vector-only recurrent KDA/KDN decode kernel.
// The launcher accepts model bf16 and converts it once to the fp16 wire format
// required by C220 vector TCVT. State is fp32 in [slot, head, V, K] layout.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>

using namespace pto;

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef KDN_V
#define KDN_V GDN_D
#endif
#ifndef KDN_BV
#define KDN_BV 32
#endif

#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                       RV, CV, pto::SLayout::NoneBox, 512>;
#endif

template <int KDim, int VDim, int VTile>
AICORE void kdn_decode_kernel(
    __gm__ half *q_ptr, __gm__ half *k_ptr,
    __gm__ half *v_ptr, __gm__ half *g_ptr,
    __gm__ half *beta_ptr, __gm__ float *state_ptr,
    __gm__ half *out_ptr, __gm__ int32_t *state_indices,
    int64_t batch_size, int64_t seq_len, int32_t num_heads,
    int32_t num_state_slots, float scale, uint64_t ffts_addr) {
  const int32_t cid = get_block_idx();
  const int32_t block_num = get_block_num();
  const int32_t vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
  static_assert(KDim % 8 == 0, "KDim must be a multiple of 8");
  static_assert(VTile % 8 == 0, "VTile must be a multiple of 8");
  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr int NumVTiles = (VDim + VTile - 1) / VTile;
  const int worker = cid * 2 + vid;
  const int workers = block_num * 2;
  const int64_t total = batch_size * static_cast<int64_t>(num_heads) * NumVTiles;

  // [state, work, reduction scratch, q, k, g, fp16 staging, v, rows, output]
  constexpr int StateAddr = 0;
  constexpr int WorkAddr = StateAddr + VTile * KDim * 4;
  constexpr int TmpAddr = WorkAddr + VTile * KDim * 4;
  constexpr int QAddr = TmpAddr + VTile * KDim * 4;
  constexpr int KAddr = QAddr + KDim * 4;
  constexpr int GAddr = KAddr + KDim * 4;
  constexpr int QBfAddr = GAddr + KDim * 4;
  constexpr int KBfAddr = QBfAddr + KDim * 2;
  constexpr int GBfAddr = KBfAddr + KDim * 2;
  constexpr int VBfAddr = GBfAddr + KDim * 2;
  constexpr int VAddr = VBfAddr + VTile * 2;
  constexpr int RowAddr = VAddr + VTile * 4;
  constexpr int OutBfAddr = RowAddr + VTile * 4;

  using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using KStride = Stride<1, 1, 1, KDim, 1>;
  using VStride = Stride<1, 1, 1, VDim, 1>;
  using BfK = GlobalTensor<half, DynShape, KStride>;
  using BfV = GlobalTensor<half, DynShape, VStride>;
  using F32K = GlobalTensor<float, DynShape, KStride>;

  for (int64_t work_id = worker; work_id < total; work_id += workers) {
    const int vt = static_cast<int>(work_id % NumVTiles);
    const int64_t bh = work_id / NumVTiles;
    const int head = static_cast<int>(bh % num_heads);
    const int64_t batch = bh / num_heads;
    const int v0 = vt * VTile;
    const int rows = (v0 + VTile <= VDim) ? VTile : (VDim - v0);
    const int slot = state_indices == nullptr ? static_cast<int>(batch)
                                               : state_indices[batch];
    if (slot < 0 || slot >= num_state_slots) continue;

    const int64_t state_off =
        ((static_cast<int64_t>(slot) * num_heads + head) * VDim + v0) * KDim;
    DynShape state_shape;
    state_shape.shape[3] = rows;
    state_shape.shape[4] = KDim;
    F32K state_gm(state_ptr + state_off, state_shape);
    UbND<float, VTile, KDim, DYNAMIC, DYNAMIC> state(rows, KDim);
    TASSIGN(state, StateAddr);
    TLOAD(state, state_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int64_t t = 0; t < seq_len; ++t) {
      const int64_t token_head = (batch * seq_len + t) * num_heads + head;
      const int64_t k_off = token_head * KDim;
      const int64_t v_off = token_head * VDim + v0;
      DynShape ks;
      ks.shape[3] = 1;
      ks.shape[4] = KDim;
      BfK q_gm(q_ptr + k_off, ks), k_gm(k_ptr + k_off, ks), g_gm(g_ptr + k_off, ks);
      UbND<half, 1, KDim> q_bf, k_bf, g_bf;
      TASSIGN(q_bf, QBfAddr); TASSIGN(k_bf, KBfAddr); TASSIGN(g_bf, GBfAddr);
      TLOAD(q_bf, q_gm); TLOAD(k_bf, k_gm); TLOAD(g_bf, g_gm);
      DynShape vs;
      vs.shape[3] = 1; vs.shape[4] = rows;
      BfV v_gm(v_ptr + v_off, vs);
      UbND<half, 1, VTile, DYNAMIC, DYNAMIC> v_bf(1, rows);
      TASSIGN(v_bf, VBfAddr); TLOAD(v_bf, v_gm);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      UbND<float, 1, KDim> q, k, decay;
      TASSIGN(q, QAddr); TASSIGN(k, KAddr); TASSIGN(decay, GAddr);
      TCVT(q, q_bf, pto::RoundMode::CAST_NONE);
      TCVT(k, k_bf, pto::RoundMode::CAST_NONE);
      TCVT(decay, g_bf, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMULS(q, q, scale); TEXP(decay, decay); pipe_barrier(PIPE_V);

      UbND<float, 1, VTile, DYNAMIC, DYNAMIC> v_row(1, rows);
      TASSIGN(v_row, VAddr); TCVT(v_row, v_bf, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      UbDN<float, VTile, 1, DYNAMIC, DYNAMIC> delta(rows, 1);
      TRESHAPE(delta, v_row);
      UbND<float, 1, VTile, DYNAMIC, DYNAMIC> delta_flat(1, rows);
      UbND<float, 1, VTile, DYNAMIC, DYNAMIC> row_flat(1, rows);
      TRESHAPE(delta_flat, delta);
      UbND<float, VTile, KDim, DYNAMIC, DYNAMIC> work(rows, KDim), tmp(rows, KDim);
      TASSIGN(work, WorkAddr); TASSIGN(tmp, TmpAddr);
      UbDN<float, VTile, 1, DYNAMIC, DYNAMIC> row(rows, 1);
      TASSIGN(row, RowAddr);
      TRESHAPE(row_flat, row);

      TCOLEXPANDMUL(state, state, decay); pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(work, state, k); pipe_barrier(PIPE_V);
      TROWSUM(row, work, tmp); pipe_barrier(PIPE_V);
      TSUB(delta_flat, delta_flat, row_flat); pipe_barrier(PIPE_V);
      TMULS(delta_flat, delta_flat, static_cast<float>(beta_ptr[token_head]));
      pipe_barrier(PIPE_V);
      TCOLEXPAND(work, k); pipe_barrier(PIPE_V);
      TROWEXPANDMUL(work, work, delta); pipe_barrier(PIPE_V);
      TADD(state, state, work); pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(work, state, q); pipe_barrier(PIPE_V);
      TROWSUM(row, work, tmp); pipe_barrier(PIPE_V);

      UbND<half, 1, VTile, DYNAMIC, DYNAMIC> out_flat(1, rows);
      TASSIGN(out_flat, OutBfAddr);
      TCVT(out_flat, row_flat, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      UbND<half, 1, VTile, DYNAMIC, DYNAMIC> out_row(1, rows);
      TRESHAPE(out_row, out_flat);
      BfV out_gm(out_ptr + v_off, vs);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(out_gm, out_row);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(state_gm, state);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  }
#endif
}

extern "C" __global__ AICORE void launch_kdn_decode(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v,
    __gm__ uint8_t *g, __gm__ uint8_t *beta, __gm__ uint8_t *state,
    __gm__ uint8_t *out, __gm__ uint8_t *indices, int64_t batch, int64_t seq,
    int32_t heads, int32_t slots, float scale, uint64_t ffts) {
  kdn_decode_kernel<GDN_D, KDN_V, KDN_BV>(
      reinterpret_cast<__gm__ half *>(q), reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(v), reinterpret_cast<__gm__ half *>(g),
      reinterpret_cast<__gm__ half *>(beta), reinterpret_cast<__gm__ float *>(state),
      reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ int32_t *>(indices),
      batch, seq, heads, slots, scale, ffts);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream, uint8_t *q, uint8_t *k, uint8_t *v,
    uint8_t *g, uint8_t *beta, uint8_t *state, uint8_t *out, uint8_t *indices,
    int64_t batch, int64_t seq, int32_t heads, int32_t slots, float scale) {
  uint32_t len{0}; uint64_t addr{0};
  rtGetC2cCtrlAddr(&addr, &len);
  launch_kdn_decode<<<block_dim, nullptr, stream>>>(q, k, v, g, beta, state,
                                                    out, indices, batch, seq,
                                                    heads, slots, scale, addr);
}
