// Minimal state-traffic ceiling for kdn_decode.
// Each worker copies every assigned [VTile, KDim] fp32 state tile from GM to
// one UB tile and back to the same GM address.  There is deliberately no
// compute, ping-pong state, or address helper in this kernel.

#include <runtime/rt_ffts.h>

#include <pto/pto-inst.hpp>

#include "acl/acl.h"

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
template <typename T, int R, int C>
using CopyTile =
    pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor, R, C,
              pto::SLayout::NoneBox, 512, pto::PadValue::Null>;
#endif

template <int KDim, int VDim, int VTile>
AICORE void kdn_memcpy_bound_kernel(__gm__ float *state_ptr, int64_t batch,
                                    int32_t heads, uint64_t ffts_addr) {
  const int32_t cid = get_block_idx();
  const int32_t block_num = get_block_num();
  const int32_t vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_VEC__)
  static_assert(KDim % 8 == 0, "KDim must be a multiple of 8");
  static_assert(VDim % VTile == 0, "VDim must be divisible by VTile");
  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr int NumVTiles = VDim / VTile;
  constexpr int UbAddr = 0;
  using ShapeT = Shape<1, 1, 1, VTile, KDim>;
  using StrideT = Stride<1, 1, 1, KDim, 1>;
  using GlobalT = GlobalTensor<float, ShapeT, StrideT>;

  const int worker = cid * 2 + vid;
  const int workers = block_num * 2;
  const int64_t total = batch * static_cast<int64_t>(heads) * NumVTiles;
  CopyTile<float, VTile, KDim> ub;
  TASSIGN(ub, UbAddr);

  for (int64_t id = worker; id < total; id += workers) {
    const int vt = static_cast<int>(id % NumVTiles);
    const int64_t bh = id / NumVTiles;
    const int v0 = vt * VTile;
    const int64_t offset = (bh * VDim + v0) * KDim;
    ShapeT shape;
    StrideT stride;
    GlobalT gm(state_ptr + offset, shape, stride);

    TLOAD(ub, gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gm, ub);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
#endif
}

extern "C" __global__ AICORE void launch_kdn_memcpy_bound(__gm__ uint8_t *state,
                                                          int64_t batch,
                                                          int32_t heads,
                                                          uint64_t ffts) {
  kdn_memcpy_bound_kernel<GDN_D, KDN_V, KDN_BV>(
      reinterpret_cast<__gm__ float *>(state), batch, heads, ffts);
}

extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *state,
                            int64_t batch, int32_t heads) {
  uint32_t len{0};
  uint64_t addr{0};
  rtGetC2cCtrlAddr(&addr, &len);
  launch_kdn_memcpy_bound<<<block_dim, nullptr, stream>>>(state, batch, heads,
                                                          addr);
}
