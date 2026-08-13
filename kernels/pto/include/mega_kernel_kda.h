// mega_kernel_kda.cpp — KDA (Kimi Delta Attention) Mega-Kernel: all PTO stages
// in one launch
//
// Fuses the six KDA stages into a single NPU launch, modelled on the GDN
// mega_kernel.cpp. Sub-kernels are reused verbatim via namespaced #include of
// their headers, and their templated device functions are invoked in sequence
// with SyncAllMegaKernel<false>() barriers between stages.
//
// Stages:
//   1. gate_cumsum (Vec)             — within-chunk prefix sum of g  [B,T,HV,K]
//   2. transpose   (Vec)             — g_sum BSND [T,HV,K] -> head-major
//   [HV,T,K]
//   3. kkt         (Cube+Vec)        — gated K·K^T lower-tri matrix L
//   4. solve_tril  (Cube)            — (I + L)^{-1}  (shared tri_inverse
//   kernel)
//   5. wy          (Vec+Cube)        — WY auxiliaries u, w
//   6. chunk_h     (Cube+Vec)        — recurrent state snapshots + v_corr
//   7. chunk_o     (Cube+Vec)        — output
//
// KDA difference from GDN: gates are per-dimension [B,T,HV,K] (not scalar
// [B,T,H]), and the sub-kernels read k/q/g_cs in head-major [HV,T,K] layout
// (beta in [HV,T]).  Static inputs (q, k, beta) are permuted to head-major in
// Python; only g_sum — produced inside the kernel by gate_cumsum — is
// transposed on-device here.

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif

#pragma once

#include <pto/pto-inst.hpp>
#include <type_traits>

#include "kernel_utils.h"

using namespace pto;
using namespace kernel_utils;

// ===================================================================
// Device-only helpers
// ===================================================================
#ifdef __CCE_AICORE__

// Strided GM->GM copy reordering a per-dimension tensor from BSND [T,HV,K] to
// head-major [HV,T,K].  K stays innermost-contiguous; only the (T,HV) axes
// swap, so this is a pure gather/scatter of K-contiguous rows (no element
// transpose).  Layout-only — independent of cu_seqlens.
template <typename T, int32_t KD>
AICORE inline void mega_permute_THK_to_HTK(__gm__ T* src, __gm__ T* dst,
                                           int64_t T_len, int32_t HV) {
  // To avoid ambiguity with bisheng intrinsic header's global `enum class
  // Stride`
  using pto::Stride;

#if defined(__DAV_VEC__)
  if (get_subblockid() != 0) return;
  set_mask_norm();
  set_vector_mask(-1, -1);

  auto cid = get_block_idx();
  auto block_num = get_block_num();

  constexpr int32_t BLOCK = 128;  // tokens per UB tile
  constexpr int32_t UB0 = 0;

  using UBTileDyn =
      Tile<TileType::Vec, T, BLOCK, KD, BLayout::RowMajor, DYNAMIC, DYNAMIC,
           SLayout::NoneBox, 512, PadValue::Zero>;
  using Gm2D = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmSrcS = Stride<1, 1, 1, DYNAMIC,
                        1>;  // row stride = HV*K (runtime; skip other heads)
  using GmDstS = Stride<1, 1, 1, KD, 1>;  // contiguous
  GmSrcS src_stride(HV * KD);

  int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;
  int64_t total = static_cast<int64_t>(HV) * num_tok_blocks;

  for (int64_t wi = static_cast<int64_t>(cid); wi < total;
       wi += static_cast<int64_t>(block_num)) {
    int64_t h = wi / num_tok_blocks;
    int64_t bi = wi % num_tok_blocks;
    int64_t t0 = bi * BLOCK;
    int32_t valid =
        (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);

    {
      Gm2D gs;
      gs.shape[3] = valid;
      gs.shape[4] = KD;
      GlobalTensor<T, Gm2D, GmSrcS> gm(
          src + (t0 * static_cast<int64_t>(HV) + h) * KD, gs, src_stride);
      UBTileDyn ld(valid, KD);
      TASSIGN(ld, UB0);
      TLOAD(ld, gm);
    }
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    {
      Gm2D gs;
      gs.shape[3] = valid;
      gs.shape[4] = KD;
      GlobalTensor<T, Gm2D, GmDstS> gm(dst + (h * T_len + t0) * KD, gs);
      UBTileDyn st(valid, KD);
      TASSIGN(st, UB0);
      TSTORE(gm, st);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
#endif
}

#endif  // __CCE_AICORE__

// ===================================================================
// Include the KDA sub-kernel device implementations in separate namespaces.
// Each header holds only device code (no `call_kernel` host wrapper), so the
// namespaces keep the per-kernel helper aliases apart (matches
// mega_kernel.cpp).
// ===================================================================

namespace mk_gc {
#include "gate_cumsum_kda.h"
}

namespace mk_kkt {
#include "kkt_kda.h"
}

namespace mk_solve {
#include "tri_inverse_impl.h"
}

namespace mk_wy {
#include "wy_kda.h"
}

namespace mk_h {
#include "chunk_h_kda.h"
}

namespace mk_o {
#include "chunk_o_kda.h"
}

// Shared triangular-inverse dispatch (identical to GDN mega_kernel.cpp).
AICORE void mega_solve_tril(__gm__ half* out, __gm__ half* in,
                            __gm__ half* minus_id, uint32_t matrix_size,
                            uint32_t num_matrices, uint32_t num_bsnd_heads,
                            __gm__ int32_t* cu_seqlens, uint32_t is_lower) {
  if (num_matrices <= get_block_num())
    mk_solve::runKernelTriInvRecUnroll<half, half, GDN_C, 1, true>(
        out, in, minus_id, num_matrices, num_bsnd_heads, is_lower, cu_seqlens);
  else if (num_matrices <= 2u * get_block_num())
    mk_solve::runKernelTriInvRecUnroll<half, half, GDN_C, 2, true>(
        out, in, minus_id, num_matrices, num_bsnd_heads, is_lower, cu_seqlens);
  else
    mk_solve::runKernelTriInvRecUnroll<half, half, GDN_C, 4, true>(
        out, in, minus_id, num_matrices, num_bsnd_heads, is_lower, cu_seqlens);
}
