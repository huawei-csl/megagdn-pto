// ============================================================================
// wy_kda.cpp — WY representation for KDA chunk recurrence (per-dim gate)
//
// Computes the two auxiliary tensors U and W per chunk:
//   U[r, d_v] = sum_c INV[r, c] * beta[c] * v[c, d_v]
//   W[r, d_k] = sum_c INV[r, c] * beta[c] * exp(g_cs[c, d_k]) * k[c, d_k]
//
// where INV = (I + L)^{-1} is full lower-triangular, beta is post-sigmoid
// scalar per token, and g_cs is per-DIMENSION cumulative gate sum (the KDA
// difference from GDN). Math reference: tests/test_kda_single_kernels.py
// `ref_wy_kda` (lines 154-200).
//
// Key insight: both U and W reuse a single beta-scaled A2 matrix:
//   A2[r, c]    = INV[r, c] * beta[c]            (column-scale by beta)
//   K_eff[c, d] = k[c, d] * exp(g_cs[c, d])      (element-wise per-dim)
//   U = A2 @ V                                    (V from BSND, fp16)
//   W = A2 @ K_eff                                (K_eff from workspace, fp16)
//
// Cube loads A2 once into L1 and reuses it across both GEMMs — saves one L1
// load and one workspace pass compared to GDN wy_fast.cpp (which needs two
// distinct reweighted-A matrices because GDN's scalar g could be folded into
// a column scale of A). In KDA g is per-dim so we must precompute K_eff
// instead.
//
//  Vec core (both sub-blocks active; each owns one HalfChunk-row stripe):
//    For each (seq, chunk, head):
//      Phase 1: Load beta [1,C], INV [HC,C] BSND -> build A2 = INV * beta_2d
//               -> fp16 -> store stripe to ws_a2 [block_dim, C, C]
//      Phase 2: Load k [HC,K], g_cs [HC,K] head-major -> K_eff = k*exp(g_cs)
//               -> fp16 -> store stripe to ws_keff [block_dim, C, K]
//      Signal Cube via cross-core flags 1 and 2.
//
//  Cube core:
//    For each (seq, chunk, head):
//      Load V from BSND -> v_l1
//      Wait flag 1 (ws_a2 ready) -> TLOAD a2_l1
//      GEMM U = A2 @ V -> u_l0 -> store BSND -> signal flag 3 (ws_a2 free)
//      Wait flag 2 (ws_keff ready) -> TLOAD keff_l1
//      GEMM W = A2 @ K_eff (a2_l1 still in L1!) -> w_l0 -> store BSND
//      Signal flag 4 (ws_keff free)
//
// FFTS flags (single-buffered with explicit free signals):
//   10 : V→C reduce (both vids must signal) "ws_a2 ready"
//   11 : V→C reduce "ws_keff ready"
//   12 : C→V broadcast "ws_a2 free"
//   13 : C→V broadcast "ws_keff free"
// Layout (matches Python kda_kernel_libs convention; v cast to fp16 wrap-side):
//   k       head-major [HV, T, K]    fp32
//   v       BSND       [B, T, HV, V] fp16   (V == K in current setup)
//   beta    head-major [HV, T]       fp32
//   g_cs    head-major [HV, T, K]    fp32
//   A (INV) BSND       [B, T, HV, C] fp32
//   ws_a2              [bd, C, C]    fp16
//   ws_keff            [bd, C, K]    fp16
//   u_out   BSND       [B, T, HV, V] fp32
//   w_out   BSND       [B, T, HV, K] fp32
//
// Compile-time template params: GDN_D = K (= V_DIM here), GDN_C = C.
// Runtime argument: num_heads = HV.
// ============================================================================

#include "wy_kda.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

extern "C" __global__ AICORE void launch_wy_kda(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle,
    __gm__ uint8_t *A_handle, __gm__ uint8_t *workspace_a2_handle,
    __gm__ uint8_t *workspace_keff_handle, __gm__ uint8_t *U_handle,
    __gm__ uint8_t *W_handle, __gm__ uint8_t *cu_seqlens, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, int32_t num_heads,
    uint64_t ffts_addr) {
  wy_kda_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ half *>(workspace_a2_handle),
      reinterpret_cast<__gm__ half *>(workspace_keff_handle),
      reinterpret_cast<__gm__ half *>(U_handle),
      reinterpret_cast<__gm__ half *>(W_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens), batch_size, seq_len,
      total_tokens, num_heads, ffts_addr);
}

extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *k,
                            uint8_t *v, uint8_t *beta, uint8_t *g_cs,
                            uint8_t *A, uint8_t *workspace_a2,
                            uint8_t *workspace_keff, uint8_t *u, uint8_t *w,
                            uint8_t *cu_seqlens, int64_t batch_size,
                            int64_t seq_len, int64_t total_tokens,
                            uint32_t num_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_wy_kda<<<block_dim, nullptr, stream>>>(
      k, v, beta, g_cs, A, workspace_a2, workspace_keff, u, w, cu_seqlens,
      batch_size, seq_len, total_tokens, static_cast<int32_t>(num_heads),
      fftsAddr);
}
