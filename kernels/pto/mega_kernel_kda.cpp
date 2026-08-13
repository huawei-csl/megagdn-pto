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

#include "mega_kernel_kda.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

// ===================================================================
// Fused launch
// ===================================================================
extern "C" __global__ AICORE void launch_mega_kernel_kda(
    // ── inputs ──────────────────────────────────────────────────────
    __gm__ uint8_t* q_hm_ptr,     // [1, HV, T, K] fp16 (head-major, scaled)
    __gm__ uint8_t* k_hm_ptr,     // [1, HV, T, K] fp16 (head-major)
    __gm__ uint8_t* v_ptr,        // [1, T, HV, V] fp16 (BSND)
    __gm__ uint8_t* g_in_ptr,     // [1, T, HV, K] fp16 (BSND, raw gate)
    __gm__ uint8_t* beta_hm_ptr,  // [1, HV, T]    fp16 (head-major)
    // ── masks / constants ───────────────────────────────────────────
    __gm__ uint8_t* mask_strict_ptr,  // [C, C] fp32 (rows >  cols)
    __gm__ uint8_t* mask_incl_ptr,    // [C, C] fp32 (rows >= cols)
    __gm__ uint8_t* minus_id_ptr,     // [C, C] fp16 (-I)
    __gm__ uint8_t* cu_seqlens_ptr,   // int32
    // ── output ──────────────────────────────────────────────────────
    __gm__ uint8_t* o_ptr,  // [1, T, HV, V] fp16 (BSND)
    // ── intermediate buffers ────────────────────────────────────────
    __gm__ uint8_t* g_sum_ptr,    // [1, T, HV, K] fp32 (BSND)
    __gm__ uint8_t* g_cs_hm_ptr,  // [1, HV, T, K] fp32 (head-major)
    __gm__ uint8_t* L_ptr,        // [1, T, HV, C] fp16
    __gm__ uint8_t* A_inv_ptr,    // [1, T, HV, C] fp16
    __gm__ uint8_t* u_ptr,        // [1, T, HV, V] fp16
    __gm__ uint8_t* w_ptr,        // [1, T, HV, K] fp16
    __gm__ uint8_t* s_ptr,        // [tc, HV, K, V] fp16
    __gm__ uint8_t* v_corr_ptr,   // [1, T, HV, V] fp16
    // ── per-core workspaces ─────────────────────────────────────────
    __gm__ uint8_t* kkt_ws_in_ptr,  // [bd*2, 2C, K] fp32 (stages exp(±g_cs))
    __gm__ uint8_t*
        kkt_ws_out_ptr,            // [bd*2, C, C]  fp32 (unmasked gated K·K^T)
    __gm__ uint8_t* wy_ws_a2_ptr,  // [bd, C, C]    fp16
    __gm__ uint8_t* wy_ws_keff_ptr,  // [bd, C, K]    fp16
    __gm__ uint8_t* h_ws_ptr,        // [bd*5, K, K]  fp16
    __gm__ uint8_t* o_ws_ptr,  // [bd*7, K, K]  fp32 (gated q/k + GEMM I/O)
    // ── scalars ─────────────────────────────────────────────────────
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint32_t num_matrices, int32_t num_heads, uint64_t ffts_addr) {
  set_ffts_base_addr(ffts_addr);

  // Head count (HV) is a runtime argument; only GDN_D (K) and GDN_C (C) stay
  // compile-time, so the fused .so is head-count-agnostic like the staged ones.
  const int32_t HV = num_heads;
  constexpr int32_t KD = GDN_D;
  constexpr int32_t C = GDN_C;

  // ── Stage 1: gate_cumsum (BSND -> BSND) ──────────────────────────
  mk_gc::gate_cumsum_kda_kernel<KD, C>(
      reinterpret_cast<__gm__ half*>(g_in_ptr),
      reinterpret_cast<__gm__ float*>(g_sum_ptr),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), batch_size, seq_len,
      HV, ffts_addr);

#ifdef MEGA_STOP_AFTER_CUMSUM
  pipe_barrier(PIPE_ALL);
  return;
#endif

  SyncAllMegaKernel<false>();

  // ── Stage 2: transpose g_sum -> head-major g_cs ──────────────────
  mega_permute_THK_to_HTK<float, KD>(
      reinterpret_cast<__gm__ float*>(g_sum_ptr),
      reinterpret_cast<__gm__ float*>(g_cs_hm_ptr), total_tokens, HV);

#ifdef MEGA_STOP_AFTER_TRANSPOSE
  pipe_barrier(PIPE_ALL);
  return;
#endif

  SyncAllMegaKernel<false>();

  // ── Stage 3: kkt (gated K·K^T lower-tri matrix) ──────────────────
  mk_kkt::kkt_kda_kernel<KD, C>(
      reinterpret_cast<__gm__ half*>(k_hm_ptr),
      reinterpret_cast<__gm__ float*>(g_cs_hm_ptr),
      reinterpret_cast<__gm__ half*>(beta_hm_ptr),
      reinterpret_cast<__gm__ float*>(mask_strict_ptr),
      reinterpret_cast<__gm__ float*>(kkt_ws_in_ptr),
      reinterpret_cast<__gm__ float*>(kkt_ws_out_ptr),
      reinterpret_cast<__gm__ half*>(L_ptr),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), batch_size, seq_len,
      total_tokens, HV, ffts_addr);

#ifdef MEGA_STOP_AFTER_KKT
  pipe_barrier(PIPE_ALL);
  return;
#endif

  SyncAllMegaKernel<false>();

  // ── Stage 4: solve_tril ((I + L)^{-1}) ───────────────────────────
  mega_solve_tril(reinterpret_cast<__gm__ half*>(A_inv_ptr),
                  reinterpret_cast<__gm__ half*>(L_ptr),
                  reinterpret_cast<__gm__ half*>(minus_id_ptr), C, num_matrices,
                  HV, reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), 1);

#ifdef MEGA_STOP_AFTER_SOLVE
  pipe_barrier(PIPE_ALL);
  return;
#endif

  SyncAllMegaKernel<false>();

  // ── Stage 5: wy (auxiliaries u, w) ───────────────────────────────
  mk_wy::wy_kda_kernel<KD, C>(reinterpret_cast<__gm__ half*>(k_hm_ptr),
                              reinterpret_cast<__gm__ half*>(v_ptr),
                              reinterpret_cast<__gm__ half*>(beta_hm_ptr),
                              reinterpret_cast<__gm__ float*>(g_cs_hm_ptr),
                              reinterpret_cast<__gm__ half*>(A_inv_ptr),
                              reinterpret_cast<__gm__ half*>(wy_ws_a2_ptr),
                              reinterpret_cast<__gm__ half*>(wy_ws_keff_ptr),
                              reinterpret_cast<__gm__ half*>(u_ptr),
                              reinterpret_cast<__gm__ half*>(w_ptr),
                              reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr),
                              batch_size, seq_len, total_tokens, HV, ffts_addr);

#ifdef MEGA_STOP_AFTER_WY
  pipe_barrier(PIPE_ALL);
  return;
#endif

  SyncAllMegaKernel<false>();

  // ── Stage 6: chunk_h (state snapshots + v_corr) ──────────────────
  mk_h::chunk_h_kda_kernel<KD, C>(
      reinterpret_cast<__gm__ half*>(k_hm_ptr),
      reinterpret_cast<__gm__ half*>(w_ptr),
      reinterpret_cast<__gm__ half*>(u_ptr),
      reinterpret_cast<__gm__ float*>(g_cs_hm_ptr),
      reinterpret_cast<__gm__ half*>(s_ptr),
      reinterpret_cast<__gm__ half*>(v_corr_ptr),
      reinterpret_cast<__gm__ half*>(h_ws_ptr),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), batch_size, seq_len,
      total_tokens, HV, ffts_addr);

#ifdef MEGA_STOP_AFTER_H
  pipe_barrier(PIPE_ALL);
  return;
#endif

  SyncAllMegaKernel<false>();

  // ── Stage 7: chunk_o (output) ────────────────────────────────────
  mk_o::chunk_o_kda_kernel<KD, C>(
      reinterpret_cast<__gm__ half*>(q_hm_ptr),
      reinterpret_cast<__gm__ half*>(k_hm_ptr),
      reinterpret_cast<__gm__ half*>(v_corr_ptr),
      reinterpret_cast<__gm__ half*>(s_ptr),
      reinterpret_cast<__gm__ float*>(g_cs_hm_ptr),
      reinterpret_cast<__gm__ float*>(mask_incl_ptr),
      reinterpret_cast<__gm__ float*>(o_ws_ptr),
      reinterpret_cast<__gm__ half*>(o_ptr),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), batch_size, seq_len,
      total_tokens, HV, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void* stream, uint8_t* q_hm, uint8_t* k_hm, uint8_t* v,
    uint8_t* g_in, uint8_t* beta_hm, uint8_t* mask_strict, uint8_t* mask_incl,
    uint8_t* minus_id, uint8_t* cu_seqlens, uint8_t* o, uint8_t* g_sum,
    uint8_t* g_cs_hm, uint8_t* L, uint8_t* A_inv, uint8_t* u, uint8_t* w,
    uint8_t* s, uint8_t* v_corr, uint8_t* kkt_ws_in, uint8_t* kkt_ws_out,
    uint8_t* wy_ws_a2, uint8_t* wy_ws_keff, uint8_t* h_ws, uint8_t* o_ws,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint32_t num_matrices, uint32_t num_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_mega_kernel_kda<<<block_dim, nullptr, stream>>>(
      q_hm, k_hm, v, g_in, beta_hm, mask_strict, mask_incl, minus_id,
      cu_seqlens, o, g_sum, g_cs_hm, L, A_inv, u, w, s, v_corr, kkt_ws_in,
      kkt_ws_out, wy_ws_a2, wy_ws_keff, h_ws, o_ws, batch_size, seq_len,
      total_tokens, num_matrices, static_cast<int32_t>(num_heads), fftsAddr);
}
