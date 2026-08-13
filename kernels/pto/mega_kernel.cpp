// mega_kernel.cpp — GDN Mega-Kernel (group-value / GQA): all PTO stages in one
// launch
//
// Same pipeline as pto_mega_kernel, but scaled_dot_kkt / wy_fast / chunk_h /
// chunk_o use runtime H/Hg dispatch from dynamic_bsnd_groupvalue; cumsum still
// uses H (value heads) like dynamic_bsnd.
//
// Stages:
//   1. cumsum      (Vec)
//   2. transpose   (Vec)
//   3. kkt         (Cube+Vec)  — K has Hg heads; β,g,A use H value heads
//   4. solve_tril  (Cube)
//   5. wy_fast     (Vec+Cube)
//   6. chunk_h     (Cube+Vec)
//   7. chunk_o     (Cube+Vec)

#include "mega_kernel.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

extern "C" __global__ AICORE void launch_mega_kernel(
    __gm__ uint8_t* q_ptr, __gm__ uint8_t* k_ptr, __gm__ uint8_t* v_ptr,
    __gm__ uint8_t* g_in_ptr, __gm__ uint8_t* beta_ptr,
    __gm__ uint8_t* msk_lower_ptr, __gm__ uint8_t* msk_full_ptr,
    __gm__ uint8_t* minus_id_ptr, __gm__ uint8_t* cu_seqlens_ptr,
    __gm__ uint8_t* o_ptr, __gm__ uint8_t* g_sum_ptr, __gm__ uint8_t* g_t_ptr,
    __gm__ uint8_t* beta_t_ptr, __gm__ uint8_t* A_ptr,
    __gm__ uint8_t* A_inv_f32_ptr, __gm__ uint8_t* A_inv_ptr,
    __gm__ uint8_t* w_ptr, __gm__ uint8_t* u_ptr, __gm__ uint8_t* s_ptr,
    __gm__ uint8_t* v_new_ptr, __gm__ uint8_t* fs_ptr, __gm__ uint8_t* h0_ptr,
    int64_t has_initial_state, __gm__ uint8_t* kkt_ws_ptr,
    __gm__ uint8_t* wy_ws_a1_ptr, __gm__ uint8_t* wy_ws_a2_ptr,
    __gm__ uint8_t* h_ws_ptr, __gm__ uint8_t* o_ws_qk_ptr,
    __gm__ uint8_t* o_ws_qs_ptr, __gm__ uint8_t* o_ws_gated_ptr,
    uint32_t num_heads, uint32_t num_key_heads, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, uint32_t num_matrices,
    uint64_t ffts_addr) {
  // num_heads is a runtime kernel argument (one .so serves every head count).
  // Guard the compile-time UB ceiling; the host validates before launch too.
  if (num_heads == 0 || num_heads > GDN_MAX_HEADS) {
    return;
  }
  mega_kernel_impl(q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr,
                   msk_full_ptr, minus_id_ptr, cu_seqlens_ptr, o_ptr, g_sum_ptr,
                   g_t_ptr, beta_t_ptr, A_ptr, A_inv_f32_ptr, A_inv_ptr, w_ptr,
                   u_ptr, s_ptr, v_new_ptr, fs_ptr, h0_ptr, has_initial_state,
                   kkt_ws_ptr, wy_ws_a1_ptr, wy_ws_a2_ptr, h_ws_ptr,
                   o_ws_qk_ptr, o_ws_qs_ptr, o_ws_gated_ptr,
                   static_cast<int32_t>(num_heads), num_key_heads, batch_size,
                   seq_len, total_tokens, num_matrices, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void* stream, uint8_t* q, uint8_t* k, uint8_t* v,
    uint8_t* g_in, uint8_t* beta, uint8_t* msk_lower, uint8_t* msk_full,
    uint8_t* minus_id, uint8_t* cu_seqlens, uint8_t* o, uint8_t* g_sum,
    uint8_t* g_t, uint8_t* beta_t, uint8_t* A, uint8_t* A_inv_f32,
    uint8_t* A_inv, uint8_t* w, uint8_t* u, uint8_t* s, uint8_t* v_new,
    uint8_t* fs, uint8_t* h0, int64_t has_initial_state, uint8_t* kkt_ws,
    uint8_t* wy_ws_a1, uint8_t* wy_ws_a2, uint8_t* h_ws, uint8_t* o_ws_qk,
    uint8_t* o_ws_qs, uint8_t* o_ws_gated, uint32_t num_heads,
    uint32_t num_key_heads, int64_t batch_size, int64_t seq_len,
    int64_t total_tokens, uint32_t num_matrices) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_mega_kernel<<<block_dim, nullptr, stream>>>(
      q, k, v, g_in, beta, msk_lower, msk_full, minus_id, cu_seqlens, o, g_sum,
      g_t, beta_t, A, A_inv_f32, A_inv, w, u, s, v_new, fs, h0,
      has_initial_state, kkt_ws, wy_ws_a1, wy_ws_a2, h_ws, o_ws_qk, o_ws_qs,
      o_ws_gated, num_heads, num_key_heads, batch_size, seq_len, total_tokens,
      num_matrices, fftsAddr);
}
