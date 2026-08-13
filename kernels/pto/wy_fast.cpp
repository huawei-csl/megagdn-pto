// ============================================================================
// wy_fast_kernel.cpp — WY representation for GatedDeltaNet chunk recurrence
//
// Computes the WY update matrices U and W for each chunk of C tokens:
//   U = A2 @ V     where A2 = A * beta_2d        (beta-scaled attention)
//   W = A1 @ K     where A1 = A * (exp(g)*beta)_2d (gate+beta-scaled attention)
//
// beta is the decay factor, g is the gate value, A is the triangular attention
// matrix (from the kkt kernel).  The column-broadcast notation x_2d means
// expanding a 1xC vector into a C/2 x C matrix by replicating across rows.
//
// Architecture: Vec+Cube cooperative kernel using cross-core synchronization.
//
//  Vec core (two sub-blocks for upper/lower C/2 rows):
//    For each chunk:
//      1. Load beta [H,T] and A [B,S,H,C], compute A2 = A * beta_2d -> ws
//      2. Load G [H,T], compute A1 = A * (exp(g)*beta)_2d -> ws
//      3. Signal Cube via cross-core flags when workspaces are ready
//
//  Cube core (waits for Vec signals):
//    For each chunk:
//      1. Load K, V from BSND layout into L1
//      2. Load A2 from workspace -> GEMM: U = A2 @ V
//      3. Load A1 from workspace -> GEMM: W = A1 @ K
//      4. Store U, W back to BSND layout
//
// NPU memory hierarchy used:
//   GM -> UB (Vec), GM -> L1 -> L0A/L0B -> L0C -> GM (Cube)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel uses BOTH the Cube engine (matrix multiply) and Vec engine
// (SIMD element-wise ops), running on SEPARATE physical cores that
// communicate via Global Memory (GM) + cross-core flags (FFTS).
//
// Execution flow:
//   Vec core:  load A,beta,G → compute A2,A1 → store to GM workspace
//   Cube core: wait for workspace → load A2/A1 + K/V → GEMM → store U,W
//
// Key PTO APIs (with numpy/torch equivalents):
//   TLOAD(ub_tile, gm)      — ub_tile = gm[...]          (DMA: GM→UB, async
//   MTE2) TSTORE(gm, ub_tile)     — gm[...] = ub_tile          (DMA: UB→GM,
//   async MTE3) TCVT(dst, src, mode)    — dst = src.float() or .half() (type
//   conversion) TMOV(dst, src)          — dst = src.clone() TMUL(d, a, b) — d =
//   a * b                   (element-wise) TEXP(d, s)              — d =
//   torch.exp(s) TCOLEXPAND(2d, row)     — 2d[i,j] = row[j]  (broadcast row
//   across all rows) TEXTRACT(l0, l1, r, c)  — L1 sub-block → L0A/L0B     (MTE1
//   for Cube GEMM) TMATMUL(C, A, B)        — C = A @ B in Cube engine
//   (fp16→fp32 accumulate) set_flag / wait_flag    — sync between pipes on SAME
//   core ffts_cross_core_sync    — signal ACROSS Cube↔Vec cores
//   wait_flag_dev(flag)     — wait for cross-core signal
// ============================================================================

#include "wy_fast.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

extern "C" __global__ AICORE void launch_wy_fast(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle,
    __gm__ uint8_t *A_handle, __gm__ uint8_t *workspace_a1_handle,
    __gm__ uint8_t *workspace_a2_handle, __gm__ uint8_t *W_handle,
    __gm__ uint8_t *U_handle, __gm__ uint8_t *cu_seqlens, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, uint32_t num_heads,
    uint32_t num_key_heads, uint64_t ffts_addr) {
  wy_fast_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ half *>(workspace_a1_handle),
      reinterpret_cast<__gm__ half *>(workspace_a2_handle),
      reinterpret_cast<__gm__ half *>(W_handle),
      reinterpret_cast<__gm__ half *>(U_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens), batch_size, seq_len,
      total_tokens, num_heads, num_key_heads, ffts_addr);
}

extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *k,
                            uint8_t *v, uint8_t *beta, uint8_t *g_sum,
                            uint8_t *A, uint8_t *workspace_a1,
                            uint8_t *workspace_a2, uint8_t *w, uint8_t *u,
                            uint8_t *cu_seqlens, int64_t batch_size,
                            int64_t seq_len, int64_t total_tokens,
                            uint32_t num_heads, uint32_t num_key_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_wy_fast<<<block_dim, nullptr, stream>>>(
      k, v, beta, g_sum, A, workspace_a1, workspace_a2, w, u, cu_seqlens,
      batch_size, seq_len, total_tokens, num_heads, num_key_heads, fftsAddr);
}
