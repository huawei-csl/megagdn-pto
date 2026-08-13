// ============================================================================
// kkt_kda.cpp — Within-chunk gated attention matrix for KDA (numerically
// stable)
//
// Mathematical operation (per chunk of C tokens, per head h):
//   L[r,c] = beta[r] * sum_d k[r,d] * k[c,d] * exp(g_cs[r,d] - g_cs[c,d])
//            for r > c (strictly lower-tri), else 0
//
// STABILITY: Kimi KDA gates (g = -exp(A_log)*softplus(...)) are unbounded; the
//   within-chunk cumulative gate g_cs can reach ~-500.  The previous factorized
//   form  A_eff=k*exp(g_cs), B_eff=k*exp(-g_cs), L=A_eff@B_eff^T  computes
//   exp(-g_cs)=exp(+500) which overflows fp32 (max e^88) -> inf -> inf*0 = NaN.
//
//   This kernel instead computes exp(g_cs[r]-g_cs[c]) as a DIFFERENCE, never
//   the product of two separate exponentials.  For the kept (lower-tri) entries
//   r>c, g_cs[r] <= g_cs[c] (g_cs monotone decreasing within a chunk) so the
//   argument is <= 0 and exp(.) <= 1 — always finite.  We clamp the argument
//   with min(., 0) so the masked (upper-tri) entries also stay finite (then
//   discarded by only storing the strict-lower part).  No pivot, no saturation,
//   exact.
//
// IMPLEMENTATION: per (head, chunk) the work is split across the two Vec
//   sub-blocks by row range (vid=0 -> rows [0,C/2), vid=1 -> rows [C/2, C)),
//   mirroring the GDN scaled_dot_kkt row split.  Each vid loops over columns c
//   and computes its rows' column-c of L via a per-column elementwise
//   reduction:
//
//     diff[r,d] = g_cs[my_row r, d] - g_cs[c, d]   (TCOLEXPANDSUB, per-dim c)
//     diff      = min(diff, 0)                       (TMINS)
//     t[r,d]    = exp(diff) * k[c,d] * k[my_row r,d] (TEXP, TCOLEXPANDMUL,
//     TMUL) L[my_row r, c] = beta[r] * sum_d t[r,d]        (TROWSUM, TMUL)
//
//   then stores the strict-lower rows (global_row > c) to L_out.  This is a
//   Vec-only kernel; the Cube pass only participates in the entry/exit
//   barriers. (A GEMM-accelerated off-diagonal path is a future optimization.)
//
// Inputs (all on GM, head-major [HV, total_tokens, K]):
//   k       [HV, total_tokens, K]  float16  — keys
//   g_cs    [HV, total_tokens, K]  float32  — within-chunk cumulative gate sum
//   beta    [HV, total_tokens]     float16  — post-sigmoid beta in (0, 1)
//   mask    [C, C]                 float32  — (unused; kept for ABI stability)
//   ws_in   [block_dim*2, 2*C, K]  float32  — (unused; kept for ABI stability)
//   ws_out  [block_dim*2, C, C]    float32  — (unused; kept for ABI stability)
//   L_out   [total_tokens, HV, C]  float16  — strictly-lower-tri L (BSND)
//
// Output:
//   L_out   [total_tokens, HV, C]  float32  — strictly-lower-tri L matrix
//   (BSND)
//
// Cross-core architecture (mirrors GDN scaled_dot_kkt pattern):
//   Both Vec sub-blocks (vid=0,1) do real work: each handles HalfChunk rows.
//     vid=0 → rows [0, C/2),  vid=1 → rows [C/2, C)
//   Vec pre:  load k, g_cs (my rows) → A_eff = k*exp(g), B_eff = k*exp(-g),
//             cast fp16 → ws_in[my rows]
//   Cube:     load full A_eff, B_eff from ws_in → GEMM A @ B^T → ws_out
//   Vec post: load ws_out[my rows], cast fp32 → apply mask + beta row-scale →
//   L_out
//
// FFTS flags (double-buffered, slot = ci & 1):
//   0, 1 : Vec → Cube  "ws_in[slot] ready"  (both vids must sig under mode-2
//   reduce) 2, 3 : Cube → Vec  "ws_out[slot] ready" (broadcast: each vid gets a
//   signal) 4, 5 : Vec → Cube  "ws_out[slot] free"  (Vec done reading L_full;
//   conditional)
//
// UB budget (per vid, HalfChunk=C/2 rows; UB ~192 KB per Vec sub-block):
//   mask fp32 [C/2, C] lives always at offset 0 (loaded once per launch).
//   The rest of UB is a shared pool reused between pre-compute and post-process
//   (they never run concurrently within a chunk).
//   Pre-compute pool (live simultaneously):
//     g_ub  fp32 [C/2, KTC],  k_ub fp32 [C/2, KTC],
//     ab_ub fp32 [C/2, KTC],  half_buf fp16 [C/2, KTC]    (scratch reused A →
//     B)
//   Post-process pool (live simultaneously; overlaps pre-compute addresses):
//     L_half fp16 [C/2, C],  L_ub fp32 [C/2, C],
//     beta_2d fp32 [C/2, C], beta fp32 [1, C/2]
//   Peak @ C=128, K=128: mask 32 + pre 112 = 144 KB ✓ (under 192 KB)
//   Peak @ C=16,  K=128: mask 0.5 + pre 14 ≈ 15 KB ✓
//
// Template parameters:
//   Compile-time: GDN_D = K, GDN_C = C.  Runtime: num_heads = HV.
// ============================================================================

#include "kkt_kda.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

// ── Device entry point
// ────────────────────────────────────────────────────────
extern "C" __global__ AICORE void launch_kkt_kda(
    __gm__ uint8_t *k_ptr, __gm__ uint8_t *g_cs_ptr, __gm__ uint8_t *beta_ptr,
    __gm__ uint8_t *mask_ptr, __gm__ uint8_t *ws_in_ptr,
    __gm__ uint8_t *ws_out_ptr, __gm__ uint8_t *L_out_ptr,
    __gm__ uint8_t *cu_seqlens, int64_t batch_size, int64_t seq_len,
    int64_t total_tokens, int32_t num_heads, uint64_t ffts_addr) {
  kkt_kda_kernel<GDN_D, GDN_C>(reinterpret_cast<__gm__ half *>(k_ptr),
                               reinterpret_cast<__gm__ float *>(g_cs_ptr),
                               reinterpret_cast<__gm__ half *>(beta_ptr),
                               reinterpret_cast<__gm__ float *>(mask_ptr),
                               reinterpret_cast<__gm__ float *>(ws_in_ptr),
                               reinterpret_cast<__gm__ float *>(ws_out_ptr),
                               reinterpret_cast<__gm__ half *>(L_out_ptr),
                               reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
                               batch_size, seq_len, total_tokens, num_heads,
                               ffts_addr);
}

// ── Host entry point (called from Python via ctypes) ─────────────────────────
extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *k_ptr,
                            uint8_t *g_cs_ptr, uint8_t *beta_ptr,
                            uint8_t *mask_ptr, uint8_t *ws_in_ptr,
                            uint8_t *ws_out_ptr, uint8_t *L_out_ptr,
                            uint8_t *cu_seqlens, int64_t batch_size,
                            int64_t seq_len, int64_t total_tokens,
                            uint32_t num_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_kkt_kda<<<block_dim, nullptr, stream>>>(
      k_ptr, g_cs_ptr, beta_ptr, mask_ptr, ws_in_ptr, ws_out_ptr, L_out_ptr,
      cu_seqlens, batch_size, seq_len, total_tokens,
      static_cast<int32_t>(num_heads), fftsAddr);
}
