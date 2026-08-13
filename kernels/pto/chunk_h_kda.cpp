// ============================================================================
// chunk_h_kda.cpp — Recurrent hidden state update for KDA (per-dim gate)
//
// Math (per chunk, matches ref_chunk_h_kda in
//   tests/test_kda_single_kernels.py:276-326):
//   v_corr  = u - w @ S                              # [c_len, V]
//   k_rest  = k * exp(g_total - g_cs)                # [c_len, K]
//   S_new   = exp(g_total).unsqueeze(-1) * S + k_rest^T @ v_corr   # [K, V]
//
// where g_total = g_cs[valid-1, :] is the chunk's per-K-dim cumulative gate
// at the last valid token, and S is the [K, V] state. Snapshots produced:
//   s_snapshots[ci_base + ci, head, :, :] = S entering chunk ci.
//
// Differences from GDN chunk_h.cpp:
//   - g is per-DIMENSION here: g_cs has shape [HV, T, K] (head-major).
//   - State decay factor is a K-vector exp(g_total[k]), not a scalar.
//   - K rescaling coeff_2d[c, k] = exp(g_total[k] - g_cs[c, k]) is element-wise
//     per (token, k-dim), not a row-broadcast scalar.
//   - No GQA — K, W, U all use HV heads with the same BSND stride.
//   - Inputs U, K, G are fp32; W arrives as fp16 (cast in the Python wrapper)
//     so Cube can use it directly.  Outputs v_corr and snapshots are fp32.
//   - v_corr (fp16 copy) lives in a dedicated workspace slot (WS_V) so the
//     Cube K_rest^T @ V_corr GEMM has an fp16 source — the BSND output is fp32.
//
// Inputs:
//   K   [HV, T, K]              fp32  — keys (head-major)
//   W   [B, T, HV, K]           fp16  — wy_kda output, cast to fp16 in wrapper
//   U   [B, T, HV, V]           fp32  — wy_kda output (BSND)
//   G   [HV, T, K]              fp32  — per-dim cumulative gate sum
//   (head-major) S   [total_chunks, HV, K, V] fp32 — snapshots (output) V_corr
//   [B, T, HV, V]        fp32  — corrected values (BSND, output) workspace
//   [per-core scratch] fp16  — 5 slots × K*V halves
//
// Workspace per AI core (5 slots, fp16; assumes K == V == HiddenSize):
//   WS_WS [C, V]   Cube writes WS = W @ S          → Vec reads
//   WS_K  [C, K]   Vec writes K_rest               → Cube reads (^T view)
//   WS_V  [C, V]   Vec writes V_corr (fp16 copy)   → Cube reads
//   WS_S  [K, V]   Vec writes fp16(S)              → Cube reads (next chunk)
//   WS_KV [K, V]   Cube writes K_rest^T @ V_corr   → Vec reads
// ============================================================================

#include "chunk_h_kda.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

extern "C" __global__ AICORE void launch_chunk_h_kda(
    __gm__ uint8_t *K, __gm__ uint8_t *W, __gm__ uint8_t *U, __gm__ uint8_t *G,
    __gm__ uint8_t *S, __gm__ uint8_t *V_corr, __gm__ uint8_t *workspace,
    __gm__ uint8_t *cu_seqlens, int64_t batch_size, int64_t seq_len,
    int64_t total_tokens, int32_t num_heads, uint64_t ffts_addr) {
  chunk_h_kda_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K), reinterpret_cast<__gm__ half *>(W),
      reinterpret_cast<__gm__ half *>(U), reinterpret_cast<__gm__ float *>(G),
      reinterpret_cast<__gm__ half *>(S),
      reinterpret_cast<__gm__ half *>(V_corr),
      reinterpret_cast<__gm__ half *>(workspace),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens), batch_size, seq_len,
      total_tokens, num_heads, ffts_addr);
}

extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *K,
                            uint8_t *W, uint8_t *U, uint8_t *G, uint8_t *S,
                            uint8_t *V_corr, uint8_t *workspace,
                            uint8_t *cu_seqlens, int64_t batch_size,
                            int64_t seq_len, int64_t total_tokens,
                            uint32_t num_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_h_kda<<<block_dim, nullptr, stream>>>(
      K, W, U, G, S, V_corr, workspace, cu_seqlens, batch_size, seq_len,
      total_tokens, static_cast<int32_t>(num_heads), fftsAddr);
}
