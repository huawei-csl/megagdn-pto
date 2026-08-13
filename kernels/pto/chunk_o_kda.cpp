// ============================================================================
// chunk_o_kda.cpp — Output stage for KDA (per-dim gate)
//
// Math (per chunk, matches ref_chunk_o_kda in
//   tests/test_kda_single_kernels.py:333-380):
//   q_eff = q * exp(g_cs)              # [c_len, K]
//   k_eff = k * exp(-g_cs)             # [c_len, K]
//   inter = q_eff @ S                  # [c_len, V]
//   Aqk   = tril(q_eff @ k_eff^T,      # [c_len, c_len], INCLUSIVE diagonal
//                diagonal=0)
//   o     = inter + Aqk @ v_corr       # [c_len, V]
//
// where S = s_snapshots[ci_base + ci, head] is the [K, V] state *entering*
// this chunk (already computed by chunk_h_kda), and v_corr = u - w @ S is
// the corrected values (also from chunk_h_kda).
//
// Differences from GDN chunk_o.cpp:
//   - Gate is per-DIMENSION (g_cs has shape [HV, T, K] head-major).
//   - Vec pre-scales Q and K element-wise (q*exp(g_cs), k*exp(-g_cs)) BEFORE
//     Cube sees them.  Cube does pure matmuls; there is no per-element gating
//     coefficient applied to QK on the Vec side.
//   - Causal mask is INCLUSIVE of the diagonal (rows >= cols), so the mask
//     tensor passed from Python differs from kkt_kda's strict-lower mask.
//   - No GQA: Q, K, V_corr, O all use HV heads.
//   - S is fp32 in GM (from chunk_h_kda's output) — Vec casts to fp16 into
//     workspace so Cube has fp16 sources for all three GEMMs.
//
// Chunks within a (seq, head) work item are fully independent (each reads
// its own s_snapshots entry).  Cube/Vec still process them sequentially per
// work item to keep the per-core 4-flag protocol simple.
//
// Inputs:
//   Q       [HV, T, K]               fp32  — queries (head-major), scale
//   pre-applied K       [HV, T, K]               fp32  — keys    (head-major)
//   V_corr  [B, T, HV, V]            fp32  — corrected values from chunk_h_kda
//   (BSND) S       [total_chunks, HV, K, V] fp32  — snapshots from chunk_h_kda
//   G_cs    [HV, T, K]               fp32  — per-dim cumulative gate
//   (head-major) Msk     [C, C]                   fp32  — inclusive lower-tri
//   mask (rows >= cols) workspace [per-core scratch]     float32 — 7 slots ×
//   K*V floats O       [B, T, HV, V]            fp32  — output (BSND)
//
// NOTE: the workspace (and all three GEMMs) are fp32, not fp16: k_eff =
//   k*exp(-g_cs) and the unmasked QK = q_eff @ k_eff^T blow up to ~e^64
//   (per-128 chunk |g_cs|≈64) which overflows fp16 (max 6.5e4) -> inf ->
//   inf*mask=NaN. fp32 (max 3.4e38) holds them, and the inclusive mask zeroes
//   the upper-tri cleanly.  q/k/v/S inputs arrive as fp16 from GM and are cast
//   up; O is cast back to fp16 on write.
//
// Workspace per AI core (7 slots, float32; assumes K == V == HiddenSize):
//   WS_Q   [C, K]   Vec writes q*exp(g_cs)  → Cube reads (GEMM1 A, GEMM2 A)
//   WS_K   [C, K]   Vec writes k*exp(-g_cs) → Cube reads (GEMM1 B, transposed)
//   WS_V   [C, V]   Vec writes V_corr fp16  → Cube reads (GEMM3 B)
//   WS_S   [K, V]   Vec writes S fp16       → Cube reads (GEMM2 B)
//   WS_QK  [C, C]   Cube writes QK fp16     → Vec masks → Cube reads (GEMM3 A)
//   WS_QS  [C, V]   Cube writes QS fp16     → Vec reads (final combine)
//   WS_QKV [C, V]   Cube writes QKV fp16    → Vec reads (final combine)
// ============================================================================

#include "chunk_o_kda.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

extern "C" __global__ AICORE void launch_chunk_o_kda(
    __gm__ uint8_t *Q, __gm__ uint8_t *K, __gm__ uint8_t *V_corr,
    __gm__ uint8_t *S, __gm__ uint8_t *G, __gm__ uint8_t *Mask,
    __gm__ uint8_t *workspace, __gm__ uint8_t *O, __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    int32_t num_heads, uint64_t ffts_addr) {
  chunk_o_kda_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(Q), reinterpret_cast<__gm__ half *>(K),
      reinterpret_cast<__gm__ half *>(V_corr),
      reinterpret_cast<__gm__ half *>(S), reinterpret_cast<__gm__ float *>(G),
      reinterpret_cast<__gm__ float *>(Mask),
      reinterpret_cast<__gm__ float *>(workspace),
      reinterpret_cast<__gm__ half *>(O),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens), batch_size, seq_len,
      total_tokens, num_heads, ffts_addr);
}

extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *Q,
                            uint8_t *K, uint8_t *V_corr, uint8_t *S, uint8_t *G,
                            uint8_t *Mask, uint8_t *workspace, uint8_t *O,
                            uint8_t *cu_seqlens, int64_t batch_size,
                            int64_t seq_len, int64_t total_tokens,
                            uint32_t num_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_o_kda<<<block_dim, nullptr, stream>>>(
      Q, K, V_corr, S, G, Mask, workspace, O, cu_seqlens, batch_size, seq_len,
      total_tokens, static_cast<int32_t>(num_heads), fftsAddr);
}
