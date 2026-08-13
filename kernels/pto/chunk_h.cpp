// ============================================================================
// chunk_h_kernel.cpp — Recurrent hidden state update for GatedDeltaNet
//
// Mathematical recurrence per chunk c:
//   S_{c+1} = exp(g_last) * S_c  +  K^T @ V
//
// where g_last = exp(g[valid-1]) is the chunk's final gate value, S is the
// D×D hidden state, K ∈ ℝ^{C×D}, V ∈ ℝ^{C×D}, and g ∈ ℝ^C is the per-token
// gate.
//
// ── Cube phase (two GEMMs per chunk, sequentially): ──────────────────────
//   1. WS = W @ S       project current state through W (wy_fast output)
//      W ∈ ℝ^{C×D}, S ∈ ℝ^{D×D}  →  WS ∈ ℝ^{C×D}
//   2. KV = K^T @ V     outer product of keys and values (transpose_A!)
//      K stored as D×C, V ∈ ℝ^{C×D}  →  KV ∈ ℝ^{D×D}
//
// ── Vec phase (two sub-blocks handle upper/lower C/2 rows): ─────────────
//   For each chunk:
//     1. Load K, G (pre-transposed), U (from wy_fast)
//     2. Compute coeff[i] = exp(g[i] - g[valid-1])  — time-decay scaling
//        Uses TROWEXPAND to broadcast coefficients across D columns
//     3. Scale K: K_scaled[i,:] = K[i,:] * coeff[i]
//     4. Load WS from Cube workspace, compute V_new = U - WS (residual)
//     5. Store V_new and K_scaled to workspace for Cube's next iteration
//     6. Update state: S = exp(g_last) * S + KV (from Cube workspace)
//     7. Store final state FS after last chunk
//
// Cross-core sync: Cube→Vec flags for WS/KV ready, Vec→Cube flags for
// K/S ready.
//
// Inputs:
//   K  [total_tokens, Hg, D] half   — keys (BSND layout; GQA/MQA group heads)
//   W  [total_tokens, H, D]  half   — wy_fast output (BSND layout)
//   U  [total_tokens, H, D]  half   — values pre-residual (BSND layout)
//   G  [H, total_tokens]     float  — pre-transposed cumulative gates
//   S  [total_chunks, H, D, D] half — per-chunk state snapshots (output)
//   V  [total_tokens, H, D]  half   — residual-corrected values (output)
//   FS [batch, H, D, D]      half   — final state per sequence (output)
//   H0 [batch, H, D, D]      half   — optional initial state per sequence
//   workspace [per-core scratch]     — Cube↔Vec communication buffer
//
// NPU memory hierarchy:
//   GM → L1 (Cube-accessible) → L0A/L0B/L0C (Cube GEMM registers)
//   GM → UB (Vec-accessible, on-chip SRAM)
//   Cross-core sync via FFTS (Fast Fine-grained Task Synchronization)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This is the most complex kernel in the GDN suite. It implements the
// recurrent state update, requiring sequential chunk processing (chunks
// within a sequence CANNOT be parallelized — each depends on the previous).
//
// Key PTO APIs (numpy/torch equivalents):
//   TLOAD(dst, gm)          — dst = gm_data        (DMA: GM→L1 or GM→UB)
//   TSTORE(gm, src)         — gm_data = src        (DMA: UB/L0C→GM)
//   TASSIGN(tile, addr)     — tile = memory[addr]   (bind tile to buffer
//   address) TCVT(dst, src, mode)    — dst = src.float()/.half() TMOV(dst, src)
//   — dst = src.clone() TADD(d, a, b)           — d = a + b TSUB(d, a, b) — d =
//   a - b TMUL(d, a, b)           — d = a * b TMULS(d, s, scalar)     — d = s *
//   scalar       (scalar multiply) TADDS(d, s, scalar)     — d = s + scalar
//   (scalar add) TEXP(d, s)              — d = torch.exp(s) TEXPANDS(tile,
//   scalar)  — tile[:] = scalar     (fill with constant) TROWEXPAND(2d, col) —
//   2d[i,j] = col[i]    (broadcast col across row dim) TFILLPAD(dst, src) —
//   zero-fill L1 tile padding (for tail chunks) TEXTRACT(l0, l1, r, c)  — L1
//   sub-tile → L0A/L0B TRESHAPE(zn, nz)        — reinterpret layout NZ↔ZN
//   (logical transpose, free) TMATMUL(C, A, B)        — C = A @ B (Cube GEMM,
//   fp16 inputs → fp32 accum) set_flag/wait_flag      — pipe sync within same
//   core ffts_cross_core_sync    — cross-core signal Cube↔Vec
//   wait_flag_dev(flag)     — wait for cross-core signal
//   GetValue(idx)           — read a single scalar from a UB tile (slow, use
//   sparingly)
//
// ── Workspace memory layout (shared between Cube and Vec via GM) ──────
// Each AI core has its own workspace region to avoid contention:
//   WS_WS [C×D]:  Cube writes WS = W @ S here → Vec reads it
//   WS_K  [D×C]:  Vec writes K_scaled here → Cube reads it for KV = K^T @ V
//   WS_S  [D×D]:  Vec writes current state S here → Cube reads it for GEMM 1
//   WS_KV [D×D]:  Cube writes KV = K^T @ V here → Vec reads it to update S
//
// Data flow per chunk (think of it as a ping-pong between Cube and Vec):
//   Vec: write S₀ to WS_S → signal Cube (flag 3)
//   Cube: read S from WS_S, load W → compute WS = W@S → write WS_WS → signal
//   Vec (flag 0) Vec: read WS, compute V_new = U - WS, compute K_scaled → write
//   WS_K → signal Cube (flag 1) Cube: read K from WS_K, load V → compute KV =
//   K^T@V → write WS_KV → signal Vec (flag 2) Vec: read KV, update S =
//   exp(g_last)*S + KV → write S to WS_S → signal Cube (flag 3)
//   ... repeat for next chunk ...
// ============================================================================

#include "chunk_h.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

extern "C" __global__ AICORE void launch_chunk_h(
    __gm__ uint8_t* K, __gm__ uint8_t* W, __gm__ uint8_t* U, __gm__ uint8_t* G,
    __gm__ uint8_t* S, __gm__ uint8_t* V, __gm__ uint8_t* FS,
    __gm__ uint8_t* H0, int64_t has_initial_state, int64_t output_final_state,
    __gm__ uint8_t* workspace, __gm__ uint8_t* cu_seqlens, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, uint32_t num_heads,
    uint32_t num_key_heads, uint64_t ffts_addr) {
  chunk_h_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half*>(K), reinterpret_cast<__gm__ half*>(W),
      reinterpret_cast<__gm__ half*>(U), reinterpret_cast<__gm__ float*>(G),
      reinterpret_cast<__gm__ half*>(S), reinterpret_cast<__gm__ half*>(V),
      reinterpret_cast<__gm__ half*>(FS), reinterpret_cast<__gm__ half*>(H0),
      has_initial_state, output_final_state,
      reinterpret_cast<__gm__ half*>(workspace),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens), batch_size, seq_len,
      total_tokens, num_heads, num_key_heads, ffts_addr);
}

extern "C" void call_kernel(uint32_t block_dim, void* stream, uint8_t* K,
                            uint8_t* W, uint8_t* U, uint8_t* G, uint8_t* S,
                            uint8_t* V, uint8_t* FS, uint8_t* H0,
                            int64_t has_initial_state,
                            int64_t output_final_state, uint8_t* workspace,
                            uint8_t* cu_seqlens, int64_t batch_size,
                            int64_t seq_len, int64_t total_tokens,
                            uint32_t num_heads, uint32_t num_key_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_h<<<block_dim, nullptr, stream>>>(
      K, W, U, G, S, V, FS, H0, has_initial_state, output_final_state,
      workspace, cu_seqlens, batch_size, seq_len, total_tokens, num_heads,
      num_key_heads, fftsAddr);
}
