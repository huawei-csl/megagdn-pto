// ============================================================================
// scaled_dot_kkt_kernel.cpp — Intra-chunk attention matrix for GatedDeltaNet
//
// Computes A = mask(KK^T · gating_coeff) per chunk, where:
//   KK^T ∈ ℝ^{C×C} = K @ K^T                  (Cube engine, GEMM)
//   coeff[i,j] = exp(clamp(g[i]+log(β[i]) - g[j], max=0))  (Vec engine)
//   A[i,j] = KK^T[i,j] · coeff[i,j] · causal_mask[i,j]
//
// Inputs:
//   K       [total_tokens, Hg, D] half  — key vectors (BSND along seq; stride
//   Hg * D) Beta    [H, total_tokens]     half  — gate bias per **value** head
//   (pre-transposed) G       [H, total_tokens]     float — cumulative gate sum
//   per **value** head Msk     [C, C]                float — lower-triangular
//   causal mask
//
// Output:
//   A       [total_tokens, H, C]  half  — gated attention matrix in BSND
//
// Architecture: Cube + Vec cross-core kernel.
//   Cube phase: K→L1, GEMM K@K^T→L0C, store to workspace (GM)
//   Vec phase:  load workspace KK^T, compute gating coefficients, apply mask
//
// Cross-core sync: Cube signals Vec via FFTS flag after each chunk's KK^T
// is written to workspace. Vec signals back when workspace buffer is free.
// Two workspace slots alternate (double-buffering via slot = ci & 1).
//
// Vec sub-blocks: Two sub-blocks (vid=0,1) process upper/lower halves of
// the C×C attention matrix in parallel (HalfChunk rows each).
//
// NPU memory hierarchy:
//   GM → L1 (Cube-accessible) → L0A/L0B (GEMM operands) → L0C (accumulator)
//   GM → UB (Vec-accessible SRAM)
//
// ── PTO / NPU Primer for This Kernel ──────────────────────────────────
// NPU Architecture (simplified):
//   Each "AI Core" (like a GPU SM) has:
//     - Cube engine: matrix multiply unit (like GPU Tensor Cores), works on
//     L0A/L0B/L0C
//     - Vec engine: SIMD vector unit (like GPU CUDA cores), works on UB
//     (Unified Buffer)
//     - MTE2: DMA engine for loading data: GM → L1 or GM → UB
//     - MTE3: DMA engine for storing data: UB → GM or L0C → GM
//     - MTE1: DMA engine for L1 → L0A/L0B transfers (internal to Cube pipeline)
//   Memory hierarchy (fast→slow): L0 registers > L1 cache > UB (SRAM) > GM
//   (HBM) Cube and Vec run on SEPARATE cores — they communicate via GM +
//   cross-core flags.
//
// Key PTO APIs used in this kernel (with numpy/torch equivalents):
//   TASSIGN(tile, addr)     — Bind tile to UB/L1/L0 address (tile =
//   memory[addr]) TLOAD(dst, gm_tensor)   — DMA load: dst = gm_tensor (async,
//   MTE2 pipe) TSTORE(gm, src)         — DMA store: gm = src (async, MTE3 pipe)
//   TFILLPAD(dst, src)      — Zero-fill padding: dst[outside valid] = 0
//   TFILLPAD_INPLACE(d, s)  — Same but in-place for UB tiles
//   TEXTRACT(l0, l1, r, c)  — Copy L1 sub-block → L0A or L0B (MTE1 pipe)
//   TRESHAPE(dst, src)      — Reinterpret L1 tile layout (NZ↔ZN for transpose)
//   TMATMUL(C, A, B)        — Matrix multiply: C = A @ B in Cube engine
//   TCVT(dst, src, mode)    — Type conversion: like dst = src.float() or
//   src.half() TMOV(dst, src)          — Copy: dst = src.clone() TADD(d, a, b)
//   — Element-wise add: d = a + b TSUB(d, a, b)           — Element-wise
//   subtract: d = a - b TMUL(d, a, b)           — Element-wise multiply: d = a
//   * b TMINS(d, s, val)        — Clamp max: d = torch.clamp(s, max=val)
//   TEXP(d, s)              — Element-wise exp: d = torch.exp(s)
//   TLOG(d, s)              — Element-wise log: d = torch.log(s)
//   TROWEXPAND(2d, col)     — Broadcast column → rows: 2d[i,j] = col[i]
//   TCOLEXPAND(2d, row)     — Broadcast row → cols: 2d[i,j] = row[j]
//   set_flag(P1, P2, EVT)   — Signal from pipe P1 to pipe P2 (like a semaphore
//   post) wait_flag(P1, P2, EVT)  — Wait for signal from P1 (like a semaphore
//   wait) pipe_barrier(PIPE_V)    — Local Vec barrier (ensure all Vec ops
//   complete) pipe_barrier(PIPE_ALL)  — Barrier for all local pipes
//   ffts_cross_core_sync()  — Cross-core signal (Cube↔Vec, different physical
//   cores) wait_flag_dev(flag)     — Wait for cross-core signal
// ============================================================================

#include "scaled_dot_kkt.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

// ── NPU kernel entry point ────────────────────────────────────────────
// extern "C" __global__ AICORE: NPU kernel entry point (like CUDA __global__).
// Parameters passed as uint8_t* and reinterpret_cast'd — standard NPU
// convention. The NPU runtime passes raw byte pointers; we cast them to typed
// pointers here.
extern "C" __global__ AICORE void launch_scaled_dot_kkt(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *Beta_handle,
    __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle,
    __gm__ uint8_t *workspace_handle, __gm__ uint8_t *A_handle,
    __gm__ uint8_t *cu_seqlens, int64_t batch_size, int64_t seq_len,
    int64_t total_tokens, uint32_t num_heads, uint32_t num_key_heads,
    uint64_t ffts_addr) {
  kkt_kernel<GDN_D, GDN_C>(reinterpret_cast<__gm__ half *>(K_handle),
                           reinterpret_cast<__gm__ half *>(Beta_handle),
                           reinterpret_cast<__gm__ float *>(G_handle),
                           reinterpret_cast<__gm__ float *>(Msk_handle),
                           reinterpret_cast<__gm__ half *>(workspace_handle),
                           reinterpret_cast<__gm__ half *>(A_handle),
                           reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
                           batch_size, seq_len, total_tokens, num_heads,
                           num_key_heads, ffts_addr);
}

// ── Host-side launcher ────────────────────────────────────────────────
// call_kernel(): Host-side launcher invoked from Python via ctypes.
//   block_dim = number of AI cores (like CUDA grid size)
//   <<<block_dim, nullptr, stream>>>: NPU kernel launch syntax
//     - block_dim: how many AI cores to use (each runs kkt_kernel
//     independently)
//     - nullptr: no shared memory (NPU doesn't have CUDA-style shared mem)
//     - stream: async execution stream (like CUDA streams)
//
// rtGetC2cCtrlAddr: Get the hardware address of the cross-core (Cube↔Vec) flag
// table. This address is passed to the kernel so it can call
// ffts_cross_core_sync.
extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *K_handle,
                            uint8_t *Beta_handle, uint8_t *G_handle,
                            uint8_t *Msk_handle, uint8_t *workspace_handle,
                            uint8_t *A_handle, uint8_t *cu_seqlens,
                            int64_t batch_size, int64_t seq_len,
                            int64_t total_tokens, uint32_t num_heads,
                            uint32_t num_key_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_scaled_dot_kkt<<<block_dim, nullptr, stream>>>(
      K_handle, Beta_handle, G_handle, Msk_handle, workspace_handle, A_handle,
      cu_seqlens, batch_size, seq_len, total_tokens, num_heads, num_key_heads,
      fftsAddr);
}
