// ============================================================================
// chunk_o_kernel.cpp — Output computation for GatedDeltaNet (chunk-wise)
//
// Mathematical operation (per chunk of C tokens, per head h):
//
//   O = (QK_gated @ V) + exp(g) * (Q @ S)
//     = intra_chunk_attention + inter_chunk_state_contribution
//
// where:
//   Q, K, V ∈ ℝ^{C×D}    — query/key/value projections for this chunk
//   S ∈ ℝ^{D×D}           — accumulated hidden state entering this chunk
//   G ∈ ℝ^{C}             — cumulative gate values (pre-transposed [H,T])
//   Msk ∈ ℝ^{C×C}         — lower-triangular causal mask
//
// Cube phase (3 GEMMs per chunk):
//   1. QK   = Q @ K^T         — intra-chunk attention scores
//   2. QS   = Q @ S           — query applied to accumulated state
//   3. QKV  = QK_gated @ V    — gated attention applied to values
//
// Vec phase (two sub-blocks process upper/lower C/2 rows):
//   a. Load G → compute gating coefficients:
//        coeff[i,j] = exp(min(g[i] - g[j], 0)) * mask[i,j]
//   b. Apply gating to QK: QK_gated = QK * coeff
//   c. Scale QS by exp(g): QS_gated = QS * exp(g_row)
//   d. Combine: O = QS_gated + QKV
//   e. Store O to GM in BSND layout
//
// Cross-core sync protocol (Cube ↔ Vec via FFTS):
//   flag 0: Cube→Vec  — QK and QS results ready in workspace
//   flag 1: Vec→Cube  — QK_gated written back, Cube can proceed to GEMM 3
//   flag 2: Cube→Vec  — QKV result ready in workspace
//   flag 3: Vec→Cube  — Vec done with this chunk, Cube can reuse workspace
//
// NPU memory hierarchy used:
//   GM → L1 (Cube-accessible) → L0A/L0B (matrix engines) → L0C (accumulator)
//   GM → UB (Vec-accessible, on-chip SRAM)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel combines matrix multiplication (Cube) with element-wise gating
// (Vec) in a tightly coordinated 3-GEMM + gating pipeline per chunk.
//
// Execution timeline for one chunk:
//   Cube: GEMM1(Q@K^T) → GEMM2(Q@S) → store QK,QS → signal Vec ──────┐
//   Vec:  (meanwhile) load G, compute gating coefficients                │
//   Vec:  ←── wait for Cube signal ──── apply gating to QK → QK_gated  │
//   Vec:  store QK_gated → signal Cube ────────────────────────────────┐│
//   Cube: ←── wait for Vec signal ──── GEMM3(QK_gated@V) → store QKV ─┘│
//   Vec:  ←── wait for Cube signal ──── scale QS, combine O=QKV+QS_g   │
//   Vec:  store O → signal Cube "done" ─────────────────────────────────┘
//
// numpy pseudocode for the entire chunk computation:
//   QK = Q @ K.T                                          # GEMM 1
//   QS = Q @ S                                            # GEMM 2
//   coeff = exp(min(g_row - g_col, 0)) * mask             # gating (dynamic
//   PTO)
//   (``static_baseline/run_chunk_o_static.py`` uses exp(g_row-g_col) without
//   min.) QK_gated = QK * coeff                                 # apply gating
//   QKV = QK_gated @ V                                    # GEMM 3
//   O = QKV + QS * np.exp(g_row).reshape(-1, 1)           # final output
//
// Key PTO APIs (with numpy/torch equivalents):
//   TLOAD(dst, gm)          — dst = gm_data      (DMA: GM→UB/L1, async)
//   TSTORE(gm, src)         — gm = src            (DMA: UB/L0C→GM, async)
//   TASSIGN(tile, addr)     — bind tile descriptor to buffer address
//   TCVT(dst, src, mode)    — type cast: dst = src.float() or .half()
//   TMOV(dst, src)          — copy: dst = src.clone()
//   TADD(d, a, b)           — d = a + b
//   TSUB(d, a, b)           — d = a - b
//   TMUL(d, a, b)           — d = a * b
//   TMINS(d, s, val)        — d = torch.clamp(s, max=val)
//   TEXP(d, s)              — d = torch.exp(s)
//   TROWEXPAND(2d, col)     — 2d[i,j] = col[i] (broadcast column→rows)
//   TCOLEXPAND(2d, row)     — 2d[i,j] = row[j] (broadcast row→columns)
//   TEXTRACT(l0, l1, r, c)  — copy L1 sub-tile → L0A/L0B (Cube input regs)
//   TRESHAPE(zn, nz)        — reinterpret L1 fractal layout (transpose, free)
//   TMATMUL(C, A, B)        — C = A @ B (Cube engine, fp16→fp32 accum)
//   set_flag / wait_flag    — synchronize pipes within same AI core
//   ffts_cross_core_sync    — signal across Cube↔Vec cores
//   wait_flag_dev(flag)     — wait for cross-core signal
// ============================================================================

#include "chunk_o.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

// ── Device kernel entry point ─────────────────────────────────────────
// extern "C" __global__ AICORE: NPU kernel function.
// Runs on each AI core independently. Args are uint8_t* (type-erased)
// because the NPU launch ABI passes all pointers as raw bytes; we
// reinterpret_cast them to the correct types before calling the template.
extern "C" __global__ AICORE void launch_chunk_o(
    __gm__ uint8_t *Q_handle, __gm__ uint8_t *K_handle,
    __gm__ uint8_t *V_handle, __gm__ uint8_t *S_handle,
    __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle,
    __gm__ uint8_t *workspace_qk, __gm__ uint8_t *workspace_qs_qkv,
    __gm__ uint8_t *workspace_qk_gated, __gm__ uint8_t *O_handle,
    __gm__ uint8_t *cu_seqlens, int64_t batch_size, int64_t seq_len,
    int64_t total_tokens, uint32_t num_heads, uint32_t num_key_heads,
    uint64_t ffts_addr) {
  chunk_o_kernel<GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(Q_handle),
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(S_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ float *>(Msk_handle),
      reinterpret_cast<__gm__ half *>(workspace_qk),
      reinterpret_cast<__gm__ half *>(workspace_qs_qkv),
      reinterpret_cast<__gm__ half *>(workspace_qk_gated),
      reinterpret_cast<__gm__ half *>(O_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens), batch_size, seq_len,
      total_tokens, num_heads, num_key_heads, ffts_addr);
}

// ── Host launcher (called from Python ctypes) ─────────────────────────
// Launches kernel on block_dim AI cores via NPU stream.
// rtGetC2cCtrlAddr obtains the FFTS (cross-core sync) control address that
// the kernel needs for Cube↔Vec flag signaling.
extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *q,
                            uint8_t *k, uint8_t *v, uint8_t *s, uint8_t *g_sum,
                            uint8_t *mask, uint8_t *workspace_qk,
                            uint8_t *workspace_qs_qkv,
                            uint8_t *workspace_qk_gated, uint8_t *o,
                            uint8_t *cu_seqlens, int64_t batch_size,
                            int64_t seq_len, int64_t total_tokens,
                            uint32_t num_heads, uint32_t num_key_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_o<<<block_dim, nullptr, stream>>>(
      q, k, v, s, g_sum, mask, workspace_qk, workspace_qs_qkv,
      workspace_qk_gated, o, cu_seqlens, batch_size, seq_len, total_tokens,
      num_heads, num_key_heads, fftsAddr);
}
