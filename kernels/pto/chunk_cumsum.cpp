// ============================================================================
// chunk_cumsum_kernel.cpp — Prefix sum of gate values G along time dimension
//
// Mathematical operation (per chunk of C tokens, independently per head h):
//   g_sum[t, h] = Σ_{i=0}^{t} g[i, h]    for t = 0 .. valid-1
//
// Input:  g     [total_tokens, H]  float, BSND layout  — raw gate values
// Output: g_sum [total_tokens, H]  float               — cumulative sums
//
// The prefix sum enables downstream kernels to compute exponential decay
// coefficients:  exp(g_sum[i] - g_sum[j])  gives the cumulative gate
// from token j to token i within a chunk.
//
// Architecture: Vec-only kernel (no Cube/GEMM). Single Vec sub-block.
// Pipeline: MTE2(load) → Vec(compute) → MTE3(store), serialized per chunk.
//
// NPU memory hierarchy used:
//   GM (Global Memory) → UB (Unified Buffer, on-chip SRAM, Vec-accessible)
//
// ─── PTO / NPU Primer for This Kernel ──────────────────────────────────────
//
// AI Core: The basic processing unit of an NPU, analogous to a Streaming
//   Multiprocessor (SM) on a GPU. A single chip has many AI cores, and each
//   core runs the same kernel code on different data (SPMD model).
//
// Memory hierarchy (outer → inner):
//   GM  (Global Memory) — Off-chip DRAM, like GPU HBM. Large (several GB)
//       but high latency. All AI cores share GM.
//   UB  (Unified Buffer) — On-chip SRAM, ~256 KB per AI core. Like GPU
//       shared memory. Very fast, but small. The Vec engine can only operate
//       on data that lives in UB, so every tensor must be DMA'd in first.
//
// Hardware pipes (execute in parallel, like independent GPU warps):
//   Vec   — SIMD vector processor. Performs element-wise math (add, mul, etc.)
//           on data already in UB. Think of it as a wide SIMD ALU.
//   MTE2  — DMA engine for loads: copies data from GM → UB.
//   MTE3  — DMA engine for stores: copies data from UB → GM.
//   Cube  — Matrix engine for GEMMs (not used in this kernel).
//
// Synchronization (set_flag / wait_flag):
//   Because Vec, MTE2, and MTE3 run in parallel on separate hardware, you
//   must explicitly synchronize them to ensure data is ready:
//     set_flag(SRC_PIPE, DST_PIPE, event): SRC signals that it is done.
//     wait_flag(SRC_PIPE, DST_PIPE, event): DST blocks until the signal.
//   Example: After MTE2 loads data into UB, Vec must wait_flag before reading
//   it. This is like a fine-grained torch.cuda.synchronize() between pipes.
//   Events (EVENT_ID0 .. EVENT_ID7) are semaphore indices.
//
// ============================================================================

#include "chunk_cumsum.h"

#include <runtime/rt_ffts.h>

#include "acl/acl.h"

// ── Device-side kernel entry point ─────────────────────────────────
// extern "C" __global__ AICORE: marks this as an NPU kernel function
//   (like __global__ in CUDA). Each AI core runs one instance of this function.
// Parameters are passed as uint8_t* (raw bytes) and reinterpret_cast'd to
// typed pointers — this is the standard NPU kernel calling convention.
extern "C" __global__ AICORE void launch_cumsum(
    __gm__ uint8_t* g_ptr, __gm__ uint8_t* g_sum_ptr,
    __gm__ uint8_t* cu_seqlens, int64_t batch_size, int64_t seq_len,
    uint32_t num_heads, uint64_t ffts_addr) {
  // NumHeads is a runtime kernel argument (one .so serves every head count).
  // Guard the compile-time UB ceiling; host also validates before launch.
  if (num_heads == 0 || num_heads > GDN_MAX_HEADS) {
    return;
  }
  cumsum_kernel<GDN_C>(reinterpret_cast<__gm__ float*>(g_ptr),
                       reinterpret_cast<__gm__ float*>(g_sum_ptr),
                       reinterpret_cast<__gm__ int32_t*>(cu_seqlens),
                       batch_size, seq_len, static_cast<int32_t>(num_heads),
                       ffts_addr);
}

// ── Host-side launcher (called from Python via ctypes) ────────────
// call_kernel(): CPU function that launches the NPU kernel.
//   block_dim = number of AI cores to use (like CUDA grid size)
//   stream = NPU stream for async execution (like CUDA stream)
//   rtGetC2cCtrlAddr: gets the FFTS control address for cross-core sync
//   <<<block_dim, nullptr, stream>>>: NPU kernel launch syntax (like CUDA
//   <<<>>>)
extern "C" void call_kernel(uint32_t block_dim, void* stream, uint8_t* g_ptr,
                            uint8_t* g_sum_ptr, uint8_t* cu_seqlens,
                            int64_t batch_size, int64_t seq_len,
                            uint32_t num_heads) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_cumsum<<<block_dim, nullptr, stream>>>(
      g_ptr, g_sum_ptr, cu_seqlens, batch_size, seq_len, num_heads, fftsAddr);
}
