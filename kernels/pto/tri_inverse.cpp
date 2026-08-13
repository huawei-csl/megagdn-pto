/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

// The build script adds kernels/pto/include/ to the include path so that
// kernel_utils.h (included by tri_inverse_impl.h) is found.
#include "tri_inverse_impl.h"

/*
 * @brief: Wrapper for the kernel, "half" type (fp16).
 *
 * @param tensor_out pointer to the global memory to store the final inverse.
 * @param tensor_in Pointer to the global tensor matrix in global memory.
 * @param minus_identity_in Pointer to global memory that contains the negative
 * identity.
 * @param matrix_size The size if each individual matrix / tile. Can take
 * values: {16, 32, 64, 128}.
 * @param num_matrices The total number of matrices / tiles in the global
 * tensor.
 * @param num_bsnd_heads The number of heads, which is only greater than zero
 * if the matrix is in BSND format, that is, the tiles need to be loaded with
 * strided accesses. If each tile is stored consecutively (and row-wise) in
 * memory, then num_bsnd_heads=0.
 */
extern "C" __global__ AICORE void tri_inv_rec_unroll_fp16(
    __gm__ void* tensor_out, __gm__ void* tensor_in,
    __gm__ void* minus_identity_in, uint32_t matrix_size, uint32_t num_matrices,
    uint32_t num_bsnd_heads, __gm__ void* cu_seqlens) {
  const uint32_t is_lower = (num_bsnd_heads >> 16) & 1u;
  const uint32_t actual_heads = num_bsnd_heads & 0xFFFFu;

  run_tri_inv_rec_unroll_per_num_matrices<half, float,
                                          1 /* NumTilesPerCubeIter */>(
      (__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
      (__gm__ half*)minus_identity_in, matrix_size, num_matrices, actual_heads,
      is_lower, (__gm__ int32_t*)cu_seqlens);
}

/**
 * @brief JIT entry point for the triangular inverse (recursive unroll) kernel.
 *
 * @param blockDim   Number of AI-Core blocks to launch.
 * @param stream     NPU stream handle.
 * @param tensor_out fp32 output buffer (same element count as tensor_in).
 * @param tensor_in  fp16 input buffer holding the upper-triangular matrices
 *                   (diagonal is assumed to be all-ones).
 * @param minus_identity_in  fp16 buffer of size matrix_size×matrix_size
 *                           pre-filled with -I (negative identity).
 * @param matrix_size   Side length of each square matrix (16 / 32 / 64 / 128).
 * @param num_matrices  Total number of matrices to invert.
 * @param num_bsnd_heads  0 for standard (B…ND) layout;
 *                        N (number of heads) for BSND layout.
 *                        Bit 16 encodes is_lower: if set, the input is
 *                        lower-triangular and the kernel transposes on
 *                        load/store. Actual heads = num_bsnd_heads & 0xFFFF.
 * @param cu_seqlens  Optional int32 pointer used only for varlen BSND. Matches
 *                    the Triton-style API and stores cumulative sequence
 *                    boundaries for the packed BSND tensor.
 */
extern "C" void call_kernel(uint32_t blockDim, void* stream, void* tensor_out,
                            void* tensor_in, void* minus_identity_in,
                            uint32_t matrix_size, uint32_t num_matrices,
                            uint32_t num_bsnd_heads, void* cu_seqlens) {
  tri_inv_rec_unroll_fp16<<<blockDim, nullptr, stream>>>(
      tensor_out, tensor_in, minus_identity_in, matrix_size, num_matrices,
      num_bsnd_heads, cu_seqlens);
}
