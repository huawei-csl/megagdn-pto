/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include <pto/pto-inst.hpp>

#include "kernel_utils.h"

// Diagonal block size for the D+N split; see DiagonalBlockSize.
#ifndef TRI_INV_DIAGONAL_BLOCK
#define TRI_INV_DIAGONAL_BLOCK 16
#endif

using namespace pto;
using namespace kernel_utils;

#define BSND_OFFSET(tile_id, N, S, D) \
  (((tile_id) / (N)) * (S) * (N) * (D) + ((tile_id) % (N)) * (D))

/*
 * For aligned BSND, tile_id enumerates chunk-major then head-major and maps to
 * a fixed-stride address inside the dense BSND tensor.
 */
AICORE inline uint32_t GetBSNDFixedTileOffset(uint32_t tile_id,
                                              uint32_t num_bsnd_heads,
                                              uint32_t matrix_size) {
  return BSND_OFFSET(tile_id, num_bsnd_heads, matrix_size, matrix_size);
}

/**
 * @brief Struct containing starting address and size of a single tile
 */
struct BSNDVarlenTileInfo {
  uint32_t bsnd_offset; /**< Contains the starting index in the global tensor */
  uint32_t valid_size;  /**< This is the size (num_rows/cols) of the tile */
};

/*
 * For cu_seqlens-based varlen BSND, tile_id still enumerates chunk-major then
 * head-major. We recover the owning sequence by scanning cu_seqlens and
 * counting chunks per sequence.
 */
AICORE inline BSNDVarlenTileInfo GetBSNDVarlenTileInfoFromCuSeqlens(
    uint32_t tile_id, uint32_t num_bsnd_heads, uint32_t matrix_size,
    __gm__ int32_t* cu_seqlens) {
  const uint32_t head_idx = tile_id % num_bsnd_heads;
  const uint32_t chunk_idx = tile_id / num_bsnd_heads;

  uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[0]);
  uint32_t accumulated_chunks = 0;
  for (uint32_t seq_idx = 0;; ++seq_idx) {
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
    const uint32_t seq_len = seq_end - seq_start;
    const uint32_t seq_num_chunks = CeilDiv(seq_len, matrix_size);
    if (chunk_idx < accumulated_chunks + seq_num_chunks) {
      const uint32_t local_chunk_idx = chunk_idx - accumulated_chunks;
      const uint32_t row_start = seq_start + local_chunk_idx * matrix_size;
      const uint32_t valid_size =
          min(static_cast<uint32_t>(seq_end - row_start), matrix_size);
      return {row_start * num_bsnd_heads * matrix_size + head_idx * matrix_size,
              valid_size};
    }
    accumulated_chunks += seq_num_chunks;
    seq_start = seq_end;
  }
}

/**
 * @brief: Takes as input two matrices of size MatrixSize * MatrixSize each,
 * and an integer block_size. The src matrix lies in L1, while the dst matrix
 * either in L0A or L0B. This method copies some of the diagonal blocks from the
 * input to the output as follows:
 * - If dst is in L0A (left): copy even diagonal blocks 0, 2, 4, ...
 * - If dst is in L0B (right): copy odd blocks 1, 3, 5, ...
 * Important note: the dst matrix should be initialized to all-zeros before
 * calling this method
 *
 * @tparam InputT Input data type (fp16).
 * @tparam FractalSize Size of each fractal matrix (diagonal block).
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam SrcL1TileT The actual tile type of the src matrix.
 * @tparam DstL0TileT The actual tile type of the dst matrix.
 *
 * @param src Tile in L1 memory.
 * @param dst Tile in L0A or L0B memory.
 * @param block_size Size of diagonal blocks. Needs: block_size >= FractalSize.
 * @param swap_parity If true, swap which parity of diagonal blocks is copied.
 */
template <typename InputT, uint32_t FractalSize, uint32_t MatrixSize,
          typename SrcL1TileT, typename DstL0TileT>
AICORE inline void CopyOddOrEvenBlocksL1ToL0(SrcL1TileT src, DstL0TileT dst,
                                             uint32_t block_size,
                                             bool swap_parity = false) {
  constexpr bool is_left =
      std::is_same_v<DstL0TileT, TileLeft<InputT, MatrixSize, MatrixSize>>;
  constexpr TileType LeftOrRight = is_left ? TileType::Left : TileType::Right;
  constexpr SLayout InnerLayout =
      is_left ? SLayout::RowMajor : SLayout::ColMajor;
  constexpr BLayout OuterLayout = kernel_utils::GetOuterLayout(is_left);
  // For left: copy even blocks 0, 2, 4, ... (starting_block=0)
  // For right: copy odd blocks 1, 3, 5, ... (starting_block=1)
  // Default: left→even(0), right→odd(1). swap_parity flips this.
  const uint32_t starting_block_index =
      (is_left ? 0u : 1u) ^ (swap_parity ? 1u : 0u);

  const uint32_t num_blocks = MatrixSize / block_size;
  const uint32_t num_fractals_per_block = block_size / FractalSize;

  // might need fewer fractals if block_size < FractalSize
  Tile<LeftOrRight, InputT, FractalSize, FractalSize, OuterLayout, FractalSize,
       FractalSize, InnerLayout, TileConfig::fractalABSize>
      fractals[MatrixSize / FractalSize];

  const std::uintptr_t starting_address =
      reinterpret_cast<std::uintptr_t>(dst.data());
  for (uint32_t i = 0; i < num_fractals_per_block; ++i) {
    for (uint32_t j = 0; j < num_fractals_per_block; ++j) {
      for (uint32_t b = starting_block_index; b < num_blocks; b += 2) {
#ifdef __DAV_C310__
        const uint32_t row_stride = is_left ? FractalSize : MatrixSize;
        const uint32_t col_stride = is_left ? MatrixSize : FractalSize;
#else
        const uint32_t row_stride = MatrixSize;
        const uint32_t col_stride = FractalSize;
#endif
        const uint32_t offset =
            b * (MatrixSize + FractalSize) * block_size /* block_offset */ +
            j * col_stride * FractalSize /* col_fractal_offset */ +
            i * row_stride * FractalSize /* row_fractal_offset */;
        TASSIGN(fractals[b], starting_address + offset * sizeof(InputT));
        TEXTRACT(fractals[b], src, b * block_size + i * FractalSize,
                 b * block_size + j * FractalSize);
      }
    }
  }
}

/**
 * @brief Copies the diagonal blocks of `block_size` from src to dst.
 *
 * Both parities of CopyOddOrEvenBlocksL1ToL0 together are exactly the
 * diagonal blocks. At block_size == MatrixSize the whole matrix is one
 * block, and the fractal path would issue (MatrixSize/FractalSize)^2
 * TEXTRACTs to copy what a single TMOV covers.
 *
 * @tparam InputT Data type of the input matrix.
 * @tparam FractalSize Size of matrix fractals.
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam SrcL1TileT The type of the source tile in L1.
 * @tparam DstL0TileT The type of the destination tile in L0.
 *
 * @param src Source tile in L1.
 * @param dst Destination tile in L0.
 * @param block_size Size of the diagonal blocks to copy.
 */
template <typename InputT, uint32_t FractalSize, uint32_t MatrixSize,
          typename SrcL1TileT, typename DstL0TileT>
AICORE inline void CopyDiagonalBlocksL1ToL0(SrcL1TileT src, DstL0TileT dst,
                                            uint32_t block_size) {
  if (block_size >= MatrixSize) {
    TMOV(dst, src);
    return;
  }
  CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(src, dst,
                                                             block_size, false);
  CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(src, dst,
                                                             block_size, true);
}

/**
 * @brief: Prepares Identity and Zeros matrix.
 *
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 *
 * @param I_neg_l1_tile Tile containing the -I (negative identity) matrix.
 * @param Zero_l1_tile Tile to store the all-zero matrix.
 * @param I_l1_tile Tile to store the identity matrix.
 * @param a_l0_tile Tile in L0A for matmuls.
 * @param b_l0_tile Tile in L0B for matmuls.
 * @param c_l0_tile Tile in L0C for matmuls.
 */
template <typename TileL1AB, typename TileL0A, typename TileL0B,
          typename TileL0C>
AICORE inline void PrepareAuxiliaryMatrices(
    TileL1AB I_neg_l1_tile, TileL1AB Zero_l1_tile, TileL1AB I_l1_tile,
    TileL0A a_l0_tile, TileL0B b_l0_tile, TileL0C c_l0_tile) {
  TMOV(a_l0_tile, I_neg_l1_tile);  // a_l0 initialized with I_neg
  TMOV(b_l0_tile, I_neg_l1_tile);  // b_l0 initialized with I_neg
  set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
  wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains I
  set_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  wait_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));

  TMOV(I_l1_tile, c_l0_tile);  // I_l1 now contains I
  set_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));

  TMOV(b_l0_tile, I_l1_tile);  // b_l0 contains I
  set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
  wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));

  TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile,
              b_l0_tile);  // c_l0 contains zeros
  set_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  wait_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));

  TMOV(Zero_l1_tile, c_l0_tile);  // Zeros_l1 now contains zeros
  set_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
}

/**
 * @brief: Inverts a single matrix / tile of the global tensor.
 * Writes M = D + N, where D is block diagonal and N is strictly block
 * triangular. The first phase obtains Xd = (I + D)^-1 by doubling inside D's
 * small diagonal blocks. The second phase obtains (I + Xd N)^-1 by doubling
 * in block steps, then multiplies it by Xd.
 *
 * @tparam InputT The type of the input elements.
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam FractalSize Size of matrix fractals.
 * @tparam DiagonalBlockSize Side length of D's diagonal blocks.
 * @tparam NumTilesPerCubeIter How many matrices to load and invert in a single
 * cube iteration.
 *
 * @param X_l1_tile Tile in L1 used for intermediate computations.
 * @param I_l1_tile Tile containing the identity matrix.
 * @param I_neg_l1_tile Tile containing the negative identity matrix.
 * @param M_neg_l1_tile Tile containing the negative input matrix.
 * @param Zero_l1_tile Tile containing the all-zero matrix.
 * @param Y_l1_tile Tile in L1 used for intermediate computations.
 * @param a_l0_tile* Array of two tiles in L0A (for double-buffering).
 * @param b_l0_tile* Array of two tiles in L0B (for double-buffering).
 * @param c_l0_tile* Tile in L0C for matmuls.
 * @param tile_id Index of the current tile (used for sync).
 */
template <typename InputT, typename TileL1AB, typename TileL0A,
          typename TileL0B, typename TileL0C, uint32_t MatrixSize,
          uint32_t FractalSize, uint32_t DiagonalBlockSize,
          uint32_t NumTilesPerCubeIter>
AICORE inline void InvertSingleTile(TileL1AB X_l1_tile, TileL1AB I_l1_tile,
                                    TileL1AB I_neg_l1_tile,
                                    TileL1AB M_neg_l1_tile,
                                    TileL1AB Zero_l1_tile, TileL1AB Y_l1_tile,
                                    TileL0A* a_l0_tile, TileL0B* b_l0_tile,
                                    TileL0C* c_l0_tile,
                                    const uint32_t tile_id) {
  const event_t event_0 = static_cast<event_t>(tile_id);
  const event_t event_1 = static_cast<event_t>(tile_id + NumTilesPerCubeIter);

  TMOV(b_l0_tile[0], Y_l1_tile);      // b_l0[0] contains M
  TMOV(a_l0_tile[0], I_neg_l1_tile);  // a_l0[0] contains I_neg
  set_flag(PIPE_MTE1, PIPE_M, event_0);
  TMOV(a_l0_tile[1], Zero_l1_tile);
  TMOV(b_l0_tile[1], Zero_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, event_1);
  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  set_flag(PIPE_M, PIPE_MTE1, event_1);
  wait_flag(PIPE_M, PIPE_MTE1, event_1);
  CopyDiagonalBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
      Y_l1_tile, a_l0_tile[1], DiagonalBlockSize);  // a_l0[1] = diag_blocks(M)
  CopyDiagonalBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
      Y_l1_tile, b_l0_tile[1], DiagonalBlockSize);  // b_l0[1] = diag_blocks(M)
  set_flag(PIPE_MTE1, PIPE_M, event_1);

  /* First Matmul: event_0 */
  wait_flag(PIPE_MTE1, PIPE_M, event_0);
  TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // c_l0[0] contains M_neg
  set_flag(PIPE_M, PIPE_FIX, event_0);
  set_flag(PIPE_M, PIPE_MTE1, event_0);

  wait_flag(PIPE_M, PIPE_FIX, event_0);
  TMOV(M_neg_l1_tile, c_l0_tile[0]);  // M_neg_l1 now contains M_neg
  set_flag(PIPE_FIX, PIPE_M, event_0);

  /* Second Matmul: event_1 */
  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  set_flag(PIPE_MTE1, PIPE_M, event_1);
  TMATMUL(c_l0_tile[1], a_l0_tile[1],
          b_l0_tile[1]);  // c_l0[1] contains diag_fractals(M)^2
  set_flag(PIPE_M, PIPE_FIX, event_1);
  wait_flag(PIPE_M, PIPE_FIX, event_1);
  TMOV(Y_l1_tile,
       c_l0_tile[1]);  // Y_l1 now contains diag_fractals(M)^2
  set_flag(PIPE_FIX, PIPE_M, event_1);
  wait_flag(PIPE_FIX, PIPE_M, event_1);

  /* Third Matmul: event_0*/
  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  TMOV(b_l0_tile[0], I_neg_l1_tile);  // b_l0[0] contains I_neg
  TMOV(a_l0_tile[0], I_neg_l1_tile);  // a_l0[0] contains I_neg
  set_flag(PIPE_MTE1, PIPE_M, event_0);

  wait_flag(PIPE_MTE1, PIPE_M, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);
  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  TMATMUL(c_l0_tile[0], a_l0_tile[1],
          b_l0_tile[0]);  // c_l0[0] = diag_fractals(M_neg)
  set_flag(PIPE_M, PIPE_FIX, event_0);
  wait_flag(PIPE_M, PIPE_FIX, event_0);
  set_flag(PIPE_FIX, PIPE_M, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);

  TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0],
              b_l0_tile[0]);  // c_l0[0] has I-diag_fractals(M)
  set_flag(PIPE_M, PIPE_FIX, event_1);
  wait_flag(PIPE_M, PIPE_FIX, event_1);
  TMOV(X_l1_tile, c_l0_tile[0]);  // X_l1 now contains I-diag_fractals(M)

  /*
   * Inv Trick part:
   * X = I - D
   * Y = D^2
   * block_size = 1
   * while block_size < DiagonalBlockSize / 2:
   *     X = X + X @ Y
   *     Y = Y @ Y
   *     block_size *= 2
   */
  set_flag(PIPE_FIX, PIPE_M, event_0);   // store c
  set_flag(PIPE_M, PIPE_MTE1, event_0);  // load matrices for matmuls
  set_flag(PIPE_FIX, PIPE_MTE1, event_0);
  set_flag(PIPE_FIX, PIPE_M, event_1);     // only for update Y
  set_flag(PIPE_M, PIPE_MTE1, event_1);    // only for update Y
  set_flag(PIPE_FIX, PIPE_MTE1, event_1);  // only for update Y
  for (uint32_t block_size = 1; block_size < DiagonalBlockSize / 2;
       block_size *= 2) {
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    TMOV(b_l0_tile[0], I_l1_tile);
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
    TMOV(a_l0_tile[0], X_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
    TMOV(b_l0_tile[1], Y_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_1);

    wait_flag(PIPE_FIX, PIPE_M, event_0);   // from previous iter
    wait_flag(PIPE_MTE1, PIPE_M, event_0);  // from loading a_l0[0], b_l0[0]
    // c_l0[0] already holds X: the previous level's TMATMUL_ACC left it
    // there and the TMOV to X_l1 only reads it, so X @ Y accumulates
    // straight onto it. X also stays in the FP32 accumulator across levels
    // instead of round-tripping through the FP16 X_l1 copy.

    if (block_size <
        DiagonalBlockSize / 4) {  // Update Y except in last iteration
      wait_flag(PIPE_M, PIPE_MTE1, event_1);  // from previous iter
      TMOV(a_l0_tile[1], Y_l1_tile);
      wait_flag(PIPE_MTE1, PIPE_M, event_1);
      set_flag(PIPE_MTE1, PIPE_M, event_1);

      wait_flag(PIPE_MTE1, PIPE_M, event_1);
      wait_flag(PIPE_FIX, PIPE_M, event_1);  // from previous iter
      TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]);
      set_flag(PIPE_M, PIPE_MTE1, event_1);  // for next iter
      set_flag(PIPE_M, PIPE_FIX, event_1);
      set_flag(PIPE_MTE1, PIPE_M, event_1);

      wait_flag(PIPE_M, PIPE_FIX, event_1);
      TMOV(Y_l1_tile, c_l0_tile[1]);
      set_flag(PIPE_FIX, PIPE_M, event_1);  // for next iter
    }
    set_flag(PIPE_FIX, PIPE_MTE1, event_1);  // for next iter

    wait_flag(PIPE_MTE1, PIPE_M, event_1);
    TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0],
                b_l0_tile[1]);  // c_l0[0] has X + X @ Y
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    set_flag(PIPE_M, PIPE_FIX, event_0);

    wait_flag(PIPE_M, PIPE_FIX, event_0);
    TMOV(X_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_FIX, PIPE_M, event_0);     // for next iter
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);  // for next iter
  }
  wait_flag(PIPE_FIX, PIPE_MTE1, event_1);  // only for update Y
  wait_flag(PIPE_M, PIPE_MTE1, event_1);    // only for update Y
  wait_flag(PIPE_FIX, PIPE_M, event_1);     // only for update Y
  wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);

  if constexpr (MatrixSize > DiagonalBlockSize) {
    constexpr uint32_t NumDiagonalBlocks = MatrixSize / DiagonalBlockSize;

    // b_l0[0] = -N: start from -M and zero D's diagonal blocks.
    TMOV(a_l0_tile[0], X_l1_tile);      // rounded Xd
    TMOV(b_l0_tile[0], M_neg_l1_tile);  // -M
    CopyDiagonalBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
        Zero_l1_tile, b_l0_tile[0], DiagonalBlockSize);
    set_flag(PIPE_MTE1, PIPE_M, event_0);
    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // -Xd N
    set_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    TMOV(M_neg_l1_tile, c_l0_tile[0]);  // rounded -Xd N
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);

    // X = I - Xd N. The negative is already in the accumulator, so the
    // identity needs only one accumulated matmul.
    TMOV(a_l0_tile[0], I_neg_l1_tile);
    TMOV(b_l0_tile[0], I_neg_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);
    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);
    set_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    TMOV(Y_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);

    if constexpr (NumDiagonalBlocks > 2) {
      // Y = (Xd N)^2. Squaring -Xd N removes its sign.
      TMOV(a_l0_tile[1], M_neg_l1_tile);
      TMOV(b_l0_tile[1], M_neg_l1_tile);
      set_flag(PIPE_MTE1, PIPE_M, event_1);
      wait_flag(PIPE_MTE1, PIPE_M, event_1);
      TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]);
      set_flag(PIPE_M, PIPE_FIX, event_1);
      set_flag(PIPE_M, PIPE_MTE1, event_1);
      wait_flag(PIPE_M, PIPE_FIX, event_1);
      wait_flag(PIPE_M, PIPE_MTE1, event_1);
      TMOV(M_neg_l1_tile, c_l0_tile[1]);
      set_flag(PIPE_FIX, PIPE_MTE1, event_1);
      set_flag(PIPE_FIX, PIPE_M, event_1);
      wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
      wait_flag(PIPE_FIX, PIPE_M, event_1);

      // X = (I - Xd N)(I + (Xd N)^2)(I + (Xd N)^4)... . Xd N
      // is strictly block triangular, so its NumDiagonalBlocks-th power is 0.
      for (uint32_t block_step = 1; block_step < NumDiagonalBlocks / 2;
           block_step *= 2) {
        TMOV(a_l0_tile[0], Y_l1_tile);
        TMOV(b_l0_tile[0], M_neg_l1_tile);
        set_flag(PIPE_MTE1, PIPE_M, event_0);
        wait_flag(PIPE_MTE1, PIPE_M, event_0);
        TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);
        set_flag(PIPE_M, PIPE_FIX, event_0);
        set_flag(PIPE_M, PIPE_MTE1, event_0);
        wait_flag(PIPE_M, PIPE_FIX, event_0);
        wait_flag(PIPE_M, PIPE_MTE1, event_0);
        TMOV(Y_l1_tile, c_l0_tile[0]);
        set_flag(PIPE_FIX, PIPE_MTE1, event_0);
        set_flag(PIPE_FIX, PIPE_M, event_0);
        wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
        wait_flag(PIPE_FIX, PIPE_M, event_0);

        if (block_step < NumDiagonalBlocks / 4) {
          TMOV(a_l0_tile[1], M_neg_l1_tile);
          TMOV(b_l0_tile[1], M_neg_l1_tile);
          set_flag(PIPE_MTE1, PIPE_M, event_1);
          wait_flag(PIPE_MTE1, PIPE_M, event_1);
          TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]);
          set_flag(PIPE_M, PIPE_FIX, event_1);
          set_flag(PIPE_M, PIPE_MTE1, event_1);
          wait_flag(PIPE_M, PIPE_FIX, event_1);
          wait_flag(PIPE_M, PIPE_MTE1, event_1);
          TMOV(M_neg_l1_tile, c_l0_tile[1]);
          set_flag(PIPE_FIX, PIPE_MTE1, event_1);
          set_flag(PIPE_FIX, PIPE_M, event_1);
          wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
          wait_flag(PIPE_FIX, PIPE_M, event_1);
        }
      }
    }

    // (I + M)^-1 Xd, M = Xd N.
    TMOV(a_l0_tile[0], Y_l1_tile);
    TMOV(b_l0_tile[0], X_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);
    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);
    set_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
  }
}

/**
 * @brief: Runs the main kernel (inverts all matrices in the tensor)
 *
 * @tparam InputT The type of the input elements. Supports fp16 and bf16.
 * @tparam OutputT The type of the output elements.
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam NumTilesPerCubeIter How many matrices to load and invert in a single
 * cube iteration.
 * @tparam IsBSND If IsBSND is false, then the last two dimensions represent a
 * 2D triangular matrix in row-major format, while the other dimensions are
 * batch dimensions. If IsBSND is true, then the dimensions represent in order:
 * B batch size, S sequence length (which is chunked in tiles of size D), N
 * number of heads (equivalent to a second batch dimension for this kernel), and
 * D chunk size. The inverse is over the dimensions S (chunked) and D, row-major
 * within each tile.
 *
 * @param M_inv pointer to the global memory to store the final inverse.
 * @param M Pointer to the global tensor matrix in global memory.
 * @param I_neg Pointer to global memory that contains the negative identity.
 * @param total_tiles The total number of matrices to invert.
 * @param num_bsnd_heads The number of heads, only for BSND format.
 * @param is_lower If input matrices are lower-triangular (is_lower == 1) or
 * upper-triangular (is_lower == 0). Default is upper triangular.
 * @param num_bsnd_heads The number of heads, only for BSND format.
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE inline void TriInvRecUnrollKernel(__gm__ OutputT* M_inv,
                                         __gm__ InputT* M, __gm__ InputT* I_neg,
                                         uint32_t total_tiles,
                                         uint32_t num_bsnd_heads = 0,
                                         uint32_t is_lower = 0,
                                         __gm__ int32_t* cu_seqlens = nullptr) {
  using pto::Stride;

  /* Initializations */
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  constexpr uint32_t FractalSize = 16;  // fractal size for half /bf16
  // The fp16 doubling operands only contain powers within a small diagonal
  // block. The cross-block matrix is nilpotent in MatrixSize / block steps, so
  // the second doubling phase is bounded independently of the input norm.
  constexpr uint32_t DiagonalBlockSize =
      MatrixSize < TRI_INV_DIAGONAL_BLOCK ? MatrixSize
                                          : TRI_INV_DIAGONAL_BLOCK;
  static_assert(DiagonalBlockSize >= FractalSize);
  static_assert((DiagonalBlockSize & (DiagonalBlockSize - 1)) == 0);
  static_assert(MatrixSize % DiagonalBlockSize == 0);
  constexpr uint32_t NumFractalsRowWise = MatrixSize / FractalSize;
  constexpr uint32_t NumL0Buffers = 2;

  if (get_block_idx() * NumTilesPerCubeIter >= total_tiles) {
    return;
  }

  using GlobalTileShapeIn =
      TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileStridesIn = typename std::conditional<
      !IsBSND, BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>,
      pto::Stride<1, 1, 1, -1, 1>>::type;
  using GlobalTileIn =
      GlobalTensor<InputT, GlobalTileShapeIn, GlobalTileStridesIn, Layout::ND>;
  using GlobalTileDynamicShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GlobalTileDynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalTileDynamicIn = GlobalTensor<InputT, GlobalTileDynamicShape,
                                           GlobalTileDynamicStride, Layout::ND>;
  using GlobalTileStridesINeg =
      BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileINeg = GlobalTensor<InputT, GlobalTileShapeIn,
                                      GlobalTileStridesINeg, Layout::ND>;

  using GlobalTileShapeOut =
      TileShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileStridesOut = typename std::conditional<
      !IsBSND, BaseShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>,
      pto::Stride<1, 1, 1, -1, 1>>::type;
  using GlobalTileOut = GlobalTensor<OutputT, GlobalTileShapeOut,
                                     GlobalTileStridesOut, Layout::ND>;
  using GlobalTileDynamicOut =
      GlobalTensor<OutputT, GlobalTileDynamicShape, GlobalTileDynamicStride,
                   Layout::ND>;
  using TileL1AB =
      Tile<TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor,
           MatrixSize, MatrixSize, SLayout::RowMajor, 512>;
  using TileL1ABDynamic =
      Tile<TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor,
           DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  // L0 Memory
  using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
  using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
  using TileL0C = TileAcc<float, MatrixSize, MatrixSize>;
  using TileL0CDynamic =
      TileAcc<float, MatrixSize, MatrixSize, DYNAMIC, DYNAMIC>;

  GlobalTileINeg I_neg_global_in(I_neg);

  TileL1AB X_l1_tile;
  TileL1AB I_l1_tile;
  TileL1AB I_neg_l1_tile;
  TileL1AB M_neg_l1_tile;
  TileL1AB Zero_l1_tile;
  TileL1AB Y_l1_tile[NumTilesPerCubeIter];

  TileL0A a_l0_tile[NumL0Buffers];
  TileL0B b_l0_tile[NumL0Buffers];
  TileL0C c_l0_tile[NumL0Buffers];

  TASSIGN(I_l1_tile, 0x0);
  TASSIGN(I_neg_l1_tile, 0x0 + TileLen * sizeof(InputT));
  TASSIGN(Zero_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));
  TASSIGN(M_neg_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
  TASSIGN(X_l1_tile, 0x0 + 4 * TileLen * sizeof(InputT));
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    TASSIGN(Y_l1_tile[tile_id], 0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
  }

  for (uint32_t buffer_num = 0; buffer_num < NumL0Buffers; ++buffer_num) {
    TASSIGN(a_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(b_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(c_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(float));
  }
  TLOAD(I_neg_l1_tile, I_neg_global_in);
  set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));

  PrepareAuxiliaryMatrices<TileL1AB, TileL0A, TileL0B, TileL0C>(
      I_neg_l1_tile, Zero_l1_tile, I_l1_tile, a_l0_tile[0], b_l0_tile[0],
      c_l0_tile[0]);

  const uint32_t max_iters_per_aic =
      CeilDiv(total_tiles, (uint32_t)(NumTilesPerCubeIter * get_block_num()));

  /* Main iteration - Compute all tiles */
  uint32_t bsnd_tile_offsets[NumTilesPerCubeIter] = {0};
  uint32_t bsnd_tile_valid_sizes[NumTilesPerCubeIter] = {0};
  uint32_t next_tile_id_that_waits_for_pipe_fix_pipe_m = 0;
  set_flag(PIPE_FIX, PIPE_M,
           static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
  }
  for (uint32_t cube_iter = 0; cube_iter < max_iters_per_aic; ++cube_iter) {
    const uint32_t global_index =
        (cube_iter * get_block_num() + get_block_idx()) * NumTilesPerCubeIter;
    if (global_index >= total_tiles) {
      break;
    }
    for (uint32_t tile_id = 0; (tile_id < NumTilesPerCubeIter) &&
                               (global_index + tile_id < total_tiles);
         ++tile_id) {
      if constexpr (IsBSND) {
        const uint32_t global_tile_id = global_index + tile_id;
        if (cu_seqlens != nullptr) {
          const BSNDVarlenTileInfo tile_info =
              GetBSNDVarlenTileInfoFromCuSeqlens(global_tile_id, num_bsnd_heads,
                                                 MatrixSize, cu_seqlens);
          bsnd_tile_offsets[tile_id] = tile_info.bsnd_offset;
          bsnd_tile_valid_sizes[tile_id] = tile_info.valid_size;
        } else {
          bsnd_tile_offsets[tile_id] = GetBSNDFixedTileOffset(
              global_tile_id, num_bsnd_heads, MatrixSize);
          bsnd_tile_valid_sizes[tile_id] = MatrixSize;
        }
        const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
        const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
        const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        if (valid_size < MatrixSize) {
          TileL1ABDynamic Y_dyn_l1_tile(valid_size, valid_size);
          TASSIGN(Y_dyn_l1_tile,
                  0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
          GlobalTileDynamicIn M_global_in_dyn(
              M + bsnd_offset,
              {1, 1, 1, static_cast<int>(valid_size),
               static_cast<int>(valid_size)},
              {1, 1, 1, row_stride, 1});
          TLOAD(Y_dyn_l1_tile, M_global_in_dyn);
          set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
          wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
          TFILLPAD(Y_dyn_l1_tile, Y_dyn_l1_tile);
        } else {
          GlobalTileIn M_global_in(M + bsnd_offset, {}, {row_stride});
          TLOAD(Y_l1_tile[tile_id], M_global_in);
        }
      } else {
        GlobalTileIn M_global_in(M + (global_index + tile_id) * TileLen);
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        TLOAD(Y_l1_tile[tile_id],
              M_global_in);  // Copies NumTilesPerCubeIter tiles at once
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
    }

    constexpr uint32_t final_c_buffer_index = 0;
    for (uint32_t tile_id = 0; (tile_id < NumTilesPerCubeIter) &&
                               (global_index + tile_id < total_tiles);
         ++tile_id) {
      // Wait for previous cube iter to write result
      wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(tile_id));
      // Wait for loading new matrices from GM
      wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));

      InvertSingleTile<InputT, TileL1AB, TileL0A, TileL0B, TileL0C, MatrixSize,
                       FractalSize, DiagonalBlockSize, NumTilesPerCubeIter>(
          X_l1_tile, I_l1_tile, I_neg_l1_tile, M_neg_l1_tile, Zero_l1_tile,
          Y_l1_tile[tile_id], a_l0_tile, b_l0_tile, c_l0_tile, tile_id);

      // Allow next cube_iter to proceed for this tile_id
      set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));

      /* Store result */
      if constexpr (IsBSND) {
        const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
        const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
        const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
        if (valid_size < MatrixSize) {
          TileL0CDynamic c_l0_tail_tile(valid_size, valid_size);
          TASSIGN(c_l0_tail_tile,
                  0x0 + final_c_buffer_index * TileLen * sizeof(float));
          GlobalTileDynamicOut M_inv_global_out_dyn(
              M_inv + bsnd_offset,
              {1, 1, 1, static_cast<int>(valid_size),
               static_cast<int>(valid_size)},
              {1, 1, 1, row_stride, 1});
          TSTORE(M_inv_global_out_dyn, c_l0_tail_tile);
        } else {
          GlobalTileOut M_inv_global_out(M_inv + bsnd_offset, {}, {row_stride});
          TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
        }
      } else {
        GlobalTileOut M_inv_global_out(M_inv +
                                       (global_index + tile_id) * TileLen);
        TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
      }
      next_tile_id_that_waits_for_pipe_fix_pipe_m =
          (tile_id + 1) % NumTilesPerCubeIter;
      set_flag(
          PIPE_FIX, PIPE_M,
          static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
    }
  }
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
  }
  wait_flag(PIPE_FIX, PIPE_M,
            static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
}

/*
 * @brief: Computes the inverses of the blocks of tensor M
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE void runKernelTriInvRecUnroll(__gm__ OutputT* M_inv, __gm__ InputT* M,
                                     __gm__ InputT* I_neg, uint32_t total_tiles,
                                     uint32_t num_bsnd_heads = 0,
                                     uint32_t is_lower = 0,
                                     __gm__ int32_t* cu_seqlens = nullptr) {
#if defined(__DAV_CUBE__)  // Cube compilation

  TriInvRecUnrollKernel<InputT, OutputT, MatrixSize, NumTilesPerCubeIter,
                        IsBSND>(M_inv, M, I_neg, total_tiles, num_bsnd_heads,
                                is_lower, cu_seqlens);
#else
// Nothing to do on AIV
#endif
}

template <typename InputT, typename OutputT, uint32_t NumTilesPerCubeIter,
          bool IsBSND>
AICORE void run_tri_inv_rec_unroll(__gm__ OutputT* tensor_out,
                                   __gm__ InputT* tensor_in,
                                   __gm__ InputT* minus_eye_in,
                                   uint32_t matrix_size, uint32_t num_matrices,
                                   uint32_t num_bsnd_heads,
                                   uint32_t is_lower = 0,
                                   __gm__ int32_t* cu_seqlens = nullptr) {
  static_assert(
      std::is_same_v<InputT, half> or std::is_same_v<InputT, bfloat16_t>,
      "tri_inv_rec_unroll supports only fp16 or bf16.");

  static_assert(
      std::is_same_v<OutputT, float> or std::is_same_v<OutputT, bfloat16_t>,
      "tri_inv_rec_unroll supports only fp32 or bf16.");
  switch (matrix_size) {
    case 16:
      runKernelTriInvRecUnroll<InputT, OutputT, 16, NumTilesPerCubeIter,
                               IsBSND>(tensor_out, tensor_in, minus_eye_in,
                                       num_matrices, num_bsnd_heads, is_lower,
                                       cu_seqlens);
      break;
    case 32:
      runKernelTriInvRecUnroll<InputT, OutputT, 32, NumTilesPerCubeIter,
                               IsBSND>(tensor_out, tensor_in, minus_eye_in,
                                       num_matrices, num_bsnd_heads, is_lower,
                                       cu_seqlens);
      break;
    case 64:
      runKernelTriInvRecUnroll<InputT, OutputT, 64, NumTilesPerCubeIter,
                               IsBSND>(tensor_out, tensor_in, minus_eye_in,
                                       num_matrices, num_bsnd_heads, is_lower,
                                       cu_seqlens);
      break;
    case 128:
      runKernelTriInvRecUnroll<InputT, OutputT, 128, NumTilesPerCubeIter,
                               IsBSND>(tensor_out, tensor_in, minus_eye_in,
                                       num_matrices, num_bsnd_heads, is_lower,
                                       cu_seqlens);
      break;
  }
}

/*
 * @brief: Wrapper for the kernel supporting fp16 and bfloat16 types.
 *
 * @param tensor_out pointer to the global memory to store the final inverse.
 * @param tensor_in Pointer to the global tensor matrix in global memory.
 * @param minus_eye_in Pointer to the global tensor matrix containing the
 * negative identity matrix.
 * @param matrix_size The size if each individual matrix / tile. Can take
 * values: {16, 32, 64, 128}.
 * @param num_matrices The total number of matrices / tiles in the global
 * tensor.
 * @param num_bsnd_heads The number of heads, which is only greater than zero
 * if the matrix is in BSND format, that is, the tiles need to be loaded with
 * strided accesses. If each tile is stored consecutively (and row-wise) in
 * memory, then num_bsnd_heads=0.
 */
template <typename InputT, typename OutputT, uint32_t NumTilesPerCubeIter>
AICORE void run_tri_inv_rec_unroll_per_num_matrices(
    __gm__ OutputT* tensor_out, __gm__ InputT* tensor_in,
    __gm__ InputT* minus_eye_in, uint32_t matrix_size, uint32_t num_matrices,
    uint32_t num_bsnd_heads, uint32_t is_lower = 0,
    __gm__ int32_t* cu_seqlens = nullptr) {
  if (num_bsnd_heads == 0) {
    if (num_matrices <= get_block_num()) {
      run_tri_inv_rec_unroll<InputT, OutputT, 1 /* NumTilesPerCubeIter */,
                             false /* IsBSND */>(
          tensor_out, tensor_in, minus_eye_in, matrix_size, num_matrices,
          num_bsnd_heads, is_lower, cu_seqlens);
    } else if (num_matrices <= 2 * get_block_num()) {
      run_tri_inv_rec_unroll<InputT, OutputT, 2 /* NumTilesPerCubeIter */,
                             false /* IsBSND */>(
          tensor_out, tensor_in, minus_eye_in, matrix_size, num_matrices,
          num_bsnd_heads, is_lower, cu_seqlens);
    } else {
      run_tri_inv_rec_unroll<InputT, OutputT, 4 /* NumTilesPerCubeIter */,
                             false /* IsBSND */>(
          tensor_out, tensor_in, minus_eye_in, matrix_size, num_matrices,
          num_bsnd_heads, is_lower, cu_seqlens);
    }
  } else {
    if (num_matrices <= get_block_num()) {
      run_tri_inv_rec_unroll<InputT, OutputT, 1 /* NumTilesPerCubeIter */,
                             true /* IsBSND */>(
          tensor_out, tensor_in, minus_eye_in, matrix_size, num_matrices,
          num_bsnd_heads, is_lower, cu_seqlens);
    } else if (num_matrices <= 2 * get_block_num()) {
      run_tri_inv_rec_unroll<InputT, OutputT, 2 /* NumTilesPerCubeIter */,
                             true /* IsBSND */>(
          tensor_out, tensor_in, minus_eye_in, matrix_size, num_matrices,
          num_bsnd_heads, is_lower, cu_seqlens);
    } else {
      run_tri_inv_rec_unroll<InputT, OutputT, 4 /* NumTilesPerCubeIter */,
                             true /* IsBSND */>(
          tensor_out, tensor_in, minus_eye_in, matrix_size, num_matrices,
          num_bsnd_heads, is_lower, cu_seqlens);
    }
  }
}

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
 * @brief Host-side launcher (called from Python via ctypes).
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
