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
// Cross-core sync protocol (Cube ↔ Vec via FFTS), three flags:
//   flag 0: Cube→Vec  — QK and QS results ready in this item's mailbox slots
//   flag 1: Vec→Cube  — QK_gated written back, Cube can proceed to GEMM 3
//   flag 2: Cube→Vec  — QKV result ready in this item's slot
//
// There is no "workspace free" flag. Each work item owns its own slot in a
// ring of GDN_O_PRE_LAUNCH-sized depth, so the Cube can run ahead of the Vec
// instead of waiting for it to release a single shared buffer.
//
// NPU memory hierarchy used:
//   GM → L1 (Cube-accessible) → L0A/L0B (matrix engines) → L0C (accumulator)
//   GM → UB (Vec-accessible, on-chip SRAM)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel combines matrix multiplication (Cube) with element-wise gating
// (Vec) in a tightly coordinated 3-GEMM + gating pipeline per chunk.
//
// Execution timeline. Each unit runs a produce step for work item `s` and a
// consume step for item `s - GDN_O_PRE_LAUNCH`, so the two units are working
// on different items at the same time:
//   Cube produce(s):  GEMM1(Q@K^T) → GEMM2(Q@S) → store QK,QS → flag 0 ──┐
//   Vec  produce(s):  load G, compute coefficients (before the wait)      │
//   Vec  produce(s):  ←── flag 0 ──── gate QK → store QK_gated → flag 1 ─┐│
//   Cube consume(s'): ←── flag 1 ──── GEMM3(QK_gated@V) → store → flag 2 │
//   Vec  consume(s'): ←── flag 2 ──── scale QS, O = QKV + QS_g → store   │
// No item hands a workspace back: each owns its slot in the mailbox ring.
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

#include <runtime/rt_ffts.h>

#include <pto/pto-inst.hpp>

#include "acl/acl.h"
#include "kernel_utils.h"
using namespace pto;
using namespace kernel_utils;

// ── Compile-time configuration (overridable at build time via -D flags) ──
// D/C stay compile-time because tile shapes depend on them. H/Hg are runtime.
#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

// ── PTO type aliases (device-only, guarded for host pass safety) ────────────
// The bisheng compiler performs 3 passes: vec core, cube core (__CCE_AICORE__
// defined), and host (__CCE_AICORE__ NOT defined). Type aliases using PTO
// tile types must be guarded so the host pass never sees them.
#ifdef __CCE_AICORE__

// UbND = Unified Buffer tile, row-major (ND) layout, for Vec SIMD ops.
//   Like torch.empty((R, C), dtype=T) in fast on-chip SRAM (~256KB).
//   RV, CV = valid region (handles dynamic shapes, partial chunks).
//   PadValue::Zero = fill with 0 outside valid region during TLOAD.
// T=dtype, R×C=static shape, RV×CV=valid region, P=pad fill for TLOAD.
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor, RV,
                       CV, pto::SLayout::NoneBox, 512, P>;

// UbDN = UB tile in column-major (DN) layout.
//   Needed as source for TROWEXPAND which requires column-format input.
//   TROWEXPAND takes a column vector and broadcasts it across all columns
//   of a destination ND tile: dst[i,j] = col[i] for all j.
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor, RV,
                       CV, pto::SLayout::NoneBox, 512>;

// L1Mat = L1 cache tile in NZ fractal format — standard Cube GEMM input.
//   Data is loaded here from GM via TLOAD, then fed to L0A/L0B via TEXTRACT.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor, RV,
                        CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1MatZN = ZN fractal format — used for transposed GEMM operands.
//   TRESHAPE(l1_zn, l1_nz) converts NZ→ZN = logical matrix transpose (free, no
//   data movement).
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN =
    pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor, RV, CV,
              pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

#endif  // __CCE_AICORE__

// ── Cube/Vec pipeline depth ───────────────────────────────────────────
// The Cube runs GDN_O_PRE_LAUNCH work items ahead of the Vec, so each unit's
// hand-off wait overlaps the other's work. Work items are independent
// (chunk, head) pairs, which is what makes the run-ahead legal.
//
// The mailbox ring is 2 * PRE_LAUNCH + 2 slots, not PRE_LAUNCH + 1. QS is
// written at the produce step and read at the consume step, so it outlives the
// other three mailboxes, and the Cube's consume of step s only establishes
// that Vec has consumed up to s - PRE_LAUNCH - 1. That puts the boundary at
// 2 * PRE_LAUNCH + 1, which frees a slot in the very iteration it is read; the
// extra slot is the margin.
//
// 0 restores the lock-step form: produce and consume in the same iteration,
// each still owning its own slot. That alone is worth most of the speedup.
//
// Validated on A2/A3 only. On A5 the Cube and both Vec sub-blocks share one
// core and the hand-offs become set_intra_block / wait_intra_block; running
// ahead needs those to count outstanding signals, which has not been checked
// on hardware. Set this to 0 if A5 misbehaves.
#ifndef GDN_O_PRE_LAUNCH
#define GDN_O_PRE_LAUNCH 1
#endif

constexpr int64_t ChunkOPreLaunch = GDN_O_PRE_LAUNCH;
constexpr int64_t ChunkOSlots = 2 * ChunkOPreLaunch + 2;
constexpr int64_t ChunkORing = ChunkOPreLaunch + 1;

// ── UB memory map (byte addresses within Unified Buffer) ──────────────
constexpr int32_t GUbAddr = 0;
constexpr int32_t MskUbAddr = 512;
constexpr int32_t QKUbAddr = 33280;
constexpr int32_t CoeffUbAddr = 66304;
constexpr int32_t QKHalfUbAddr = 99072;
constexpr int32_t QSHalfUbAddr = 115456;
constexpr int32_t QSUbAddr = 131840;
constexpr int32_t OHalfUbAddr = 164608;
constexpr int32_t OUbAddr = QKUbAddr;

// One (chunk, head) work item, carried from its produce step to its consume
// step ChunkOPreLaunch iterations later.
struct ChunkOItem {
  int32_t valid_rows;
  int32_t local_rows;
  int32_t head_idx;
  int64_t qk_off;
  int64_t v_off;
  int64_t s_offset;
  int64_t chunk_token_start;
};

// Mailbox slot offsets. qk and gated get one ring per core; the qs workspace
// holds two, because QS must survive until its consume step while QKV is
// written into the same buffer at that step.
AICORE inline int64_t ChunkOSlotOff(int64_t cid, int64_t step, int32_t size) {
  return (cid * ChunkOSlots + step % ChunkOSlots) * static_cast<int64_t>(size);
}

AICORE inline int64_t ChunkOSlotOff2(int64_t cid, int64_t step, int32_t size,
                                     bool second) {
  int64_t base = cid * 2 * ChunkOSlots + (second ? ChunkOSlots : 0);
  return (base + step % ChunkOSlots) * static_cast<int64_t>(size);
}

// seq_len tokens each; otherwise the bounds come from cu_seqlens.
AICORE inline void SeqBounds(__gm__ int32_t *cu_seqlens, int64_t si,
                             int64_t seq_len, int64_t &bos, int64_t &slen) {
  if (cu_seqlens == nullptr) {
    bos = si * seq_len;
    slen = seq_len;
  } else {
    bos = static_cast<int64_t>(cu_seqlens[si]);
    slen = static_cast<int64_t>(cu_seqlens[si + 1]) - bos;
  }
}

#if defined(__DAV_CUBE__)

// Q @ K^T and Q @ S for one work item, into that item's mailbox slots.
template <int32_t HiddenSize, int32_t ChunkSize>
AICORE inline void ChunkOCubeProduce(
    __gm__ half *Q_handle, __gm__ half *K_handle, __gm__ half *S_handle,
    __gm__ half *workspace_qk_handle, __gm__ half *workspace_qs_qkv_handle,
    int32_t BSND_QK_STRIDE, int32_t valid_rows, int64_t qk_off,
    int64_t s_offset, int64_t ws_qk_off, int64_t ws_qs_off) {
  using pto::Stride;
  L1Mat<half, ChunkSize, HiddenSize> q_l1;
  TASSIGN(q_l1, 0);
  L1Mat<half, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 32768);
  L1Mat<half, HiddenSize, HiddenSize> s_l1;
  TASSIGN(s_l1, 65536);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> qk_l0;
  TASSIGN(qk_l0, 0);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qs_l0;
  TASSIGN(qs_l0, 65536);

  // ── Load Q [valid_rows × D] from GM → L1 ────────────────────────
  // GlobalTensor describes the GM layout with BSND strides.
  // TLOAD performs DMA (MTE2 pipe). TFILLPAD zero-pads tail rows so
  // downstream GEMMs see a clean C×D matrix.
  {
    L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows,
                                                             HiddenSize);
    TASSIGN(_l1, 0);
    GmShape2D _gs(valid_rows, HiddenSize);
    GmStride2D _stride(BSND_QK_STRIDE);
    GmTensor2D<half> _gm(Q_handle + qk_off, _gs, _stride);
    TLOAD(_l1, _gm);
    if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
  }
  // ── Load K [valid_rows × D] from GM → L1 ────────────────────────
  {
    L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows,
                                                             HiddenSize);
    TASSIGN(_l1, 32768);
    GmShape2D _gs(valid_rows, HiddenSize);
    GmStride2D _stride(BSND_QK_STRIDE);
    GmTensor2D<half> _gm(K_handle + qk_off, _gs, _stride);
    TLOAD(_l1, _gm);
    if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
  }

  // ── GEMM 1: QK = Q @ K^T  (intra-chunk attention scores) ────────
  // ── GEMM 1: QK = Q @ K^T ─────────────────────────────────────────
  // numpy: QK = Q @ K.T  →  [C×D] @ [D×C] = [C×C]
  //
  // How transpose works on NPU:
  //   K is loaded into L1 in NZ (col-major fractal) format.
  //   TRESHAPE(l1_zn, k_l1) reinterprets it as ZN (row-major fractal) =
  //   K^T. This is a ZERO-COST operation — no data movement, just metadata
  //   change. TEXTRACT then loads the transposed view into L0B.
  //
  // Cube GEMM pipeline:
  //   TEXTRACT(l0a, q_l1, 0, 0)  — Q → L0A (left operand)
  //   TEXTRACT(l0b, k_zn, 0, 0)  — K^T → L0B (right operand)
  //   TMATMUL(qk_l0, l0a, l0b)   — QK = L0A × L0B → L0C accumulator
  //
  // transpose_B: TRESHAPE converts k_l1 from NZ → ZN fractal layout,
  // effectively transposing K before TEXTRACT loads it into L0B.
  {
    TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
    TileRight<half, HiddenSize, ChunkSize, HiddenSize, ChunkSize> _l0b;
    TASSIGN(_l0a, 0x0);
    TASSIGN(_l0b, 0x0);
    auto _we = EVENT_ID1;
    set_flag(PIPE_MTE2, PIPE_MTE1, _we);
    wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
    set_flag(PIPE_M, PIPE_MTE1, _we);
    wait_flag(PIPE_M, PIPE_MTE1, _we);
    TEXTRACT(_l0a, q_l1, 0, 0);
    L1MatZN<half, HiddenSize, ChunkSize> _bzn;
    TRESHAPE(_bzn, k_l1);
    TEXTRACT(_l0b, _bzn, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, _we);
    wait_flag(PIPE_MTE1, PIPE_M, _we);
    TMATMUL(qk_l0, _l0a, _l0b);
    set_flag(PIPE_MTE1, PIPE_MTE2, _we);
    wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
    set_flag(PIPE_M, PIPE_FIX, _we);
    wait_flag(PIPE_M, PIPE_FIX, _we);
  }

  // ── Load S [D × D] from GM → L1  (accumulated hidden state) ─────
  {
    L1Mat<half, HiddenSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(HiddenSize,
                                                              HiddenSize);
    TASSIGN(_l1, 65536);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = HiddenSize;
    _gs.shape[4] = HiddenSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
        S_handle + s_offset, _gs);
    TLOAD(_l1, _gm);
  }

  // ── GEMM 2: QS = Q @ S  (query applied to accumulated state) ────
  {
    TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
    TileRight<half, HiddenSize, HiddenSize, HiddenSize, HiddenSize> _l0b;
    TASSIGN(_l0a, 0x0);
    TASSIGN(_l0b, 0x0);
    auto _we = EVENT_ID1;
    set_flag(PIPE_MTE2, PIPE_MTE1, _we);
    wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
    set_flag(PIPE_M, PIPE_MTE1, _we);
    wait_flag(PIPE_M, PIPE_MTE1, _we);
    TEXTRACT(_l0a, q_l1, 0, 0);
    TEXTRACT(_l0b, s_l1, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, _we);
    wait_flag(PIPE_MTE1, PIPE_M, _we);
    TMATMUL(qs_l0, _l0a, _l0b);
    set_flag(PIPE_MTE1, PIPE_MTE2, _we);
    wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
    set_flag(PIPE_M, PIPE_FIX, _we);
    wait_flag(PIPE_M, PIPE_FIX, _we);
  }

  // ── Store QK [C × C] from L0C → GM workspace (fp32→fp16 cast) ───
  // TSTORE on TileAcc triggers MTE3 DMA with implicit type conversion.
  {
    TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                               ChunkSize);
    TASSIGN(_l0, 0);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = ChunkSize;
    _gs.shape[4] = ChunkSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        workspace_qk_handle + ws_qk_off, _gs);
    TSTORE(_gm, _l0);
  }

  // ── Store QS [C × D] from L0C → GM workspace ────────────────────
  {
    TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                                HiddenSize);
    TASSIGN(_l0, 65536);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = ChunkSize;
    _gs.shape[4] = HiddenSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
        workspace_qs_qkv_handle + ws_qs_off, _gs);
    TSTORE(_gm, _l0);
  }

  // Signal Vec: QK and QS are ready (flag 0, Cube→Vec)
  // ── Cross-core sync protocol ──────────────────────────────────────
  // Cube and Vec are SEPARATE physical cores. They exchange data through GM
  // and coordinate via FFTS flags. Think of it as two processes
  // communicating through shared memory with semaphores.
  //
  // ffts_cross_core_sync(PIPE_FIX, config):
  //   config = 1 | (mode << 4) | (flag_id << 8)
  //   mode=2: broadcast signal to all cores in this block
  //   flag_id: identifies which signal (0, 1, 2)
  //
  // Protocol for this kernel:
  //   flag 0: Cube→Vec "QK and QS are ready in this item's slots"
  //   flag 1: Vec→Cube "QK_gated is ready for GEMM 3"
  //   flag 2: Cube→Vec "QKV (GEMM 3 result) is ready"
  // ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));
#if __CCE_AICORE__ == 220
  SetCrossFlag<PIPE_FIX>(0);
#else
  pipe_barrier(PIPE_ALL);
  SignalBothVecOnA5<PIPE_FIX>(0);
#endif
}

// QK_gated @ V for one work item, from that item's gated slot.
template <int32_t HiddenSize, int32_t ChunkSize>
AICORE inline void ChunkOCubeConsume(__gm__ half *V_handle,
                                     __gm__ half *workspace_qk_gated_handle,
                                     __gm__ half *workspace_qs_qkv_handle,
                                     int32_t BSND_V_STRIDE, int32_t valid_rows,
                                     int64_t v_off, int64_t ws_gated_off,
                                     int64_t ws_qkv_off) {
  using pto::Stride;
  L1Mat<half, ChunkSize, ChunkSize> qk_gated_l1;
  TASSIGN(qk_gated_l1, 98304);
  L1Mat<half, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 131072);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qkv_l0;
  TASSIGN(qkv_l0, 0);

  // Wait for Vec to write QK_gated back (flag 1, Vec→Cube)
#if __CCE_AICORE__ == 220
  wait_flag_dev(1);
#else
  WaitBothVecOnA5<PIPE_MTE2>(1);
  pipe_barrier(PIPE_ALL);
#endif

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  // ── Load QK_gated [C × C] from GM workspace → L1 ────────────────
  {
    L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize,
                                                            ChunkSize);
    TASSIGN(_l1, 98304);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = ChunkSize;
    _gs.shape[4] = ChunkSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        workspace_qk_gated_handle + ws_gated_off, _gs);
    TLOAD(_l1, _gm);
  }
  // ── Load V [valid_rows × D] from GM → L1 ────────────────────────
  {
    L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows,
                                                             HiddenSize);
    TASSIGN(_l1, 131072);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = valid_rows;
    _gs.shape[4] = HiddenSize;
    GmStride2D _stride(BSND_V_STRIDE);
    GmTensor2D<half> _gm(V_handle + v_off, _gs, _stride);
    TLOAD(_l1, _gm);
    if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
  }

  // ── GEMM 3: QKV = QK_gated @ V  (gated attention → values) ──────
  {
    TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
    TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
    TASSIGN(_l0a, 0x0);
    TASSIGN(_l0b, 0x0);
    auto _we = EVENT_ID1;
    set_flag(PIPE_MTE2, PIPE_MTE1, _we);
    wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
    set_flag(PIPE_M, PIPE_MTE1, _we);
    wait_flag(PIPE_M, PIPE_MTE1, _we);
    TEXTRACT(_l0a, qk_gated_l1, 0, 0);
    TEXTRACT(_l0b, v_l1, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, _we);
    wait_flag(PIPE_MTE1, PIPE_M, _we);
    TMATMUL(qkv_l0, _l0a, _l0b);
    set_flag(PIPE_MTE1, PIPE_MTE2, _we);
    wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
    set_flag(PIPE_M, PIPE_FIX, _we);
    wait_flag(PIPE_M, PIPE_FIX, _we);
  }

  // ── Store QKV [C × D] from L0C → GM workspace ───────────────────
  // ── Workspace buffer reuse ────────────────────────────────────────
  // workspace_qs_qkv_handle is shared between QS (GEMM 2 output) and QKV
  // (GEMM 3 output). This is safe because:
  //   1. Vec reads QS BEFORE Cube writes QKV to the same buffer
  //   2. The cross-core flags ensure proper ordering:
  //      - flag 0: QS ready (Vec reads QS)
  //      - flag 1: QK_gated ready (Vec done reading QS, Cube can write QKV)
  //      - flag 2: QKV ready (Vec reads QKV from same buffer)
  {
    TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                                HiddenSize);
    TASSIGN(_l0, 0);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = ChunkSize;
    _gs.shape[4] = HiddenSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
        workspace_qs_qkv_handle + ws_qkv_off, _gs);
    TSTORE(_gm, _l0);
  }

  // Signal Vec: QKV is ready (flag 2, Cube→Vec)
  // ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
#if __CCE_AICORE__ == 220
  SetCrossFlag<PIPE_FIX>(2);
#else
  pipe_barrier(PIPE_ALL);
  SignalBothVecOnA5<PIPE_FIX>(2);
#endif
}

#endif  // __DAV_CUBE__

#if defined(__DAV_VEC__)

// Gate coefficients for one work item, applied to that item's QK slot.
template <int32_t HiddenSize, int32_t ChunkSize>
AICORE inline void ChunkOVecProduce(
    __gm__ float *G_handle, __gm__ half *workspace_qk_handle,
    __gm__ half *workspace_qk_gated_handle, int64_t total_tokens, int32_t H,
    int32_t vid, int32_t head_idx, int64_t chunk_token_start,
    int32_t valid_rows, int32_t local_rows, int64_t ws_qk_off,
    int64_t ws_gated_off, int32_t gexp_addr) {
  using pto::Stride;
  constexpr int32_t HalfChunk = ChunkSize / 2;
  UbND<float, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  UbND<float, HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  UbND<float, HalfChunk, ChunkSize> qk_ub;
  TASSIGN(qk_ub, QKUbAddr);
  UbND<float, 1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, gexp_addr);
  UbND<float, HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  UbND<half, HalfChunk, ChunkSize, HalfChunk, ChunkSize, PadValue::Zero>
      qk_ub_half;
  TASSIGN(qk_ub_half, QKHalfUbAddr);

  // The previous step's vector reads (V) and workspace stores (MTE3) are
  // still in flight when this one starts, and they read the same UB tiles the
  // loads below overwrite. In the lock-step form the item boundary ordered
  // them; with a pipeline it has to be said.
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

  if (local_rows > 0) {
    // ── Load G [1 × valid_rows] — gate values for this chunk ────────
    // G is pre-transposed to [H, total_tokens], contiguous per head.
    {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
      _gs.shape[3] = 1;
      _gs.shape[4] = valid_rows;
      GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
          G_handle + static_cast<int64_t>(head_idx) * total_tokens +
              chunk_token_start,
          _gs);
      UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
          1, valid_rows);
      TASSIGN(_ld, GUbAddr);
      TLOAD(_ld, _gm);
      if (valid_rows != ChunkSize) {
        UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
        TASSIGN(_pd, GUbAddr);
        TFILLPAD_INPLACE(_pd, _ld);
      }
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ── Compute gating coefficients ──────────────────────────────────
    // ── Gating coefficient computation (numpy pseudocode) ─────────────
    // For this sub-block's rows (vid=0: rows 0..C/2-1, vid=1: rows
    // C/2..C-1):
    //
    //   g_row = g[my_start:my_start+C/2]    # my gates (shape [C/2])
    //   g_col = g[0:C]                       # full chunk gates (shape [C])
    //
    //   # Broadcast to 2D matrices:
    //   g_r_2d = g_row[:, None] * np.ones((1, C))    # TROWEXPAND: [C/2, C]
    //   g_c_2d = np.ones((C/2, 1)) * g_col[None, :]  # TCOLEXPAND: [C/2, C]
    //   coeff = exp(min(g_r_2d - g_c_2d, 0)) * mask
    //
    //   # Also compute exp(g_row) for QS scaling:
    //   exp_g_row = np.exp(g_row)                     # TEXP
    UbND<float, 1, HalfChunk> g_ub_temp_0;
    TASSIGN(g_ub_temp_0, GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                                       static_cast<int32_t>(sizeof(float)));
    TMOV(g_v_ub, g_ub_temp_0);

    // Broadcast g_row into [C/2 × C] and g_col into [C/2 × C]
    UbND<float, HalfChunk, ChunkSize> g_r_2d;
    TASSIGN(g_r_2d, QSUbAddr);
    UbDN<float, HalfChunk, 1> g_v_col;
    TASSIGN(g_v_col, gexp_addr);
    TROWEXPAND(g_r_2d, g_v_col);       // g_r_2d[i,j] = g_row[i]
    TCOLEXPAND(coeff_ub, g_ub);        // coeff[i,j] = g_col[j]
    TSUB(coeff_ub, g_r_2d, coeff_ub);  // d = g_row - g_col
    PipeBarrierVec();
    TMINS(coeff_ub, coeff_ub, 0.0f);
    PipeBarrierVec();
    TEXP(coeff_ub, coeff_ub);
    PipeBarrierVec();
    TMUL(coeff_ub, coeff_ub, msk_ub);
    PipeBarrierVec();
    TEXP(g_v_ub, g_v_ub);  // exp(g_row) for QS scaling
  }

  // ── Wait for Cube→Vec flag 0: QK & QS ready ─────────────────────
#if __CCE_AICORE__ == 220
  wait_flag_dev(0);
#else
  wait_intra_block(PIPE_MTE3, 0);
  pipe_barrier(PIPE_ALL);
#endif
  if (local_rows == 0) {
    // No rows here — still signal, so the Cube's wait counts stay balanced.
#if __CCE_AICORE__ == 220
    SetCrossFlag<PIPE_MTE3>(1);
#else
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_MTE3, 1);
#endif
    return;
  }

  // ── Load QK [C/2 × C] from workspace → UB ───────────────────────
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = local_rows;
    _gs.shape[4] = ChunkSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        workspace_qk_handle + ws_qk_off +
            static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
        _gs);
    UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
        local_rows, ChunkSize);
    TASSIGN(_ld, QKHalfUbAddr);
    TLOAD(_ld, _gm);
    if (local_rows != HalfChunk) {
      TFILLPAD_INPLACE(qk_ub_half, _ld);
    }
  }

  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

  // ── Apply gating: QK_gated = QK * exp(d*mask)*mask
  TMUL(qk_ub, qk_ub, coeff_ub);
  TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

  // ── Store QK_gated [C/2 × C] → workspace for Cube's GEMM 3 ─────
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = local_rows;
    _gs.shape[4] = ChunkSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        workspace_qk_gated_handle + ws_gated_off +
            static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
        _gs);
    UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(local_rows,
                                                           ChunkSize);
    TASSIGN(_st, QKHalfUbAddr);
    TSTORE(_gm, _st);
  }
  // Vec→Cube: QK_gated ready (flag 1)
  // ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
#if __CCE_AICORE__ == 220
  SetCrossFlag<PIPE_MTE3>(1);
#else
  pipe_barrier(PIPE_ALL);
  set_intra_block(PIPE_MTE3, 1);
#endif
}

// exp(g) * QS + QKV for one work item, into its rows of O.
template <int32_t HiddenSize, int32_t ChunkSize>
AICORE inline void ChunkOVecConsume(__gm__ half *O_handle,
                                    __gm__ half *workspace_qs_qkv_handle,
                                    int32_t BSND_V_STRIDE, int32_t H,
                                    int32_t vid, int32_t head_idx,
                                    int64_t chunk_token_start,
                                    int32_t local_rows, int64_t ws_qs_off,
                                    int64_t ws_qkv_off, int32_t gexp_addr) {
  using pto::Stride;
  constexpr int32_t HalfChunk = ChunkSize / 2;
  UbND<half, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
      qs_ub_half;
  TASSIGN(qs_ub_half, QSHalfUbAddr);
  UbND<float, HalfChunk, HiddenSize> qs_ub;
  TASSIGN(qs_ub, QSUbAddr);
  UbND<half, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
      o_ub_half;
  TASSIGN(o_ub_half, OHalfUbAddr);
  UbND<float, HalfChunk, HiddenSize> o_ub;
  TASSIGN(o_ub, OUbAddr);

  // The previous step's vector reads (V) and workspace stores (MTE3) are
  // still in flight when this one starts, and they read the same UB tiles the
  // loads below overwrite. In the lock-step form the item boundary ordered
  // them; with a pipeline it has to be said.
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

  if (local_rows == 0) {
#if __CCE_AICORE__ == 220
    wait_flag_dev(2);
#else
    wait_intra_block(PIPE_MTE3, 2);
    pipe_barrier(PIPE_ALL);
#endif
    return;
  }

  // ── Load QS [C/2 × D] from workspace → UB ───────────────────────
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = local_rows;
    _gs.shape[4] = HiddenSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
        workspace_qs_qkv_handle + ws_qs_off +
            static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
        _gs);
    UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
        local_rows, HiddenSize);
    TASSIGN(_ld, QSHalfUbAddr);
    TLOAD(_ld, _gm);
    if (local_rows != HalfChunk) {
      TFILLPAD_INPLACE(qs_ub_half, _ld);
    }
  }

  // ── Scale QS by exp(g): QS_gated = QS * exp(g_row) ──────────────
  // ── Scale QS by exp(g): inter-chunk state contribution ────────────
  // numpy: QS_scaled = QS * np.exp(g_row)[:, None]   (broadcast across D
  // columns) TROWEXPAND broadcasts the scalar exp(g[i]) for each row i
  // across all D columns, then TMUL applies it element-wise. This gates how
  // much the accumulated state contributes to each token's output.
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);
  UbND<float, HalfChunk, HiddenSize> g_exp_2d;
  TASSIGN(g_exp_2d, CoeffUbAddr);
  UbDN<float, HalfChunk, 1> g_v_col2;
  TASSIGN(g_v_col2, gexp_addr);
  TROWEXPAND(g_exp_2d, g_v_col2);  // broadcast exp(g_row) across columns
  PipeBarrierVec();
  TMUL(qs_ub, qs_ub, g_exp_2d);  // QS_gated = QS * exp(g_row)

  // ── Wait for Cube→Vec flag 2: QKV ready ─────────────────────────
#if __CCE_AICORE__ == 220
  wait_flag_dev(2);
#else
  wait_intra_block(PIPE_MTE3, 2);
  pipe_barrier(PIPE_ALL);
#endif

  // ── Load QKV [C/2 × D] from workspace → UB ──────────────────────
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = local_rows;
    _gs.shape[4] = HiddenSize;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
        workspace_qs_qkv_handle + ws_qkv_off +
            static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
        _gs);
    UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
        local_rows, HiddenSize);
    TASSIGN(_ld, OHalfUbAddr);
    TLOAD(_ld, _gm);
    if (local_rows != HalfChunk) {
      TFILLPAD_INPLACE(o_ub_half, _ld);
    }
  }

  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  // ── Combine: O = QS_gated + QKV ─────────────────────────────────
  // ── Final output: O = QKV + QS_scaled ─────────────────────────────
  // numpy: O = (QK_gated @ V) + (Q @ S) * exp(g)[:, None]
  //       = intra_chunk_attention + inter_chunk_state_contribution
  // TCVT half→float for QKV, then TADD, then TCVT float→half for output.
  TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
  TADD(o_ub, qs_ub, o_ub);
  TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

  // ── Store O [C/2 × D] → GM in BSND layout ───────────────────────
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  int64_t o_offset = (chunk_token_start * static_cast<int64_t>(H) +
                      static_cast<int64_t>(head_idx)) *
                         static_cast<int64_t>(HiddenSize) +
                     static_cast<int64_t>(vid) * HalfChunk *
                         static_cast<int64_t>(BSND_V_STRIDE);

  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = local_rows;
    _gs.shape[4] = HiddenSize;
    GmStride2D _stride(BSND_V_STRIDE);
    GmTensor2D<half> _gm(O_handle + o_offset, _gs, _stride);
    UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC> _st(local_rows,
                                                            HiddenSize);
    TASSIGN(_st, OHalfUbAddr);
    TSTORE(_gm, _st);
  }
}

#endif  // __DAV_VEC__

template <int32_t HiddenSize, int32_t ChunkSize>
AICORE void chunk_o_kernel(__gm__ half *Q_handle, __gm__ half *K_handle,
                           __gm__ half *V_handle, __gm__ half *S_handle,
                           __gm__ float *G_handle, __gm__ float *Msk_handle,
                           __gm__ half *workspace_qk_handle,
                           __gm__ half *workspace_qs_qkv_handle,
                           __gm__ half *workspace_qk_gated_handle,
                           __gm__ half *O_handle, __gm__ int32_t *cu_seqlens,
                           int64_t batch_size, int64_t seq_len,
                           int64_t total_tokens, uint32_t num_heads,
                           uint32_t num_key_heads, uint64_t ffts_addr) {
  // To avoid ambiguity with bisheng intrinsic header's global `enum class
  // Stride`
  using pto::Stride;

  // Half the chunk — each Vec sub-block handles C/2 rows independently.
  constexpr int32_t HalfChunk = ChunkSize / 2;
  // KTail / CTail: the number of valid elements in the last 128-element tile
  // when D or C isn't a multiple of 128. Used internally by PTO for partial
  // tiles.
  constexpr uint32_t KTail = (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);
  constexpr uint32_t CTail = (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

  const int32_t H = static_cast<int32_t>(num_heads);
  const int32_t Hg = static_cast<int32_t>(num_key_heads);
  if (H <= 0 || Hg <= 0 || (H % Hg) != 0) return;
  const int32_t GROUP = H / Hg;
  const int32_t BSND_V_STRIDE = H * HiddenSize;
  const int32_t BSND_QK_STRIDE = Hg * HiddenSize;

  // Workspace sizes (in elements) shared between Cube and Vec via GM
  constexpr int32_t WsQKSize = ChunkSize * ChunkSize;
  constexpr int32_t WsQSSize = ChunkSize * HiddenSize;
  constexpr int32_t WsGatedSize = ChunkSize * ChunkSize;

  // Initialize the cross-core FFTS signaling base address for this AI core.
  set_ffts_base_addr(ffts_addr);
  // cid = which AI core am I? (0..block_num-1). Used to partition work items.
  auto cid = get_block_idx();
  // block_num = total number of AI cores running this kernel in parallel.
  auto block_num = get_block_num();
  // vid = Vec sub-block ID (0 or 1). Each Vec core has 2 sub-blocks that
  // process the upper (vid=0) and lower (vid=1) halves of C/2 rows.
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  // The gate row exp(g) is produced at the produce step and read at the consume
  // step, so it lives in a small UB ring above the output tile. One row is
  // HalfChunk floats, which is 256 B at C = 128.
  constexpr int32_t GExpSlot = ((HalfChunk * 4 + 31) / 32) * 32;
  constexpr int32_t GExpUbAddr =
      OHalfUbAddr + HalfChunk * HiddenSize * static_cast<int32_t>(sizeof(half));

// =====================================================================
// CUBE CORE — Q@K^T and Q@S for item `step`, QK_gated@V for the item
// ChunkOPreLaunch steps behind it. A work item is one (chunk, head) pair;
// item gi runs on core gi % block_num. Items are independent, so the Cube
// can run ahead of the Vec; the mailbox ring keeps their slots apart.
// =====================================================================
#if defined(__DAV_CUBE__)
  {
    int64_t gi = 0;                // work item index, head-fastest
    int64_t chunk_global_idx = 0;  // chunk counter across all sequences
    int64_t step = 0;              // this core's item counter
    ChunkOItem ring[ChunkORing];

    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos, slen;
      SeqBounds(cu_seqlens, si, seq_len, bos, slen);
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci, ++chunk_global_idx) {
        for (int32_t head_idx = 0; head_idx < H; ++head_idx, ++gi) {
          if (gi % static_cast<int64_t>(block_num) != static_cast<int64_t>(cid))
            continue;

          int64_t chunk_start = ci * ChunkSize;
          int64_t remaining = slen - chunk_start;
          ChunkOItem &it = ring[step % ChunkORing];
          it.head_idx = head_idx;
          it.valid_rows = static_cast<int32_t>(
              remaining < ChunkSize ? remaining : ChunkSize);
          it.chunk_token_start = bos + chunk_start;
          it.qk_off = (it.chunk_token_start * static_cast<int64_t>(Hg) +
                       static_cast<int64_t>(head_idx / GROUP)) *
                      static_cast<int64_t>(HiddenSize);
          it.v_off = (it.chunk_token_start * static_cast<int64_t>(H) +
                      static_cast<int64_t>(head_idx)) *
                     static_cast<int64_t>(HiddenSize);
          it.s_offset = (chunk_global_idx * H + head_idx) *
                        static_cast<int64_t>(HiddenSize) *
                        static_cast<int64_t>(HiddenSize);

          set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
          ChunkOCubeProduce<HiddenSize, ChunkSize>(
              Q_handle, K_handle, S_handle, workspace_qk_handle,
              workspace_qs_qkv_handle, BSND_QK_STRIDE, it.valid_rows, it.qk_off,
              it.s_offset, ChunkOSlotOff(cid, step, WsQKSize),
              ChunkOSlotOff2(cid, step, WsQSSize, false));

          if (step >= ChunkOPreLaunch) {
            int64_t back = step - ChunkOPreLaunch;
            const ChunkOItem &done = ring[back % ChunkORing];
            ChunkOCubeConsume<HiddenSize, ChunkSize>(
                V_handle, workspace_qk_gated_handle, workspace_qs_qkv_handle,
                BSND_V_STRIDE, done.valid_rows, done.v_off,
                ChunkOSlotOff(cid, back, WsGatedSize),
                ChunkOSlotOff2(cid, back, WsQSSize, true));
          }
          ++step;
        }
      }
    }

    // Drain the items still in flight.
    for (int64_t back = (step > ChunkOPreLaunch ? step - ChunkOPreLaunch : 0);
         back < step; ++back) {
      const ChunkOItem &done = ring[back % ChunkORing];
      ChunkOCubeConsume<HiddenSize, ChunkSize>(
          V_handle, workspace_qk_gated_handle, workspace_qs_qkv_handle,
          BSND_V_STRIDE, done.valid_rows, done.v_off,
          ChunkOSlotOff(cid, back, WsGatedSize),
          ChunkOSlotOff2(cid, back, WsQSSize, true));
    }
  }
#endif

// =====================================================================
// VEC CORE — Gating, element-wise ops, output assembly
// Two Vec sub-blocks (vid=0,1) process upper/lower C/2 rows in parallel.
// Each sub-block independently:
//   1. Computes gating coefficients from G and the causal mask
//   2. Applies gating to the Cube's QK result → QK_gated
//   3. Scales the Cube's QS result by exp(g)
//   4. Combines QKV + scaled QS → final output O
// =====================================================================
#if defined(__DAV_VEC__)
  // Vec engine initialization: set_mask_norm selects "normal" masking mode,
  // and set_vector_mask(-1, -1) enables ALL SIMD lanes (no masking).
  set_mask_norm();
  set_vector_mask(-1, -1);

  // ── Load causal mask once (reused across all chunks) ─────────────────
  // ── Causal mask (loaded once, reused) ─────────────────────────────────
  // The causal mask is a C×C lower-triangular matrix of 0s and 1s:
  //   mask[i,j] = 1 if i >= j else 0
  // Each sub-block loads its C/2 rows. Applied via TMUL to zero out
  // non-causal (future) attention scores.
  //
  // Each sub-block (vid=0,1) loads its C/2 rows of the C×C lower-tri mask.
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = HalfChunk;
    _gs.shape[4] = ChunkSize;
    GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        Msk_handle + static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
    UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
        HalfChunk, ChunkSize);
    TASSIGN(_ld, MskUbAddr);
    TLOAD(_ld, _gm);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  {
    int64_t gi = 0;    // work item index, head-fastest
    int64_t step = 0;  // this core's item counter
    ChunkOItem ring[ChunkORing];

    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos, slen;
      SeqBounds(cu_seqlens, si, seq_len, bos, slen);
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t head_idx = 0; head_idx < H; ++head_idx, ++gi) {
          if (gi % static_cast<int64_t>(block_num) != static_cast<int64_t>(cid))
            continue;

          int64_t chunk_start = ci * ChunkSize;
          int64_t remaining = slen - chunk_start;
          ChunkOItem &it = ring[step % ChunkORing];
          it.head_idx = head_idx;
          it.valid_rows = static_cast<int32_t>(
              remaining < ChunkSize ? remaining : ChunkSize);
          it.chunk_token_start = bos + chunk_start;
          int32_t rows = it.valid_rows - static_cast<int32_t>(vid) * HalfChunk;
          if (rows < 0) rows = 0;
          if (rows > HalfChunk) rows = HalfChunk;
          it.local_rows = rows;

          ChunkOVecProduce<HiddenSize, ChunkSize>(
              G_handle, workspace_qk_handle, workspace_qk_gated_handle,
              total_tokens, H, static_cast<int32_t>(vid), it.head_idx,
              it.chunk_token_start, it.valid_rows, it.local_rows,
              ChunkOSlotOff(cid, step, WsQKSize),
              ChunkOSlotOff(cid, step, WsGatedSize),
              GExpUbAddr + static_cast<int32_t>(step % ChunkORing) * GExpSlot);

          if (step >= ChunkOPreLaunch) {
            int64_t back = step - ChunkOPreLaunch;
            const ChunkOItem &done = ring[back % ChunkORing];
            ChunkOVecConsume<HiddenSize, ChunkSize>(
                O_handle, workspace_qs_qkv_handle, BSND_V_STRIDE, H,
                static_cast<int32_t>(vid), done.head_idx,
                done.chunk_token_start, done.local_rows,
                ChunkOSlotOff2(cid, back, WsQSSize, false),
                ChunkOSlotOff2(cid, back, WsQSSize, true),
                GExpUbAddr +
                    static_cast<int32_t>(back % ChunkORing) * GExpSlot);
          }
          ++step;
        }
      }
    }

    // Drain the items still in flight.
    for (int64_t back = (step > ChunkOPreLaunch ? step - ChunkOPreLaunch : 0);
         back < step; ++back) {
      const ChunkOItem &done = ring[back % ChunkORing];
      ChunkOVecConsume<HiddenSize, ChunkSize>(
          O_handle, workspace_qs_qkv_handle, BSND_V_STRIDE, H,
          static_cast<int32_t>(vid), done.head_idx, done.chunk_token_start,
          done.local_rows, ChunkOSlotOff2(cid, back, WsQSSize, false),
          ChunkOSlotOff2(cid, back, WsQSSize, true),
          GExpUbAddr + static_cast<int32_t>(back % ChunkORing) * GExpSlot);
    }
  }
#endif
}

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
