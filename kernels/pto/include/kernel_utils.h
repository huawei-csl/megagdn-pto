/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#include <pto/pto-inst.hpp>
#include <type_traits>

namespace kernel_utils {
/**
 * @brief Do a sync step (set-wait flag) between two pipes.
 *
 * @tparam SrcPipe The pipe that sets the flag.
 * @tparam DstPipe The pipe that waits for the flag.
 * @param [in] id The event id to sync for.
 */
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetWaitFlag(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

/**
 * @brief Performs a division on two integral numbers and rounds the result up
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
AICORE inline T1 CeilDiv(T1 value, T2 divisor) {
  return (value + divisor - 1) / divisor;
}

template <pipe_t Pipe, uint8_t VEC_NUM = 2>
AICORE inline void SetCrossFlag(int32_t flag) {
  ffts_cross_core_sync(Pipe, 1 | (VEC_NUM << 4) | (flag << 8));
}

template <pipe_t Pipe>
AICORE inline void SignalBothVecOnA5(uint16_t flag) {
  // A5: the flag offset is 16 on new core.
  constexpr uint16_t VEC_FLAG_OFFSET = 16;

  set_intra_block(Pipe, flag);
  set_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}

template <pipe_t Pipe>
AICORE inline void WaitBothVecOnA5(uint16_t flag) {
  // A5: the flag offset is 16 on new core.
  constexpr uint16_t VEC_FLAG_OFFSET = 16;

  wait_intra_block(Pipe, flag);
  wait_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}


}  // namespace kernel_utils
