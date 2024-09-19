#ifndef GPU_TO_TRITON_CONVERSION_PASSES_H
#define GPU_TO_TRITON_CONVERSION_PASSES_H

#include "comet/Conversion/GpuToTriton/GpuToTritonPass.h"

namespace mlir {
namespace comet {

#define GEN_PASS_REGISTRATION
#include "comet/Conversion/GpuToTriton/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
