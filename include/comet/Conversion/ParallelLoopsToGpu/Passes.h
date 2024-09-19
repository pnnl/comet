#ifndef COMET_PARALLEL_LOOPS_TO_GPU_CONVERSION_PASSES
#define COMET_PARALLEL_LOOPS_TO_GPU_CONVERSION_PASSES

#include "comet/Conversion/ParallelLoopsToGpu/ParallelLoopsToGpu.h"

namespace mlir {
namespace comet {
#define GEN_PASS_REGISTRATION
#include "comet/Conversion/ParallelLoopsToGpu/Passes.h.inc"
}
}

#endif