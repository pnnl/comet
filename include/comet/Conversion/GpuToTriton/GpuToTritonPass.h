#ifndef GPU_CONVERSION_GPUTOTRITON_H
#define GPU_CONVERSION_GPUTOTRITON_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace comet {


std::unique_ptr<OperationPass<ModuleOp>> createConvertGpuKernelToTritonPass();

} // namespace comet
} // namespace mlir

#endif
