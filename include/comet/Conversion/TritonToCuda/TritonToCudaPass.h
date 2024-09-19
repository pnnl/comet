#ifndef TRITON_CONVERSION_TRITONTOCUDA_H
#define TRITON_CONVERSION_TRITONTOCUDA_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace comet {


std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerGpuHostToCudaPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerTritonDeviceToCudaPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerTritonDeviceToCudaPass(int numWarps,
                                                int threadsPerWarp,
                                                int numCTAs,
                                                int numStages,
                                                int computeCapability);

} // namespace triton
} // namespace mlir

#endif