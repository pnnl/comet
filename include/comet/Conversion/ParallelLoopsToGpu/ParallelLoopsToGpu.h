#ifndef COMET_CONVERSION_PARALLELLOOPSTOGPU_H
#define COMET_CONVERSION_PARALLELLOOPSTOGPU_H

#include <memory>
namespace mlir {
class ModuleOp;
namespace func{
class FuncOp;
}
template <typename T> class OperationPass;
namespace comet {
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createConvertParallelLoopsToGpuPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createConvertParallelLoopsToGpuPass(int blockX, int blockY, int blockR);
}
}

#endif