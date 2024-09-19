#ifndef GPU_TO_TRITON_CONVERSION_PASSES
#define GPU_TO_TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertGpuKernelToTritonPass: Pass<"convert-gpu-kernel-to-triton", "mlir::ModuleOp"> {
    let summary = "Convert Gpu Kernel to Triton Kernel";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertGpuKernelToTritonPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::affine::AffineDialect"];
}
#endif
