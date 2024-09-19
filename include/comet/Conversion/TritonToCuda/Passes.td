#ifndef GPU_TO_TRITON_CONVERSION_PASSES
#define GPU_TO_TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LowerHostToCuda: Pass<"lower-gpu-host-to-cuda", "mlir::ModuleOp"> {
       let summary = "Lower Gpu dialect host to cuda";
       let description = [{}];
       let constructor = "mlir::comet::createLowerGpuHostToCudaPass()";

       let dependentDialects = ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}

def LowerTritonDeviceToCuda: Pass<"lower-triton-device-to-cuda", "mlir::ModuleOp"> {
       let summary = "Lower Triton dialect host to cuda";
       let description = [{}];
       let constructor = "mlir::comet::createLowerTritonDeviceToCudaPass()";

       let options = [
              Option<"numWarps", "numWarps",
                     "int32_t", /*default*/ "4",
                     "Number of warps">,
              
              Option<"threadsPerWarp", "threadsPerWarp",
                     "int32_t", /*default*/ "32",
                     "Number of threads per warp">,

              Option<"numStages", "numStages",
                     "int32_t", /*default*/ "3",
                     "Number of stages">,

              Option<"numCTAs", "numCTAs",
                     "int32_t", /*default*/ "1",
                     "Number of CTAs">,
              
              Option<"computeCapability", "computeCapability",
                     "int32_t", "80",
                     "Target compute capability">,
       ];

       let dependentDialects = ["mlir::arith::ArithDialect", "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::scf::SCFDialect",  "mlir::tensor::TensorDialect","mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect", "mlir::triton::TritonDialect", "mlir::triton::gpu::TritonGPUDialect", "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect", "mlir::triton::nvgpu::NVGPUDialect", "mlir::NVVM::NVVMDialect"];
}
#endif
