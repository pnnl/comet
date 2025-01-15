//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

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
