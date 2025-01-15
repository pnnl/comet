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

#ifndef COMET_PARALLEL_LOOPS_TO_GPU_CONVERSION_PASSES
#define COMET_PARALLEL_LOOPS_TO_GPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def CometParallelLoopsToGpu : Pass<"comet-parallel-loops-to-gpu", "mlir::func::FuncOp"> {
  let summary = "Map loops to gpu";

  let constructor = "mlir::comet::createConvertParallelLoopsToGpuPass()";

  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::gpu::GPUDialect", "mlir::affine::AffineDialect", "mlir::DLTIDialect"];


     let options = [
       Option<"blockX", "blockX",
              "int32_t", /*default*/"32",
              "triton block size for dim X">,

       Option<"blockY", "blockY",
              "int32_t", /*default*/"8",
              "triton block size for dim Y">,
              
       Option<"blockR", "blockR",
       "int32_t", /*default*/"32",
       "triton block size for reduction dim">,

   ];
}

#endif
