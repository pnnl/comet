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
