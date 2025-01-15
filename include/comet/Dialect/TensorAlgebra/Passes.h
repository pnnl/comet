//===- Passes.h - Conversion Pass Construction and Registration -----------===//
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
//===----------------------------------------------------------------------===//

#ifndef COMET_DIALECT_TENSORALGEBRA_PASSES_H
#define COMET_DIALECT_TENSORALGEBRA_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace comet
    {
/// Generate the code for registering conversion passes.
#define GEN_PASS_DECL
#include "comet/Dialect/TensorAlgebra/Passes.h.inc"

        /// Check if it is needed to add tensor declarations introduced by compound expressions
        std::unique_ptr<Pass> createTensorAlgebraCheckImplicitTensorDeclPass();

        void populateDenseTensorDeclLoweringPatterns(RewritePatternSet &patterns);
        void populateSparseOutputTensorDeclLoweringPatterns(RewritePatternSet &patterns);

        /// Create a pass to lower dense input and output tensor declarations
        std::unique_ptr<Pass> createDenseTensorDeclLoweringPass();

        /// Create a pass to lower sparse tensor declarations and create sparse output tensor declarations
        std::unique_ptr<Pass> createSparseTensorDeclLoweringPass();

        /// Create a pass to lower sparse output tensor declarations
        std::unique_ptr<Pass> createSparseOutputTensorDeclLoweringPass();

        /// Create a pass to lower temporary sparse output tensor declarations - temporary sparse output is introduced in compound expressions
        std::unique_ptr<Pass> createSparseTempOutputTensorDeclLoweringPass();

        std::unique_ptr<Pass> createFindOptimalTCFactorizationPass();
        std::unique_ptr<Pass> createLowerTAMulChainPass();

        /// Create a pass for lowering TA operations to TTGT
        /// This pass selects either the best permutation among all
        /// or pass can specify the iteration order of the permutation, ith permutation
        std::unique_ptr<Pass> createLoweringTTGTPass(bool enableBestPerm,
                                                     int whatPermID = 1,
                                                     bool printFlops = false);

        std::unique_ptr<Pass> createLinAlgMatmulTilingPass();
        std::unique_ptr<Pass> createLinAlgMatmulMicroKernelPass();

        // Optimize dense transpose (linalg.copy) based on the following paper:
        // HPTT: A High-Performance Tensor Transposition C++ Library
        // https://arxiv.org/abs/1704.04374
        std::unique_ptr<Pass> createOptDenseTransposePass(uint64_t tile_size = 1,
                                                          bool seperate_tiles = false);

        std::unique_ptr<Pass> createLowerLinAlgFillPass();

        /// Create a pass for lowering tensor fill operation to linalg.fill
        std::unique_ptr<Pass> createTensorFillLoweringPass();

        /// Create a pass for lowering to the rest of the operations in `Std` dialects,
        /// such as printOp, constantOp, ReturnOp..
        std::unique_ptr<mlir::Pass> createLateLoweringPass();

        /// Create a pass for lowering sparse TA operations to SCFDimAttr
        std::unique_ptr<Pass> createSTCRemoveDeadOpsPass();

        /// Create a pass for lowering sparse TA operations to SCFDimAttrGPU
        std::unique_ptr<Pass> createSCFToSCFParallelPass();

        /// Create a pass for lowering programming constructs to SCF ops
        std::unique_ptr<Pass> createPCToLoopsLoweringPass(); // Conversion

        // TODO(gkestor): this pass is a workout to handle redundant LabeledTensor operations
        std::unique_ptr<Pass> createRemoveLabeledTensorOpsPass();

        std::unique_ptr<Pass> createFuncOpLoweringPass(); // Conversion

        std::unique_ptr<Pass> createDimOpLoweringPass();
        std::unique_ptr<Pass> createTABufferizeFunc();
    }

}

#endif // COMET_DIALECT_TENSORALGEBRA_PASSES_H
