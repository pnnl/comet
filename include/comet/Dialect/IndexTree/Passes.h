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

#ifndef COMET_DIALECT_INDEXTREE_PASSES_H
#define COMET_DIALECT_INDEXTREE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace comet
    {
/// Generate the code for registering conversion passes.
#define GEN_PASS_DECL
#include "comet/Dialect/IndexTree/Passes.h.inc"
        // Create a pass for infering the domain from the use of the index variables
        std::unique_ptr<Pass> createIndexTreeDomainInferencePass();

        // Create a pass for concretizing the domain from the tensor definitions 
        std::unique_ptr<Pass> createIndexTreeDomainConcretizationPass();

        // Create a pass for creating the symbolic pass 
        std::unique_ptr<Pass> createIndexTreeSymbolicComputePass();

        // Create a pass for inlining the index tree
        std::unique_ptr<Pass> createIndexTreeInliningPass();

        /// Create a pass for applying compressed workspace transformation into IndexTreeIR
        std::unique_ptr<Pass> createIndexTreeWorkspaceTransformationsPass();

        /// Create a pass for the redundancy-aware kernel fusion on index tree dialect for some compound expressions
        // std::unique_ptr<Pass> createIndexTreeKernelFusionPass();
    }

}

#endif // COMET_DIALECT_INDEXTREE_PASSES_H
