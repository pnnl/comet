//===-- Passes.td - IndexTree dialect pass definition file--------*- tablegen -*-===//
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

#ifndef COMET_DIALECT_INDEXTREE_PASSES
#define COMET_DIALECT_INDEXTREE_PASSES

include "mlir/Pass/PassBase.td"

///===----------------------------------------------------------------------===///
/// Kernel Fusion
///===----------------------------------------------------------------------===///

//def IndexTreeKernelFusion : Pass<"indextree-kernel-fusion"> {
//  let summary = "The redundancy-aware kernel fusion on index tree dialect for compound expressions";
//  let description = [{

//      }];
//  let constructor = "comet::createIndexTreeKernelFusionPass()";
//  let dependentDialects = [
//    "comet::IndexTreeDialect",
//    "memref::MemRefDialect",
//    "scf::SCFDialect"
//  ];
//}

///===----------------------------------------------------------------------===///
/// Workspace Transformations
///===----------------------------------------------------------------------===///

def IndexTreeWorkspaceTranformations: Pass<"indextree-workspace-transformations", "func::FuncOp"> {
  let summary = "Compressed workspace transformation on IndexTree dialect"
                "to produce sparse output";
  let description = [{}];
  let constructor = "comet::createIndexTreeWorkspaceTransformationsPass()";
  let dependentDialects = ["comet::IndexTreeDialect"];
}

def IndexTreeDomainInference : Pass<"indextree-domain-inference", "func::FuncOp"> {
    let summary = "Infer domain of index variables";
    let description = [{
        Propogate domain values from tensors, through compute operations to index variables.
        Necessary for eventual lowering of index variables to for loops
    }];

    let constructor = "comet::createIndexTreeDomainInferencePass()";
}

def IndexTreeDomainConcretization : Pass<"indextree-domain-concretization", "func::FuncOp"> {
    let summary = "Transform tensor domains into concrete descriptions of dense or sparse domains";
    let description = [{
        Transform tensor domains into concrete descriptions of dense or sparse domains
    }];

    let constructor = "comet::createIndexTreeDomainConcretizationPass()";
}

def IndexTreeSymbolicComputePass : Pass<"indextree-symbolic-compute", "func::FuncOp"> {
    let summary = "Create index-tree for symbolic computations for sparse output";
    let description = [{
        Inserting values into a sparse output tensor cannot be done in parallel because it requires
        sequential access to the row pointers and dynamic allocation. The symbolic pass
        computes the size of allocations and row pointers without performing the computations
        so the sparse tensors can be allocated and the computations can be done in parallel.
    }];

    let constructor = "comet::createIndexTreeSymbolicComputePass()";
}

def IndexTreeInliningPass : Pass <"indextree-inlining", "func::FuncOp"> {
    let summary = "Inline index tree";
    let description = [{}];
    let constructor = "comet::createIndexTreeInliningPass()";
}

#endif /// COMET_DIALECT_INDEXTREE_PASSES
