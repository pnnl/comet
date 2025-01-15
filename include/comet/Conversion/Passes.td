//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
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

#ifndef COMET_CONVERSION_PASSES
#define COMET_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

///===----------------------------------------------------------------------===//
/// IndexTreeToSCF
///===----------------------------------------------------------------------===//

def ConvertIndexTreeToSCF : Pass<"convert-it-to-scf"> {
  let summary = " "
                "";
  let description = [{

      }];
  let constructor = "comet::createLowerIndexTreeToSCFPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def ConvertSymbolicDomains : Pass<"convert-symbolic-domains"> {
  let summary = " "
                "";
  let description = [{

      }];
  let constructor = "comet::createConvertSymbolicDomainsPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

//===----------------------------------------------------------------------===//
// TensorAlgebraToIndexTree
//===----------------------------------------------------------------------===//

// def ConvertTensorAlgebraToIndexTree : Pass<"convert-ta-to-it"> {
//   let summary = " "
//                 "";
//   let description = [{

//       }];
//   let constructor = "comet::createLowerTensorAlgebraToIndexTreePass()";
//   let dependentDialects = [
//     "memref::MemRefDialect",
//     "scf::SCFDialect"
//   ];
// }

///===----------------------------------------------------------------------===//
/// TensorAlgebraToSCF
///===----------------------------------------------------------------------===//

def ConvertTensorAlgebraToSCF : Pass<"convert-ta-to-scf"> {
  let summary = " "
                "";
  let description = [{

      }];
  let constructor = "comet::createLowerTensorAlgebraToSCFPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "index::IndexDialect"
  ];
}

def SparseTensorConversionPass : Pass<"convert-sparse-tensor"> {
  let summary = "Lowers operations on Tensor Algebra sparse tensors to mlir tensor operations";
  let description = [{}];
  
  let constructor = "comet::createSparseTensorConversionPass()";

  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "index::IndexDialect",
    "tensor::TensorDialect"
  ];
}

#endif /// COMET_CONVERSION_PASSES
