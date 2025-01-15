//===-- Passes.td - TensorAlgebra dialect pass definition file--------*- tablegen -*-===//
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

#ifndef COMET_DIALECT_TENSORALGEBRA_PASSES
#define COMET_DIALECT_TENSORALGEBRA_PASSES

include "mlir/Pass/PassBase.td"

//TODO(gkestor): add notes for each pass

///===----------------------------------------------------------------------===//
/// If needed, add tensor declaration for temporary tensors introduced by compound expressions
///===----------------------------------------------------------------------===//
def TensorAlgebraCheckImplicitTensorDecl : Pass<"check-implicit-tensor-decl"> {
  let summary = "check if it is needed to add missing tensor declarations introduced by compound expressions";
  let description = [{

      }];
  let constructor = "comet::createTensorAlgebraCheckImplicitTensorDeclPass()";
  let dependentDialects = [
    "comet::TensorAlgebraDialect"
  ];
}

def TensorAlgebraDenseTensorDeclLowering : Pass<"lower-dense-tensor-decl"> {
  let summary = "";
  let description = [{

      }];
  let constructor = "comet::createDenseTensorDeclLoweringPass()";
  let dependentDialects = [
    "comet::TensorAlgebraDialect"
  ];
}

def TensorAlgebraSparseTensorDeclLowering : Pass<"lower-sparse-tensor-decl"> {
  let summary = "";
  let description = [{

      }];
  let constructor = "comet::createSparseTensorDeclLoweringPass()";
  let dependentDialects = [
    "comet::TensorAlgebraDialect"
  ];
}

def TensorAlgebraSparseTempOutputTensorDeclLowering : Pass<"lower-sparse-temp-output-tensor-decl"> {
  let summary = "";
  let description = [{

      }];
  let constructor = "comet::createSparseTempOutputTensorDeclLoweringPass()";
  let dependentDialects = [
    "comet::TensorAlgebraDialect"
  ];
}

def TensorAlgebraSparseOutputTensorDeclLowering : Pass<"lower-sparse-output-tensor-decl"> {
  let summary = "";
  let description = [{

      }];
  let constructor = "comet::createSparseOutputTensorDeclLoweringPass()";
  let dependentDialects = [
    "comet::TensorAlgebraDialect"
  ];
}
def TensorAlgebraBufferizeFunc : Pass<"ta-bufferize-func"> {
  let summary = "";
  let description = [{

      }];
  let constructor = "comet::createTABufferizeFunc()";
  let dependentDialects = [
    "comet::TensorAlgebraDialect", "func::FuncDialect"
  ];
}


#endif /// COMET_DIALECT_TENSORALGEBRA_PASSES
