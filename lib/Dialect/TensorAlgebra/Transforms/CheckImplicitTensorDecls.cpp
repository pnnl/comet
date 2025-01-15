//===- CheckImplicitTensorDecl.cpp - check if it is needed to add tensor declarations introduced by compound expressions------------------===//
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
//
/// This file implements a pass that adds temporary tensor declaration introduced by compound expressions in the COMET DSL
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <limits>
#include <map>
#include <set>
#include <unordered_map>

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tensorAlgebra;

#define DEBUG_TYPE "check-implicit-tensor-decl"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

using namespace mlir;

namespace
{
  /// Add tensor decl ops for tmp result
  struct TensorAlgebraCheckImplicitTensorDeclPass
      : public PassWrapper<TensorAlgebraCheckImplicitTensorDeclPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorAlgebraCheckImplicitTensorDeclPass)
    void runOnOperation() override;
  };
} /// namespace

template <typename t>
bool isNeedTensorDecl(t op)
{
  bool isUsedInSetSource = true;
  mlir::Value result = op->getResult(0);
  comet_debug() << " ";
  comet_vdump(result);
  for (auto u1 : result.getUsers())
  {
    comet_debug() << " ";
    comet_pdump(u1);
    /// If not used as source tensor of set_op, it is tmp result
    if (isa<tensorAlgebra::TensorSetOp>(u1))
    {
      auto p = cast<tensorAlgebra::TensorSetOp>(u1).getOperation();
      if(result == p->getOperand(0))
      {
        isUsedInSetSource = false;
      }
    }
  }
  return isUsedInSetSource;
}

template <typename t>
void addTensorDecl(t op)
{
  OpBuilder builder(op);
  auto location = op.getLoc();
  op->getOperands();
  std::vector<mlir::Value> lbls_value;
  mlir::Value ret_value;
  std::string ret_format;
  ArrayAttr imaps = op.getIndexingMaps();

  /// Retrieve the size of the tensor based on the affinity maps
  /// 1. Check the affinity map results of the LHS e.g.
  /// 2. Find which dimensions match on the RHS(1,2) e.g. (d0,d1,d2) -> (d0, d1), (d0,d1,d2) -> (d1, d2)
  /// 3. The result will be the dimension of the matching indices
  /// e.g For matmulop RHS1: (d0,d1,d2) -> (d0, d1), RHS2: (d0,d1,d2) -> (d1, d2), LHS : (d0,d1,d2) -> (d0,d2)
  ///     So we get (dim(RHS1, 0), dim(RHS2, 1)) because of d0, d2 respectively
  auto res_map = imaps[imaps.size() - 1].cast<AffineMapAttr>().getValue();
  for (auto v : res_map.getResults())
  {
    for (size_t i = 0; i < imaps.size() - 1; i++)
    {
      auto map = imaps[i].cast<AffineMapAttr>().getValue();
      if (auto pos = map.getResultPosition(v))
      {
        lbls_value.push_back(builder.create<TensorDimOp>(location, op->getOperand(i), *pos));
      }
    }
  }
  ret_value = op.getOperation()->getResult(0);

  mlir::ArrayAttr opFormatsArrayAttr = op.getFormats();
  unsigned int i = opFormatsArrayAttr.size() - 1;
  std::string ret_format_local(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
  ret_format = ret_format_local;

  mlir::Value itensor;
  if (ret_format.compare("Dense") == 0)
  {
    itensor = builder.create<DenseTensorDeclOp>(location, ret_value.getType(), lbls_value, ret_format);
    builder.create<TensorFillOp>(location, itensor, builder.getF64FloatAttr(0));
  }
  else
  {
    /// It is a temporal tensor declaration generated by compound expressions, BoolAttr is true
    /// to identify SparseTensorDeclOp is for temporaries
    itensor = builder.create<SparseTensorDeclOp>(location, ret_value.getType(), lbls_value, ret_format, true);
  }
  comet_debug() << "PreLowering SparseTensorDeclaration creation\n";
  comet_vdump(itensor);
  op.replaceAllUsesWith(itensor);

  builder.setInsertionPointAfter(op);
#ifdef DEBUG_MODE_ADDTEMPTENSORDECLARATION
  auto setop = builder.create<TensorSetOp>(location, ret_value, itensor);
  comet_debug() << " ";
  comet_vdump(setop);

  comet_debug() << " ";
  comet_vdump(ret_value);
#else
  builder.create<TensorSetOp>(location, ret_value, itensor);
#endif
}

void TensorAlgebraCheckImplicitTensorDeclPass::runOnOperation()
{
  func::FuncOp func = getOperation();
  comet_debug() << "Before TensorAlgebraCheckImplicitTensorDeclPass\n";

  /// The walker proceeds in post-order, but we need to process outer loops first
  /// to control the number of outer parallel loops, so push candidate loops to
  /// the front of a deque.
  func.walk([&](mlir::Operation *cur_op)
            {
           comet_debug() <<  " find a transpose op\n";
           /// if the output is not used as a source tensor of a set op
           /// Need to store use a sparse/dense tensor decl op to store the result
           if (isa<tensorAlgebra::TransposeOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TransposeOp>(cur_op);
             if (isNeedTensorDecl<tensorAlgebra::TransposeOp>(op))
             {
               addTensorDecl<tensorAlgebra::TransposeOp>(op);
             }
           }
           else if (isa<tensorAlgebra::TensorMultOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TensorMultOp>(cur_op);
             if (isNeedTensorDecl<TensorMultOp>(op))
             {
               addTensorDecl<TensorMultOp>(op);
             }
           }
           else if (isa<tensorAlgebra::TensorElewsMultOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TensorElewsMultOp>(cur_op);
             if (isNeedTensorDecl<TensorElewsMultOp>(op))
             {
               addTensorDecl<TensorElewsMultOp>(op);
             }
           }
           else if (isa<tensorAlgebra::TensorAddOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TensorAddOp>(cur_op);
             if (isNeedTensorDecl<TensorAddOp>(op))
             {
               addTensorDecl<TensorAddOp>(op);
             }
           }
           else if (isa<tensorAlgebra::TensorSubtractOp>(cur_op))
           {
             auto op = cast<tensorAlgebra::TensorSubtractOp>(cur_op);
             if (isNeedTensorDecl<TensorSubtractOp>(op))
             {
               addTensorDecl<TensorSubtractOp>(op);
             }
           } });
  comet_debug() << "After TensorAlgebraCheckImplicitTensorDeclPass\n";
}

std::unique_ptr<Pass> mlir::comet::createTensorAlgebraCheckImplicitTensorDeclPass()
{
  return std::make_unique<TensorAlgebraCheckImplicitTensorDeclPass>();
}
