//===- TensorDeclLowering.cpp -- Lower dense and sparse tensor declarations for input and output tensors----===//
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
// This file implements a lowering of sparse and dense tensor declarations
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LLVM.h"

#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::affine;

using namespace mlir::tensorAlgebra;
using namespace mlir::indexTree;

#define DEBUG_TYPE "tensor-decl-lowering"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

//===----------------------------------------------------------------------===//
/// Lowering Passes for sparse/dense tensor declarations
//===----------------------------------------------------------------------===//
namespace
{
  struct TensorFillLowering : public ConversionPattern
  {
    TensorFillLowering(MLIRContext *ctx)
        : ConversionPattern(tensorAlgebra::TensorFillOp::getOperationName(), 1,
                            ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TensorFillOp>(op));

      auto loc = op->getLoc();
      auto tensorFillOp = cast<tensorAlgebra::TensorFillOp>(op);

      auto tensorOperand = operands[0];
      ToTensorOp tensorLoadOp;
      if (!mlir::isa_and_present<ToTensorOp>(tensorOperand.getDefiningOp()))
      {
        /// TODO: may need to re-visit when doing reduction support.
        /// the user declared output to have zeros.
        rewriter.eraseOp(op);
        return success();
      }
      tensorLoadOp = cast<ToTensorOp>(tensorOperand.getDefiningOp());
      auto memref = tensorLoadOp.getMemref();
      auto valueAttr = tensorFillOp.getValue();
      
      rewriter.setInsertionPoint(tensorLoadOp);
      Value constantOp = rewriter.create<ConstantOp>(loc, llvm::cast<TypedAttr>(valueAttr));
      rewriter.create<linalg::FillOp>(loc, constantOp, memref);
      rewriter.eraseOp(op);

      return success();
    }
  };

  struct FuncOpLowering : public OpConversionPattern<tensorAlgebra::FuncOp>
  {
    using OpConversionPattern<tensorAlgebra::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(tensorAlgebra::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final
    {
      /// We only lower the main function as we expect that all other functions
      /// have been inlined.
      /// [TODO] Make sure this is indeed the case
      if (op.getName() != "main")
        return failure();

      /// Verify that the given main has no inputs and results.
      if (op.getNumArguments() || op.getFunctionType().getNumResults())
      {
        return rewriter.notifyMatchFailure(op, [](Diagnostic &diag)
                                           { diag << "expected 'main' to have 0 inputs and 0 results"; });
      }

      /// Create a new non-toy function, with the same region.
      auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                      op.getFunctionType());
      rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
      rewriter.eraseOp(op);
      return success();
    }
  };

}
//===----------------------------------------------------------------------===//
/// Early Lowering Passes end
//===----------------------------------------------------------------------===//

namespace
{
  struct TensorFillLoweringPass
      : public PassWrapper<TensorFillLoweringPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorFillLoweringPass)
    void runOnOperation() override;
  };

} /// end anonymous namespace.

void TensorFillLoweringPass::runOnOperation()
{
  comet_debug() << "---------------TensorFillLoweringPass start\n";
  /// this is a simple pass that replaces tensor decl with linalg.fill
  func::FuncOp func = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<LinalgDialect,
                         ArithDialect,
                         scf::SCFDialect,
                         AffineDialect,
                         memref::MemRefDialect,
                         bufferization::BufferizationDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<TensorFillLowering>(&getContext());

  if (failed(applyPartialConversion(func, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to Lower STCOutputLowering\n";
    signalPassFailure();
  }
  comet_debug() << "---------------TensorFillLoweringPass end\n";
}

/// Create a pass for lowering tensor fill operation
std::unique_ptr<Pass> mlir::comet::createTensorFillLoweringPass()
{
  return std::make_unique<TensorFillLoweringPass>();
}

namespace
{

  struct DimOpLowering : public OpRewritePattern<tensorAlgebra::TensorDimOp>
  {
    using OpRewritePattern<tensorAlgebra::TensorDimOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions and ta.sptensor_construct operation.
     * Step 3: Remove the TensorDimOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::TensorDimOp op,
                                  PatternRewriter &rewriter) const final
    {
      /// Sparse output tensor declaration happens after lowering to index tree dialect
      assert(isa<tensorAlgebra::TensorDimOp>(op));
      comet_debug() << "TensorDimOpLowering in format begin\n";
      comet_vdump(op);
      auto tensor = op.getTensor();
      if (tensor.getType().isa<SparseTensorType>())
      {
        ::mlir::TypedValue<::mlir::IndexType> idx = op.getIndex();
        auto realIndex = getConstantIntValue(idx);
        rewriter.replaceOpWithNewOp<SpTensorGetDimSize>(op, rewriter.getIndexType(), tensor, *realIndex);
      }
      else if (tensor.getType().isa<TensorType>())
      {
        ::mlir::TypedValue<::mlir::IndexType> idx = op.getIndex();
        auto dim = rewriter.create<tensor::DimOp>(op.getLoc(), tensor, idx);
        rewriter.replaceAllUsesWith(op, dim);
        rewriter.eraseOp(op);
      }
      else
      {
        return failure();
      }

      comet_debug() << "--------------TensorDimOpLowering in format end\n";
      return success();
    }
  };

  class DimOpLoweringPass
      : public PassWrapper<DimOpLoweringPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DimOpLoweringPass)
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      patterns.insert<DimOpLowering>(patterns.getContext());

      func::FuncOp function = getOperation();
      ConversionTarget target(getContext());
      target.addLegalDialect<ArithDialect,
                             memref::MemRefDialect,
                             tensor::TensorDialect,
                             scf::SCFDialect,
                             bufferization::BufferizationDialect,
                             IndexTreeDialect>();
      target.addIllegalDialect<TADialect>();

      target.addLegalOp<tensorAlgebra::PrintOp,
                        tensorAlgebra::ReduceOp,
                        tensorAlgebra::TransposeOp,
                        tensorAlgebra::TensorFillOp,
                        tensorAlgebra::GetTimeOp,
                        tensorAlgebra::PrintElapsedTimeOp,
                        tensorAlgebra::TensorSetOp,
                        tensorAlgebra::IndexLabelOp,
                        tensorAlgebra::DenseConstantOp,
                        tensorAlgebra::TensorMultOp,
                        tensorAlgebra::ScalarOp,
                        tensorAlgebra::SpTensorAliasOp,
                        tensorAlgebra::SpTensorGetDimSize,
                        tensorAlgebra::SpTensorGetDimPos,
                        tensorAlgebra::SpTensorGetDimCrd,
                        tensorAlgebra::SpTensorGetVals,
                        tensorAlgebra::SparseTensorConstructOp>();

      if (failed(applyPartialConversion(function, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to applyPartialConversion in DenseTensorDeclLoweringPass\n";
        signalPassFailure();
      }
    }
  };
}

std::unique_ptr<Pass> mlir::comet::createDimOpLoweringPass()
{
  return std::make_unique<DimOpLoweringPass>();
}