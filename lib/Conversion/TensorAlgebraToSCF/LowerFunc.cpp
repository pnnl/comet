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

#include "mlir/IR/BuiltinDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
using namespace mlir;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

//===----------------------------------------------------------------------===//
/// tensorAlgebra::FuncOp to  func::FuncOp RewritePatterns
//===----------------------------------------------------------------------===//

namespace
{
  //===----------------------------------------------------------------------===//
  /// TensorAlgebra RewritePatterns: Func operations
  //===----------------------------------------------------------------------===//
  struct FuncOpLowering : public OpConversionPattern<tensorAlgebra::FuncOp>
  {
    using OpConversionPattern<tensorAlgebra::FuncOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(tensorAlgebra::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final
    {
      /// We only lower the main function as we expect that all other functions
      /// have been inlined.
      if (op.getName() == "main")
      {
        /// return failure();
        /// Verify that the given main has no inputs and results.
        if (op.getNumArguments() || op.getFunctionType().getNumResults())
        {
          return rewriter.notifyMatchFailure(op, [](Diagnostic &diag)
                                             { diag << "expected 'main' to have 0 inputs and 0 results"; });
        }
      }

      /// Create a new non-tensorAlgebra function, with the same region.
      auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                      op.getFunctionType());
      rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
      rewriter.eraseOp(op);
      return success();
    }
  };

} /// namespace

//===----------------------------------------------------------------------===//
/// FuncOpLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace
{
  struct FuncOpLoweringPass
      : public PassWrapper<FuncOpLoweringPass, OperationPass<ModuleOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuncOpLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override
    {
      registry.insert<func::FuncDialect, memref::MemRefDialect>();
    }
    void runOnOperation() final;
  };
} /// namespace

struct ReturnOpLowering : public OpRewritePattern<tensorAlgebra::TAReturnOp>
{
  using OpRewritePattern<tensorAlgebra::TAReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensorAlgebra::TAReturnOp op,
                                PatternRewriter &rewriter) const final
  {
    /// TODO(gkestor): During this lowering, do we expect that all function calls have been inlined?
    if (op.hasOperand())
    {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
    }
    else
    {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    }

    return success();
  }
};

struct GenericCallOpLowering : public OpRewritePattern<mlir::tensorAlgebra::GenericCallOp>
{
  using OpRewritePattern<mlir::tensorAlgebra::GenericCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tensorAlgebra::GenericCallOp op,
                                PatternRewriter &rewriter) const final
  {
    /// TODO(gkestor): During this lowering, do we expect that all function calls have been inlined?

    /// We lower "toy.return" directly to "func.return".
    if (op.getResults().size() > 0)
    {
      rewriter.replaceOpWithNewOp<func::CallOp>(op, op->getAttrOfType<SymbolRefAttr>("callee"), mlir::UnrankedTensorType::get(rewriter.getF64Type()), op.getOperands());
    }
    else
    {
      rewriter.replaceOpWithNewOp<func::CallOp>(op, op->getAttrOfType<SymbolRefAttr>("callee"), mlir::TypeRange(), op.getOperands());
    }

    return success();
  }
};

void FuncOpLoweringPass::runOnOperation()
{
  /// The first thing to define is the conversion target. This will define the
  /// final target for this lowering.
  ConversionTarget target(getContext());

  /// We define the specific operations, or dialects, that are legal targets for
  /// this lowering. In our case, we are lowering to a combination of the
  /// `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<BuiltinDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();

  /// We also define the Toy dialect as Illegal so that the conversion will fail
  /// if any of these operations are *not* converted. Given that we actually want
  /// a partial lowering, we explicitly mark the Toy operations that don't want
  /// to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  /// to be updated though (as we convert from TensorType to MemRefType), so we
  /// only treat it as `legal` if its operands are legal.
  /// target.addIllegalDialect<tensorAlgebra::TADialect>();

  /// Now that the conversion target has been defined, we just need to provide
  /// the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpLowering, ReturnOpLowering, GenericCallOpLowering>(
      &getContext());

  /// With the target and rewrite patterns defined, we can now attempt the
  /// conversion. The conversion will signal failure if any of our `illegal`
  /// operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::comet::createFuncOpLoweringPass()
{
  return std::make_unique<FuncOpLoweringPass>();
}
