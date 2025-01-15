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

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/IndexTree/Patterns.h"

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using llvm::SmallVector;

namespace mlir {
    namespace comet{
      #define GEN_PASS_DEF_INDEXTREE_INLINING
      #include "comet/Conversion/Passes.h.inc"
    }
}


namespace {
class ConvertIndexTreeYieldOpTypes : public OpConversionPattern<indexTree::YieldOp> {
  public:
    using OpConversionPattern<indexTree::YieldOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> unpacked;
    for (Value v : adaptor.getOperands()) {
      if (auto cast =
              dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp())) {
        if (cast.getInputs().size() != 1) {
          unpacked.append(cast.getInputs().begin(), cast.getInputs().end());
          continue;
        }
      }
      // 1 : 1 type conversion.
      unpacked.push_back(v);
    }
    rewriter.replaceOpWithNewOp<indexTree::YieldOp>(op, unpacked);
    return success();
  }
};


class ConvertIndexTreeTypes : public OpConversionPattern<indexTree::IndexTreeOp>{
  public:
    using OpConversionPattern<indexTree::IndexTreeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(indexTree::IndexTreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the operands
    SmallVector<Value> unpacked;
    for (Value v : adaptor.getOperands()) {
      if (auto cast =
              dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp())) {
        if (cast.getInputs().size() != 1) {
          unpacked.append(cast.getInputs().begin(), cast.getInputs().end());
          continue;
        }
      }
      // 1 : 1 type conversion.
      unpacked.push_back(v);
    }

    SmallVector<Type> dstTypes;
    SmallVector<unsigned> offsets;
    offsets.push_back(0);
    // Do the type conversion and record the offsets.
    for (Type type : op.getResultTypes()) {
      if (failed(typeConverter->convertTypes(type, dstTypes)))
        return rewriter.notifyMatchFailure(op, "could not convert result type");
      offsets.push_back(dstTypes.size());
    }

    // Calls the actual converter implementation to convert the operation.
    auto newOp = rewriter.create<indexTree::IndexTreeOp>(op.getLoc(), dstTypes, unpacked);
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());

    // Packs the return value.
    SmallVector<Value> packedRets;
    for (unsigned i = 1, e = offsets.size(); i < e; i++) {
      unsigned start = offsets[i - 1], end = offsets[i];
      unsigned len = end - start;
      ValueRange mappedValue = newOp->getResults().slice(start, len);
      if (len != 1) {
        // 1 : N type conversion.
        Type origType = op.getResultTypes()[i - 1];
        Value mat = typeConverter->materializeSourceConversion(
            rewriter, op.getLoc(), origType, mappedValue);
        if (!mat) {
          return rewriter.notifyMatchFailure(
              op, "Failed to materialize 1:N type conversion");
        }
        packedRets.push_back(mat);
      } else {
        // 1 : 1 type conversion.
        packedRets.push_back(mappedValue.front());
      }
    }

    rewriter.replaceOp(op, packedRets);
    return success();
  }
};

class InlineIndexTreeOp : public OpConversionPattern<indexTree::IndexTreeOp>{
  public:
    using OpConversionPattern<indexTree::IndexTreeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(indexTree::IndexTreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    Block& body = op.getRegion().front();
    Operation* terminator = body.getTerminator();
    rewriter.replaceOp(op, terminator->getOperands());
    rewriter.inlineBlockBefore(&body, op);
    return success();
  }
};

class InlineIndexTreeYieldOp : public OpConversionPattern<indexTree::YieldOp>{
  public:
    using OpConversionPattern<indexTree::YieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(indexTree::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} //namespace

void mlir::indexTree::populateIndexTreeInliningPatterns(MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<InlineIndexTreeOp, InlineIndexTreeYieldOp>(context);
}

void mlir::indexTree::populateIndexTreeTypeConversionPatterns(MLIRContext *context, RewritePatternSet &patterns, TypeConverter &typeConverter, ConversionTarget& target) {
  target.addDynamicallyLegalOp<indexTree::IndexTreeOp>([&](Operation *op) {
      return typeConverter.isLegal(op->getResultTypes());
    });
    target.addDynamicallyLegalOp<indexTree::YieldOp>([&](indexTree::YieldOp op) {
      return typeConverter.isLegal(op->getOperandTypes());
    });

  patterns.add<ConvertIndexTreeTypes, ConvertIndexTreeYieldOpTypes>(typeConverter, context);
}

struct IndexTreeInliningPass
    : public PassWrapper<IndexTreeInliningPass, OperationPass<func::FuncOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexTreeInliningPass)

  void runOnOperation() override
  {
    // Convert the rest of the index tree dialect to SCF
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<index::IndexDialect, func::FuncDialect>();
    target.addIllegalOp<indexTree::IndexTreeOp, indexTree::YieldOp>();

    mlir::RewritePatternSet patterns(&getContext());
    mlir::indexTree::populateIndexTreeInliningPatterns(&getContext(), patterns);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

/// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createIndexTreeInliningPass()
{
  return std::make_unique<IndexTreeInliningPass>();
}