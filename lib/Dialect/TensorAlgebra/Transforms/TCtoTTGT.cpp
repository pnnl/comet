//===- TCtoTTGT.cpp ------===//
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
// This file implements reformulation of tensor contraction operations as Transpose-Transpose-GEMM-Transpose
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include <limits>
#include <map>
#include <set>
#include <unordered_map>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::arith;
using namespace mlir::bufferization;

using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

const StringLiteral kLinalgTransformMarker = "__with_tiling__";

template <typename T>
static bool arePermutations(const std::vector<T> &vec1,
                            const std::vector<T> &vec2)
{
  if (vec1.size() != vec2.size())
  {
    return false;
  }
  std::vector<bool> taken(vec1.size(), false);
  for (size_t i = 0; i < vec1.size(); i++)
  {
    auto it = std::find(vec2.begin(), vec2.end(), vec1[i]);
    if (it == vec2.end())
    {
      return false;
    }
    if (taken[std::distance(vec2.begin(), it)] == true)
    {
      return false;
    }
    taken[std::distance(vec2.begin(), it)] = true;
  }
  return true;
}

/// Detect whether memref dims [dim, dim + extent) can be reshaped without
/// copies.
[[maybe_unused]] static bool isReshapableDimBand(unsigned dim, unsigned extent,
                                                 ArrayRef<int64_t> sizes,
                                                 ArrayRef<AffineExpr> strides)
{
  assert(sizes.size() == strides.size() && "mismatched ranks");
  /// off by 1 indexing to avoid out of bounds
  for (auto idx = dim, e = dim + extent; idx + 1 < e; ++idx)
  {
    /// Only bands of static shapes are reshapable. This is due to the fact that
    /// there is no relation between dynamic sizes and dynamic strides: we do not
    /// have enough information to know whether a "-1" size corresponds to the
    /// proper symbol in the AffineExpr of a stride.
    if (ShapedType::isDynamic(sizes[dim + 1]))
      return false;
    /// simplify on the fly and catch more reshapable cases.
    if (strides[idx] != strides[idx + 1] * sizes[idx + 1])
      return false;
  }
  return true;
}

static IndexVector getIndexRange(unsigned lo, unsigned hi, unsigned step = 1)
{
  IndexVector result;
  for (unsigned i = lo; i < hi; i += step)
  {
    result.push_back(i);
  }
  return result;
}

//===----------------------------------------------------------------------===//
/// TAEarlyLoweringTTGTPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to linear algebra of the tensor algebra
/// operations that are computationally intensive (like matmul for example...)
/// while keeping the rest of the code in the TA dialect.
namespace
{

  struct TensorContractionOpLoweringTTGT : public ConversionPattern
  {
    TensorContractionOpLoweringTTGT(MLIRContext *ctx, bool isSelectBestPerm, int whatPerm, bool printFlops)
        : ConversionPattern(tensorAlgebra::TensorMultOp::getOperationName(), 1, ctx),
          isSelectBestPerm(isSelectBestPerm), whatPerm(whatPerm), printFlops{printFlops} {}

    /**
     * @brief Latest implementation with following optimizations:
     *        - if no transpose is required there won't be any copy operations
     *        - if any operand is 2 dimensional no reshape
     *        - does not copy C
     */
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final
    {
      comet_pdump(op);
      assert(isa<tensorAlgebra::TensorMultOp>(op));

      auto ctx = rewriter.getContext();
      auto loc = op->getLoc();
      auto multop = cast<tensorAlgebra::TensorMultOp>(op);
      auto alphaAttr = multop.getOperation()->getAttr("__alpha__");
      auto betaAttr = multop.getOperation()->getAttr("__beta__");

      Operation *startTime;
      std::string getTimeStr = "getTime";
      auto f64Type = rewriter.getF64Type();
      if (printFlops)
      {
        startTime = rewriter.create<func::CallOp>(
            op->getLoc(), getTimeStr, SmallVector<Type, 2>{f64Type});
      }

      ArrayAttr indexMaps = multop.getIndexingMaps();
      std::vector<std::vector<unsigned>> allPerms;

      /// Find summation indices
      for (const auto &map : indexMaps)
      {
        auto affineMap = map.cast<AffineMapAttr>().getValue();
        std::vector<unsigned> perm;
        for (size_t i = 0; i < affineMap.getNumResults(); i++)
        {
          auto expr = affineMap.getResult(i);
          perm.push_back(llvm::cast<AffineDimExpr>(expr).getPosition());
        }

        allPerms.push_back(perm);
      }

      comet_pdump(op);
      comet_debug() << "\n";
      // auto rhs1Tensor = cast<ToTensorOp>(operands[0].getDefiningOp());
      // auto rhs2Tensor = cast<ToTensorOp>(operands[1].getDefiningOp());
      comet_debug() << "\n";
      Value lhsDef;
      tensorAlgebra::TensorSetOp setnewop;
      for (auto u : multop.getOperation()->getResult(0).getUsers())
      {
        comet_pdump(u);
        if (isa<tensorAlgebra::TensorSetOp>(u))
        {
          setnewop = cast<tensorAlgebra::TensorSetOp>(u);
          Value dstTensor = u->getOperand(1);

          lhsDef = dstTensor;
          comet_vdump(lhsDef);
        }
      }
      // auto lhsTensor = cast<ToTensorOp>(lhsDef.getDefiningOp());

      comet_vdump(setnewop);
      comet_debug() << "\n";

      Value rhs1Memref, rhs2Memref, lhsMemref;
      if(auto tensor_rhs1 = operands[0].getType().dyn_cast<TensorType>())
      {
        rhs1Memref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(tensor_rhs1.getShape(), tensor_rhs1.getElementType()), operands[0]);
      }
      else if(operands[0].getType().isa<MemRefType>())
      {
        rhs1Memref = operands[0];
      }
      else 
      {
        assert(false && "Unexpected type");
      }

      if(auto tensor_rhs2 = operands[1].getType().dyn_cast<TensorType>())
      {
        rhs2Memref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(tensor_rhs2.getShape(), tensor_rhs2.getElementType()), operands[1]);
      }
      else if(operands[1].getType().isa<MemRefType>())
      {
        rhs2Memref = operands[1];
      }
      else 
      {
        assert(false && "Unexpected type");
      }

      if(auto tensor_lhs = lhsDef.getType().dyn_cast<TensorType>())
      {
        lhsMemref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(tensor_lhs.getShape(), tensor_lhs.getElementType()), lhsDef);
      }
      else if(lhsDef.getType().isa<MemRefType>())
      {
        lhsMemref = lhsDef;
      }
      else 
      {
        assert(false && "Unexpected type");
      }

      auto rhs1MemrefType = rhs1Memref.getType().cast<MemRefType>();
      auto rhs2MemrefType = rhs2Memref.getType().cast<MemRefType>();
      auto lhsMemrefType = lhsMemref.getType().cast<MemRefType>();

      std::vector<TensorShape> allShapes{rhs1MemrefType.getShape(),
                                         rhs2MemrefType.getShape(),
                                         lhsMemrefType.getShape()};

      ContractionPlan plan{allPerms[0], allShapes[0], allPerms[1],
                           allShapes[1], allPerms[2], allShapes[2]};

      /// computeBestPermutations identifies the optimal index permutation for TTGT
      /// it should enable and disable to heuristic
      IndexVector rhs1OutPerm, rhs2OutPerm, lhsOutPerm;
      std::tie(rhs1OutPerm, rhs2OutPerm, lhsOutPerm) = plan.computePermutations(isSelectBestPerm, whatPerm);

      comet_debug() << "Best permutation : " << plan.bestPermStr_ << "\n";

      std::set<unsigned> rhsIndices(allPerms[0].begin(), allPerms[0].end());
      rhsIndices.insert(allPerms[1].begin(), allPerms[1].end());
      std::set<unsigned> lhsIndices(allPerms[2].begin(), allPerms[2].end());

      std::vector<unsigned> sumIndices;

      std::set_difference(rhsIndices.begin(), rhsIndices.end(),
                          lhsIndices.begin(), lhsIndices.end(),
                          std::inserter(sumIndices, sumIndices.begin()));

      std::vector<unsigned int> rhs1InPerm = getIdentityPermutation(allPerms[0].size());
      std::vector<unsigned int> rhs2InPerm = getIdentityPermutation(allPerms[1].size());
      std::vector<unsigned int> lhsInPerm = getIdentityPermutation(allPerms[2].size());

      AffineMapAttr rhs1OutMapAttr = AffineMapAttr::get(AffineMap::getPermutationMap(rhs1OutPerm, ctx));
      [[maybe_unused]] AffineMap rhs1InMap = AffineMap::getPermutationMap(rhs1InPerm, ctx);   /// [PT] Remove?
      [[maybe_unused]] AffineMap rhs1OutMap = AffineMap::getPermutationMap(rhs1OutPerm, ctx); /// [PT] Remove?

      AffineMapAttr rhs2OutMapAttr =
          AffineMapAttr::get(AffineMap::getPermutationMap(rhs2OutPerm, ctx));
      [[maybe_unused]] AffineMap rhs2InMap = AffineMap::getPermutationMap(rhs2InPerm, ctx);   /// [PT] Remove?
      [[maybe_unused]] AffineMap rhs2OutMap = AffineMap::getPermutationMap(rhs2OutPerm, ctx); /// [PT] Remove?

      AffineMapAttr lhsOutMapAttr =
          AffineMapAttr::get(AffineMap::getPermutationMap(lhsOutPerm, ctx));
      [[maybe_unused]] AffineMap lhsInMap = AffineMap::getPermutationMap(lhsInPerm, ctx);   /// [PT] Remove?
      [[maybe_unused]] AffineMap lhsOutMap = AffineMap::getPermutationMap(lhsOutPerm, ctx); /// [PT] Remove?

      Value rhs1Alloc = rhs1Memref;
      Value rhs2Alloc = rhs2Memref;
      Value lhsAlloc = lhsMemref;

      std::vector<int64_t> rhs1OutPerm_int64(rhs1OutPerm.begin(), rhs1OutPerm.end());
      std::vector<int64_t> rhs2OutPerm_int64(rhs2OutPerm.begin(), rhs2OutPerm.end());
      std::vector<int64_t> lhsOutPerm_int64(lhsOutPerm.begin(), lhsOutPerm.end());

      /// Do transpose if needed
      if (!rhs1OutMapAttr.getValue().isIdentity())
      {
        auto shape = rhs1MemrefType.getShape();
        std::vector<Value> operands;
        std::vector<int64_t> rhs1Dims;
        for (auto idx : rhs1OutPerm)
        {
          rhs1Dims.push_back(shape[idx]);
          if (rhs1MemrefType.isDynamicDim(idx))
          {
            operands.push_back(rewriter.create<memref::DimOp>(loc, rhs1Memref, idx));
          }
        }

        rhs1Alloc = insertAllocAndDeallocDynamic(
            MemRefType::get(rhs1Dims, rhs1MemrefType.getElementType()), operands, loc,
            rewriter);

#ifdef DEBUG_MODE_TTGT
        auto rhs1LinalgCopy = rewriter.create<linalg::TransposeOp>(loc, rhs1Memref, rhs1Alloc, llvm::ArrayRef<int64_t>(rhs1OutPerm_int64));
        comet_debug() << "\n";
        comet_vdump(rhs1LinalgCopy);
#else
        rewriter.create<linalg::TransposeOp>(loc, rhs1Memref, rhs1Alloc, llvm::ArrayRef<int64_t>(rhs1OutPerm_int64));
#endif
      }

      if (!rhs2OutMapAttr.getValue().isIdentity())
      {
        std::vector<Value> operands;
        std::vector<int64_t> rhs2Dims;
        auto shape = rhs2MemrefType.getShape();
        for (auto idx : rhs2OutPerm)
        {
          rhs2Dims.push_back(shape[idx]);
          if (rhs2MemrefType.isDynamicDim(idx))
          {
            operands.push_back(rewriter.create<memref::DimOp>(loc, rhs2Memref, idx));
          }
        }

        rhs2Alloc = insertAllocAndDeallocDynamic(
            MemRefType::get(rhs2Dims, rhs2MemrefType.getElementType()), operands, loc,
            rewriter);
#ifdef DEBUG_MODE_TTGT
        auto rhs2LinalgCopy = rewriter.create<linalg::TransposeOp>(loc, rhs2Memref, rhs2Alloc, llvm::ArrayRef<int64_t>(rhs2OutPerm_int64));
        comet_debug() << " rhs2LinalgCopy op: " << __LINE__ << "\n";
        comet_vdump(rhs2LinalgCopy);
#else
        rewriter.create<linalg::TransposeOp>(loc, rhs2Memref, rhs2Alloc, llvm::ArrayRef<int64_t>(rhs2OutPerm_int64));

#endif
      }

      bool useLHSTranspose = false;
      if (!lhsOutMapAttr.getValue().isIdentity())
      {
        std::vector<Value> operands;
        std::vector<int64_t> lhsDims;
        auto shape = lhsMemrefType.getShape();
        for (auto idx : lhsOutPerm)
        {
          lhsDims.push_back(shape[idx]);
          if (lhsMemrefType.isDynamicDim(idx))
          {
            operands.push_back(rewriter.create<memref::DimOp>(loc, lhsMemref, idx));
          }
        }

        lhsAlloc = insertAllocAndDeallocDynamic(
            MemRefType::get(lhsDims, lhsMemrefType.getElementType()), operands, loc,
            rewriter);
        useLHSTranspose = true;
        double beta_val = betaAttr.cast<FloatAttr>().getValueAsDouble();

        if (beta_val == 0)
        {
          Value constantOp = rewriter.create<ConstantOp>(loc, rewriter.getF64FloatAttr(0.0));
          rewriter.create<linalg::FillOp>(loc, constantOp, lhsAlloc);
        }
        else
        {
          rewriter.create<linalg::TransposeOp>(loc, lhsMemref, lhsAlloc, llvm::ArrayRef<int64_t>(lhsOutPerm_int64));
        }
      }

      Value rhs1Reshape = rhs1Alloc;
      Value rhs2Reshape = rhs2Alloc;
      Value lhsReshape = lhsAlloc;

      unsigned mIdxSize = plan.m_indices_.size();
      unsigned nIdxSize = plan.n_indices_.size();
      unsigned kIdxSize = plan.k_indices_.size();

      bool isRHS1SumPermutation = arePermutations(allPerms[0], sumIndices);
      bool isRHS2SumPermutation = arePermutations(allPerms[1], sumIndices);

      comet_debug() << __LINE__ << "mIdxSize, nIdxSize, kIdxSize: "
                    << mIdxSize << ", "
                    << nIdxSize << ", "
                    << kIdxSize << " isRHS1SumPermutation, isRHS2SumPermutation: "
                    << isRHS1SumPermutation << ", "
                    << isRHS2SumPermutation << "\n";

      /// Do reshape if needed
      if (isRHS1SumPermutation)
      {
        auto resultShape = rhs1MemrefType.getShape();

        auto rhs1AffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);

        SmallVector<AffineMap, 2> rhs1IndexingMap{rhs1AffineMap};

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(rhs1IndexingMap);

        comet_debug() << "\n";
        rhs1Reshape = rewriter.create<memref::CollapseShapeOp>(
            loc, rhs1Alloc, reassociationIndices);
        comet_vdump(rhs1Reshape);
      }
      else if (rhs1MemrefType.getShape().size() != 2)
      {
        auto resultShape = rhs1MemrefType.getShape();
        /// Construct combined shape of 2D memrefc
        std::vector<unsigned> rhs1_0, rhs1_1;

        if (plan.swapAB_)
        {
          rhs1_0 = getIndexRange(0, kIdxSize);
          rhs1_1 = getIndexRange(kIdxSize, kIdxSize + mIdxSize);
        }
        else
        {
          rhs1_0 = getIndexRange(0, mIdxSize);
          rhs1_1 = getIndexRange(mIdxSize, mIdxSize + kIdxSize);
        }

        auto rhs1AffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);
        auto rhs1Subset0 = rhs1AffineMap.getSubMap(rhs1_0);
        auto rhs1Subset1 = rhs1AffineMap.getSubMap(rhs1_1);

        SmallVector<AffineMap, 2> rhs1IndexingMap;

        rhs1IndexingMap.push_back(rhs1Subset0);
        rhs1IndexingMap.push_back(rhs1Subset1);

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(rhs1IndexingMap);

        comet_debug() << "\n";
        comet_debug() << " rhs1Alloc: \n";
        comet_vdump(rhs1Alloc);
        comet_vdump(rhs1MemrefType);

        rhs1Reshape = rewriter.create<memref::CollapseShapeOp>(
            loc, rhs1Alloc, reassociationIndices);
        comet_debug() << " Before rhs1Reshape: \n";
        comet_vdump(rhs1Reshape);
        comet_debug() << " After rhs1Reshape: \n";
      }

      if (isRHS2SumPermutation && rhs2MemrefType.getShape().size() != 1)
      {
        auto resultShape = rhs2MemrefType.getShape();

        auto rhs2AffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);

        SmallVector<AffineMap, 2> rhs2IndexingMap{rhs2AffineMap};

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(rhs2IndexingMap);

        rhs2Reshape = rewriter.create<memref::CollapseShapeOp>(
            loc, rhs2Alloc, reassociationIndices);

        comet_debug() << "\n";
        comet_vdump(rhs2Reshape);
      }
      else if (rhs2MemrefType.getShape().size() != 2 && rhs2MemrefType.getShape().size() != 1)
      {
        auto resultShape = rhs2MemrefType.getShape();

        /// Construct combined shape of 2D memref
        std::vector<unsigned> rhs2_0, rhs2_1;

        if (plan.swapAB_)
        {
          rhs2_0 = getIndexRange(0, nIdxSize);
          rhs2_1 = getIndexRange(nIdxSize, nIdxSize + kIdxSize);
        }
        else
        {
          rhs2_0 = getIndexRange(0, kIdxSize);
          rhs2_1 = getIndexRange(kIdxSize, kIdxSize + nIdxSize);
        }

        auto rhs2AffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);
        auto rhs2Subset0 = rhs2AffineMap.getSubMap(rhs2_0);
        auto rhs2Subset1 = rhs2AffineMap.getSubMap(rhs2_1);

        SmallVector<AffineMap, 2> rhs2IndexingMap;

        rhs2IndexingMap.push_back(rhs2Subset0);
        rhs2IndexingMap.push_back(rhs2Subset1);

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(rhs2IndexingMap);

        rhs2Reshape = rewriter.create<memref::CollapseShapeOp>(
            loc, rhs2Alloc, reassociationIndices);

        comet_debug() << "\n";
        comet_vdump(rhs2Reshape);
      }

      bool expandLHS = false;
      /// Keep the reassociation indices that will be used for collapsing the LHS tensor
      /// The exact same indices can be used to re-expand it back to its original rank (after the potential transpose operation)
      SmallVector<ReassociationIndices> lhsReassociationIndices;

      comet_debug() << "\n";
      if (isRHS1SumPermutation || (isRHS2SumPermutation && rhs2MemrefType.getShape().size() != 1))
      {
        comet_debug() << "\n";
        auto resultShape = lhsMemrefType.getShape();

        auto lhsAffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);

        SmallVector<AffineMap, 2> lhsIndexingMap{lhsAffineMap};

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(lhsIndexingMap);

        /// TODO(gkestor): should it be expandop?
        lhsReshape = rewriter.create<memref::CollapseShapeOp>(
            loc, lhsAlloc, reassociationIndices);

        comet_debug() << "\n";
        comet_vdump(lhsReshape);
        expandLHS = true;
        lhsReassociationIndices = reassociationIndices;
      }
      else if (lhsMemrefType.getShape().size() != 2 && lhsMemrefType.getShape().size() != 1)
      {
        comet_debug() << "\n";
        auto resultShape = lhsMemrefType.getShape();
        /// Construct combined shape of 2D memref
        std::vector<unsigned> lhs_0, lhs_1;
        if (plan.swapAB_)
        {
          lhs_0 = getIndexRange(0, nIdxSize);
          lhs_1 = getIndexRange(nIdxSize, nIdxSize + mIdxSize);
        }
        else
        {
          lhs_0 = getIndexRange(0, mIdxSize);
          lhs_1 = getIndexRange(mIdxSize, mIdxSize + nIdxSize);
        }

        auto lhsAffineMap = AffineMap::getPermutationMap(
            getIdentityPermutation(resultShape.size()), ctx);
        auto lhsSubset0 = lhsAffineMap.getSubMap(lhs_0);
        auto lhsSubset1 = lhsAffineMap.getSubMap(lhs_1);

        SmallVector<AffineMap, 2> lhsIndexingMap;

        lhsIndexingMap.push_back(lhsSubset0);
        lhsIndexingMap.push_back(lhsSubset1);

        SmallVector<ReassociationIndices> reassociationIndices =
            getReassociationIndices(lhsIndexingMap);

        lhsReshape = rewriter.create<memref::CollapseShapeOp>(
            loc, lhsAlloc, reassociationIndices);
        comet_debug() << "\n";
        comet_vdump(lhsReshape);

        expandLHS = true;
        lhsReassociationIndices = reassociationIndices;
      }

      comet_debug() << "\n";
      /// Create linalg matmul op
      linalg::MatmulOp matmulop;
      linalg::MatvecOp matvecop;

      if (isRHS1SumPermutation)
      {
        comet_debug() << "\n";
        matvecop = rewriter.create<linalg::MatvecOp>(
            loc, ValueRange{rhs2Reshape, rhs1Reshape},
            ValueRange{lhsReshape});
        comet_debug() << "\n";
        comet_vdump(matvecop);

        matvecop.getOperation()->setAttr("__alpha__", alphaAttr);
        matvecop.getOperation()->setAttr("__beta__", betaAttr);

        /// TODO(gkestor): Add attribute to the linalg.matvec operations
        /// matvecop.setAttr(kLinalgTransformMarker, rewriter.getStringAttr(kLinalgTransformMarker));
      }
      else if (isRHS2SumPermutation)
      {
        comet_debug() << "\n";
        matvecop = rewriter.create<linalg::MatvecOp>(
            loc, ValueRange{rhs1Reshape, rhs2Reshape},
            ValueRange{lhsReshape});
        comet_debug() << "\n";
        comet_vdump(rhs1Reshape);
        comet_vdump(rhs2Reshape);
        comet_vdump(lhsReshape);
        comet_vdump(matvecop);

        matvecop.getOperation()->setAttr("__alpha__", alphaAttr);
        matvecop.getOperation()->setAttr("__beta__", betaAttr);

        /// TODO(gkestor): Add attribute to the linalg.matvec operations
        /// matvecop.setAttr(kLinalgTransformMarker, rewriter.getStringAttr(kLinalgTransformMarker));
      }
      else
      {
        comet_debug() << "\n";
        if (plan.swapAB_)
        {
          /// TODO(gkestor) - there is error with the building process
          matmulop = rewriter.create<linalg::MatmulOp>(
              loc, ValueRange{rhs2Reshape, rhs1Reshape},
              ValueRange{lhsReshape});
          comet_debug() << "\n";
          comet_vdump(matmulop);
        }
        else
        {
          matmulop = rewriter.create<linalg::MatmulOp>(
              loc, ValueRange{rhs1Reshape, rhs2Reshape},
              ValueRange{lhsReshape});
          comet_debug() << "\n";
          comet_vdump(rhs1Reshape);
          comet_vdump(rhs2Reshape);
          comet_vdump(lhsReshape);
          comet_vdump(matmulop);
        }
        comet_debug() << "\n";
        /// Add attribute to the linalg.matmul operations
        matmulop.getOperation()->setAttr(kLinalgTransformMarker,
                                         rewriter.getStringAttr(kLinalgTransformMarker));
        matmulop.getOperation()->setAttr("__alpha__", alphaAttr);
        matmulop.getOperation()->setAttr("__beta__", betaAttr);
      }

      Value lhsExpand = lhsReshape;
      if (expandLHS) /// LHS tensor was collapsed and now needs to be re-expanded using the same reassociation indices
      {
        auto expandedTensorType = MemRefType::get(lhsAlloc.getType().cast<MemRefType>().getShape(), lhsAlloc.getType().cast<MemRefType>().getElementType());

        comet_debug() << "\nExpanded:\n";
        lhsExpand = rewriter.create<memref::ExpandShapeOp>(
            loc, expandedTensorType, lhsReshape, lhsReassociationIndices);
#ifdef DEBUG_MODE_TTGT
        comet_debug() << "\n";
        comet_vdump(lhsExpand);
#endif
      }

      /// Copy back the result if needed
      if (lhsAlloc != lhsMemref && useLHSTranspose)
      {
        std::vector<int64_t> revLhsOutPerm(lhsOutPerm_int64.size());
        for (size_t i = 0; i < revLhsOutPerm.size(); i++)
          revLhsOutPerm[lhsOutPerm_int64[i]] = i;

#ifdef DEBUG_MODE_TTGT
        auto lhsFinalCopy =
            rewriter.create<linalg::TransposeOp>(loc, lhsExpand, lhsMemref, llvm::ArrayRef<int64_t>(revLhsOutPerm));
        comet_debug() << "\n";
        comet_vdump(lhsFinalCopy);
#else
        rewriter.create<linalg::TransposeOp>(loc, lhsExpand, lhsMemref, llvm::ArrayRef<int64_t>(revLhsOutPerm));

#endif
      }

      if (printFlops)
      {
        auto endTime = rewriter.create<func::CallOp>(
            loc, getTimeStr, SmallVector<Type, 2>{f64Type});

        auto start = startTime->getResult(0);
        auto end = endTime.getResult(0);

        Value totalTimeValue =
            rewriter.create<SubFOp>(loc, f64Type, end, start);

        double opNums = 2.0 * plan.m_size_ * plan.n_size_ * plan.k_size_;

        Value numFlopsOp =
            rewriter.create<ConstantOp>(loc, FloatAttr::get(f64Type, opNums));

        Value flopsOp =
            rewriter.create<DivFOp>(loc, f64Type, numFlopsOp, totalTimeValue);

        ///   call @print_flops(%flops) : (f64) -> ()
        std::string printFlopsStr = "print_flops";
        /// auto printFlopsCall =
        rewriter.create<func::CallOp>(
            loc, printFlopsStr, SmallVector<Type, 2>{}, ValueRange{flopsOp});
      }

      rewriter.eraseOp(setnewop);
      rewriter.eraseOp(op);
      return success();
    }

  private:
    bool isSelectBestPerm;
    int whatPerm;
    bool printFlops;
  }; /// namespace

  struct TALoweringTTGTPass
      : public PassWrapper<TALoweringTTGTPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TALoweringTTGTPass)
    TALoweringTTGTPass(bool isSelectBestPerm, int whatPerm, bool printFlops) : isSelectBestPerm(isSelectBestPerm), whatPerm(whatPerm), printFlops{printFlops} {};
    void runOnOperation() override;

  private:
    bool isSelectBestPerm;
    int whatPerm;
    bool printFlops;
  };

} /// end anonymous namespace.

void TALoweringTTGTPass::runOnOperation()
{
  func::FuncOp function = getOperation();
  auto module = function.getOperation()->getParentOfType<ModuleOp>();
  auto *ctx = &getContext();

  auto getTimeFunc = FunctionType::get(ctx, {}, {FloatType::getF64(ctx)});
  auto printFlopFunc = FunctionType::get(ctx, {FloatType::getF64(ctx)}, {});

  /// func @getTime() -> f64
  if (!hasFuncDeclaration(module, "getTime"))
  {
    mlir::func::FuncOp func1 = mlir::func::FuncOp::create(function.getLoc(), "getTime", getTimeFunc,
                                                          ArrayRef<NamedAttribute>{});
    func1.setPrivate();
    module.push_back(func1);
  }

  /// func @print_flops(%flops) : (f64) -> ()
  if (!hasFuncDeclaration(module, "print_flops"))
  {
    mlir::func::FuncOp func1 = mlir::func::FuncOp::create(function.getLoc(), "print_flops",
                                                          printFlopFunc, ArrayRef<NamedAttribute>{});
    func1.setPrivate();
    module.push_back(func1);
  }

  RewritePatternSet patterns(&getContext());
  patterns.insert<TensorContractionOpLoweringTTGT>(&getContext(), isSelectBestPerm, whatPerm, printFlops);

  ConversionTarget target(getContext());
  target.addLegalDialect<LinalgDialect, ArithDialect, memref::MemRefDialect, bufferization::BufferizationDialect>();

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    llvm::errs() << "Failed to applyPartialConversion in TALoweringTTGTPass\n";
    signalPassFailure();
  }
}

/// Create a pass for lowering operations in the `LinAlg` and `Std` dialects,
/// for a subset of the TA IR (e.g. matmul).
/// ordering of permutation starts with one
std::unique_ptr<Pass> mlir::comet::createLoweringTTGTPass(bool isSelectBestPerm, int whatPerm, bool printFlops)
{
  return std::make_unique<TALoweringTTGTPass>(isSelectBestPerm, whatPerm, printFlops);
}
