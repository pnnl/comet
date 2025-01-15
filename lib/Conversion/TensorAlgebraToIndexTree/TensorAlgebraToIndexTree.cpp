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

#include <algorithm>
#include <cassert>
#include <unordered_set>

#include "comet/Conversion/TensorAlgebraToIndexTree/TensorAlgebraToIndexTree.h"

#include "comet/Dialect/IndexTree/Transforms/Tensor.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Transforms/UnitExpression.h"
#include "comet/Dialect/IndexTree/IR/IndexTree.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

using namespace mlir;

namespace
{
  struct LowerTensorAlgebraToIndexTreePass
      : public PassWrapper<LowerTensorAlgebraToIndexTreePass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorAlgebraToIndexTreePass)
    LowerTensorAlgebraToIndexTreePass(TargetDevice device) : device(device){};
    void runOnOperation() override;
  
    TargetDevice device;
  };

} /// namespace

/**
 * @brief Check if the given allPerms is from one of the chosen operations.
 * Current algorithm is to check the pattern of the allPerms to see if it is tensor contraction.
 * Current chosen operations includes:
 * dense matrix-matrix multiplication (MM),
 * dense matrix-vector multiplication (MV),
 * sparse matrix-dense matrix multiplication (SpMM),
 * sparse matrix-dense vector multiplication (SpMV)
 *
 * @param allPerms allPerms from the operation. For example, [[d0, d1], [d1, d2], [d0, d2]]
 * @return true : it is one of the chosen operations.
 * @return false : it is not.
 */
bool check_chosen_operations(const std::vector<std::vector<int64_t>> &allPerms,
                             const std::vector<std::vector<std::string>> &allFormats)
{
  if (allPerms.size() != 3)
  {
    return false;
  }

  /// The output tensor should be dense.
  auto lhs_formats = allFormats[2];
  for (auto &f : lhs_formats)
  {
    if (f != "D")
    {
      return false;
    }
  }

  /// do lhs = op(rhs1, rhs2)
  const std::vector<int64_t> &rhs1_perms = allPerms[0];
  const std::vector<int64_t> &rhs2_perms = allPerms[1];
  const std::vector<int64_t> &lhs_perms = allPerms[2];

  if (rhs1_perms.size() == 2)
  {
    if (rhs2_perms.size() == 2 && lhs_perms.size() == 2)
    {
      /// If it is op(matrix, matrix)
      if (rhs1_perms[0] == lhs_perms[0] &&
          rhs1_perms[1] == rhs2_perms[0] &&
          rhs2_perms[1] == lhs_perms[1])
      {
        /// Then op is MM or SpMM
        return true;
      }
    }
    else if (rhs2_perms.size() == 1 && lhs_perms.size() == 1)
    {
      /// If it is op(matrix, vector)
      if (rhs1_perms[0] == lhs_perms[0] &&
          rhs1_perms[1] == rhs2_perms[0])
      {
        /// Then op is MV or SpMV
        return true;
      }
    }
  }

  return false;
}

Value getRealLhs(Operation *op)
{
  assert(isa<TensorMultOp>(op) || isa<TensorElewsMultOp>(op) || isa<TensorAddOp>(op) || isa<TensorSubtractOp>(op));
  Operation *firstUser = nullptr;
  for (auto user : op->getResult(0).getUsers())
  {
    firstUser = user;
    break;
  }

  comet_pdump(firstUser);
  assert(isa<TensorSetOp>(firstUser));
  TensorSetOp setOp = cast<TensorSetOp>(firstUser);
  return setOp.getOperand(1);
}

Value getRealRhs(Value val)
{
  /// this will return set_op for transpose, but messes up getUsers() or subsequent calls to it.

  /// TODO(gkestor): need to find out why user set_op is not showing up in users of TransposeOp
  ///       from the for loop below. once resolved, remove getNextNode().
  /// Operation *firstUser;
  /// for (auto user : op->getResult(0).getUsers())
  //{
  ///  firstUser = user;
  ///  break;
  //}
  if(Operation* op = val.getDefiningOp())
  {
    Operation *firstUser = op->getNextNode();
    comet_pdump(firstUser);

    if (isa<tensorAlgebra::TransposeOp>(op))
    {
      if (isa<TensorSetOp>(firstUser))
      {
        TensorSetOp setOp = cast<TensorSetOp>(firstUser);
        return setOp.getOperand(1);
      }
      else
      {
        llvm::errs() << "ERROR: Transpose has no set_op after it!\n";
      }
      
    }
  }

  return val;
}

// void buildDefUseInfo(UnitExpression *e)
// {
//   auto lhs = e->getLHS();
//   lhs->setDefiningExpr(e);
//   for (auto operand : e->getOperands())
//   {
//     if (auto def = operand->getDefiningExpr())
//     {
//       def->addUser(e);
//     }
//   }
// }

// IndicesType getUnion(IndicesType indices1, IndicesType indices2)
// {
//   sort(indices1.begin(), indices1.end());
//   sort(indices2.begin(), indices2.end());

//   IndicesType allIndices(indices1.size() * 4);

//   IndicesType::iterator it = set_union(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(), allIndices.begin());
//   allIndices.resize(it - allIndices.begin());
//   return allIndices;
// }

// IndicesType gpuIndices(IndicesType indices1, IndicesType indices2)
// {
//   sort(indices1.begin(), indices1.end());
//   sort(indices2.begin(), indices2.end());

//   IndicesType interIndices;
//   IndicesType unIndices;
//   IndicesType difIndices;
//   IndicesType allIndices;

//   std::set_intersection(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(),  std::back_inserter(interIndices));
//   set_union(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(), std::back_inserter(unIndices));
//   std::set_difference(unIndices.begin(), unIndices.end(), interIndices.begin(), interIndices.end(), std::back_inserter(difIndices));
//   allIndices = difIndices;
//   allIndices.insert(allIndices.end(), interIndices.begin(), interIndices.end());

//   // allIndices.resize(it - allIndices.begin());
//   return allIndices;  
// }


template<class TATensorOp>
mlir::LogicalResult generalIndexOperationRewrite(
    mlir::Operation* op, 
    ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter,
    bool compute_missing = false) {

  auto loc = op->getLoc();
  auto context = rewriter.getContext();
  TATensorOp mult_op = llvm::dyn_cast<TATensorOp>(op);

  Value rhs1_tensor = getRealRhs(mult_op.getRhs1());
  Value rhs2_tensor = getRealRhs(mult_op.getRhs2());
  Value lhs_tensor = getRealLhs(op);

  Value mask_tensor = nullptr;
  if(llvm::isa<TensorMultOp>(op)){
    mask_tensor = llvm::cast<TensorMultOp>(op).getMask();
    if(mask_tensor && (mask_tensor == rhs1_tensor || mask_tensor == rhs2_tensor))
    {
      mask_tensor = rewriter.create<SpTensorAliasOp>(loc, mask_tensor.getType(), mask_tensor);
    }
  }

  auto indexing_maps = mult_op.getIndexingMaps();
  auto semiring = mult_op.getSemiringAttr().template cast<mlir::StringAttr>().getValue();
  auto MaskingTypeAttr = mult_op.getMaskTypeAttr();

  auto tensor_type = op->getResultTypes()[0];
  auto itree_op = rewriter.create<IndexTreeOp>(loc, tensor_type, lhs_tensor);
  Region* body = &itree_op.getRegion();
  loc = body->getLoc();
  Block* block = rewriter.createBlock(body);

  indexTree::IndexTreeType tree_type = indexTree::IndexTreeType::get(context);
  Value parent = rewriter.create<indexTree::IndexTreeRootOp>(loc, tree_type);

  //Construct each index variable
  auto lhsMap = cast<AffineMapAttr>(indexing_maps[2]).getValue();
  indexTree::IndexNodeType index_node_type = indexTree::IndexNodeType::get(context); 
  std::vector<Value> index_nodes;
  bool is_parallel = true; // Outer-most, non-reduction dimensions are parallel

  // TODO: For now, we do not support outputting sparse tensors from a parallel loop.
  if(llvm::isa<SparseTensorType>(tensor_type))
  {
    is_parallel = false;
  }
  
  for (unsigned i = 0; i < lhsMap.getNumDims(); i++)
  {
    if(!lhsMap.isFunctionOfDim(i)){
      is_parallel = false;
    }
    parent = rewriter.create<indexTree::IndexTreeIndicesOp>(loc, index_node_type, parent, nullptr, is_parallel);
    index_nodes.push_back(parent);
  }

  //Construct LHS Operand
  llvm::SmallVector<Value> pos;
  llvm::SmallVector<Value> crds;
  Value prev_dim = nullptr;
  llvm::SmallVector<Value> mask_pos;
  llvm::SmallVector<Value> mask_crds;
  Value mask_prev_dim;
  auto access_type = rewriter.getIndexType();
  for (size_t i = 0; i < lhsMap.getNumResults(); i++)
  {
    auto expr = lhsMap.getResult(i);
    IndexTreeIndexToTensorOp access_op = rewriter.create<IndexTreeIndexToTensorOp>(
      loc,
      TypeRange({access_type, access_type}),
      lhs_tensor,
      index_nodes[expr.template cast<AffineDimExpr>().getPosition()],
      rewriter.getUI32IntegerAttr((unsigned)i),
      prev_dim
    );
    pos.push_back(access_op.getPos());
    crds.push_back(access_op.getCrd());
    prev_dim = pos[pos.size() - 1];

    if(mask_tensor != nullptr)
    {
      IndexTreeIndexToTensorOp access_op = rewriter.create<IndexTreeIndexToTensorOp>(
        loc,
        TypeRange({access_type, access_type}),
        mask_tensor,
        index_nodes[expr.template cast<AffineDimExpr>().getPosition()],
        rewriter.getUI32IntegerAttr((unsigned)i),
        mask_prev_dim
      );
      mask_pos.push_back(access_op.getPos());
      mask_crds.push_back(access_op.getCrd());
      mask_prev_dim = mask_pos[mask_pos.size() - 1];
    }
  }

  indexTree::OperandType operand_type = indexTree::OperandType::get(context);
  Value lhs_operand = rewriter.create<indexTree::IndexTreeLHSOperandOp>(loc, operand_type,
                                                                        lhs_tensor, pos,
                                                                        crds);
  Value mask_operand = nullptr;
  if(mask_tensor != nullptr)
  {
    mask_operand = rewriter.create<indexTree::IndexTreeOperandOp>(loc, operand_type,
                                                                      mask_tensor, mask_pos,
                                                                      mask_crds);
  }

  //Construct RHS operands
  std::vector<Value> rhs_operands;
  pos.clear();
  crds.clear();
  prev_dim = nullptr;
  auto affineMap = indexing_maps[0].template cast<AffineMapAttr>().getValue();
  for (size_t i = 0; i < affineMap.getNumResults(); i++)
  {
    auto expr = affineMap.getResult(i);
    IndexTreeIndexToTensorOp access_op = rewriter.create<indexTree::IndexTreeIndexToTensorOp>(
      loc,
      TypeRange({access_type, access_type}),
      rhs1_tensor,
      index_nodes[expr.template cast<AffineDimExpr>().getPosition()],
      rewriter.getUI32IntegerAttr((unsigned)i),
      prev_dim
    );
    pos.push_back(access_op.getPos());
    crds.push_back(access_op.getCrd());
    prev_dim = pos[pos.size() - 1];
  }
  rhs_operands.push_back(rewriter.create<IndexTreeOperandOp>(
                          loc, operand_type, rhs1_tensor, pos, crds));
  
  pos.clear();
  crds.clear();
  prev_dim = nullptr;
  affineMap = indexing_maps[1].template cast<AffineMapAttr>().getValue();
  for (size_t i = 0; i < affineMap.getNumResults(); i++)
  {
    auto expr = affineMap.getResult(i);
    IndexTreeIndexToTensorOp access_op = rewriter.create<IndexTreeIndexToTensorOp>(
      loc,
      TypeRange({access_type, access_type}),
      rhs2_tensor,
      index_nodes[expr.template cast<AffineDimExpr>().getPosition()],
      rewriter.getUI32IntegerAttr((unsigned)i),
      prev_dim
    );
    pos.push_back(access_op.getPos());
    crds.push_back(access_op.getCrd());
    prev_dim = pos[pos.size() - 1];
  }
  rhs_operands.push_back(rewriter.create<indexTree::IndexTreeOperandOp>(
                          loc, operand_type, rhs2_tensor, pos, crds));

  Value compute_op = rewriter.create<indexTree::IndexTreeComputeOp>(
      loc,
      tensor_type,
      parent,
      lhs_operand,
      rhs_operands,
      mask_operand,
      rewriter.getStringAttr(semiring),
      rewriter.getBoolAttr(compute_missing)
  );

  rewriter.create<indexTree::YieldOp>(loc, TypeRange(), compute_op);
  rewriter.replaceOp(op, itree_op->getResults());
  return success();
}

struct TensorMultOpLowering : public mlir::ConversionPattern {
  TensorMultOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TensorMultOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    return generalIndexOperationRewrite<TensorMultOp>(op, operands, rewriter);
  }
};

struct TensorElewsMultOpLowering : public mlir::ConversionPattern {
  TensorElewsMultOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TensorElewsMultOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    return generalIndexOperationRewrite<TensorElewsMultOp>(op, operands, rewriter);
  }
};

struct TensorAddOpLowering : public mlir::ConversionPattern {
  TensorAddOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TensorAddOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    return generalIndexOperationRewrite<TensorAddOp>(op, operands, rewriter, true);
  }
};

struct TensorSubtractOpLowering : public mlir::ConversionPattern {
  TensorSubtractOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TensorSubtractOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    return generalIndexOperationRewrite<TensorSubtractOp>(op, operands, rewriter, true);
  }
};

void LowerTensorAlgebraToIndexTreePass::runOnOperation()
{
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<indexTree::IndexTreeDialect>();
  target.addLegalOp<tensorAlgebra::SpTensorAliasOp>();
  target.addIllegalOp<tensorAlgebra::TensorMultOp, tensorAlgebra::TensorElewsMultOp,
                      tensorAlgebra::TensorAddOp, tensorAlgebra::TensorSubtractOp>();
  
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<TensorMultOpLowering, 
               TensorElewsMultOpLowering,
               TensorAddOpLowering,
               TensorSubtractOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// create all the passes.
std::unique_ptr<Pass> mlir::comet::createLowerTensorAlgebraToIndexTreePass(TargetDevice device)
{
  comet_debug() << " Calling createLowerTensorAlgebraToIndexTreePass\n";
  return std::make_unique<LowerTensorAlgebraToIndexTreePass>(device);
}
