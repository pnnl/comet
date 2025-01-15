//===- WorkspaceTransforms.cpp  ------===//
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
// This pass performs workspace transformations on index tree dialect for sparse-sparse computation
//===----------------------------------------------------------------------===//

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/IndexTree/Patterns.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


#include "llvm/Support/Debug.h"
#include <iostream>
#include <algorithm>
#include <vector>

#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <queue>
#include <tuple>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::arith;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

using llvm::SmallVector;
using llvm::StringRef;

#define DEBUG_TYPE "workspace-transformations"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//



///  Apply workspace transformation on the lhs
///  Consider CSR first
///  ikj Cij += Aik * Bkj ===> i (j Wj = 0; kj Wj += Aik * Bkj; kj Cij=Wj)
///  ij Cij = Aij * Bij =====> i (j Wj = 0; j Wj = Aij * Bij; j Cij=Wj)

///  Apply workspace transformation on the rhs
///  Consider CSR first
///  j Wj = Aij * Bij ===> j Vj = 0; j Vj = Bij; j Wj = Aij * Vj;

//===----------------------------------------------------------------------===//
/// WorkspaceTransforms Pass
//===----------------------------------------------------------------------===//

///  Apply workspace transformations on the ta.tc and tc.elews_mul
namespace
{
  struct IndexTreeWorkspaceTransformationsPass
      : public PassWrapper<IndexTreeWorkspaceTransformationsPass, OperationPass<mlir::func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexTreeWorkspaceTransformationsPass)
    void runOnOperation() override;
  };
} /// end anonymous namespace.

struct TransformSparseOutput : public OpRewritePattern<IndexTreeComputeOp> {
  TransformSparseOutput(MLIRContext *context)
    : OpRewritePattern<IndexTreeComputeOp>(context, /*benefit=*/0.5) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeComputeOp compute_op, mlir::PatternRewriter &rewriter) const override {
    IndexTreeLHSOperandOp lhs_op = compute_op.getLhs().getDefiningOp<IndexTreeLHSOperandOp>();
    Value old_output = lhs_op.getTensor();

    // Check to see if output is sparse
    if(!llvm::isa<SparseTensorType>(old_output.getType()))
      return failure();

    // Check to see if there are "redundant" inserts
    llvm::SmallDenseMap<Value, IndexTreeIndexToTensorOp> index_vars;
    for(auto pos : lhs_op.getPos()) {
      auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
      if(index_to_tensor){
        index_vars.insert(std::make_pair(
          index_to_tensor.getIndex(),
          index_to_tensor
        ));
      }
    }

    // Find last output dimension
    Value parent = compute_op.getParent();
    auto node = parent.getDefiningOp<IndexTreeIndicesOp>();
    while(index_vars.find(parent) == index_vars.end()) {
      parent = node.getParent();
      node = parent.getDefiningOp<IndexTreeIndicesOp>();
    }
    // node contains output domain
    
    // Find output dimensions after reduction variable
    // to include in workspace
    unsigned workspace_rank = 0;
    llvm::SmallVector<IndexTreeIndexToTensorOp> accesses;
    llvm::SmallVector<int32_t> dims;
    while(index_vars.find(parent) != index_vars.end()){
      auto access_op = index_vars[parent];
      accesses.push_back(access_op);
      dims.push_back(access_op.getDim());
      workspace_rank++;

      parent = node.getParent();
      node = parent.getDefiningOp<IndexTreeIndicesOp>();
      if(!node){
        return failure();
      }    
    }
    //Match success!
    // Parent contains reduction variable

    // Declare the workspace outside of the tree
    auto loc = compute_op.getLoc();
    auto tree_op = compute_op->getParentOfType<IndexTreeOp>();
    rewriter.setInsertionPoint(tree_op);
    SparseTensorType spType = llvm::cast<SparseTensorType>(old_output.getType());
    llvm::SmallVector<int64_t> dim_sizes(workspace_rank, ShapedType::kDynamic);
    Type workspace_type = WorkspaceType::get(compute_op.getContext(), spType.getElementType(), spType.getIndicesType(), dim_sizes);
    std::reverse(dims.begin(), dims.end());
    Value workspace = rewriter.create<AllocWorkspaceOp>(loc, workspace_type, old_output, rewriter.getI32ArrayAttr(dims));
    
    // Clean the workspace before use
    rewriter.setInsertionPoint(node);
    Value clean_workspace = rewriter.create<IndexTreeCleanWorkspaceOp>(loc, workspace_type, node.getParent(), workspace);

    // Create new compute op
    auto context = getContext();
    rewriter.setInsertionPoint(compute_op);
    Type index_type = rewriter.getIndexType();
    llvm::SmallVector<Value> pos;
    llvm::SmallVector<Value> crds;
    std::reverse(accesses.begin(), accesses.end());
    int32_t dim = 0;
    Value prev_dim = nullptr;
    for(auto access_op : accesses)
    {
      auto new_access_op = rewriter.create<IndexTreeIndexToTensorOp>(
        loc,
        TypeRange({index_type, index_type}),
        clean_workspace,
        access_op.getIndex(),
        rewriter.getUI32IntegerAttr(dim),
        prev_dim
      );

      pos.push_back(new_access_op.getPos());
      crds.push_back(new_access_op.getCrd());
      prev_dim = new_access_op.getPos();
      dim++;
    }

    Type operand_type = OperandType::get(context);
    Value new_lhs = rewriter.create<IndexTreeLHSOperandOp>(
      loc, 
      operand_type,
      clean_workspace,
      pos,
      crds
    );
    Value new_workspace = rewriter.create<IndexTreeComputeOp>(
      loc,
      workspace_type,
      compute_op.getParent(),
      new_lhs,
      compute_op.getRhs(),
      nullptr,
      compute_op.getSemiringAttr()
    );

    pos.clear();
    crds.clear();
    dim = 0;
    prev_dim = nullptr;
    for(auto access_op : accesses)
    {
      auto new_access_op = rewriter.create<IndexTreeIndexToTensorOp>(
        loc,
        TypeRange({index_type, index_type}),
        new_workspace,
        access_op.getIndex(),
        rewriter.getUI32IntegerAttr(dim),
        prev_dim
      );

      pos.push_back(new_access_op.getPos());
      crds.push_back(new_access_op.getCrd());
      prev_dim = new_access_op.getPos();
      dim++;
    }

    Value new_rhs = rewriter.create<IndexTreeOperandOp>(
      loc,
      operand_type,
      new_workspace,
      pos,
      crds
    );

    rewriter.replaceOpWithNewOp<IndexTreeComputeOp>(
      compute_op,
      old_output.getType(),
      compute_op.getParent(),
      compute_op.getLhs(),
      ValueRange{new_rhs,},
      nullptr,
      "noop_noop"
    );


    // Update the index tree op
    SmallVector<Value> tree_args(tree_op->getOperands());
    SmallVector<Type> tree_types(tree_op->getResultTypes());
    tree_args.push_back(workspace);
    tree_types.push_back(workspace_type);
    rewriter.setInsertionPoint(tree_op);
    auto newOp = rewriter.create<IndexTreeOp>(loc, tree_types, tree_args);
    rewriter.inlineRegionBefore(tree_op.getRegion(), newOp.getRegion(), newOp.getRegion().end());
    indexTree::YieldOp yield = cast<indexTree::YieldOp>(newOp.getRegion().getBlocks().front().getTerminator());
    rewriter.updateRootInPlace(yield, [&]() {
      yield->insertOperands(yield->getNumOperands(), ValueRange{workspace});
    });
    for(unsigned i = 0; i < tree_op.getNumResults(); i++){
      rewriter.replaceAllUsesWith(tree_op.getResult(i), newOp.getResult(i));
    }
    rewriter.eraseOp(tree_op);

    return success();    
  }
};

struct MoveInvariantComputeOp : public OpRewritePattern<IndexTreeComputeOp> {
  MoveInvariantComputeOp (MLIRContext *context)
    : OpRewritePattern<IndexTreeComputeOp>(context, /*benefit=*/2) {}

  mlir::LogicalResult
  matchAndRewrite(IndexTreeComputeOp compute_op, mlir::PatternRewriter &rewriter) const override {
    // Collect all indices used in this compute expression
    llvm::SmallDenseSet<Value> used_indices;
    IndexTreeLHSOperandOp lhs_op = compute_op.getLhs().getDefiningOp<IndexTreeLHSOperandOp>();
    for(auto pos : lhs_op.getPos()) {
      auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
      if(!index_to_tensor)
        return failure();
      used_indices.insert(index_to_tensor.getIndex());
    }
    
    auto rhs_operands = compute_op.getRhs();
    for(Value rhs : rhs_operands) {
      IndexTreeOperandOp operand_op = rhs.getDefiningOp<IndexTreeOperandOp>();
      for(auto pos : operand_op.getPos()) {
        auto index_to_tensor = pos.getDefiningOp<IndexTreeIndexToTensorOp>();
        if(!index_to_tensor)
          return failure();
        used_indices.insert(index_to_tensor.getIndex());
      }
    }

    // We want to find all of the indices that this compute op is nested under
    // and check if they are used in this compute expression. Every time we come
    // across an unused index, the index nodes that we have seen so far need to be copied
    // to form a new branch of the tree. We also keep track of the parent at the fork
    llvm::SmallVector<Value> seen_indices;
    llvm::SmallVector<Value> indices_to_copy;
    Value parent = compute_op.getParent();
    Value fork = parent;
    IndexTreeIndicesOp node = parent.getDefiningOp<IndexTreeIndicesOp>();
    while(node) {
      if(used_indices.find(parent) != used_indices.end()) {
        // Used index variable
        seen_indices.push_back(parent);
      } else {
        // Unused index variable
        fork = node.getParent();
        indices_to_copy.insert(indices_to_copy.begin(), seen_indices.rbegin(), seen_indices.rend());
        seen_indices.clear();
      }

      parent = node.getParent();
      node = parent.getDefiningOp<IndexTreeIndicesOp>();
    }

    if(fork == compute_op.getParent()) {
      return failure(); // Match failed, no indces to move.
    }

    // Success!
    IRMapping map;
    auto context = rewriter.getContext();
    auto loc = compute_op.getLoc();
    IndexNodeType index_node_type = IndexNodeType::get(context); 
    parent = fork;
    for(auto index : indices_to_copy)
    {
      Value new_index = rewriter.create<IndexTreeIndicesOp>(loc, index_node_type, parent);
      map.map(index, new_index);
      parent = new_index;
    }

    for(auto pos : lhs_op.getPos()) {
      Operation* index_to_tensor = pos.getDefiningOp();
      rewriter.clone(*index_to_tensor, map);
    }
    rewriter.clone(*lhs_op.getOperation(), map);
    
    for(Value rhs : rhs_operands) {
      IndexTreeOperandOp operand_op = rhs.getDefiningOp<IndexTreeOperandOp>();
      for(auto pos : operand_op.getPos()) {
        Operation* index_to_tensor = pos.getDefiningOp();
        rewriter.clone(*index_to_tensor, map);
      }
      rewriter.clone(*operand_op.getOperation(), map);
    }
    Operation* new_compute_op = rewriter.clone(*compute_op.getOperation(), map);
    rewriter.replaceOp(compute_op, new_compute_op->getResults());
    return success();    
  }
};

void IndexTreeWorkspaceTransformationsPass::runOnOperation()
{
  comet_debug() << __FILE__ << " " << __LINE__ << " starting CompressedWorkspaceTransforms pass \n";
  mlir::RewritePatternSet workspace_transformation_patterns(&getContext());

  workspace_transformation_patterns.add<TransformSparseOutput, MoveInvariantComputeOp>(&getContext());
  indexTree::populateDomainInferencePatterns(&getContext(), workspace_transformation_patterns); //For new index variables
  indexTree::populateDomainConcretizationPatterns(&getContext(), workspace_transformation_patterns);
  mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(workspace_transformation_patterns));
  comet_debug() << __FILE__ << " " << __LINE__ << " ending CompressedWorkspaceTransforms pass \n";
}

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeWorkspaceTransformationsPass()
{
  return std::make_unique<IndexTreeWorkspaceTransformationsPass>();
}