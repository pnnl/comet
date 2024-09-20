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
#undef COMET_DEBUG_MODE
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
 * @brief Check if the given allPerms and allFormats are from one of the chosen operations.
 * Current algorithm is to check the pattern of the allPerms to see if it is tensor contraction.
 * Current chosen operations includes:
 * dense matrix-matrix multiplication (MM),
 * dense matrix-vector multiplication (MV),
 * sparse matrix-dense matrix multiplication (SpMM),
 * sparse matrix-dense vector multiplication (SpMV)
 *
 * @param allPerms allPerms from the operation. For example, [[d0, d1], [d1, d2], [d0, d2]]. RHS is allPerms[0,1], LHS is allPerms[2].
 * @param allFormats allFormats from the operation; the same size with allPerms. For example, [["D", "D"], ["D", "CU"], ["D", "D"]]. RHS is allFormats[0,1], LHS is allFormats[2].
 * @return true : it is one of the chosen operations.
 * @return false : it is not.
 */
bool checkChosenDenseMixedOperations(const std::vector<std::vector<int64_t>> &allPerms,
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


/**
 * @brief Check if the given allPerms and allFormats are from SpGEMM operation.
 * @param allPerms allPerms from the operation. For example, [[d0, d1], [d1, d2], [d0, d2]]. RHS is allPerms[0,1], LHS is allPerms[2].
 * @param allFormats allFormats from the operation; the same size with allPerms. For example, [["D", "CU"], ["D", "CU"], ["D", "CU"]]. RHS is allFormats[0,1], LHS is allFormats[2].
 * @return true : it is one of the chosen operations.
 * @return false : it is not.
 */
bool checkChosenSpGEMMOperation(const std::vector<std::vector<int64_t>> &allPerms,
                                const std::vector<std::vector<std::string>> &allFormats)
{
  if (allPerms.size() != 3)
  {
    return false;
  }

  /// The output tensor should be sparse.
  auto lhs_formats = allFormats[2];
  if (!(lhs_formats[0] == "D" && lhs_formats[1] == "CU")) {
    return false;
  }

  /// do lhs = op(rhs1, rhs2)
  const std::vector<int64_t> &rhs1_perms = allPerms[0];
  const std::vector<int64_t> &rhs2_perms = allPerms[1];
  const std::vector<int64_t> &lhs_perms = allPerms[2];

  if (rhs1_perms.size() == 2 && rhs2_perms.size() == 2 && lhs_perms.size() == 2) {
    /// SpGEMM is C[i,j] = A[i,k] * B[k,j]
    if (rhs1_perms[0] == lhs_perms[0] &&
        rhs1_perms[1] == rhs2_perms[0] &&
        rhs2_perms[1] == lhs_perms[1])
    {
      return true;
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

Value getRealRhs(Operation *op)
{
  /// this will return set_op for transpose, but messes up getUsers() or subsequent calls to it.
  Operation *firstUser = op->getNextNode();
  comet_pdump(firstUser);
  /// TODO(gkestor): need to find out why user set_op is not showing up in users of TransposeOp
  ///       from the for loop below. once resolved, remove getNextNode().
  /// Operation *firstUser;
  /// for (auto user : op->getResult(0).getUsers())
  //{
  ///  firstUser = user;
  ///  break;
  //}

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
  else
  {
    /// do nothing
    return op->getResult(0);
  }
  return op->getResult(0);
}

void buildDefUseInfo(UnitExpression *e)
{
  auto lhs = e->getLHS();
  lhs->setDefiningExpr(e);
  for (auto operand : e->getOperands())
  {
    if (auto def = operand->getDefiningExpr())
    {
      def->addUser(e);
    }
  }
}

IndicesType getUnion(IndicesType indices1, IndicesType indices2)
{
  sort(indices1.begin(), indices1.end());
  sort(indices2.begin(), indices2.end());

  IndicesType allIndices(indices1.size() * 4);

  IndicesType::iterator it = set_union(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(), allIndices.begin());
  allIndices.resize(it - allIndices.begin());
  return allIndices;
}

IndicesType gpuIndices(IndicesType indices1, IndicesType indices2)
{
  sort(indices1.begin(), indices1.end());
  sort(indices2.begin(), indices2.end());

  IndicesType interIndices;
  IndicesType unIndices;
  IndicesType difIndices;
  IndicesType allIndices;

  std::set_intersection(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(),  std::back_inserter(interIndices));
  set_union(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(), std::back_inserter(unIndices));
  std::set_difference(unIndices.begin(), unIndices.end(), interIndices.begin(), interIndices.end(), std::back_inserter(difIndices));
  allIndices = difIndices;
  allIndices.insert(allIndices.end(), interIndices.begin(), interIndices.end());

  // allIndices.resize(it - allIndices.begin());
  return allIndices;  
}

void doTensorMultOp(TensorMultOp op, unique_ptr<Index_Tree> &tree, TargetDevice device = CPU)
{
  Value rhs1_tensor = getRealRhs(op.getRhs1().getDefiningOp());
  Value rhs2_tensor = getRealRhs(op.getRhs2().getDefiningOp());
  Value lhs_tensor = getRealLhs(op);
  Value mask_tensor = op.getMask();
  // rhs1_tensor.getDefiningOp

  comet_debug() << "LowerTensorAlgebraToIndexTreePass: doTensorMultOp\n";
  comet_debug() << "rhs1-tensor\n";
  comet_vdump(rhs1_tensor);
  comet_debug() << "rhs2-tensor\n";
  comet_vdump(rhs2_tensor);
  comet_debug() << "lhs-tensor\n";
  comet_vdump(lhs_tensor);
  comet_debug() << "mask-tensor\n";
  comet_vdump(mask_tensor);

  std::vector<mlir::Value> rhs1_labels = op.getRhs1IndexLabels();
  std::vector<mlir::Value> rhs2_labels = op.getRhs2IndexLabels();
  std::vector<mlir::Value> lhs_labels = op.getResultIndexLabels();

  auto allPerms = getAllPerms(op.getIndexingMaps());
  assert(allPerms.size() == 3);

#ifdef COMET_DEBUG_MODE
  comet_debug() << "\n";
  llvm::errs() << "[";
  for (auto &perm : allPerms)
  {
    llvm::errs() << "[";
    for (auto &i : perm)
    {
      llvm::errs() << i << ",";
    }
    llvm::errs() << "],";
  }
  llvm::errs() << "]\n";
  comet_debug() << "";
// comet_debug() << allPerms;
#endif

  auto allFormats = getAllFormats(op.getFormatsAttr(), allPerms);
  auto SemiringOp = op.getSemiringAttr();
  auto MaskingTypeAttr = op.getMaskTypeAttr();

  /// If the operation is one of the chosen operations, then record output indices as parallel interators.
  bool is_chosen_dense_mixed_operations = checkChosenDenseMixedOperations(allPerms, allFormats);
  bool is_chosen_spgemm_operation = checkChosenSpGEMMOperation(allPerms, allFormats);
  if (mask_tensor != nullptr)
  {
    is_chosen_spgemm_operation = false; /// currently omp.parallel not support masking
  }

  auto B = tree->getOrCreateTensor(rhs1_tensor, rhs1_labels, allFormats[0]);
  auto C = tree->getOrCreateTensor(rhs2_tensor, rhs2_labels, allFormats[1]);
  auto A = tree->getOrCreateTensor(lhs_tensor, lhs_labels, allFormats[2]);

  Tensor *M;
  std::unique_ptr<UnitExpression> e;
  std::vector<mlir::Value> empty;
  if (mask_tensor != nullptr) /// mask is an optional input
  {
    comet_debug() << "mask input provided by user\n";
    M = tree->getOrCreateTensor(mask_tensor, empty, allFormats[2]); /// We don't need indexlabel info for the mask
    e = make_unique<UnitExpression>(A, B, C, M, "*");
  }
  else
  {
    comet_debug() << "no mask input provided by user\n";
    e = make_unique<UnitExpression>(A, B, C, "*");
  }

  e->setSemiring(SemiringOp.cast<mlir::StringAttr>().getValue());
  e->setMaskType(MaskingTypeAttr.cast<mlir::StringAttr>().getValue());

  e->setOperation(op);
  buildDefUseInfo(e.get());

  auto inputDomains = e->computeInputIterDomains();
  auto outputDomains = e->computeOutputIterDomains();

  IndicesType rhs1_indices = tree->getIndices(rhs1_labels);
  IndicesType rhs2_indices = tree->getIndices(rhs2_labels);
  IndicesType allIndices;
  int available_parallel_iterators = 0;  /// The number of parallel iterators (Index Nodes); architecture dependent.

  switch (device) 
  {
    case mlir::tensorAlgebra::CPU:
    {
      allIndices = getUnion(rhs1_indices, rhs2_indices);
      available_parallel_iterators = 1;   /// For CPU, only allow one-level parallel for-loop (i.e., no nested parallelism).
    }
    break;
    case mlir::tensorAlgebra::GPU:
    {
      allIndices = gpuIndices(rhs1_indices, rhs2_indices);
      available_parallel_iterators = 2;  /// TODO(zhen.peng): For GPU, it needs further consideration if the number of parallel iterators should be 2 or 3.
    }
    break;
  }
//  tree->setSizeOfIteratorTypesByIndices(allIndices);  // Set the total number of iterators

  auto lhsIndices = A->getIndices();

  TreeNode *parent = tree->getRoot();
  for (unsigned long i = 0; i < allIndices.size(); ++i)
  {
    int index = allIndices[i];
    auto &idomain = inputDomains.at(index);

    auto node = tree->addIndexNode(index, parent, idomain);

    /// If this index appears on the lhs too, set output domain for the index node
    /// and also set the index as a parallel iterator
    unique_ptr<IteratorType> iteratorType(new IteratorType);
    comet_debug() << iteratorType->dump() << "\n";
    if (std::find(lhsIndices.begin(), lhsIndices.end(), index) != lhsIndices.end())
    {
      auto &odomain = outputDomains.at(index);
      node->setOutputDomain(odomain);
      if (is_chosen_dense_mixed_operations && available_parallel_iterators)
      {
        /// If the operation is one of the chosen ones, and the index appears on the lhs,
        /// then the index has "parallel" as its iterator type.
        iteratorType->setType("parallel");
        --available_parallel_iterators;
      }
      else if (is_chosen_spgemm_operation && available_parallel_iterators)
      {
        /// If the operation is SpGEMM,
        /// then the index has "omp.parallel" as its iterator type.
        iteratorType->setType("omp.parallel");
        --available_parallel_iterators;
      }
    }
    comet_debug() << "index " << index << "\n";
    /// Set the iterator type of the node
    tree->setIteratorTypeByIndex(index, std::move(iteratorType));
    node->setIteratorType(tree->getIteratorTypeByIndex(index));
    comet_debug() << "tree: " << tree->getIteratorTypeByIndex(index)->dump() << " ptr: " << tree->getIteratorTypeByIndex(index) <<  "\n";
    comet_debug() << "node: " << node->getIteratorType()->dump() << " ptr: " << node->getIteratorType() << "\n";

    parent = node;
  }

  tree->addComputeNode(std::move(e), parent);
}

template <typename T>
void doElementWiseOp(T op, unique_ptr<Index_Tree> &tree)
{
  std::vector<mlir::Value> rhs1_labels = op.getRhs1IndexLabels();
  std::vector<mlir::Value> rhs2_labels = op.getRhs2IndexLabels();
  std::vector<mlir::Value> lhs_labels = op.getResultIndexLabels();

  Value rhs1_tensor = getRealRhs(op.getRhs1().getDefiningOp());
  Value rhs2_tensor = getRealRhs(op.getRhs2().getDefiningOp());
  Value lhs_tensor = getRealLhs(op);

  comet_debug() << "LowerTensorAlgebraToIndexTreePass: doElementWiseMultOp\n";
  comet_debug() << "rhs1-tensor\n";
  comet_vdump(rhs1_tensor);
  comet_debug() << "rhs2-tensor\n";
  comet_vdump(rhs2_tensor);
  comet_debug() << "lhs-tensor\n";
  comet_vdump(lhs_tensor);

  auto allPerms = getAllPerms(op.getIndexingMaps());
  auto allFormats = getAllFormats(op.getFormatsAttr(), allPerms);
  auto SemiringOp = op.getSemiringAttr();
  auto maskAttr = "none";

  assert(allPerms.size() == 3);

  auto B = tree->getOrCreateTensor(rhs1_tensor, rhs1_labels, allFormats[0]);
  auto C = tree->getOrCreateTensor(rhs2_tensor, rhs2_labels, allFormats[1]);
  auto A = tree->getOrCreateTensor(lhs_tensor, lhs_labels, allFormats[2]);

  auto e = make_unique<UnitExpression>(A, B, C, "*");

  e->setOperation(op);
  e->setSemiring(SemiringOp.template cast<mlir::StringAttr>().getValue()); /// for element-wise multiplication
  e->setMaskType(maskAttr);                                                /// for element-wise multiplication
  buildDefUseInfo(e.get());

  auto inputDomains = e->computeInputIterDomains();
  auto outputDomains = e->computeOutputIterDomains();

  /// RHS and LHS indices must be the same for elementwise multiplication
  IndicesType allIndices = tree->getIndices(rhs1_labels);
//  tree->setSizeOfIteratorTypesByIndices(allIndices); // Set the total number of iterators

  auto lhsIndices = A->getIndices();
  TreeNode *parent = tree->getRoot();
  for (unsigned long i = 0; i < allIndices.size(); i++)
  {
    int index = allIndices[i];
    auto &idomain = inputDomains.at(index);

    auto node = tree->addIndexNode(index, parent, idomain);
//    unique_ptr<IteratorType> iteratorType(new IteratorType);

    /// If this index appears on the lhs too, set output domain for the index node
    if (std::find(lhsIndices.begin(), lhsIndices.end(), index) != lhsIndices.end())
    {
      auto &odomain = outputDomains.at(index);
      node->setOutputDomain(odomain);
//      iteratorType->setType("parallel");
    }

    /// Set iterator type. Currently "default" for all elementwise operations.
    unique_ptr<IteratorType> iteratorType(new IteratorType);
    tree->setIteratorTypeByIndex(index, std::move(iteratorType));
    node->setIteratorType(tree->getIteratorTypeByIndex(index));
    comet_debug() << "tree: " << tree->getIteratorTypeByIndex(index)->dump() << " ptr: " << tree->getIteratorTypeByIndex(index) <<  "\n";
    comet_debug() << "node: " << node->getIteratorType()->dump() << " ptr: " << node->getIteratorType() << "\n";
    parent = node;
  }
  tree->addComputeNode(std::move(e), parent);
  /// cout << "print tree after tc\n";
  /// tree->print();
}

/// helper for treeToDialect()
Operation *getSetOpForTC(Operation *op)
{
  assert(isa<TensorMultOp>(op) || isa<TensorElewsMultOp>(op) || isa<TensorAddOp>(op) || isa<TensorSubtractOp>(op));
  /// TODO(gkestor): fix the issue with getUsers() after getRealRhs().
  comet_debug() << "The following loop may cause issue!\n";
  Operation *firstUser = nullptr;
  for (auto user : op->getResult(0).getUsers())
  {
    firstUser = user;
    break;
  }

  assert(isa<TensorSetOp>(firstUser));
  return firstUser;
}

/// helper for treeToDialect()
IndexTreeComputeOp createComputeNodeOp(OpBuilder &builder, TreeNode *node, Location &loc)
{
  auto context = builder.getContext();
  IntegerType i64Type = IntegerType::get(context, 64);
  auto expr = node->getExpression();
  SmallVector<Attribute, 8> allIndices_rhs;

  for (auto t : expr->getOperands())
  {
    SmallVector<int64_t, 8> indices;
    for (auto index : t->getIndices())
    {
      indices.push_back(index);
    }
    allIndices_rhs.push_back(builder.getI64ArrayAttr(indices));
  }

  SmallVector<Attribute, 8> allIndices_lhs;
  for (auto t : expr->getResults())
  {
    SmallVector<int64_t, 8> indices;
    for (auto index : t->getIndices())
    {
      comet_debug() << index << " \n";
      indices.push_back(index);
    }
    allIndices_lhs.push_back(builder.getI64ArrayAttr(indices));
  }

  SmallVector<Attribute, 8> allFormats_rhs;
  for (auto t : expr->getOperands())
  {
    SmallVector<StringRef, 8> formats;
    for (auto &f : t->getFormats())
    {
      formats.push_back(f);
    }
    allFormats_rhs.push_back(builder.getStrArrayAttr(formats));
  }

  SmallVector<Attribute, 8> allFormats_lhs;
  for (auto t : expr->getResults())
  {
    SmallVector<StringRef, 8> formats;
    for (auto &f : t->getFormats())
    {
      formats.push_back(f);
    }
    allFormats_lhs.push_back(builder.getStrArrayAttr(formats));
  }

  std::vector<Value> t_rhs;
  Value t_lhs = expr->getLHS()->getValue();
  for (auto o : expr->getOperands())
  {
    t_rhs.push_back(o->getValue());
  }

  /// check if mask exists and add to t_rhs
  if (expr->getMask() != nullptr)
  {
    comet_debug() << "user has provided mask input\n";
    t_rhs.push_back(expr->getMask()->getValue()); /// add mask to IndexTreeComputeRHSOp
  }

  Value leafop_rhs = builder.create<indexTree::IndexTreeComputeRHSOp>(loc,
                                                                      mlir::UnrankedTensorType::get(builder.getF64Type()), t_rhs,
                                                                      builder.getArrayAttr(allIndices_rhs),
                                                                      builder.getArrayAttr(allFormats_rhs));
  comet_vdump(leafop_rhs);
  Value leafop_lhs = builder.create<indexTree::IndexTreeComputeLHSOp>(loc,
                                                                      mlir::UnrankedTensorType::get(builder.getF64Type()), t_lhs,
                                                                      builder.getArrayAttr(allIndices_lhs),
                                                                      builder.getArrayAttr(allFormats_lhs));
  comet_vdump(leafop_lhs);

  bool comp_worksp_opt = false; /// non-compressed workspace, this is a place-holder and it is updated in workspace transform pass.
  llvm::StringRef semiring = expr->getSemiring();
  llvm::StringRef maskType = expr->getMaskType();
  auto leafop = builder.create<IndexTreeComputeOp>(loc, i64Type, leafop_rhs, leafop_lhs, builder.getBoolAttr(comp_worksp_opt), builder.getStringAttr(semiring), builder.getStringAttr(maskType));

  comet_pdump(leafop);
  return leafop;
}

/**
 * This function performs the actual removal of the ta operations in the tree,
 * and add corresponding ta.itree operations.›
 * @param tree
 */
void treeToDialect(Index_Tree *tree)
{
  vector<mlir::Operation *> TAOps = tree->getContainingTAOps();
  unsigned int TAOpsID = 0;
  OpBuilder builder(TAOps[TAOpsID]);
  auto loc = TAOps[TAOpsID]->getLoc();
  auto context = builder.getContext();

  std::map<TreeNode *, Value> nodeToOp;

  IntegerType i64Type = IntegerType::get(context, 64);

  for (auto &node : tree->getNodesInReverseTopoOrder())
  {
    if (node->isComputeNode())
    {
      assert(nodeToOp.count(node) == 0);
      builder.setInsertionPoint(TAOps[TAOpsID]);
      nodeToOp[node] = createComputeNodeOp(builder, node, loc);
      TAOpsID++;
    }
    else if (node->isRealIndexNode())
    {
      if (node->getChildren().empty())
      {
        continue; /// to skip nodes that become childless after fusion
      }
      SmallVector<Value, 8> children;
      for (auto c : node->getChildren())
      {
        assert(nodeToOp.count(c) > 0);
        children.push_back(nodeToOp[c]);
      }
      /// assert(!children.empty());
      SmallVector<int64_t, 1> indices;
      indices.push_back(node->getIndex());
      auto indicesAttr = builder.getI64ArrayAttr(indices);

      SmallVector<int64_t, 1> ids;
      ids.push_back(node->getId());

      /// new attribute iterator_type
      auto dumb_iterator_type = builder.getStringAttr(node->getIteratorType()->getType());
      Value indexNodeOp = builder.create<indexTree::IndexTreeIndicesOp>(loc,
                                                                        i64Type,
                                                                        children,
                                                                        indicesAttr,
                                                                        dumb_iterator_type);

      nodeToOp[node] = indexNodeOp;

      if (node->getParent() != nullptr && node->getParent()->isFillerIndexNode())
      {
#ifdef DEBUG_MODE_LowerTensorAlgebraToIndexTreePass
        Value op = builder.create<indexTree::IndexTreeOp>(loc, i64Type, indexNodeOp);
        comet_vdump(op);
#else
        builder.create<indexTree::IndexTreeOp>(loc, i64Type, indexNodeOp);
#endif
      }
    }
  }

  for (auto op : tree->getContainingTAOps())
  {
    auto setOp = getSetOpForTC(op);
    setOp->erase();
    op->erase();
  }
}

void LowerTensorAlgebraToIndexTreePass::runOnOperation()
{
  unique_ptr<Index_Tree> tree;
  func::FuncOp func = getOperation();
  // #ifdef COMET_DEBUG_MODE
  //   comet_debug() << "Before LowerTensorAlgebraToIndexTreePass\n";
  //   func.dump();
  // #endif

  tree = Index_Tree::createTreeWithRoot();
  bool formIndexTreeDialect = false;

  comet_debug() << "IndexTree pass running on Function\n";
  for (Block &B : func.getBody())
  {
    for (Operation &op : B)
    {
      if (isa<TensorMultOp>(&op))
      {
        doTensorMultOp(cast<TensorMultOp>(&op), tree, device);
        formIndexTreeDialect = true;
      }
      else if (isa<TensorElewsMultOp>(&op))
      {
#ifdef COMET_DEBUG_MODE
        comet_debug() << "\n !!! doElementWiseOp<TensorElewsMultOp>\n";
#endif
        doElementWiseOp<TensorElewsMultOp>(cast<TensorElewsMultOp>(&op), tree);
        formIndexTreeDialect = true;
      }
      else if (isa<TensorAddOp>(&op) || isa<TensorSubtractOp>(&op))
      {
        /// elementwise addition and subtraction
        if (isa<TensorAddOp>(&op))
        {
#ifdef COMET_DEBUG_MODE
          comet_debug() << "\n !!! doElementWiseOp<TensorAddOp>\n";
#endif
          doElementWiseOp<TensorAddOp>(cast<TensorAddOp>(&op), tree);
        }

        if (isa<TensorSubtractOp>(&op))
        {
#ifdef COMET_DEBUG_MODE
          comet_debug() << "\n !!! doElementWiseOp<TensorSubtractOp>\n";
#endif
          doElementWiseOp<TensorSubtractOp>(cast<TensorSubtractOp>(&op), tree);
        }
        formIndexTreeDialect = true;
      }
    }
  }

  if (formIndexTreeDialect)
  {
    comet_debug() << " Dumping Index tree IR\n";
    /// only do this for TensorMultOp or TensorElewsMultOp
    treeToDialect(tree.get());
  }
}

/// create all the passes.
std::unique_ptr<Pass> mlir::comet::createLowerTensorAlgebraToIndexTreePass(TargetDevice device)
{
  comet_debug() << " Calling createLowerTensorAlgebraToIndexTreePass\n";
  return std::make_unique<LowerTensorAlgebraToIndexTreePass>(device);
}
