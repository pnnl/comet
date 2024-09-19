//===- Fusion.cpp  ------===//
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
// This pass performs redundancy-aware kernel Fusion on index tree dialect
// The details of the partial fusion can be found in the following paper.
// ReACT: Redundancy-Aware Code Generation for Tensor Expressions.
// Tong Zhou, Ruiqin Tian, Rizwan A Ashraf, Roberto Gioiosa, Gokcen Kestor, Vivek Sarkar.
// 2022 31st International Conference on Parallel Architectures and Compilation Techniques (PACT). October 2022.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/IndexTree/Transforms/UnitExpression.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

using llvm::SmallVector;
using llvm::StringRef;

#define DEBUG_TYPE "partial-fusion"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

//===----------------------------------------------------------------------===//
/// KernelFusion PASS
//===----------------------------------------------------------------------===//

namespace
{
  class IndexTreeKernelFusionPass
      : public mlir::PassWrapper<IndexTreeKernelFusionPass, OperationPass<mlir::func::FuncOp>>
  {
  private:
    static void test(mlir::func::FuncOp &funcop);

    static std::vector<mlir::Operation *> getAllItrees(mlir::func::FuncOp &funcop);

    static std::vector<mlir::Operation *> getAllComputeLHSs(mlir::func::FuncOp &funcop);

    static int getIndicesOpsIndex(mlir::Operation *op);

    static std::vector<mlir::Operation *> getPathFromRoot(mlir::Operation *op);

    static std::vector<mlir::Operation *> getLongestCommonPrefix(std::vector<std::vector<mlir::Operation *>> &paths);

    static mlir::Value createNewTensorDecl(const mlir::Value &old_dense_tensor_decl,
                                           uint32_t rank_base);

    static void createNewTensor(const mlir::Value &old_tensor_alloc,
                                const mlir::Value &old_tensor_load,
                                uint32_t rank_base,
                                mlir::Value &new_tensor_alloc,
                                mlir::Value &new_tensor_load);

    static mlir::Value createReducedComputeLHS(mlir::Operation *lhs_op,
                                               mlir::Value &new_tensor_load,
                                               uint32_t rank_base);

    static mlir::Value createReducedComputeRHS(mlir::Operation *rhs_op,
                                               mlir::Value &new_tensor_load,
                                               mlir::Value &old_tensor_load,
                                               uint32_t rank_base);

    static void replaceOldOperandToNew(mlir::Operation *old_operand, mlir::Value &new_val);

    static void replaceOldTensorFillOp(const mlir::Value &old_dense_tensor_decl,
                                       const mlir::Value &new_dense_tensor_decl);

    static void replaceOldLinalgFillOp(mlir::Value &old_tensor_alloc, mlir::Value &old_tensor_load, mlir::Value &new_tensor_load);

    static mlir::Value createResetComputeRHS(const mlir::Value &new_dense_tensor_decl,
                                             mlir::Operation *last_common_prefix);

    static mlir::Value createResetComputeLHS(const mlir::Value &new_tensor_load,
                                             mlir::Operation *last_common_prefix,
                                             int &lcp_index,
                                             int64_t &rank);

    static void insertTensorReset(
        const std::vector<mlir::Operation *> &lcp,
        const mlir::Value &new_dense_tensor_decl);

    static void doKernelFusion(std::vector<mlir::Operation *> &itrees, mlir::func::FuncOp &funcop);

    static void reduceTensorDimension(std::vector<mlir::Operation *> &LHSs, mlir::func::FuncOp &funcop);

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexTreeKernelFusionPass)
    void runOnOperation() override;

    void RedundancyAwareFusion(mlir::func::FuncOp &funcop);
  }; /// class IndexTreeKernelFusionPass
} /// End anonymous namespace

[[maybe_unused]] void IndexTreeKernelFusionPass::test(mlir::func::FuncOp &funcop)
{
  int level = 0;
  funcop.walk([&](mlir::Operation *op)
              {
                if (llvm::isa<indexTree::IndexTreeOp>(*op)) {
                  comet_debug() << "level: " << level++ << "\n";
                  level = 0;
                  comet_pdump(op);
                  auto rootOp = dyn_cast<indexTree::IndexTreeOp>(*op);
                  [[maybe_unused]] Value computeOp = rootOp.getOperation()->getOperand(0);
                  comet_vdump(computeOp);
                } else if (llvm::isa<indexTree::IndexTreeIndicesOp>(*op)) {
                  comet_debug() << "level: " << level++ << "\n";
                  comet_pdump(op);
 
                  auto indicesOp = dyn_cast<indexTree::IndexTreeIndicesOp>(*op);
                  int numOperands = indicesOp.getOperation()->getNumOperands();
                  for (int oper_i = 0; oper_i < numOperands; ++oper_i) {
                    // Value oper = indicesOp.getOperation()->getOperand(oper_i);
                    comet_debug() << "Operand " << oper_i << ": ";
                    comet_vdump(indicesOp.getOperation()->getOperand(oper_i));
                  }
                  ArrayAttr idsArrayAttr = indicesOp.getIndices();
                  for (auto ida: idsArrayAttr) {
                    int id = ida.cast<mlir::IntegerAttr>().getInt();
                    comet_debug() << "id: " << id;
                  }
                  comet_debug() << "\n";
                } else if (llvm::isa<indexTree::IndexTreeComputeOp>(*op)) {
                  comet_debug() << "level: " << level++ << "\n";
                  comet_pdump(op);
                  auto computeOp = dyn_cast<indexTree::IndexTreeComputeOp>(*op);
                  int num_operands = computeOp.getOperation()->getNumOperands();
                  for (int op_i = 0; op_i < num_operands; ++op_i) {

                    comet_debug() << "Operand " << op_i << ": ";
                    comet_vdump(computeOp.getOperand(op_i));
                  }

                  bool is_comp_worksp_opt = computeOp.getCompWorkspOpt();
                  std::string semiring(computeOp.getSemiring().data());
                  comet_debug() <<  " is_comp_worksp_opt: " << is_comp_worksp_opt << " semiring: " << semiring << "\n";

                  std::vector<std::vector<std::string>> opFormats;
                  std::vector<std::vector<int>> opPerms;
                  std::vector<std::vector<bool> > inputOutputMapping;
                  getFormatsPermsOfComputeOp(computeOp, opFormats, opPerms, inputOutputMapping);
                  /// opFormats
                  comet_debug() << "[";
                  for (auto strings: opFormats) {
                    comet_debug() << "[";
                    for (auto fmt: strings) {
                      comet_debug() << fmt << " ";
                    }
                    comet_debug() << "]";
                  }
                  comet_debug() << "]\n";

                  /// opPerms
                  comet_debug() << "[";
                  for (auto ints: opPerms) {
                    comet_debug() << "[";
                    for (auto perm: ints) {
                      comet_debug() << perm << " ";
                    }
                    comet_debug() << "]";
                  }
                  comet_debug() << "]\n";

                  /// inputOutputMapping
                  comet_debug() << "[";
                  for (auto bools: inputOutputMapping) {
                    comet_debug() << "[";
                    for (auto mp: bools) {
                      comet_debug() << mp << " ";
                    }
                    comet_debug() << "]";
                  }
                  comet_debug() << "]\n";
                } else if (llvm::isa<indexTree::IndexTreeComputeRHSOp>(*op)) {
                  comet_debug() << "level: " << level << "\n";
                  comet_pdump(op);
                } else if (llvm::isa<indexTree::IndexTreeComputeLHSOp>(*op)) {
                 comet_debug() << "level: " << level << "\n";
                  comet_pdump(op);
                } });
}

std::vector<mlir::Operation *> IndexTreeKernelFusionPass::getAllItrees(mlir::func::FuncOp &funcop)
{
  std::vector<mlir::Operation *> itrees;
  funcop.walk([&](indexTree::IndexTreeOp op)
              { itrees.push_back(op.getOperation()); });

  return itrees;
}

std::vector<mlir::Operation *> IndexTreeKernelFusionPass::getAllComputeLHSs(mlir::func::FuncOp &funcop)
{
  std::vector<mlir::Operation *> lhss;
  funcop.walk([&](indexTree::IndexTreeComputeLHSOp op)
              { lhss.push_back(op.getOperation()); });

  return lhss;
}

int IndexTreeKernelFusionPass::getIndicesOpsIndex(mlir::Operation *op)
{
  assert(llvm::isa<indexTree::IndexTreeIndicesOp>(op) && "Error: op is not IndexTreeIndicesOp.");
  auto indices_op = llvm::dyn_cast<indexTree::IndexTreeIndicesOp>(*op);
  int index = indices_op.getIndices()[0].cast<mlir::IntegerAttr>().getInt();
  return index;
}

std::vector<mlir::Operation *> IndexTreeKernelFusionPass::getPathFromRoot(mlir::Operation *op)
{
  std::vector<mlir::Operation *> path;

  while (!llvm::isa<indexTree::IndexTreeOp>(*op))
  {
    { /// test
      comet_debug() << "op\n";
      comet_pdump(op);
    }
    path.push_back(op);
    op = *(op->getUsers().begin());
  }

  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<mlir::Operation *> IndexTreeKernelFusionPass::
    getLongestCommonPrefix(std::vector<std::vector<mlir::Operation *>> &paths)
{
  std::vector<mlir::Operation *> lcp;
  assert(paths.size() == 2 && "Error: too many paths to handle with.");

  auto &path0 = paths[0];
  auto &path1 = paths[1];

  uint32_t index = 0;
  while (index < path0.size() && index < path1.size())
  {
    if (path0[index] == path1[index])
    {
      lcp.push_back(path0[index]);
      { /// test
        ///        comet_debug();
        comet_pdump(path0[index]);
      }
    }
    else
    {
      break;
    }
    ++index;
  }

  return lcp;
}

mlir::Value IndexTreeKernelFusionPass::createNewTensorDecl(
    const mlir::Value &old_dense_tensor_decl,
    uint32_t rank_base)
{
  mlir::Operation *old_tensor_op = old_dense_tensor_decl.getDefiningOp();
  auto loc = old_dense_tensor_decl.getLoc();
  OpBuilder builder(old_tensor_op);

  /// Get operands
  std::vector<mlir::Value> operands;

  TensorType old_tensor = old_dense_tensor_decl.getType().cast<TensorType>();

  for (int64_t i = rank_base; i < old_tensor.getRank(); i++)
  {
    if (old_tensor.isDynamicDim(i))
    {
      operands.push_back(old_tensor_op->getOperands()[i]);
    }
    else
    {
      operands.push_back(builder.create<arith::ConstantIndexOp>(loc, old_tensor.getDimSize(i)));
    }
  }

  /// Get format
  std::string format;
  {
    mlir::tensorAlgebra::DenseTensorDeclOp old_tensor_decl_op = llvm::dyn_cast<mlir::tensorAlgebra::DenseTensorDeclOp>(
        old_tensor_op);
    format = std::string(old_tensor_decl_op.getFormat());
    assert(format == "Dense" && ("Error: only support for Dense old tensor, not " + format + ".\n").c_str());
  }

  /// Create ta.dense_tensor_decl
  auto float_type = mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                                builder.getF64Type());
  mlir::Value new_dense_tensor_decl = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc,
                                                                                       float_type,
                                                                                       operands,
                                                                                       format);

  comet_debug() << "new_dense_tensor_decl\n";
  comet_vdump(new_dense_tensor_decl);

  return new_dense_tensor_decl;
}

[[maybe_unused]] void IndexTreeKernelFusionPass::createNewTensor(
    const mlir::Value &old_tensor_alloc,
    const mlir::Value &old_tensor_load,
    uint32_t rank_base,
    mlir::Value &new_tensor_alloc,
    mlir::Value &new_tensor_load)
{
  mlir::Operation *old_tensor_alloc_op = old_tensor_alloc.getDefiningOp();
  if (old_tensor_alloc_op->getNumOperands() == 2 &&
      rank_base == 1)
  {
    /// TODO(zpeng): here only considered reduce a 2D tensor to an 1D tensor in SDDMM kernel
    /// Get the previous memref.load operand
    auto loc = old_tensor_alloc.getLoc();
    OpBuilder builder(old_tensor_alloc.getDefiningOp());

    /// Create ta.index_label
    mlir::Value index_label_op = builder.create<tensorAlgebra::IndexLabelOp>(loc);

    comet_debug() << "index_label_op\n";
    comet_vdump(index_label_op);

    /// Create ta.dense_tensor_decl
    auto float_type = mlir::RankedTensorType::get({mlir::ShapedType::kDynamic},
                                                  builder.getF64Type());
    std::vector<mlir::Value> operands = {index_label_op};
    std::string format = "Dense";
    mlir::Value dense_tensor_decl_op = builder.create<tensorAlgebra::DenseTensorDeclOp>(loc,
                                                                                        float_type,
                                                                                        operands,
                                                                                        format);

    comet_debug() << "dense_tensor_decl_op\n";
    comet_vdump(dense_tensor_decl_op);

    new_tensor_alloc = index_label_op;
    new_tensor_load = dense_tensor_decl_op;
  }
  else if (old_tensor_alloc_op->getNumOperands() <= 1)
  {
    mlir::TensorType tensor_ty = old_tensor_load.getType().cast<mlir::TensorType>();
    uint32_t rank = tensor_ty.getRank();
    { /// test
      ///      comet_debug();
      comet_debug() << "rank: " << rank << "\n";
      for (uint32_t r_i = 0; r_i < rank; ++r_i)
      {
        comet_debug() << r_i << ": " << tensor_ty.getDimSize(r_i) << " ";
      }
      comet_debug() << "\n";
    }

    mlir::OpBuilder builder(old_tensor_alloc.getDefiningOp());
    auto loc = old_tensor_alloc.getLoc();
    std::vector<int64_t> dims_vec; /// int64_t is required for MemRefType
    for (uint32_t r_i = rank_base; r_i < rank; ++r_i)
    {
      dims_vec.push_back(tensor_ty.getDimSize(r_i));
    }
    auto dims_ty = mlir::MemRefType::get(
        llvm::ArrayRef<int64_t>(dims_vec),
        tensor_ty.getElementType());

    std::vector<mlir::Value> operands;
    operands.insert(operands.end(),
                    old_tensor_alloc_op->getOperands().begin() + rank_base,
                    old_tensor_alloc_op->getOperands().end());
    new_tensor_alloc = builder.create<memref::AllocOp>(loc, dims_ty, operands, builder.getI64IntegerAttr(32));
    new_tensor_load = builder.create<ToTensorOp>(loc, new_tensor_alloc);
  }
  else
  {
    llvm::errs() << "Error: IndexTreeKernelFusionPass::createNewTensor() does not support tensors whose rank is larger than 2.";
  }
}

mlir::Value IndexTreeKernelFusionPass::createReducedComputeLHS(
    mlir::Operation *lhs_op,
    mlir::Value &new_tensor_load,
    uint32_t rank_base)
{
  assert(
      llvm::isa<indexTree::IndexTreeComputeLHSOp>(*lhs_op) && "Error: not a type of indexTree::IndexTreeComputeLHSOp");
  OpBuilder builder(lhs_op);

  /// Get old formats and perms
  mlir::indexTree::IndexTreeComputeLHSOp it_compute_lhs_op = llvm::dyn_cast<mlir::indexTree::IndexTreeComputeLHSOp>(
      lhs_op);
  ArrayAttr op_formats_ArrayAttr = it_compute_lhs_op.getAllFormats();
  ArrayAttr op_perms_ArrayAttr = it_compute_lhs_op.getAllPerms();
  std::vector<std::vector<std::string>> old_formats_strs = convertArrayAttrStrTo2DVector(op_formats_ArrayAttr);
  std::vector<std::vector<int>> old_perms_ints = convertArrayAttrIntTo2DVector(op_perms_ArrayAttr);

  /// Create the new formats
  /// i.g., convert [["D", "D"]] to [["D"]]
  SmallVector<Attribute, 8> new_formats;
  SmallVector<StringRef, 8> formats;
  formats.insert(formats.end(), old_formats_strs[0].begin() + rank_base, old_formats_strs[0].end());
  new_formats.push_back(builder.getStrArrayAttr(formats));

  /// Create the new perms
  /// i.g., convert [[1, 0]] to [[0]]
  SmallVector<Attribute, 8> new_perms;
  SmallVector<int64_t, 8> perms;
  perms.insert(perms.end(), old_perms_ints[0].begin() + rank_base, old_perms_ints[0].end());
  new_perms.push_back(builder.getI64ArrayAttr(perms));

  /// Create the new ComputeLHS
  std::vector<Value> tensors;
  tensors.push_back(new_tensor_load);
  mlir::Value new_lhs_op = builder.create<indexTree::IndexTreeComputeLHSOp>(
      lhs_op->getLoc(),
      mlir::UnrankedTensorType::get(builder.getF64Type()),
      tensors,
      builder.getArrayAttr(new_perms),
      builder.getArrayAttr(new_formats));

  return new_lhs_op;
}

mlir::Value IndexTreeKernelFusionPass::createReducedComputeRHS(
    mlir::Operation *rhs_op,
    mlir::Value &new_tensor_load,
    mlir::Value &old_tensor_load,
    uint32_t rank_base)
{
  assert(
      llvm::isa<indexTree::IndexTreeComputeRHSOp>(*rhs_op) && "Error: not a type of indexTree::IndexTreeComputeRHSOp");
  OpBuilder builder(rhs_op);

  /// Get old formats and perms
  mlir::indexTree::IndexTreeComputeRHSOp it_compute_rhs_op = llvm::dyn_cast<mlir::indexTree::IndexTreeComputeRHSOp>(
      rhs_op);
  ArrayAttr op_formats_ArrayAttr = it_compute_rhs_op.getAllFormats();
  ArrayAttr op_perms_ArrayAttr = it_compute_rhs_op.getAllPerms();
  std::vector<std::vector<std::string>> old_formats_strs = convertArrayAttrStrTo2DVector(op_formats_ArrayAttr);
  std::vector<std::vector<int>> old_perms_ints = convertArrayAttrIntTo2DVector(op_perms_ArrayAttr);

  /// Locate the operand to be reduced
  uint32_t tensor_id = 0;
  for (auto val : rhs_op->getOperands())
  {
    if (val.getDefiningOp() == old_tensor_load.getDefiningOp())
    {
      break;
    }
    ++tensor_id;
  }

  /// Create the new formats
  /// i.g., convert [["D", "D"], ["D", "D"]] to [["D"], ["D", "D"]]
  SmallVector<Attribute, 8> new_formats;
  for (uint32_t f_i = 0; f_i < old_formats_strs.size(); ++f_i)
  {
    SmallVector<StringRef, 8> formats;
    if (f_i == tensor_id)
    { /// for the new reduced tensor
      formats.insert(formats.end(), old_formats_strs[f_i].begin() + rank_base, old_formats_strs[f_i].end());
    }
    else
    { /// for other remaining old operands
      formats.insert(formats.end(), old_formats_strs[f_i].begin(), old_formats_strs[f_i].end());
    }
    new_formats.push_back(builder.getStrArrayAttr(formats));
  }

  /// Create the new perms
  /// i.g., convert [[1, 0], [0, 2]] to [[0], [0, 2]]
  SmallVector<Attribute, 8> new_perms;
  for (uint32_t p_i = 0; p_i < old_perms_ints.size(); ++p_i)
  {
    SmallVector<int64_t, 8> perms;
    if (p_i == tensor_id)
    { /// for the new reduced tensor
      perms.insert(perms.end(), old_perms_ints[p_i].begin() + rank_base, old_perms_ints[p_i].end());
    }
    else
    { /// for other remaining old operands
      perms.insert(perms.end(), old_perms_ints[p_i].begin(), old_perms_ints[p_i].end());
    }
    new_perms.push_back(builder.getI64ArrayAttr(perms));
  }

  /// Create the new ComputeRHS
  std::vector<Value> tensors;
  uint32_t val_i = 0;
  for (auto val : rhs_op->getOperands())
  {
    if (val_i == tensor_id)
    {
      tensors.push_back(new_tensor_load); /// new reduced tensor
    }
    else
    {
      tensors.push_back(val); /// other remaining old operands
    }
    ++val_i;
  }
  mlir::Value new_rhs_op = builder.create<indexTree::IndexTreeComputeRHSOp>(
      rhs_op->getLoc(),
      mlir::UnrankedTensorType::get(builder.getF64Type()),
      tensors,
      builder.getArrayAttr(new_perms),
      builder.getArrayAttr(new_formats));

  return new_rhs_op;
}

void IndexTreeKernelFusionPass::replaceOldOperandToNew(mlir::Operation *old_operand, mlir::Value &new_val)
{
  for (auto user : old_operand->getUsers())
  {
    int op_i = 0;
    for (auto op : user->getOperands())
    {
      if (op.getDefiningOp() == old_operand)
      {
        /// Replace the old operand to the new one
        user->setOperand(op_i, new_val);
        break;
      }
      ++op_i;
    }
  }

  /// Erase the old operand
  old_operand->erase();
}

void IndexTreeKernelFusionPass::replaceOldTensorFillOp(
    const mlir::Value &old_dense_tensor_decl,
    const mlir::Value &new_dense_tensor_decl)
{
  for (auto user : old_dense_tensor_decl.getUsers())
  {
    if (mlir::isa<tensorAlgebra::TensorFillOp>(user))
    {
      /// Get value
      mlir::tensorAlgebra::TensorFillOp fill_op = llvm::dyn_cast<mlir::tensorAlgebra::TensorFillOp>(user);
      auto value_attr = fill_op.getValueAttr();

      OpBuilder builder(user);
      [[maybe_unused]] auto new_fill_op = builder.create<tensorAlgebra::TensorFillOp>(
          user->getLoc(),
          new_dense_tensor_decl,
          value_attr);
      ///          builder.getF64FloatAttr(0));

      comet_debug() << "new_fill_op\n";
      comet_vdump(new_fill_op);

      user->erase();
      break;
    }
    else
    {
      comet_debug() << "Error: user's type is not supported, yet.\n";
      continue;
    }
  }
}

[[maybe_unused]] void IndexTreeKernelFusionPass::replaceOldLinalgFillOp(
    mlir::Value &old_tensor_alloc,
    mlir::Value &old_tensor_load,
    mlir::Value &new_tensor_load)
{
  for (auto user : old_tensor_alloc.getUsers())
  {
    OpBuilder builder(user);
    if (mlir::isa<linalg::FillOp>(user))
    {

      comet_debug() << "user\n";
      comet_pdump(user);

      /// Get the ConstantOp
      Value constant_op;
      for (auto op : user->getOperands())
      {
        comet_debug() << "op\n";
        comet_vdump(op);

        if (mlir::isa<ConstantOp>(op.getDefiningOp()))
        {
          constant_op = llvm::dyn_cast<ConstantOp>(*op.getDefiningOp());
          break;
        }
      }

      comet_debug() << "constant_op\n";
      comet_vdump(constant_op);

      /// Create MemRef
      ///  mlir::Value memref;
      if (llvm::isa<tensorAlgebra::DenseTensorDeclOp>(new_tensor_load.getDefiningOp()))
      {
        [[maybe_unused]] auto new_fill_op = builder.create<tensorAlgebra::TensorFillOp>(
            user->getLoc(),
            new_tensor_load,
            builder.getF64FloatAttr(0));
        {
          comet_debug() << "new_fill_op\n";
          comet_vdump(new_fill_op);
        }
      }
      else if (llvm::isa<ToTensorOp>(new_tensor_load.getDefiningOp()))
      {
        /// For GNN kernel.
        mlir::Value memref = llvm::dyn_cast<ToTensorOp>(new_tensor_load.getDefiningOp()).getMemref();
        [[maybe_unused]] auto new_fill_op = builder.create<linalg::FillOp>(
            user->getLoc(),
            constant_op,
            memref);

        comet_debug() << "new_fill_op\n";
        comet_pdump(new_fill_op);
      }

      /// Erase the user
      user->erase();
      break;
    }
    else
    {
      comet_debug() << "Error: some unsupported users.\n";
      continue;
    }
  }
}

mlir::Value IndexTreeKernelFusionPass::createResetComputeRHS(
    const mlir::Value &new_dense_tensor_decl,
    mlir::Operation *last_common_prefix)
{
  auto loc = last_common_prefix->getLoc();
  OpBuilder builder(last_common_prefix);
  /// Generate itComputeRHS, operand is constant 0, allFormats = [[]], allPerms = [[]]
  /// Operand is constant 0
  ConstantOp constant_zero;
  for (mlir::Operation *user : new_dense_tensor_decl.getUsers())
  {
    if (!mlir::isa<tensorAlgebra::TensorFillOp>(user))
    {
      continue;
    }
    /// Get the ConstantOp
    for (mlir::Value op : user->getOperands())
    {
      if (mlir::isa<ConstantOp>(op.getDefiningOp()))
      {
        constant_zero = llvm::dyn_cast<ConstantOp>(*op.getDefiningOp());
        break;
      }
    }
    break;
  }
  if (constant_zero.getOperation() == nullptr)
  {
    comet_debug() << "create ConstantOp\n";
    constant_zero = builder.create<ConstantOp>(loc,
                                               builder.getF64Type(),
                                               builder.getF64FloatAttr(0));
  }

  comet_debug() << "constant_zero\n";
  comet_vdump(constant_zero);

  std::vector<mlir::Value> tensors_rhs;
  tensors_rhs.push_back(constant_zero);

  SmallVector<Attribute, 1> indices_rhs;
  SmallVector<int64_t, 8> empty_index;
  indices_rhs.push_back(builder.getI64ArrayAttr(empty_index));

  SmallVector<Attribute, 1> formats_rhs;
  SmallVector<StringRef, 1> empty_format;
  formats_rhs.push_back(builder.getStrArrayAttr(empty_format));

  /// TODO(zpeng): What if the type is not F64?
  mlir::Value compute_rhs = builder.create<indexTree::IndexTreeComputeRHSOp>(
      loc,
      mlir::UnrankedTensorType::get(builder.getF64Type()),
      tensors_rhs,
      builder.getArrayAttr(indices_rhs),
      builder.getArrayAttr(formats_rhs));

  return compute_rhs;
}

mlir::Value IndexTreeKernelFusionPass::createResetComputeLHS(
    const mlir::Value &new_dense_tensor_decl,
    mlir::Operation *last_common_prefix,
    int &lcp_index,
    int64_t &rank)
{
  auto loc = last_common_prefix->getLoc();
  OpBuilder builder(last_common_prefix);
  /// Generate itComputeLHS, operand is tensor_load, allFormats = [["D"]], allPerms = [[1]]
  /// Operand is tensor_load
  std::vector<mlir::Value> tensors_lhs;
  tensors_lhs.push_back(new_dense_tensor_decl);
  lcp_index = getIndicesOpsIndex(last_common_prefix);
  mlir::TensorType tensor_ty = new_dense_tensor_decl.getType().cast<mlir::TensorType>();
  rank = tensor_ty.getRank();

  /// Get indices [[1]]
  SmallVector<Attribute, 1> indices_lhs;
  SmallVector<int64_t, 8> one_index;
  for (uint32_t r_i = 0; r_i < rank; ++r_i)
  {
    one_index.push_back(lcp_index + 1 + r_i);
  }
  indices_lhs.push_back(builder.getI64ArrayAttr(one_index));

  /// Get formats [["D"]]
  SmallVector<Attribute, 1> formats_lhs;
  SmallVector<StringRef, 1> one_format(rank, "D");
  formats_lhs.push_back(builder.getStrArrayAttr(one_format));

  mlir::Value compute_lhs = builder.create<indexTree::IndexTreeComputeLHSOp>(
      loc,
      mlir::UnrankedTensorType::get(builder.getF64Type()),
      tensors_lhs,
      builder.getArrayAttr(indices_lhs),
      builder.getArrayAttr(formats_lhs));

  return compute_lhs;
}

mlir::Value createResetIndicesOps(
    int lcp_index,
    int64_t rank,
    mlir::Operation *last_common_prefix,
    const mlir::Value &compute_op)
{
  auto loc = last_common_prefix->getLoc();
  OpBuilder builder(last_common_prefix);

  /// Create index nodes
  int bound_index = lcp_index + rank;
  auto i64_type = builder.getI64Type();
  mlir::Value last_indices_op;

  for (int index = bound_index; index > lcp_index; --index)
  {
    SmallVector<int64_t, 1> indices = {index};
    auto indices_attr = builder.getI64ArrayAttr(indices);
    mlir::Value indices_op;
    if (index == bound_index)
    {
      /// The IndicesOp node closest to the ComputeOp node
      /// TODO(zhen.peng): new attribute iterator_type
      auto dumb_iterator_type = builder.getStringAttr("default");
      indices_op = builder.create<indexTree::IndexTreeIndicesOp>(
          loc,
          i64_type,
          compute_op,
          indices_attr,
          dumb_iterator_type);
    }
    else
    {
      /// TODO(zhen.peng): new attribute iterator_type
      auto dumb_iterator_type = builder.getStringAttr("default");
      indices_op = builder.create<indexTree::IndexTreeIndicesOp>(
          loc,
          i64_type,
          last_indices_op,
          indices_attr,
          dumb_iterator_type);
    }
    last_indices_op = indices_op;
  }

  return last_indices_op;
}

void IndexTreeKernelFusionPass::insertTensorReset(
    const std::vector<mlir::Operation *> &lcp,
    const mlir::Value &new_dense_tensor_decl)
{
  /// Generate itComputeRHS, operand is constant 0, allFormats = [[]], allPerms = [[]]
  mlir::Operation *last_common_prefix = lcp.back();
  auto loc = last_common_prefix->getLoc();
  OpBuilder builder(last_common_prefix);
  /// Operand is constant 0
  mlir::Value compute_rhs = createResetComputeRHS(new_dense_tensor_decl,
                                                  last_common_prefix);

  comet_debug() << "compute_rhs\n";
  comet_vdump(compute_rhs);

  /// Generate itComputeLHS, operand is tensor_load, allFormats = [["D"]], allPerms = [[1]]
  int lcp_index;
  int64_t rank;
  mlir::Value compute_lhs = createResetComputeLHS(new_dense_tensor_decl,
                                                  last_common_prefix,
                                                  lcp_index,
                                                  rank);

  comet_debug() << "compute_lhs\n";
  comet_vdump(compute_lhs);

  auto comp_worksp_opt = builder.getBoolAttr(false);
  mlir::StringAttr semiring = builder.getStringAttr("noop_times");
  mlir::StringAttr maskType = builder.getStringAttr("none");

  IntegerType i64Type = IntegerType::get(builder.getContext(), 64);
  mlir::Value compute_op = builder.create<indexTree::IndexTreeComputeOp>(
      loc,
      i64Type,
      compute_rhs,
      compute_lhs,
      comp_worksp_opt,
      semiring,
      maskType);
  {
    comet_debug() << "compute_op\n";
    comet_vdump(compute_op);
  }

  /// Create index nodes
  mlir::Value indices_root = createResetIndicesOps(lcp_index,
                                                   rank,
                                                   last_common_prefix,
                                                   compute_op);
  /// Add to the last common prefix
  std::vector<mlir::Value> operands;
  operands.insert(operands.end(), last_common_prefix->getOperands().begin(), last_common_prefix->getOperands().end());
  operands.push_back(indices_root);
  last_common_prefix->setOperands(operands);
}

void IndexTreeKernelFusionPass::doKernelFusion(
    std::vector<mlir::Operation *> &itrees, mlir::func::FuncOp &funcop)
{

  std::deque<std::vector<mlir::Operation *>> buffer;
  {
    /// Collects all itrees' operands and initialize the buffer.
    std::vector<mlir::Operation *> operands;
    for (auto itree : itrees)
    {
      for (auto operand : itree->getOperands())
      {
        if (llvm::isa<indexTree::IndexTreeIndicesOp>(operand.getDefiningOp()))
        {
          comet_pdump(operand.getDefiningOp());
          operands.push_back(operand.getDefiningOp());
        }
      }
    }

    buffer.push_back(std::move(operands));
  }
  comet_debug() << "Buffer size " << buffer.size() << "\n";
  while (!buffer.empty())
  {
    std::vector<mlir::Operation *> operands = buffer.front();
    buffer.pop_front();
    comet_debug() << "Buffer size " << buffer.size() << "\n";
    if (operands.size() < 2)
    {
      /// Too few nodes to fuse
      continue;
    }

    /// Nodes to be fused are clustered and then fused to the host node
    std::vector<bool> is_clustered(operands.size(), false);
    for (int host_i = operands.size() - 1; host_i >= 0; --host_i)
    {
      if (is_clustered[host_i])
      {
        continue;
      }

      /// Get the host
      /// Note: The host is the latest one, otherwise the fused nodes are not in correct usage order and got Error:
      mlir::Operation *host = operands[host_i];
      comet_debug() << host << "host\n";
      comet_debug() << "host\n";
      comet_pdump(host);

      is_clustered[host_i] = true;
      int host_index = getIndicesOpsIndex(host);

      /// Cluster other nodes to the host
      std::vector<mlir::Operation *> cluster;
      for (int node_i = 0; node_i < host_i; ++node_i)
      {
        if (is_clustered[node_i])
        {
          continue;
        }
        mlir::Operation *node = operands[node_i];
        int node_index = getIndicesOpsIndex(node);
        comet_debug() << "node\n";
        comet_pdump(node);
        /// Check if node_i can be fused with host_i
        if (node_index == host_index)
        {
          cluster.push_back(node);
          is_clustered[node_i] = true;
        }
      }
      cluster.push_back(host);

      /// Set the operands of the host to the operands of nodes in the cluster
      std::vector<mlir::Value> sub_operands;
      for (mlir::Operation *node : cluster)
      {
        for (auto operand : node->getOperands())
        {
          sub_operands.push_back(operand);
        }
      }
      host->setOperands(sub_operands);

      /// Update the buffer
      std::vector<mlir::Operation *> tmp_operands;
      for (auto &op : sub_operands)
      {
        tmp_operands.push_back(op.getDefiningOp());
      }
      buffer.push_back(std::move(tmp_operands));

      /// Erase other nodes in the cluster and their users
      for (int node_i = cluster.size() - 2; node_i >= 0; --node_i)
      {
        mlir::Operation *node = cluster[node_i];
        comet_debug() << "Erasing \n";
        for (auto u : node->getUsers())
        {
          comet_pdump(u);
          u->erase();
        }
        comet_pdump(node);
        node->erase();
      }
    }
  }
}

void IndexTreeKernelFusionPass::reduceTensorDimension(std::vector<mlir::Operation *> &LHSs, mlir::func::FuncOp &funcop)
{
  for (mlir::Operation *lhs_op : LHSs)
  {
    comet_debug() << "lhs_op\n";
    comet_pdump(lhs_op);

    mlir::Value tensor = lhs_op->getOperand(0);
    comet_debug() << "tensor\n";
    comet_vdump(tensor);

    auto users = tensor.getUsers();

    comet_debug() << "users\n";
    for ([[maybe_unused]] auto u : users)
    {
      comet_pdump(u);
    }

    int num_users = 0;
    mlir::Operation *rhs_op = nullptr;
    for (auto u : users)
    {
      if (llvm::isa<indexTree::IndexTreeComputeLHSOp>(*u))
      {
        ++num_users;
      }
      else if (llvm::isa<indexTree::IndexTreeComputeRHSOp>(*u))
      {
        ++num_users;
        rhs_op = u;
      }
    }
    if (num_users < 2)
    {
      /// No need to reduce its dimension
      continue;
    }
    else if (num_users > 2)
    {
      comet_debug() << "Error: should not have more than 2 users for the tensor.\n";
      continue;
    }

    comet_debug() << "rhs_op\n";
    comet_pdump(rhs_op);

    /// Get paths of lsh_op and rhs_op
    std::vector<std::vector<mlir::Operation *>> paths;
    paths.push_back(getPathFromRoot(lhs_op));
    paths.push_back(getPathFromRoot(rhs_op));

    /// Get the longest common prefix
    std::vector<mlir::Operation *> lcp = getLongestCommonPrefix(paths);
    if (lcp.empty())
    {
      /// No common prefix for reducing tensor dimension
      continue;
    }

    uint32_t rank_base = lcp.size();
    mlir::Value new_dense_tensor_decl = createNewTensorDecl(tensor, rank_base);

    comet_debug() << "new_dense_tensor_decl\n";
    comet_vdump(new_dense_tensor_decl);

    mlir::Value new_compute_lhs = createReducedComputeLHS(lhs_op,
                                                          new_dense_tensor_decl,
                                                          rank_base);
    comet_debug() << "new_compute_lhs\n";
    comet_vdump(new_compute_lhs);

    mlir::Value new_compute_rhs = createReducedComputeRHS(rhs_op,
                                                          new_dense_tensor_decl,
                                                          tensor,
                                                          rank_base);

    comet_debug() << "new_compute_rhs\n";
    comet_vdump(new_compute_rhs);

    /// Switch ComputeLHS
    comet_debug() << "replace LHS to new LSH\n";
    replaceOldOperandToNew(lhs_op, new_compute_lhs);

    /// Switch ComputeRHS
    comet_debug() << "replace RHS to new RHS\n";
    replaceOldOperandToNew(rhs_op, new_compute_rhs);

    comet_debug() << "replace ta.fill\n";
    replaceOldTensorFillOp(tensor, new_dense_tensor_decl);

    /// Erase the old dense_tensor_decl
    tensor.getDefiningOp()->erase();

    /// Generate T = 0 to reset the intermediate tensor. The location is under the last common prefix.
    insertTensorReset(lcp, new_dense_tensor_decl);
  }
}

void IndexTreeKernelFusionPass::RedundancyAwareFusion(mlir::func::FuncOp &funcop)
{
  comet_vdump(funcop);
  comet_debug() << "ParitalFusionIT pass\n";

  /// Basic partial fusion
  std::vector<mlir::Operation *> itrees = getAllItrees(funcop);
  if (itrees.size() < 2)
  {
    /// Only one itree node cannot do fusion.
    return;
  }

  doKernelFusion(itrees, funcop);

  /// Reduce tensor dimension
  comet_vdump(funcop);
  std::vector<mlir::Operation *> LHSs = getAllComputeLHSs(funcop);
  reduceTensorDimension(LHSs, funcop);
}

void IndexTreeKernelFusionPass::runOnOperation()
{
  LLVM_DEBUG(llvm::dbgs() << "start IndexTreeKernelFusionPass\n");
  comet_debug() << " start KernelFusion pass \n";
  func::FuncOp func = getOperation();
  RedundancyAwareFusion(func);
}

//// Apply the partial fusion on the index tree dialect
std::unique_ptr<Pass> mlir::comet::createIndexTreeKernelFusionPass()
{
  return std::make_unique<IndexTreeKernelFusionPass>();
}
