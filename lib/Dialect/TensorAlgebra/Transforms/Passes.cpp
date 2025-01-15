//===- Passes.cpp ---------------------------===//
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

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <map>
#include <set>
#include <stack>
// #include <iostream>
#define DEBUG_TYPE "comet-passes"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace tensorAlgebra;
using namespace mlir::affine;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

namespace
{
  class FindOptimalTCFactorizationPass
      : public mlir::PassWrapper<FindOptimalTCFactorizationPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FindOptimalTCFactorizationPass)
    void runOnOperation() override;

    void FindOptimalTCFactorization(tensorAlgebra::TensorSetOp op);
  }; ///  class FindOptimalTCFactorizationPass
} ///  End anonymous namespace

namespace
{
  struct OptDenseTransposePass
      : public PassWrapper<OptDenseTransposePass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptDenseTransposePass)
    void runOnOperation() override;
  };

  struct STCRemoveDeadOpsPass
      : public PassWrapper<STCRemoveDeadOpsPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(STCRemoveDeadOpsPass)
    void runOnOperation() override;
  };

} ///  end anonymous namespace.

void removeAllUsers(Operation *op)
{
  for (auto u : op->getUsers())
  {
    comet_debug() << "Users\n";
    comet_pdump(u);
    removeAllUsers(u);
  }
  comet_debug() << "Deleting started\n";
  comet_pdump(op);
  op->erase();
  comet_debug() << "Deleting ends\n";
}

std::vector<int64_t> getTensorShape(const std::vector<Operation *> &labels,
                                    const std::map<Operation *, int64_t> &lblSizes)
{
  std::vector<int64_t> result;

  for (auto lbl : labels)
  {
    result.push_back(lblSizes.at(lbl));
  }

  return result;
}

IndexVector getLabelPerm(const std::vector<Operation *> &labels,
                         const std::map<Operation *, unsigned> &labelIdMap)
{
  IndexVector result;
  for (auto lbl : labels)
  {
    result.push_back(labelIdMap.at(lbl));
  }
  return result;
}

std::vector<Operation *> findOutput(const std::vector<Operation *> &rhs1Labels,
                                    const std::vector<Operation *> &rhs2Labels,
                                    const std::set<Operation *> &outLabels)
{
  std::vector<Operation *> result_labels;

  std::set<Operation *> inLabels(rhs1Labels.begin(), rhs1Labels.end());
  inLabels.insert(rhs2Labels.begin(), rhs2Labels.end());

  std::set_intersection(inLabels.begin(), inLabels.end(), outLabels.begin(),
                        outLabels.end(),
                        std::back_inserter(result_labels));

  return result_labels;
}

std::tuple<IndexVector, std::vector<std::vector<Operation *>>, std::vector<std::vector<int64_t>>>
optimalOrder(ArrayRef<void *> inLTOps, Operation *outLTOp,
             const std::map<Operation *, int64_t> &lblSizes,
             const std::map<void *, std::vector<Operation *>> &lblMaps)
{
  IndexVector result;
  for (size_t i = 0; i < inLTOps.size(); i++)
  {
    result.push_back(i);
  }

  double minCost = std::numeric_limits<double>::max();
  std::vector<unsigned> minResult;
  std::vector<std::vector<Operation *>> minSumLabels;
  std::vector<std::vector<int64_t>> minLHSTensorShapes;
  double totalCost = 0;
  std::map<Operation *, unsigned> labelIdMap;

  unsigned id = 0;
  for (auto op_size_pair : lblSizes)
  {
    auto op = op_size_pair.first;
    labelIdMap[op] = id++;
  }

  ///  go through each and every permutation of result vector.
  do
  {
    totalCost = 0;

    std::vector<std::vector<Operation *>> sumLabels;
    std::vector<std::vector<int64_t>> lhsTensorShapes;
    std::vector<Operation *> rhs1Labels = lblMaps.at(inLTOps[result[0]]);
    for (size_t i = 1; i < result.size(); i++)
    {
      std::vector<Operation *> rhs2Labels = lblMaps.at(inLTOps[result[i]]);

      std::set<Operation *> remainingLabels(lblMaps.at(outLTOp).begin(),
                                            lblMaps.at(outLTOp).end());

      for (size_t j = i + 1; j < result.size(); j++)
      {
        auto lblSet = lblMaps.at(inLTOps[result[j]]);
        remainingLabels.insert(lblSet.begin(), lblSet.end());
      }

      ///  find intersection of {rhs1Labels, rhs2Labels}, {remainingLabels}
      auto lhsLabels = findOutput(rhs1Labels, rhs2Labels, remainingLabels);
      ///  find difference of {rhs1Labels, rhs2Labels}, {lhsLabels}
      sumLabels.push_back(getSumLabels(rhs1Labels, rhs2Labels, lhsLabels));
      auto permA = getLabelPerm(rhs1Labels, labelIdMap);
      auto permB = getLabelPerm(rhs2Labels, labelIdMap);
      auto permC = getLabelPerm(lhsLabels, labelIdMap);

      auto tensorShapeA = getTensorShape(rhs1Labels, lblSizes);
      auto tensorShapeB = getTensorShape(rhs2Labels, lblSizes);
      auto tensorShapeC = getTensorShape(lhsLabels, lblSizes);
      lhsTensorShapes.push_back(tensorShapeC);

      ContractionPlan plan{permA, tensorShapeA,
                           permB, tensorShapeB,
                           permC, tensorShapeC};
      ///  make getTotal optional to include only the operation count or
      ///  plus the cost  operation transpose
      totalCost += plan.getTotalTime();
      rhs1Labels = lhsLabels;
    }

    ///  update global
    if (totalCost <= minCost)
    {
      minCost = totalCost;
      minResult = result;
      minSumLabels = sumLabels;
      minLHSTensorShapes = lhsTensorShapes;
    }

  } while (std::next_permutation(result.begin(), result.end()));

  return std::make_tuple(minResult, minSumLabels, minLHSTensorShapes);
}

void FindOptimalTCFactorizationPass::FindOptimalTCFactorization(tensorAlgebra::TensorSetOp op)
{
  OpBuilder builder(op);
  comet_pdump(op);
  auto operands = op->getOperands();
  auto loc = op->getLoc();
  auto lhsOp = operands[0].getDefiningOp(); ///  TensorMultOp

  std::vector<Operation *> MultOpsToRemove;
  std::vector<Operation *> LTOpsToRemove;

  std::vector<void *> inLTOps;
  std::map<void *, Value> inLTValues;

  comet_debug() << "Chain Multiplication Factorization begin...\n";
  std::map<Operation *, int64_t> lblSizes;
  std::map<Operation *, Value> labelValues;
  std::map<void*, std::vector<Operation *>> lblMaps;

  ///  collect all operands from series of ta.tc ops
  if (isa<tensorAlgebra::TensorMultOp>(lhsOp))
  {
    std::stack<Operation *> stack;

    Value currValue = operands[0];
    comet_vdump(currValue);
    Operation *curr = currValue.getDefiningOp();
    while ((curr && isa<tensorAlgebra::TensorMultOp>(curr)) || !stack.empty())
    {
      while (curr && isa<tensorAlgebra::TensorMultOp>(curr))
      {
        auto multop = cast<tensorAlgebra::TensorMultOp>(curr);
        stack.push(curr);
        MultOpsToRemove.push_back(multop.getOperation());
        std::vector<mlir::Value> labels = multop.getRhs2IndexLabels();
        std::vector<Operation *> labelVec;

        for (size_t i = 0; i < labels.size(); i++)
        {
          auto lblOp = labels[i].getDefiningOp();
          if (lblSizes.count(lblOp) == 0)
          {

            /// If dynamic dimension, we need to retrieve the value of the constantIndexOp that was used to create it
            /// If static, just get it from the tensor type
              if (multop.getRhs2().getType().cast<TensorType>().isDynamicDim(i))
              {
                if (DenseTensorDeclOp dense_decl_op = llvm::dyn_cast_if_present<tensorAlgebra::DenseTensorDeclOp>(multop.getRhs2().getDefiningOp()))
                {
                  ConstantIndexOp val = cast<ConstantIndexOp>(dense_decl_op->getOperand(multop.getRhs2().getType().cast<TensorType>().getDynamicDimIndex(i)).getDefiningOp());
                  lblSizes[lblOp] = val.value();
                }
                else 
                {
                  assert(false && "Factorization optimization can only be applied when tensor size can be inferred statically");
                }

              }
              else
              {
                lblSizes[lblOp] = multop.getRhs2().getType().cast<TensorType>().getDimSize(i);
              }
            labelValues[lblOp] = labels[i];
          }
          labelVec.push_back(lblOp);
        }
        if(!multop.getRhs2().getDefiningOp())
        {
          lblMaps[multop.getRhs2().getAsOpaquePointer()] = labelVec;
        }
        else if (isa<tensorAlgebra::DenseTensorDeclOp>(multop.getRhs2().getDefiningOp()))
        {
          lblMaps[multop.getRhs2().getDefiningOp()] = labelVec;
        }

        labelVec.clear();
        labels = multop.getRhs1IndexLabels();
        for (size_t i = 0; i < labels.size(); i++)
        {
          auto lblOp = labels[i].getDefiningOp();
          if (lblSizes.count(lblOp) == 0)
          {
            /// If dynamic dimension, we need to retrieve the value of the constantIndexOp that was used to create it
            /// If static, just get it from the tensor type
              if (multop.getRhs1().getType().cast<TensorType>().isDynamicDim(i))
              {
                if (isa<tensorAlgebra::DenseTensorDeclOp>(multop.getRhs1().getDefiningOp()))
                {
                  ConstantIndexOp val = cast<ConstantIndexOp>(multop.getRhs1().getDefiningOp()->getOperand(multop.getRhs1().getType().cast<TensorType>().getDynamicDimIndex(i)).getDefiningOp());
                  lblSizes[lblOp] = val.value();
                }
                else 
                {
                  assert(false && "Factorization optimization can only be applied when tensor size can be inferred statically");
                }
              }
              else
              {
                lblSizes[lblOp] = multop.getRhs1().getType().cast<TensorType>().getDimSize(i);
              }

            labelValues[lblOp] = labels[i];
          }
          labelVec.push_back(lblOp);
        }

        if(!multop.getRhs1().getDefiningOp())
        {
          lblMaps[multop.getRhs1().getAsOpaquePointer()] = labelVec;
        }
        else if (isa<tensorAlgebra::DenseTensorDeclOp>(multop.getRhs1().getDefiningOp()))
        {
          lblMaps[multop.getRhs1().getDefiningOp()] = labelVec;
        }

        currValue = cast<tensorAlgebra::TensorMultOp>(curr).getOperation()->getOperand(1);
        curr = currValue.getDefiningOp();
      }
      if(curr)
      {
        inLTOps.push_back(curr);
        inLTValues[curr] = currValue;
      }
      else 
      {
        inLTOps.push_back(currValue.getAsOpaquePointer());
        inLTValues[currValue.getAsOpaquePointer()] = currValue;
      }

      curr = stack.top();
      stack.pop();
      currValue = cast<tensorAlgebra::TensorMultOp>(curr).getOperation()->getOperand(0);
      curr = currValue.getDefiningOp();
    }
      if(curr)
      {
        inLTOps.push_back(curr);
        inLTValues[curr] = currValue;
      }
      else 
      {
        inLTOps.push_back(currValue.getAsOpaquePointer());
        inLTValues[currValue.getAsOpaquePointer()] = currValue;
      }
  }

  auto outLabels = cast<tensorAlgebra::TensorMultOp>(lhsOp).getResultIndexLabels();
  std::vector<Operation *> outLabelVec;
  for (auto lbl : outLabels)
  {
    auto lblOp = lbl.getDefiningOp();
    outLabelVec.push_back(lblOp);
  }
  lblMaps[lhsOp] = outLabelVec;

  IndexVector order;
  std::vector<std::vector<Operation *>> sumLabels;
  std::vector<std::vector<int64_t>> lhsTensorShapes;
  std::tie(order, sumLabels, lhsTensorShapes) = optimalOrder(inLTOps, lhsOp, lblSizes, lblMaps);

  bool same_order = hasSameOrder(getReverseIdentityPermutation(order.size()), order);

  comet_debug() << "Same order " << same_order << "\n";
  ///  updated ta dialect generation.
  if (!same_order)
  {

    Value newRhs1, newRhs2;
    std::vector<Value> ___newSumLabels; ///  needed for intermediate tensor information

    newRhs1 = inLTValues[inLTOps[order[0]]]; ///  updated later for subsequent ta.tc ops in the chain
    for (size_t i = 1; i < order.size(); i++)
    {
      newRhs2 = inLTValues[inLTOps[order[i]]];
      auto elType = newRhs1.getType().dyn_cast<RankedTensorType>().getElementType();
      auto newType = RankedTensorType::get(lhsTensorShapes[i - 1], elType);
      std::vector<Value> newSumLabels;

      std::vector<Operation *> rhs1Labels = lblMaps.at(inLTOps[order[i - 1]]);
      std::vector<Operation *> rhs2Labels = lblMaps.at(inLTOps[order[i]]);
      std::set<Operation *> remainingLabels(lblMaps.at(lhsOp).begin(),
                                            lblMaps.at(lhsOp).end());
      for (size_t j = i + 1; j < order.size(); j++)
      {
        auto lblSet = lblMaps.at(inLTOps[order[j]]);
        remainingLabels.insert(lblSet.begin(), lblSet.end());
      }
      auto lhsLabels = findOutput(rhs1Labels, rhs2Labels, remainingLabels);
      ///  find difference of {rhs1Labels, rhs2Labels}, {lhsLabels}
      auto tempLabelsOps = getSumLabels(rhs1Labels, rhs2Labels, lhsLabels);

      for (auto lbl : lhsLabels)
      {
        newSumLabels.push_back(labelValues[lbl]);
      }
      if (!isa_and_present<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()))
      { ///  store the output label values for subsequent ta.tc ops
        ___newSumLabels = newSumLabels;
      }

      std::vector<Value> new_all_lbls_value;
      std::vector<Value> new_lhs_lbls_value;
      std::vector<Value> new_rhs_lbls_value;

      if (isa_and_present<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()))
      { ///  retrieve the labels from prev iteration.
        new_rhs_lbls_value = ___newSumLabels;
        for (auto lbl : new_rhs_lbls_value)
        {
          comet_vdump(lbl);

          auto result1 = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), lbl);
          if (result1 == new_all_lbls_value.end())
          {
            new_all_lbls_value.push_back(lbl);
          }
        }
      }
      else
      { ///  retrieve the labels from lblMaps
        for (auto lbl : rhs1Labels)
        {
          comet_pdump(lbl);
          comet_debug() << labelValues[lbl] << "\n";
          if (labelValues.find(lbl) == labelValues.end())
          {
            exit(11);
          }
          new_rhs_lbls_value.push_back(labelValues[lbl]);
          auto result1 = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), labelValues[lbl]);
          if (result1 == new_all_lbls_value.end())
          {
            new_all_lbls_value.push_back(labelValues[lbl]);
          }
        }
      }

      for (auto lbl : rhs2Labels)
      {
        comet_pdump(lbl);
        comet_debug() << labelValues[lbl] << "\n";

        if (labelValues.find(lbl) == labelValues.end())
        {
          exit(11);
        }
        new_lhs_lbls_value.push_back(labelValues[lbl]);
        auto result1 = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), labelValues[lbl]);
        if (result1 == new_all_lbls_value.end())
        {
          new_all_lbls_value.push_back(labelValues[lbl]);
        }
      }

      ///  formats
      SmallVector<mlir::StringRef, 8> formats = {"Dense", "Dense", "Dense"};
      // StringRef format = "Dense";
      // if (isa<DenseTensorDeclOp>(newRhs2.getDefiningOp()))
      // {
      //   auto lhs_format = dyn_cast<DenseTensorDeclOp>(newRhs2.getDefiningOp()).getFormat();
      //   formats.push_back(lhs_format);
      // }
      // if (isa<DenseTensorDeclOp>(newRhs1.getDefiningOp()))
      // {
      //   auto rhs_format = dyn_cast<DenseTensorDeclOp>(newRhs1.getDefiningOp()).getFormat();
      //   formats.push_back(rhs_format);
      // }
      // if (isa<DenseTensorDeclOp>(newRhs1.getDefiningOp()) &&
      //     isa<DenseTensorDeclOp>(newRhs2.getDefiningOp()))
      // {
      //   auto rhs_format = dyn_cast<DenseTensorDeclOp>(newRhs1.getDefiningOp()).getFormat();
      //   formats.push_back(rhs_format);
      // }
      // if (isa<tensorAlgebra::TensorMultOp>(newRhs1.getDefiningOp()))
      // { ///  for series of ta.mul case
      //   auto lhs_format = dyn_cast<DenseTensorDeclOp>(newRhs2.getDefiningOp()).getFormat();
      //   formats.push_back(lhs_format);
      //   formats.push_back(lhs_format);
      // }
      auto strAttr = builder.getStrArrayAttr(formats);

      std::vector<int> lhs_lbls;
      std::vector<int> rhs_lbls;
      std::vector<Value> all_labels;

      for (auto e : new_rhs_lbls_value)
      {
        all_labels.push_back(e);
        auto index = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), e) - new_all_lbls_value.begin();
        comet_debug() << index << " " << new_all_lbls_value[index] << "\n";

        rhs_lbls.push_back(index);
      }

      for (auto e : new_lhs_lbls_value)
      {
        all_labels.push_back(e);
        auto index = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), e) - new_all_lbls_value.begin();
        comet_debug() << index << " " << new_all_lbls_value[index] << "\n";
        lhs_lbls.push_back(index);
      }

      std::vector<int> ret_lbls;

      for (auto e : newSumLabels)
      {
        all_labels.push_back(e);
        auto index = std::find(new_all_lbls_value.begin(), new_all_lbls_value.end(), e) - new_all_lbls_value.begin();
        comet_debug() << index << " " << new_all_lbls_value[index] << "\n";

        ret_lbls.push_back(index);
      }

      comet_debug() << "\n";
      auto sorted_lhs = lhs_lbls;
      std::sort(sorted_lhs.begin(), sorted_lhs.end());
      auto sorted_rhs = rhs_lbls;
      std::sort(sorted_rhs.begin(), sorted_rhs.end());
      std::vector<int> all_lbls;
      std::set_union(sorted_lhs.begin(), sorted_lhs.end(), sorted_rhs.begin(), sorted_rhs.end(), std::back_inserter(all_lbls));

      std::map<int, mlir::AffineExpr> expr_map;
      unsigned dim = 0;
      for (const auto &lbl : all_lbls)
      {
        comet_debug() << "All LBLS: " << lbl << "\n";
        expr_map[lbl] = getAffineDimExpr(dim++, builder.getContext());
      }

      std::vector<mlir::AffineExpr> lhs_exprs;
      std::vector<mlir::AffineExpr> rhs_exprs;
      std::vector<mlir::AffineExpr> ret_exprs;
      for (const auto &lbl : lhs_lbls)
      {
        lhs_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : rhs_lbls)
      {
        rhs_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : ret_lbls)
      {
        ret_exprs.push_back(expr_map[lbl]);
      }

      auto context = builder.getContext();
      SmallVector<mlir::AffineMap, 8> affine_maps{
          mlir::AffineMap::get(dim, 0, rhs_exprs, context),
          mlir::AffineMap::get(dim, 0, lhs_exprs, context),
          mlir::AffineMap::get(dim, 0, ret_exprs, context)};
      auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);
      auto SemiringAttr = builder.getStringAttr("plusxy_times");
      auto MaskingAttr = builder.getStringAttr("none");
      Value tcop = builder.create<tensorAlgebra::TensorMultOp>(loc, newType, newRhs1, newRhs2,
                                                               all_labels, affineMapArrayAttr, strAttr, SemiringAttr,
                                                               MaskingAttr, nullptr);
      tcop.getDefiningOp()->setAttr("__alpha__", builder.getF64FloatAttr(1.0));
      tcop.getDefiningOp()->setAttr("__beta__", builder.getF64FloatAttr(0.0));
      comet_debug() << "New operation " << tcop << "\n";
      newRhs1 = tcop;
    }

    mlir::tensorAlgebra::TensorSetOp newSetOp = builder.create<tensorAlgebra::TensorSetOp>(loc, newRhs1, operands[1]);
    newSetOp->setAttr("__beta__", builder.getF64FloatAttr(0.0));

    comet_debug() << "are they previous multop\n";
    for (auto oldTcOp : MultOpsToRemove)
    {
      comet_debug() << "Calling removeAllUsers\n";
      removeAllUsers(oldTcOp);
    }
  }
  comet_debug() << "MulOpFactorization end\n";
  return;
}

void FindOptimalTCFactorizationPass::runOnOperation()
{
  comet_debug() << " start FindOptimalTCFactorizationPass pass \n";
  func::FuncOp func = getOperation();

  func.walk([&](tensorAlgebra::TensorSetOp op)
            { FindOptimalTCFactorization(op); });
}

void STCRemoveDeadOpsPass::runOnOperation()
{
  comet_debug() << " start STCRemoveDeadOpsPass \n";
  ConversionTarget target(getContext());

  func::FuncOp func = getOperation();
  target.addLegalDialect<mlir::linalg::LinalgDialect,
                         ArithDialect,
                         scf::SCFDialect,
                         AffineDialect, memref::MemRefDialect,
                         bufferization::BufferizationDialect>();

  target.addLegalOp<tensorAlgebra::TensorMultOp>();
  RewritePatternSet patterns(&getContext());
  populateSTCRemoveDeadOpsPatterns(patterns, &getContext());
  if (failed(applyPartialConversion(func, target, std::move(patterns))))
  {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::comet::createFindOptimalTCFactorizationPass()
{
  return std::make_unique<FindOptimalTCFactorizationPass>();
}

///  Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createSTCRemoveDeadOpsPass()
{
  return std::make_unique<STCRemoveDeadOpsPass>();
}
