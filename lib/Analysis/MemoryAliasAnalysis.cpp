//===- AliasAnalysis.cpp  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains passes for constructing and performing alias analysis
//
//===----------------------------------------------------------------------===//

#include "comet/Analysis/MemoryAliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
//#define COMET_DEBUG_COMET_ANALYSIS_MEMORYALIASANALYSIS_DUMP_ON
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

/// Print a value that is used as an operand of an alias query.
static void printAliasOperand(Operation *op)
{
  llvm::errs() << op->getAttrOfType<StringAttr>("test.ptr").getValue();
}
static void printAliasOperand(Value value)
{
  if (BlockArgument arg = dyn_cast<BlockArgument>(value))
  {
    Region *region = arg.getParentRegion();
    unsigned parentBlockNumber =
        std::distance(region->begin(), arg.getOwner()->getIterator());
    llvm::errs() << region->getParentOp()
                        ->getAttrOfType<StringAttr>("test.ptr")
                        .getValue()
                 << ".region" << region->getRegionNumber();
    if (parentBlockNumber != 0)
      llvm::errs() << ".block" << parentBlockNumber;
    llvm::errs() << "#" << arg.getArgNumber();
    return;
  }
  OpResult result = cast<OpResult>(value);
  printAliasOperand(result.getOwner());
  llvm::errs() << "#" << result.getResultNumber();
}

namespace mlir
{
  namespace comet
  {
    void printAliasResult(AliasResult result, Value lhs, Value rhs)
    {
      printAliasOperand(lhs);
      llvm::errs() << " <-> ";
      printAliasOperand(rhs);
      llvm::errs() << ": " << result << "\n";
    }

    /// Print the result of an alias query.
    void printModRefResult(ModRefResult result, Operation *op, Value location)
    {
      printAliasOperand(op);
      llvm::errs() << " -> ";
      printAliasOperand(location);
      llvm::errs() << ": " << result << "\n";
    }

    void AliasAnalysisBase::runAliasAnalysisOnOperation(
        Operation *op, AliasAnalysis &aliasAnalysis)
    {
      comet_debug() << "Before analysis:\n";
      comet_pdump(op);

      llvm::errs() << "Testing : " << *op->getInherentAttr("sym_name") << "\n";

      // Collect all of the values to check for aliasing behavior.
      SmallVector<Value, 32> valsToCheck;
      op->walk([&](Operation *op)
      {
        if (!op->getDiscardableAttr("test.ptr"))
          return;
        valsToCheck.append(op->result_begin(), op->result_end());
        for (Region &region : op->getRegions())
          for (Block &block : region)
            valsToCheck.append(block.args_begin(), block.args_end());
      });

      // Check for aliasing behavior between each of the values.
      for (auto it = valsToCheck.begin(), e = valsToCheck.end(); it != e; ++it)
      {
        for (auto *innerIt = valsToCheck.begin(); innerIt != it; ++innerIt)
        {
#ifdef COMET_DEBUG_COMET_ANALYSIS_MEMORYALIASANALYSIS_DUMP_ON
          llvm::errs() << "\n";
          llvm::errs() << "lhs: ";
          comet_pdump(innerIt);
          printAliasOperand(*innerIt);
          llvm::errs() << "\n";

          llvm::errs() << "rhs: ";
          comet_pdump(it);
          printAliasOperand(*it);
          llvm::errs() << "\n";
#endif
          printAliasResult(aliasAnalysis.alias(*innerIt, *it), *innerIt, *it);
        }
      }

      comet_debug() << "After analysis:\n";
      comet_pdump(op);
    }

    void AliasAnalysisModRefBase::runAliasAnalysisOnOperation(
        Operation *op, AliasAnalysis &aliasAnalysis)
    {
      llvm::errs() << "Testing : " << *op->getInherentAttr("sym_name") << "\n";

      // Collect all of the values to check for aliasing behavior.
      SmallVector<Value, 32> valsToCheck;
      op->walk([&](Operation *op)
      {
        if (!op->getDiscardableAttr("test.ptr"))
          return;
        valsToCheck.append(op->result_begin(), op->result_end());
        for (Region &region : op->getRegions())
          for (Block &block : region)
            valsToCheck.append(block.args_begin(), block.args_end());
      });

      // Check for aliasing behavior between each of the values.
      for (auto &it : valsToCheck)
      {
        op->walk([&](Operation *op)
        {
          if (!op->getDiscardableAttr("test.ptr"))
            return;
          printModRefResult(aliasAnalysis.getModRef(op, it), op, it);
        });
      }
    }

  } // namespace comet
} // namespace mlir

//===----------------------------------------------------------------------===//
// AliasResult
//===----------------------------------------------------------------------===//

namespace
{
  struct AliasAnalysisPass
      : public comet::AliasAnalysisBase,
        PassWrapper<AliasAnalysisPass, OperationPass<>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AliasAnalysisPass)

    StringRef getArgument() const final
    {
      return "memory-alias-gokcen";
    }

    StringRef getDescription() const final
    {
      return "Memory alias analysis results.";
    }

    void runOnOperation() override
    {
      AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
      runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
    }
  };
} // namespace

//===----------------------------------------------------------------------===//
// ModRefResult
//===----------------------------------------------------------------------===//

namespace
{
  struct AliasAnalysisModRefPass
      : public comet::AliasAnalysisModRefBase,
        PassWrapper<AliasAnalysisModRefPass, OperationPass<>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AliasAnalysisModRefPass)
    StringRef getArgument() const final { return "mem-alias-analysis-modref"; }
    StringRef getDescription() const final
    {
      return "Memory alias analysis ModRef results.";
    }
    void runOnOperation() override
    {
      AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
      runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
    }
  };
} // namespace

//===----------------------------------------------------------------------===//
// LocalAliasAnalysis extending
//===----------------------------------------------------------------------===//

/// Check if value is function argument.
static bool isFuncArg(Value val)
{
  auto blockArg = dyn_cast<BlockArgument>(val);
  if (!blockArg)
    return false;

  return mlir::isa_and_nonnull<FunctionOpInterface>(
      blockArg.getOwner()->getParentOp());
}

/// Check if value has "restrict" attribute. Value must be a function argument.
static bool isRestrict(Value val)
{
  auto blockArg = cast<BlockArgument>(val);
  auto func =
      mlir::cast<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
  return !!func.getArgAttr(blockArg.getArgNumber(),
                           "local_alias_analysis.restrict");
}

namespace
{
  /// LocalAliasAnalysis extended to support "restrict" attreibute.
  class LocalAliasAnalysisRestrict : public LocalAliasAnalysis
  {
  protected:
    AliasResult aliasImpl(Value lhs, Value rhs) override
    {
      if (lhs == rhs)
        return AliasResult::MustAlias;

      // Assume no aliasing if both values are function arguments and any of them
      // have restrict attr.
      if (isFuncArg(lhs) && isFuncArg(rhs))
        if (isRestrict(lhs) || isRestrict(rhs))
          return AliasResult::NoAlias;

      return LocalAliasAnalysis::aliasImpl(lhs, rhs);
    }
  };

  /// This pass performs additional analysis impls to the AliasAnalysis.
  struct AliasAnalysisExtendingPass
      : public comet::AliasAnalysisBase,
        PassWrapper<AliasAnalysisExtendingPass, OperationPass<>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AliasAnalysisExtendingPass)
    StringRef getArgument() const final
    {
      return "mem-alias-analysis-extending";
    }
    StringRef getDescription() const final
    {
      return "Memory alias analysis extending.";
    }
    void runOnOperation() override
    {
      AliasAnalysis aliasAnalysis(getOperation());
      aliasAnalysis.addAnalysisImplementation(LocalAliasAnalysisRestrict());
      runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
    }
  };
} // namespace

// static PassRegistration<AliasAnalysisModRefPass> pass;
// // //===----------------------------------------------------------------------===//
// // // Pass Registration
// // //===----------------------------------------------------------------------===//
// namespace mlir
// {
//   namespace comet
//   {
//     void registerTestAliasAnalysisPass()
//     {
//       PassRegistration<AliasAnalysisPass>();
//     }
//   } // namespace test
// } // namespace mlir

/// pass performs alias analysis
std::unique_ptr<Pass> mlir::comet::createAliasAnalysisPass()
{
  return std::make_unique<AliasAnalysisPass>();
}

/// pass performs ModRef.
std::unique_ptr<Pass> mlir::comet::createAliasAnalysisModRefPass()
{
  return std::make_unique<AliasAnalysisModRefPass>();
}

/// pass performs additional analysis impls to the AliasAnalysis.
std::unique_ptr<Pass> mlir::comet::createAliasAnalysisExtendingPass()
{
  return std::make_unique<AliasAnalysisExtendingPass>();
}
