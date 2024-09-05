//===- AliasAnalysis.h  ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a common facility that will be used for the
// aliasing analyses
//
//===----------------------------------------------------------------------===//

#ifndef COMET_ANALYSIS_ALIASANALYSIS_H
#define COMET_ANALYSIS_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"

namespace mlir
{

  class Pass;
  class RewritePatternSet;
  namespace comet
  {

#define GEN_PASS_DECL_MEMORYANALYSIS
#include "comet/Analysis/Passes.h.inc"

    /// Print the result of an alias query.
    void printAliasResult(AliasResult result, Value lhs, Value rhs);
    void printModRefResult(ModRefResult result, Operation *op, Value location);

    struct AliasAnalysisBase
    {
      void runAliasAnalysisOnOperation(Operation *op, AliasAnalysis &aliasAnalysis);
    };

    struct AliasAnalysisModRefBase
    {
      void runAliasAnalysisOnOperation(Operation *op, AliasAnalysis &aliasAnalysis);
    };

    /// pass performs alias analysis
    std::unique_ptr<Pass> createAliasAnalysisPass();

    /// pass performs ModRef.
    std::unique_ptr<Pass> createAliasAnalysisModRefPass();

    /// pass performs additional analysis impls to the AliasAnalysis.
    std::unique_ptr<Pass> createAliasAnalysisExtendingPass();

  } // namespace comet
} // namespace mlir

#endif // COMET_ANALYSIS_ALIASANALYSIS_H
