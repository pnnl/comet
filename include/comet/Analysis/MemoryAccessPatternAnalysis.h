//===- MemoryAccessPatternAnalysis.h  ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memory access pattern analysis
//
//===----------------------------------------------------------------------===//

#ifndef COMET_MEMORY_ACCESS_PATTERN_ANALYSIS_H
#define COMET_MEMORY_ACCESS_PATTERN_ANALYSIS_H

#include <string>

namespace mlir
{
  class Pass;
  class RewritePatternSet;
  namespace comet
  {

#define GEN_PASS_DECL_MEMORYANALYSIS
#include "comet/Analysis/Passes.h.inc"

    /// pass performs additional analysis impls to the AliasAnalysis.
    std::unique_ptr<Pass> createMemoryAccessFrequencyAnalysisPass();
  } // namespace comet
} // namespace mlir

#endif // COMET_MEMORY_ACCESS_PATTERN_ANALYSIS_H
