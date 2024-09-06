//===-- Passes.td - Analysis pass definition file --------*- tablegen -*-===//
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

#ifndef COMET_ANALYSIS_PASSES_H
#define COMET_ANALYSIS_PASSES_H

include "mlir/Pass/PassBase.td"

///===----------------------------------------------------------------------===//
/// Memory Alias Analysis
///===----------------------------------------------------------------------===//

def AliasAnalysis : Pass<"memory-alias-analysis"> {
  let summary = " "
                "";
  let description = [{
      "Alias analysis."
      }];

  let constructor = "comet::createAliasAnalysisPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
  ];
}

def AliasAnalysisModRef : Pass<"memory-alias-modref-analysis"> {
  let summary = " "
                "";
  let description = [{
      "Alias analysis ModRef."
      }];

  let constructor = "comet::createAliasAnalysisExtendingPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
  ];
}

def AliasAnalysisExtending : Pass<"memory-alias-extending-analysis"> {
  let summary = " "
                "";
  let description = [{
      "Alias analysis extending."
      }];

  let constructor = "comet::createAliasAnalysisExtendingPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
  ];
}


///===----------------------------------------------------------------------===//
/// Memory Access Analysis
///===----------------------------------------------------------------------===//

def MemoryAccessFrequencyAnalysis : Pass<"mem-access-frequency-analysis"> {
  let summary = " "
                "";
  let description = [{
      "memory access frequency analysis."
      }];

  let constructor = "comet::createMemoryAccessFrequencyAnalysisPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "affine::AffineDialect",
  ];
}

def MemoryAccessPatternAnalysis : Pass<"mem-access-pattern-analysis"> {
  let summary = " "
                "";
  let description = [{
      "memory access pattern analysis."
      }];

  let constructor = "comet::createMemoryAccessPatternAnalysisPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "affine::AffineDialect",
  ];
}

#endif /// COMET_ANALYSIS_PASSES_H
