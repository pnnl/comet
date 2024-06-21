//===- AbstractLoopOp.h - A helper class AbstractLoopOp to lower Index Tree dialect to SCF dialect  --*- C++ -*-===//
//
// Copyright 2024 Battelle Memorial Institute
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

#ifndef COMET_CONVERSION_INDEXTREETOSCF_ABSTRACTLOOPOP
#define COMET_CONVERSION_INDEXTREETOSCF_ABSTRACTLOOPOP

// #include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"

// #include "llvm/Support/Debug.h"
// #include "llvm/ADT/StringSet.h"
#include <string>
#include <unordered_set>
#include <variant>

using namespace mlir;

/**
 * @brief A helper class to hold `scf::ForOp` and `scf::ParallelOp`. Both operations are stored as
 * an `Operation *`. A string `iteratorType` tells the operation type. For a specific API provided
 * by `scf::ForOp` and `scf::ParallelOp`, `AbstractLoopOp` will provide the same API with a type
 * check to obtain the required operation type.
 *
 */
class AbstractLoopOp
{
private:
  mlir::Operation *op = nullptr;
  std::string iteratorType = "default";
  static std::unordered_set<std::string> supported_types;

public:
  AbstractLoopOp() = default;

  AbstractLoopOp(scf::ForOp forOp, std::string iterator_type)
  {
    setOp(forOp, iterator_type);
  }

  AbstractLoopOp(scf::ParallelOp parallelOp, std::string iterator_type)
  {
    setOp(parallelOp, iterator_type);
  }

  AbstractLoopOp(const std::string &iteratorType,
                 mlir::OpBuilder &builder,
                 const mlir::Location &loc,
                 const mlir::Value &lowerBound,
                 const mlir::Value &upperBound,
                 const mlir::Value &step)
  {
    buildLoopOp(iteratorType,
                builder,
                loc,
                lowerBound,
                upperBound,
                step);
  }

  void setIteratorType(std::string type);

  void setOp(scf::ForOp forOp, std::string iterator_type);

  void setOp(scf::ParallelOp parallelOp, std::string iterator_type);

  void setLowerBound(mlir::Value &lowerBound);

  void setUpperBound(mlir::Value &upperBound);

  std::string getIteratorType()
  {
    return iteratorType;
  }

  mlir::Operation *getOp()
  {
    return op;
  }

  mlir::Block *getBody();

  mlir::Value getInductionVar();

  /// implicit conversion function
  operator Operation *() const
  {
    return op;
  }

  /// Build the loop operation at the location.
  void buildLoopOp(const std::string &iteratorType,
                   mlir::OpBuilder &builder,
                   const mlir::Location &loc,
                   const mlir::Value &lowerBound,
                   const mlir::Value &upperBound,
                   const mlir::Value &step);

  void dump();
};
#endif // COMET_CONVERSION_INDEXTREETOSCF_ABSTRACTLOOPOP