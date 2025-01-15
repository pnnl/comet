//===- Dialect.cpp - Index Tree  Dialect registration in MLIR ------------------===//
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
// =============================================================================
//
// This file implements the dialect for the Index Tree Dialect
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "llvm/ADT/TypeSwitch.h"

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"

using namespace mlir;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

//===----------------------------------------------------------------------===//
// IndexTreeDialect
//===----------------------------------------------------------------------===//

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "comet/Dialect/IndexTree/IR/IndexTreeTypes.cpp.inc"

// Include the op interface definitions
#include "comet/Dialect/IndexTree/IR/IndexTreeOpInterfaces.cpp.inc"


#define GET_OP_CLASSES
#include "comet/Dialect/IndexTree/IR/IndexTreeOps.cpp.inc"

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void IndexTreeDialect::initialize()
{
  addTypes<
#define GET_TYPEDEF_LIST
#include "comet/Dialect/IndexTree/IR/IndexTreeTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "comet/Dialect/IndexTree/IR/IndexTreeOps.cpp.inc"
      >();
}

using namespace mlir;
using namespace mlir::indexTree;
