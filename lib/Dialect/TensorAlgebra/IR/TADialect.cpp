//===- Dialect.cpp - TA IR Dialect registration in MLIR ------------------===//
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
// This file implements the dialect for the TA IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tensorAlgebra;

#include "comet/Dialect/TensorAlgebra/IR/TADialect.cpp.inc"

//===----------------------------------------------------------------------===//
/// TADialect
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// ConstantOp

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void DenseConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                            double value)
{
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  DenseConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
// mlir::ParseResult DenseConstantOp::parse(mlir::OpAsmParser &parser,
//                                          mlir::OperationState &result)
// {
//   mlir::DenseElementsAttr value;
//   if (parser.parseOptionalAttrDict(result.attributes) ||
//       parser.parseAttribute(value, "value", result.attributes))
//     return failure();

//   result.addTypes(value.getType());
//   return success();
// }

// /// The 'OpAsmPrinter' class is a stream that allows for formatting
// /// strings, attributes, operands, types, etc.
// void DenseConstantOp::print(mlir::OpAsmPrinter &printer)
// {
//   printer << " ";
//   printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
//   printer << getValue();
// }

/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
mlir::LogicalResult DenseConstantOp::verify()
{
  /// If the return type of the constant is not an unranked tensor, the shape
  /// must match the shape of the attribute holding the data.
  auto resultType = getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  /// Check that the rank of the attribute type matches the rank of the constant
  /// result type.
  auto attrType = getValue().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank())
  {
    if(!(attrType.getRank() == 1 && attrType.getDimSize(0) == 1))
    {
      return emitOpError("return type must match the one of the attached value "
                        "attribute: ")
            << attrType.getRank() << " != " << resultType.getRank();
    }
    else
    {
      return mlir::success();
    }
  }

  /// Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim)
  {
    if (attrType.getShape()[dim] != resultType.getShape()[dim])
    {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
//===----------------------------------------------------------------------===//
/// GenericCallOp
//===----------------------------------------------------------------------===//
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments)
{
  /// Generic call always returns an unranked Tensor initially.
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee()
{
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

//===----------------------------------------------------------------------===//
/// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs)
{
  /// FunctionOpInterface provides a convenient `build` method that will populate
  /// the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result)
{
  /// Dispatch to the FunctionOpInterface provided utility method that parses the
  /// function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &)
  { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p)
{
  /// Dispatch to the FunctionOpInterface provided utility method that prints the
  /// function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
/// DivOp
void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::TensorType resultType,
                  mlir::Value lhs, mlir::Value rhs)
{
  state.addTypes(resultType);
  state.addOperands({lhs, rhs});
}

mlir::LogicalResult TAReturnOp::verify()
{
  /// We know that the parent operation is a function, because of the 'HasParent'
  /// trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  /// The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  /// If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  /// Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
/// TensorDimOp
//===----------------------------------------------------------------------===//

// Helper builder to simplify building fron integer index
void TensorDimOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                        Value source, int64_t index)
{
  auto loc = result.location;
  Value indexValue = builder.create<mlir::arith::ConstantIndexOp>(loc, index);
  build(builder, result, builder.getIndexType(), source, indexValue);
}

//===----------------------------------------------------------------------===//
/// TA Types
//===----------------------------------------------------------------------===//

// Implements the shaped type interface for the workspace type
ShapedType WorkspaceType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const
{
  // TODO: This may (?) require converting dimensions? Not sure 
  assert(false && "Workspace tensor cannot not be closed into another type");
  return NULL;
}

bool WorkspaceType::hasRank() const
{
  return true;
}

llvm::ArrayRef<int64_t> WorkspaceType::getShape() const
{
  return getDims();
}

// Implements the shaped type interface for the sparse tensor type
ShapedType SparseTensorType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const
{
  // TODO: This may (?) require converting dimensions? Not sure 
  assert(false && "Sparse tensor cannot not be closed into another type");
  return NULL;
}

bool SparseTensorType::hasRank() const
{
  return true;
}

llvm::ArrayRef<int64_t> SparseTensorType::getShape() const
{
  return getDims();
}

::mlir::Type SparseTensorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::FailureOr<::mlir::Type> _result_element_type; 
  ::mlir::FailureOr<::mlir::IntegerType> indices_type;
  SmallVector<int64_t> _result_dims;
  SmallVector<TensorFormatEnum> result_formats;

  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'element_type'
  _result_element_type = FieldParser<Type>::parse(odsParser);
  if (failed(_result_element_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensor parameter 'element_type' which is to be a `Type`");
    return {};
  }

    // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'indices_type'
  indices_type = ::mlir::FieldParser<::mlir::IntegerType>::parse(odsParser);
  if (::mlir::failed(indices_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensor parameter 'indices_type' which is to be a `::mlir::IntegerType`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'dims'
  if(odsParser.parseDimensionList(_result_dims, true, false)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensor parameter 'dims' which is to be a `ArrayRef<int64_t>`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'format'
  do {
    auto format = ::mlir::FieldParser<TensorFormatEnum>::parse(odsParser);
    if (mlir::failed(format))  // Parse each integer
      return {};
    result_formats.push_back(*format);
  } while (odsParser.parseOptionalComma().succeeded()); 

  // Parse literal '>'
  if (odsParser.parseGreater()) return {};

   return SparseTensorType::get(odsParser.getContext(),
      ::mlir::Type((*_result_element_type)),
      ::mlir::IntegerType((*indices_type)),
      ::llvm::ArrayRef<int64_t>((_result_dims)),
      ::llvm::ArrayRef<TensorFormatEnum>((result_formats)));
}


void SparseTensorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getIndicesType());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printDimensionList(getDims());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getFormat());
  odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
/// TableGen'd type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "comet/Dialect/TensorAlgebra/IR/TATypes.cpp.inc"

//===----------------------------------------------------------------------===//
/// TableGen'd enum definitions
//===----------------------------------------------------------------------===//
#include "comet/Dialect/TensorAlgebra/IR/TAEnums.cpp.inc"

//===----------------------------------------------------------------------===//
/// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "comet/Dialect/TensorAlgebra/IR/TAOps.cpp.inc"

//===----------------------------------------------------------------------===//
/// Tensor Algebra Dialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void TADialect::initialize()
{
  addTypes<
#define GET_TYPEDEF_LIST
#include "comet/Dialect/TensorAlgebra/IR/TATypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "comet/Dialect/TensorAlgebra/IR/TAAttrs.cpp.inc"
  >();

  addOperations<
#define GET_OP_LIST
#include "comet/Dialect/TensorAlgebra/IR/TAOps.cpp.inc"
      >();
}
