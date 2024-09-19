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

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tensorAlgebra;

#include "comet/Dialect/TensorAlgebra/IR/TADialect.cpp.inc"

//===----------------------------------------------------------------------===//
/// TADialect
//===----------------------------------------------------------------------===//

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

namespace mlir
{
  namespace tensorAlgebra
  {
    namespace detail
    {
      /// This class represents the internal storage of the tensorAlgebra `SparseTensorType`.
      struct SparseTensorTypeStorage : public mlir::TypeStorage
      {
        /// The `KeyTy` is a required type that provides an interface for the storage
        /// instance. This type will be used when uniquing an instance of the type
        /// storage. For our struct type, we will unique each instance structurally on
        /// the elements that it contains.
        using KeyTy = llvm::ArrayRef<mlir::Type>;

        /// A constructor for the type storage instance.
        SparseTensorTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
            : elementTypes(elementTypes) {}

        /// Define the comparison function for the key type with the current storage
        /// instance. This is used when constructing a new instance to ensure that we
        /// haven't already uniqued an instance of the given key.
        bool operator==(const KeyTy &key) const { return key == elementTypes; }

        /// Define a hash function for the key type. This is used when uniquing
        /// instances of the storage, see the `SparseTensorType::get` method.
        /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
        /// have hash functions available, so we could just omit this entirely.
        static llvm::hash_code hashKey(const KeyTy &key)
        {
          return llvm::hash_value(key);
        }

        /// Define a construction function for the key type from a set of parameters.
        /// These parameters will be provided when constructing the storage instance
        /// itself.
        /// Note: This method isn't necessary because KeyTy can be directly
        /// constructed with the given parameters.
        static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes)
        {
          return KeyTy(elementTypes);
        }

        /// Define a construction method for creating a new instance of this storage.
        /// This method takes an instance of a storage allocator, and an instance of a
        /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
        /// allocations used to create the type storage and its internal.
        static SparseTensorTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                                  const KeyTy &key)
        {
          /// Copy the elements from the provided `KeyTy` into the allocator.
          llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

          /// Allocate the storage instance and construct it.
          return new (allocator.allocate<SparseTensorTypeStorage>())
              SparseTensorTypeStorage(elementTypes);
        }

        /// The following field contains the element types of the struct.
        llvm::ArrayRef<mlir::Type> elementTypes;
      };

    } /// end namespace detail
  } /// end namespace tensoralgebra
} /// end namespace mlir

/// Create an instance of a `SparseTensorType` with the given element types. There
/// *must* be at least one element type.
SparseTensorType SparseTensorType::get(llvm::ArrayRef<mlir::Type> elementTypes)
{
  assert(!elementTypes.empty() && "expected at least 1 element type");

  /// Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  /// of this type. The first two parameters are the context to unique in and the
  /// kind of the type. The parameters after the type kind are forwarded to the
  /// storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

/// Returns the element types of this sparse tensor type.
llvm::ArrayRef<mlir::Type> SparseTensorType::getElementTypes()
{
  /// 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

Type mlir::tensorAlgebra::TADialect::parseType(DialectAsmParser &parser) const
{
  /// Parse the main keyword for the type.
  StringRef keyword;
  /// for "indexlabel" and "spTensor" type
  if (parser.parseKeyword(&keyword))
    return Type();

  MLIRContext *context = getContext();

  /// Handle 'range' types.
  if (keyword == "indexlabel")
  {
    return IndexLabelType::get(context);
  }

  /// Parse the element types of the spTensor.
  if (keyword == "spTensor")
  {
    if (parser.parseLess())
    {
      return Type();
    }

    SmallVector<mlir::Type, 1> elementTypes;
    do
    {
      /// Parse the current element type.
      llvm::SMLoc typeLoc = parser.getCurrentLocation();
      mlir::Type elementType;

      if (parser.parseType(elementType))
        return nullptr;

      /// Check that the type is either a TensorType or another SparseTensorType.
      if (!elementType.isa<mlir::TensorType, SparseTensorType, IndexType>())
      {
        parser.emitError(typeLoc, "element type for a struct must either "
                                  "be a TensorType or a SparseTensorType, got: ")
            << elementType;
        return Type();
      }
      elementTypes.push_back(elementType);

      /// Parse the optional: `,`
    } while (succeeded(parser.parseOptionalComma()));

    /// Parse: `>`
    if (parser.parseGreater())
      return Type();

    return SparseTensorType::get(elementTypes);
  }

  parser.emitError(parser.getNameLoc(),
                   "unknown TensorAlgebra type: " + keyword);
  return Type();
}

/// IndexLabelType prints as just "indexlabel".
static void print(IndexLabelType type, DialectAsmPrinter &printer)
{
  printer << "indexlabel";
}

void mlir::tensorAlgebra::TADialect::printType(
    Type type, DialectAsmPrinter &printer) const
{
  if (type.isa<IndexLabelType>())
  {
    print(type.cast<IndexLabelType>(), printer);
  }
  else if (type.isa<SparseTensorType>())
  {
    /// Currently the only sparse tensor type is a struct type.
    SparseTensorType sparseTensorType = type.cast<SparseTensorType>();

    /// Print the struct type according to the parser format.
    printer << "spTensor<";
    llvm::interleaveComma(sparseTensorType.getElementTypes(), printer);
    printer << '>';
  }
  else
  {
    llvm_unreachable("Unhandled TensorAlgebra type");
  }
}

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
  addOperations<
#define GET_OP_LIST
#include "comet/Dialect/TensorAlgebra/IR/TAOps.cpp.inc"
      >();
  // addTypes<IndexLabelType, SparseTensorType>();
  addTypes<SparseTensorType, IndexLabelType>();
}
