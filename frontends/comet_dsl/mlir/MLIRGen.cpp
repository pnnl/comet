//===- MLIRGen.cpp - MLIR Generation from a COMET language AST
//
/// Copyright 2022 Battelle Memorial Institute
//
/// Redistribution and use in source and binary forms, with or without modification,
/// are permitted provided that the following conditions are met:
//
/// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
/// and the following disclaimer.
//
/// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
/// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
/// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
/// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
/// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
/// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
/// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
/// =============================================================================
//
/// This file implements a simple IR generation targeting MLIR from a Module AST
/// for COMET DSL.
//
//===----------------------------------------------------------------------===//

#include "Lexer.h"

#include "MLIRGen.h"
#include "AST.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"

#include "comet/Dialect/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <cstdlib> /// for random num generation
#include <random>  /// for seed of random num generation

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"

using namespace mlir::tensorAlgebra;
using namespace tensorAlgebra;
using namespace mlir::arith;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;
int32_t defaultSpTensorIndiceBitWidth = 64; // TODO: We should be able to pass this from the DSL

using StringSet = std::set<std::string>;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

namespace
{

  enum class ElType
  {
    inv = 0b1000,
    i32 = 0b0000,
    i64 = 0b0001,
    fp32 = 0b0010,
    fp64 = 0b0011,
    cfp32 = 0b0110,
    cfp64 = 0b0111
  };

  std::string getSemiringOpName(int num)
  {
    std::string opName;
    switch (num)
    {
    case 100:
      opName = "lor";
      break;
    case 101:
      opName = "land";
      break;
    case 102:
      opName = "minxy";
      break;
    case 103:
      opName = "first";
      break;
    case 104:
      opName = "plusxy";
      break;
    case 105:
      opName = "times";
      break;
    case 106:
      opName = "any";
      break;
    case 107:
      opName = "pairxy";
      break;
    case 108:
      opName = "second";
      break;
    case 109:
      opName = "minus";
      break;
    case 110:
      opName = "noop";
      break;
    default:
      llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: Semiring operator number not defined \n";
    }
    comet_debug() << "Semiring op name: " << opName << "\n";
    return opName;
  };

  bool is_number(const std::string &s)
  {
    return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c)
                                      { return !std::isdigit(c); }) == s.end();
  }

  constexpr ElType lub(ElType first, ElType second)
  {
    return ElType(static_cast<int>(first) | static_cast<int>(second));
  }

  template <typename T>
  std::vector<mlir::Value> getOperationResultIndexLabels(mlir::Operation *op)
  {
    auto cast_op = cast<T>(op);
    return cast_op.getResultIndexLabels();
  }
  // Implementation of a simple MLIR emission from the Tensor Algebra AST.
  ///
  // This will emit operations that are specific to the Tensor Algebra language,
  // preserving the semantics of the language and (hopefully) allow to perform
  // accurate analysis and transformation based on these high level semantics.
  class MLIRGenImpl
  {
  public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    // Public API: convert the AST for a Toy module (source file) to an MLIR
    // Module operation.
    mlir::ModuleOp mlirGen(ModuleAST &moduleAST)
    {
      /// We create an empty MLIR module and codegen functions one at a time and
      /// add them to the module.
      theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

      for (FunctionAST &funcAST : moduleAST)
      {
        mlir::tensorAlgebra::FuncOp func = mlirGen(funcAST);
        if (!func)
          return nullptr;
        functionMap.insert({func.getName(), func});
      }

      /// Verify the module after we have finished constructing it, this will check
      /// the structural properties of the IR and invoke any specific verifiers we
      /// have on the Toy operations.
      if (failed(mlir::verify(theModule)))
      {
        theModule.emitError("module verification error");
        return nullptr;
      }

      return theModule;
    }

  private:
    /// A "module" matches a Tensor Algebra source file: containing a list of
    /// functions.
    mlir::ModuleOp theModule;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    /// The symbol table maps a variable name to a value in the current scope.
    /// Entering a function creates a new scope, and the function arguments are
    /// added to the mapping. When the processing of a function is terminated, the
    /// scope is destroyed and the mappings created in this scope are dropped.
    llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
    std::map<std::string, int64_t> ilabel_symbolTable;   // [TODO] This needs to use scopedhashtable
    std::map<std::string, mlir::Value> labelToTensorDim; // [TODO] This needs to use scopedhashtable

    /// A mapping for the functions that have been code generated to MLIR.
    llvm::StringMap<mlir::tensorAlgebra::FuncOp> functionMap;

    // Helper conversion for a Tensor Algebra AST location to an MLIR location.
    mlir::Location loc(const Location &loc)
    {
      return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                       loc.col);
    }

    /// For scalar-arithmetic-op:
    /// var a = 0; and c = a + b; are all specified in VarDeclExprAST
    /// in which all var are added to the symboltable; in this case,
    /// a is added twice. So, we cannot return failure for this case if found
    /// an existing variable from the symbol table.
    /// Change to:
    /// If not exist, add the variable to symbol table; return success anyway
    mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value)
    {
      if (!symbolTable.count(var))
      {
        symbolTable.insert(var, value);
      }
      return mlir::success();
    }

    void declarelabeltotensordim(std::string var, mlir::Value value)
    {
      labelToTensorDim[var] = value;
    }

    void declarelabel(IndexLabelDeclExprAST &label)
    {
      {
        if (label.getEnd() == mlir::ShapedType::kDynamic)
        {
          ilabel_symbolTable[label.getName().str()] = mlir::ShapedType::kDynamic;
        }
        else
        {
          ilabel_symbolTable[label.getName().str()] = (label.getEnd() - label.getBegin()) / label.getIncrement();
        }
      }
    }

    /// Create the prototype for an MLIR function with as many arguments as the
    /// provided Tensor Algebra AST prototype.
    mlir::tensorAlgebra::FuncOp mlirGen(PrototypeAST &proto)
    {
      auto location = loc(proto.loc());

      /// This is a generic function, the return type will be inferred later.
      /// Arguments type are uniformly unranked tensors.
      llvm::SmallVector<mlir::Type, 4> arg_types(proto.getArgs().size(),
                                                 getType(VarType{}));
      auto func_type = builder.getFunctionType(arg_types, std::nullopt);
      return builder.create<mlir::tensorAlgebra::FuncOp>(location, proto.getName(),
                                                         func_type);
    }

    /// Emit a new function and add it to the MLIR module.
    mlir::tensorAlgebra::FuncOp mlirGen(FunctionAST &funcAST)
    {
      /// Create a scope in the symbol table to hold variable declarations.
      /// SymbolTableScopeT varScope(symbolTable);
      /// Create a scope in the symbol table to hold variable declarations.
      ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

      /// Create an MLIR function for the given prototype.
      builder.setInsertionPointToEnd(theModule.getBody());
      mlir::tensorAlgebra::FuncOp function = mlirGen(*funcAST.getProto());
      if (!function)
        return nullptr;

      /// Let's start the body of the function now!
      mlir::Block &entryBlock = function.front();
      auto &protoArgs = funcAST.getProto()->getArgs();

      // function.
      /// Declare all the function arguments in the symbol table.
      for (const auto nameValue :
           llvm::zip(protoArgs, entryBlock.getArguments()))
      {
        comet_debug() << "Proto Args " << std::get<1>(nameValue) << "\n";
        if (failed(declare(std::get<0>(nameValue)->getName(),
                           std::get<1>(nameValue))))
          return nullptr;
      }

      /// Set the insertion point in the builder to the beginning of the function
      /// body, it will be used throughout the codegen to create operations in this
      /// function.
      builder.setInsertionPointToStart(&entryBlock);

      /// Emit the body of the function.
      if (mlir::failed(mlirGen(*funcAST.getBody())))
      {
        function.erase();
        return nullptr;
      }

      /// Implicitly return void if no return statement was emitted.
      /// FIXME: we may fix the parser instead to always return the last expression
      /// (this would possibly help the REPL case later)
      TAReturnOp returnOp;
      if (!entryBlock.empty())
        returnOp = dyn_cast<TAReturnOp>(entryBlock.back());
      if (!returnOp)
      {
        builder.create<TAReturnOp>(loc(funcAST.getProto()->loc()));
      }
      else if (returnOp.hasOperand())
      {
        /// Otherwise, if this return operation has an operand then add a result to
        /// the function.
        function.setType(
            builder.getFunctionType(function.getFunctionType().getInputs(),
                                    *returnOp.operand_type_begin()));
      }

      /// If this function isn't main, then set the visibility to private.
      if (funcAST.getProto()->getName() != "main")
        function.setPrivate();

      return function;
    }

    // Emit a binary operation
    mlir::Value mlirGen(BinaryExprAST &binop,
                        const std::set<std::string> &out_lbls = {}, std::string out_format = "")
    {
      comet_debug() << " mlirGen for  BinaryExprAST \n";
      /// First emit the operations for each side of the operation before emitting
      /// the operation itself. For example if the expression is `a + foo(a)`
      /// 1) First it will visiting the LHS, which will return a reference to the
      ///   value holding `a`. This value should have been emitted at declaration
      ///   time and registered in the symbol table, so nothing would be
      ///   codegen'd. If the value is not in the symbol table, an error has been
      ///   emitted and nullptr is returned.
      /// 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
      ///   and the result value is returned. If an error occurs we get a nullptr
      ///   and propagate.
      ///
      std::set<std::string> rhs_lbls{out_lbls};
      std::set<std::string> lhs_lbls{out_lbls};
      auto *lhsAST = binop.getLHS();
      auto *rhsAST = binop.getRHS();

      if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor)
      {
        comet_debug() << " lhsAST is Expr_LabeledTensor \n";
        auto lblsVec = cast<LabeledTensorExprAST>(lhsAST)->getLabelNames();
        for (const auto &lbl : lblsVec)
        {
          lhs_lbls.insert(lbl);
        }
      }
      else if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_BinOp)
      {
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST is Expr_BinOp \n";
        auto lblsSet = cast<BinaryExprAST>(lhsAST)->getLabels();
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST.getOp(): " << cast<BinaryExprAST>(lhsAST)->getOp() << "\n";

        for (const auto &lbl : lblsSet)
        {
          comet_debug() << lbl << "\n";
          lhs_lbls.insert(lbl);
        }
      }
      else if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_Call)
      {
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST is Expr_Call  \n";
      }
      else if (lhsAST->getKind() == ExprAST::ExprASTKind::Expr_Transpose)
      {
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST is Expr_Transpose  \n";
      }
      else
      {
        comet_debug() << "\n"
                      << __LINE__ << " lhsAST is not Expr_BinOp/Expr_LabeledTensor/CallExprAST/TransposeExprAST  \n";
      }

      if (rhsAST->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor)
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST is Expr_LabeledTensor \n";
        auto lblsVec = cast<LabeledTensorExprAST>(rhsAST)->getLabelNames();
        for (const auto &lbl : lblsVec)
        {
          comet_debug() << lbl << "\n";
          rhs_lbls.insert(lbl);
        }
      }
      else if (rhsAST->getKind() == ExprAST::ExprASTKind::Expr_BinOp)
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST is Expr_BinOp \n";
        auto lblsSet = cast<BinaryExprAST>(rhsAST)->getLabels();
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST.getOp(): " << cast<BinaryExprAST>(rhsAST)->getOp() << "\n";

        for (const auto &lbl : lblsSet)
        {
          rhs_lbls.insert(lbl);
        }
      }
      else if (rhsAST->getKind() == ExprAST::ExprASTKind::Expr_Transpose)
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST is Expr_Transpose  \n";
      }
      else
      {
        comet_debug() << "\n"
                      << __LINE__ << " rhsAST is not Expr_BinOp/Expr_LabeledTensor/CallExprAST/TransposeExprAST  \n";
      }

      mlir::Value lhs = mlirGen(*binop.getLHS(), rhs_lbls);
      if (!lhs)
        return nullptr;
      mlir::Value rhs = mlirGen(*binop.getRHS(), lhs_lbls);
      if (!rhs)
        return nullptr;
      auto location = loc(binop.loc());

      comet_vdump(lhs);
      comet_vdump(rhs);
      comet_debug() << " " << lhsAST->getKind() << " " << rhsAST->getKind() << "\n";

      if ((lhsAST->getKind() == ExprAST::ExprASTKind::Expr_Var &&
           rhsAST->getKind() == ExprAST::ExprASTKind::Expr_Var) ||
          rhsAST->getKind() == ExprAST::ExprASTKind::Expr_Num)
      {
        // Scalar operations
        comet_debug() << " rhsAST and lhsAST are all Expr_Var OR rhsAST is Expr_Num\n";

        std::string op;
        switch (binop.getOp())
        {
        case '+':
          op = '+';
          break;
        case '-':
          op = '-';
          break;
        case '*':
          op = '*';
          break;
        case '/':
          op = '/';
          break;
        default:
          comet_debug() << "ERROR: unsupported operator type: ASCII Code(" << binop.getOp() << ")\n";
        }

        mlir::StringAttr opAttr = builder.getStringAttr(op);
        mlir::RankedTensorType returnDataType;
        if(lhs.getType().cast<mlir::RankedTensorType>().getShape() != rhs.getType().cast<mlir::RankedTensorType>().getShape())
        {
          returnDataType = lhs.getType().cast<mlir::RankedTensorType>();
          auto bcastRhs = builder.create<DenseConstantOp>(location, returnDataType, mlir::cast<DenseConstantOp>(rhs.getDefiningOp()).getValueAttr());
          comet_vdump(bcastRhs);
          rhs.replaceAllUsesWith(bcastRhs);
          rhs = bcastRhs;
        }
        else {
          mlir::Type elementType = builder.getF64Type();
          returnDataType = mlir::RankedTensorType::get(1, elementType);
        }
        comet_vdump(rhs);
        comet_vdump(lhs);

        /// lookup the output of the binary operation
        auto theOutput = symbolTable.lookup(out_format);
        if (theOutput == nullptr)
        { /// the variable for output of binary operation was not declared by user,
          /// we will create a new DenseConstantOp here.
          comet_debug() << "creating a new variable declaration, since the user did not declare it\n";

          double data = 0.0;
          auto dataAttribute = mlir::DenseElementsAttr::get(mlir::RankedTensorType::get({1}, builder.getF64Type()), llvm::ArrayRef(data));
          auto denseConst = builder.create<DenseConstantOp>(location, returnDataType, dataAttribute);

          theOutput = denseConst;
        }
        comet_vdump(theOutput);
        auto scalarOp = builder.create<ScalarOp>(location, returnDataType, rhs, lhs, opAttr);
        comet_vdump(scalarOp);
        builder.create<TensorSetOp>(location, scalarOp, theOutput);

        /// the value returned here will be used in subsequent ops.
        /// for example, in the code below, 'g' should be returned.
        ///   $ var g = a + b;
        ///   $ print(g);
        return theOutput;
      }

      else if (isa<DenseConstantOp>(lhs.getDefiningOp()))
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Unexpected case\n";
      }
      else if (isa<DenseConstantOp>(rhs.getDefiningOp()))
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Unexpected case\n";
      }
      else
      {
        comet_debug() << " lhs or rhs are binaryop\n";
        /// lhs && rhs are binaryop
 

        if (rhsAST->getKind() == ExprAST::ExprASTKind::Expr_BinOp)
        {
          comet_debug() << " rhs is binaryop\n";
          auto rhsOp = rhs.getDefiningOp();

          comet_pdump(rhsOp);
          if (isa<TensorAddOp>(rhsOp))
          {
            // TODO(gkestor) check for AddOp
            llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Not supported RHS operation - Tensor addition\n";
          }
          else if (isa<TensorSubtractOp>(rhsOp))
          {
            // TODO(gkestor) check for subtract operation
            llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Not supported RHS operation - Tensor subtraction\n";
          }
          else if (isa<mlir::tensorAlgebra::TransposeOp>(rhsOp))
          { // * transpose(A[i,j])
            comet_debug() << " "
                          << "\n";
          }
        }
        else
        {
          comet_debug() << " rhs is not binaryop\n";
        }

        auto in_labels = binop.getLabels();
        comet_debug() << __LINE__ << " in_labels: ";
        for (auto n : in_labels)
        {
          comet_debug() << n << " ";
        }
        comet_debug() << "\n";

        std::vector<std::string> sum_labels;
        std::set_difference(in_labels.begin(), in_labels.end(), out_lbls.begin(),
                            out_lbls.end(), std::back_inserter(sum_labels));
        comet_debug() << __LINE__ << " sum_labels: ";
        for (auto n : sum_labels)
        {
          comet_debug() << n << " ";
        }
        comet_debug() << "\n";

        std::vector<std::string> result_lbls;
        std::set<std::string> sum_lbls_set(sum_labels.begin(), sum_labels.end());
        std::set_difference(in_labels.begin(), in_labels.end(),
                            sum_lbls_set.begin(), sum_lbls_set.end(),
                            std::back_inserter(result_lbls));
        comet_debug() << __LINE__ << " result_lbls: ";
        for (auto n : result_lbls)
        {
          comet_debug() << n << " ";
        }
        comet_debug() << "\n";

        auto lhs_tensor = lhs.getDefiningOp()->getOpResult(0).getType();

        comet_pdump(lhs.getDefiningOp());
        
        auto lhs_labeledtensor = lhs.getDefiningOp()->getOpResult(0);

        comet_vdump(lhs_labeledtensor); // ta.labeled_tensor
        mlir::Type lhs_el_type;
        if(auto tensor_type = llvm::dyn_cast<mlir::TensorType>(lhs_tensor)){
          lhs_el_type = tensor_type.getElementType();
        }
        else if(auto tensor_type = llvm::dyn_cast<SparseTensorType>(lhs_tensor)){
          lhs_el_type = tensor_type.getElementType();
        }
        else {
          assert(false && "Expected a tensor input");
        }

        auto rhs_tensor = rhs.getDefiningOp()->getOpResult(0).getType();

        comet_pdump(rhs.getDefiningOp());

        auto rhs_labeledtensor = rhs.getDefiningOp()->getOpResult(0);

        comet_vdump(rhs_labeledtensor);
        mlir::Type rhs_el_type;
        if(auto tensor_type = llvm::dyn_cast<mlir::TensorType>(rhs_tensor)){
          rhs_el_type = tensor_type.getElementType();
        }
        else if(auto tensor_type = llvm::dyn_cast<SparseTensorType>(rhs_tensor)){
          rhs_el_type = tensor_type.getElementType();
        }
        else {
          assert(false && "Expected a tensor input");
        }

        auto result_type = getBinOpResultType(lhs_el_type, rhs_el_type);
        comet_debug() << __LINE__ << " ";
        comet_vdump(result_type);

        comet_debug() << __LINE__ << " binop.getOp(): " << binop.getOp() << "\n";
        std::vector<mlir::Value> lhs_lbls_value;

        // TODO(gkestor): urgent refactor the following code
        if (isa<SparseTensorDeclOp, DenseTensorDeclOp>(lhs_labeledtensor.getDefiningOp()))
        {
          auto lblsVec = cast<LabeledTensorExprAST>(lhsAST)->getLabelNames();
          for (const auto &lbl : lblsVec)
          {
            lhs_lbls_value.push_back(symbolTable.lookup(lbl));
          }
        }
        else if (isa<TensorMultOp>(lhs_labeledtensor.getDefiningOp()))
        {
          lhs_lbls_value = getOperationResultIndexLabels<TensorMultOp>(lhs_labeledtensor.getDefiningOp());
        }
        else if (isa<TensorElewsMultOp>(lhs_labeledtensor.getDefiningOp()))
        {
          lhs_lbls_value = getOperationResultIndexLabels<TensorElewsMultOp>(lhs_labeledtensor.getDefiningOp());
        }
        else if (isa<TensorAddOp>(lhs_labeledtensor.getDefiningOp()))
        {
          lhs_lbls_value = getOperationResultIndexLabels<TensorAddOp>(lhs_labeledtensor.getDefiningOp());
        }
        else if (isa<TensorSubtractOp>(lhs_labeledtensor.getDefiningOp()))
        {
          lhs_lbls_value = getOperationResultIndexLabels<TensorSubtractOp>(lhs_labeledtensor.getDefiningOp());
        }
        else if (isa<mlir::tensorAlgebra::TransposeOp>(lhs_labeledtensor.getDefiningOp()))
        {
          lhs_lbls_value = getOperationResultIndexLabels<mlir::tensorAlgebra::TransposeOp>(lhs_labeledtensor.getDefiningOp());
        }
        else
        {
          llvm::errs() << __FILE__ << ":" << __LINE__ << " unknown lhs \n";
        }

        std::vector<mlir::Value> rhs_lbls_value;

        if (isa<SparseTensorDeclOp, DenseTensorDeclOp>(rhs_labeledtensor.getDefiningOp()))
        {
          auto lblsVec = cast<LabeledTensorExprAST>(rhsAST)->getLabelNames();
          for (const auto &lbl : lblsVec)
          {
            rhs_lbls_value.push_back(symbolTable.lookup(lbl));
          }
        }
        else if (isa<TensorMultOp>(rhs_labeledtensor.getDefiningOp()))
        {
          rhs_lbls_value = getOperationResultIndexLabels<TensorMultOp>(rhs_labeledtensor.getDefiningOp());
        }
        else if (isa<TensorElewsMultOp>(rhs_labeledtensor.getDefiningOp()))
        {
          rhs_lbls_value = getOperationResultIndexLabels<TensorElewsMultOp>(rhs_labeledtensor.getDefiningOp());
        }
        else if (isa<TensorAddOp>(rhs_labeledtensor.getDefiningOp()))
        {
          rhs_lbls_value = getOperationResultIndexLabels<TensorAddOp>(rhs_labeledtensor.getDefiningOp());
        }
        else if (isa<TensorSubtractOp>(rhs_labeledtensor.getDefiningOp()))
        {
          rhs_lbls_value = getOperationResultIndexLabels<TensorSubtractOp>(rhs_labeledtensor.getDefiningOp());
        }
        else if (isa<mlir::tensorAlgebra::TransposeOp>(rhs_labeledtensor.getDefiningOp()))
        {
          rhs_lbls_value = getOperationResultIndexLabels<mlir::tensorAlgebra::TransposeOp>(rhs_labeledtensor.getDefiningOp());
        }
        else
        {
          llvm::errs() << __FILE__ << ":" << __LINE__ << " unknown rhs \n";
        }

        std::vector<mlir::Value> all_lbls_value;

        for (auto n : lhs_lbls_value)
        {
          all_lbls_value.push_back(n);
          comet_vdump(n);
        }
        for (auto m : rhs_lbls_value)
        {
          comet_vdump(m);
          auto result1 = std::find(all_lbls_value.begin(), all_lbls_value.end(), m);
          if (result1 == all_lbls_value.end())
          {
            all_lbls_value.push_back(m);
          }
        }
        comet_debug() << " " << all_lbls_value.size() << "\n";

        std::vector<int> lhs_lbls;
        std::vector<int> rhs_lbls;
        for (unsigned int i = 0; i < all_lbls_value.size(); i++)
        {
          auto result1 = std::find(lhs_lbls_value.begin(), lhs_lbls_value.end(), all_lbls_value[i]);
          if (result1 != lhs_lbls_value.end())
          {
            lhs_lbls.push_back(i);
          }

          auto result2 = std::find(rhs_lbls_value.begin(), rhs_lbls_value.end(), all_lbls_value[i]);
          if (result2 != rhs_lbls_value.end())
          {
            rhs_lbls.push_back(i);
          }
        }

        comet_debug() << " print lhs_lbls\n";
        for (auto n : lhs_lbls)
        {
          comet_debug() << n << " \n";
        }
        comet_debug() << "\n";
        comet_debug() << " print rhs_lbls\n";
        for (auto n : rhs_lbls)
        {
          comet_debug() << n << " \n";
        }
        comet_debug() << "\n";

        std::vector<int> sum_lbls;
        std::set_intersection(lhs_lbls.begin(), lhs_lbls.end(), rhs_lbls.begin(), rhs_lbls.end(), std::back_inserter(sum_lbls));
        std::vector<int> all_lbls;
        std::set_union(lhs_lbls.begin(), lhs_lbls.end(), rhs_lbls.begin(), rhs_lbls.end(), std::back_inserter(all_lbls));
        std::vector<int> ret_lbls;
        if (binop.getOp() == '*')
        {
          std::set_difference(all_lbls.begin(), all_lbls.end(), sum_lbls.begin(), sum_lbls.end(), std::back_inserter(ret_lbls));
        }
        else if (binop.getOp() == tok_elews || binop.getOp() == '+' || binop.getOp() == '-')
        {
          std::copy(lhs_lbls.begin(), lhs_lbls.end(), std::back_inserter(ret_lbls));
        }
        comet_debug() << " print ret_lbls\n";
        for (auto n : ret_lbls)
        {
          comet_debug() << n << " \n";
        }
        comet_debug() << "\n";

        comet_debug() << " print all_lbls\n";
        for (auto n : all_lbls)
        {
          comet_debug() << n << " \n";
        }
        comet_debug() << "\n";

        std::map<int, mlir::AffineExpr> expr_map;
        unsigned dim = 0;
        for (const auto &lbl : all_lbls)
        {
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
            mlir::AffineMap::get(dim, 0, lhs_exprs, context),
            mlir::AffineMap::get(dim, 0, rhs_exprs, context),
            mlir::AffineMap::get(dim, 0, ret_exprs, context)};

        std::vector<mlir::Value> ret_lbls_value;
        for (auto n : ret_lbls)
        {
          ret_lbls_value.push_back(all_lbls_value[n]);
        }

        // std::vector<int64_t> result_dims = getDimSizes(ret_lbls_value);
        auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);

        auto res_map = affineMapArrayAttr[affineMapArrayAttr.size() - 1].cast<mlir::AffineMapAttr>().getValue();

        /// get return-type based on affine-maps
        std::vector<int64_t> result_dims;

        for (auto v : res_map.getResults())
        {
          for (size_t i = 0; i < affineMapArrayAttr.size() - 1; i++)
          {
            auto map = affineMapArrayAttr[i].cast<mlir::AffineMapAttr>().getValue();
            if (auto pos = map.getResultPosition(v))
            {
              mlir::Value operand = i == 0 ? lhs_labeledtensor : rhs_labeledtensor;
              if(auto spTensorType = mlir::dyn_cast<mlir::tensorAlgebra::SparseTensorType>(operand.getType()))
              {
                result_dims.push_back(spTensorType.getDimSize(*pos));
              }
              else if(auto denseTensorType = mlir::dyn_cast<mlir::TensorType>(operand.getType()))
              {
                result_dims.push_back(denseTensorType.getDimSize(*pos));
              }
              else{
                assert(false && "Unexpected Tensor type");
              }
              break;
            }
          }
        }

        SmallVector<mlir::StringRef, 8> formats;
        std::vector<mlir::Value> exprs{lhs_labeledtensor, rhs_labeledtensor};
        std::vector<mlir::Value> tensors;
        // TODO(gkestor): URGENT refactor the following code -  too much repetition
        for (auto e : exprs)
        {
          if (isa<DenseTensorDeclOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";
            /// infer the format
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(e.getDefiningOp()).getFormat();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<DenseTensorDeclOp>(e.getDefiningOp()));
          }
          else if (isa<SparseTensorDeclOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";
            /// infer the format
            auto lhs_format = dyn_cast<SparseTensorDeclOp>(e.getDefiningOp()).getFormat();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<SparseTensorDeclOp>(e.getDefiningOp()));
          }
          else if (isa<TensorMultOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorMultOp\n";
            /// infer the format
            mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorMultOp>(e.getDefiningOp()).getFormats();
            unsigned int i = opFormatsArrayAttr.size() - 1;
            mlir::StringRef lhs_format = opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue();
            comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<TensorMultOp>(e.getDefiningOp()).getOperation()->getResult(0));
          }
          else if (isa<TensorElewsMultOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorElewsMultOp\n";
            /// infer the format
            mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorElewsMultOp>(e.getDefiningOp()).getFormats();
            unsigned int i = opFormatsArrayAttr.size() - 1;
            mlir::StringRef lhs_format = opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue();
            comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";

            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<TensorElewsMultOp>(e.getDefiningOp()).getOperation()->getResult(0));
          }
          else if (isa<TensorAddOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorAddOp\n";
            /// infer the format
            mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorAddOp>(e.getDefiningOp()).getFormats();
            unsigned int i = opFormatsArrayAttr.size() - 1;
            mlir::StringRef lhs_format = opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue();
            comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";

            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<TensorAddOp>(e.getDefiningOp()).getOperation()->getResult(0));
          }
          else if (isa<TensorSubtractOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TensorSubstract Op\n";
            /// infer the format
            mlir::ArrayAttr opFormatsArrayAttr = dyn_cast<TensorSubtractOp>(e.getDefiningOp()).getFormats();
            unsigned int i = opFormatsArrayAttr.size() - 1;
            mlir::StringRef lhs_format = opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue();
            comet_debug() << __LINE__ << " lhs_format: " << lhs_format << "\n";

            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<TensorSubtractOp>(e.getDefiningOp()).getOperation()->getResult(0));
          }
          else if (isa<mlir::tensorAlgebra::TransposeOp>(e.getDefiningOp()))
          {
            comet_debug() << " is TransposeOp\n";

            /// get the real transpose op output via the set op.
            mlir::Value transposeOut;
            mlir::Operation *firstUser = e.getDefiningOp()->getNextNode();
            if (isa<TensorSetOp>(firstUser))
            {
              TensorSetOp setOp = cast<TensorSetOp>(firstUser);
              transposeOut = setOp.getOperand(1);
            }
            else
            {
              llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Transpose has no set_op after it!\n";
            }

            /// get the format of transposeOut tensor
            if (isa<DenseTensorDeclOp>(transposeOut.getDefiningOp()))
            {
              auto denseFormat = dyn_cast<DenseTensorDeclOp>(transposeOut.getDefiningOp()).getFormat();
              formats.push_back(denseFormat);
            }
            else if (isa<SparseTensorDeclOp>(transposeOut.getDefiningOp()))
            {
              auto sparseFormat = dyn_cast<SparseTensorDeclOp>(transposeOut.getDefiningOp()).getFormat();
              formats.push_back(sparseFormat);
            }
            else
            {
              llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Can not determine tensor format with transpose op\n";
            }
            tensors.push_back(transposeOut);
          }
          else
          {
            comet_debug() << " not DenseTensorDecl op, not SparseTensorDecl op, not TensorMultOp, not TensorElewsMultOp\n";
          }
        }
        comet_debug() << __LINE__ << " formats.size(): " << formats.size() << "\n";
        assert(formats.size() == 2 && " less than 2 input tensors\n");
        mlir::Type ret_tensor_type;
        if (formats[0].compare("CSR") == 0 && formats[1].compare("CSR") == 0)
        {
          formats.push_back("CSR");
          std::vector format_array = getFormats("CSR", result_dims.size(), builder.getContext());
          ret_tensor_type = SparseTensorType::get(builder.getContext(), result_type, builder.getIntegerType(defaultSpTensorIndiceBitWidth), result_dims, format_array);
        }
        else if (formats[0].compare("Dense") == 0 && formats[1].compare("Dense") == 0)
        {
          formats.push_back("Dense");
          ret_tensor_type = mlir::RankedTensorType::get(result_dims, result_type);
        }
        else if (out_format.length() > 0) // non-empty format string provided.
        {
          comet_debug() << " Output Format: " << out_format << "\n";
          formats.push_back(out_format);
          if(out_format.compare("Dense") == 0)
          {
            ret_tensor_type = mlir::RankedTensorType::get(result_dims, result_type);
          } else {
            std::vector format_array = getFormats(out_format, result_dims.size(), builder.getContext());
            ret_tensor_type = SparseTensorType::get(builder.getContext(), result_type, builder.getIntegerType(defaultSpTensorIndiceBitWidth), result_dims, format_array);
          }
        }
        else
        {
          llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: the format of output tensor could not be determined during generation of binOp\n";
        }
        comet_debug() << " formats.size(): " << formats.size() << "\n";
        auto strAttr = builder.getStrArrayAttr(formats);

        assert(tensors.size() == 2 && " less than 2 input tensors for ta.mul or ta.elews_mul\n");

        std::vector<mlir::Value> labels;
        for (auto i : lhs_lbls)
        {
          labels.push_back(all_lbls_value[i]);
        }
        for (auto i : rhs_lbls)
        {
          labels.push_back(all_lbls_value[i]);
        }
        for (auto i : ret_lbls)
        {
          labels.push_back(all_lbls_value[i]);
        }

        mlir::StringAttr SemiringAttr;
        mlir::StringAttr MaskingAttr;
        /// Derive the operation name from the binary operator. At the moment we
        /// only support '+', '-','*'.
        switch (binop.getOp())
        {
        case '+':
          comet_debug() << "creating TensorAddOp\n";
          SemiringAttr = builder.getStringAttr("noop_plusxy"); // this is for standard elementwise addition
          MaskingAttr = builder.getStringAttr("none");         // default for standard elementwise addition
          return builder.create<TensorAddOp>(location, ret_tensor_type, tensors[0], tensors[1],
                                             labels, affineMapArrayAttr, strAttr, SemiringAttr,
                                             MaskingAttr);
        case '-':
          comet_debug() << "creating TensorSubtractOp\n";
          SemiringAttr = builder.getStringAttr("noop_minus"); // this is for standard elementwise subtraction
          MaskingAttr = builder.getStringAttr("none");        // default for standard elementwise subtraction
          return builder.create<TensorSubtractOp>(location, ret_tensor_type, tensors[0], tensors[1],
                                                  labels, affineMapArrayAttr, strAttr, SemiringAttr,
                                                  MaskingAttr);
        case '*':
        {
          comet_vdump(lhs_tensor);
          comet_debug() << "\n";

          comet_vdump(rhs_tensor);
          comet_debug() << "\n";
          SemiringAttr = builder.getStringAttr("plusxy_times"); // this is for standard matrix multiplication
          MaskingAttr = builder.getStringAttr("none");          // default for standard matrix multiplication
          mlir::Value tcop = builder.create<TensorMultOp>(location, ret_tensor_type, tensors[0], tensors[1],
                                                          labels, affineMapArrayAttr, strAttr, SemiringAttr,
                                                          MaskingAttr, nullptr); // TODO: masking is an optional operand
          tcop.getDefiningOp()->setAttr("__alpha__", builder.getF64FloatAttr(1.0));
          tcop.getDefiningOp()->setAttr("__beta__", builder.getF64FloatAttr(0.0));

          comet_vdump(tcop);
          return tcop;
        }

        case tok_elews:
          comet_vdump(lhs_tensor);
          comet_debug() << "\n";

          comet_vdump(rhs_tensor);
          comet_debug() << "\n";
          auto SemiringAttr = builder.getStringAttr("noop_times"); /// this is for standard element-wise multiplication
          MaskingAttr = builder.getStringAttr("none");             /// default for standard element-wise multiplication
          mlir::Value tcop = builder.create<TensorElewsMultOp>(location, ret_tensor_type, tensors[0], tensors[1], labels,
                                                               affineMapArrayAttr, strAttr, SemiringAttr,
                                                               MaskingAttr);

          comet_vdump(tcop);
          return tcop;
        }
      }

      emitError(location, "invalid binary operator '") << binop.getOp() << "'";
      return nullptr;
    }

    /// This is a reference to a variable in an expression. The variable is
    /// expected to have been declared and so should have a value in the symbol
    /// table, otherwise emit an error and return nullptr.
    mlir::Value mlirGen(VariableExprAST &expr)
    {
      if (auto variable = symbolTable.lookup(expr.getName()))
        return variable;

      emitError(loc(expr.loc()), "Unknown variable VariableExprAST '")
          << expr.getName() << "'";
      return nullptr;
    }

    /// Emit a return operation. This will return failure if any generation fails.
    mlir::LogicalResult mlirGen(ReturnExprAST &ret)
    {
      auto location = loc(ret.loc());

      /// 'return' takes an optional expression, handle that case here.
      mlir::Value expr = nullptr;
      if (ret.getExpr().has_value())
      {
        if (!(expr = mlirGen(**ret.getExpr())))
          return mlir::failure();

        expr = builder.create<mlir::tensor::CastOp>(location, mlir::UnrankedTensorType::get(builder.getF64Type()), expr);
      }

      /// Otherwise, this return operation has zero operands.
      builder.create<TAReturnOp>(location, expr ? ArrayRef(expr)
                                                : ArrayRef<mlir::Value>());
      return mlir::success();
    }

    mlir::Value mlirGen(LiteralExprAST &lit)
    {
      comet_debug() << " mlirGen for LiteralExprAST.\n";

      auto type = getType(lit.getDims());

      /// The attribute is a vector with a floating point value per element
      /// (number) in the array, see `collectData()` below for more details.
      std::vector<double> data;
      data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                   std::multiplies<int>()));
      collectData(lit, data);

      /// The type of this attribute is tensor of 64-bit floating-point with the
      /// shape of the literal.
      mlir::Type elementType = builder.getF64Type();
      auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

      /// This is the actual attribute that holds the list of values for this
      /// tensor literal.
      auto dataAttribute =
          mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

      /// Build the MLIR op `ta.constant`. This invokes the `DenseConstantOp::build`
      /// method.
      return builder.create<DenseConstantOp>(loc(lit.loc()), type, dataAttribute);
    }

    /// Recursive helper function to accumulate the data that compose an array
    /// literal. It flattens the nested structure in the supplied vector. For
    /// example with this array:
    /// [[1, 2], [3, 4]]
    /// we will generate:
    /// [ 1, 2, 3, 4 ]
    /// Individual numbers are represented as doubles.
    /// Attributes are the way MLIR attaches constant to operations.
    void collectData(ExprAST &expr, std::vector<double> &data)
    {
      if (auto *lit = dyn_cast<LiteralExprAST>(&expr))
      {
        for (auto &value : lit->getValues())
          collectData(*value, data);
        return;
      }

      assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
      data.push_back(cast<NumberExprAST>(expr).getValue());
    }

    /// Emit a call expression. It emits specific operations for the `sum`
    /// builtin. Other identifiers are assumed to be user-defined functions.
    mlir::Value mlirGen(CallExprAST &call)
    {
      comet_debug() << "CallExprAST\n";

      llvm::StringRef callee = call.getCallee();
      auto location = loc(call.loc());

      mlir::Value sumVal;
      if (callee == "SUM")
      {
        auto *expr = call.getArg(0);
        /// Check if it SUM(A[i,j]) or SUM(A[i,j] * B[j,k])
        /// Case 1: SUM(A[i,j])
        if (llvm::isa<LabeledTensorExprAST>(expr))
        {
          auto *rhsLT = llvm::cast<LabeledTensorExprAST>(expr);
          auto name = rhsLT->getTensorName();
          mlir::Value tensorValue = symbolTable.lookup(name);
          mlir::ShapedType shapedT = mlir::cast<mlir::ShapedType>(tensorValue.getType());
          comet_debug() << " generate ta.sum op\n";
          /// TODO(gkestor): look at reduceOp in linalg
          sumVal = builder.create<mlir::tensorAlgebra::ReduceOp>(location, shapedT.getElementType(), tensorValue);
        }

        /// Case 2: SUM(A[i,j]*B[j,k])
        if (llvm::isa<BinaryExprAST>(expr))
        {
          comet_debug() << " SUM parameter is a BinaryExprAST, Generate ta.SUM() \n";
          /// Generate ta.SUM
          /// parse binary
          std::set<std::string> out_lbls = {};
          mlir::Value tensorValue = mlirGen(*expr, out_lbls);
          comet_debug() << " generate ta.sum op\n";
          sumVal = builder.create<mlir::tensorAlgebra::ReduceOp>(location, builder.getF64Type(), tensorValue);
        }
      }
      else
      {
        std::vector<mlir::Value> expr_args;
        comet_debug() << "Generic Call\n";
        comet_debug() << "Num args: " << call.getNumArgs() << "\n";
        if (call.getNumArgs() > 0)
        {
          for (size_t i = 0; i < call.getNumArgs(); i++)
          {
            auto res = builder.create<mlir::tensor::CastOp>(location, mlir::UnrankedTensorType::get(builder.getF64Type()), mlirGen(*call.getArg(i)));
            expr_args.push_back(res);
          }
          comet_debug() << "Num args: " << call.getNumArgs() << "\n";
        }
        mlir::Value tensorValue;
        tensorValue = mlir::Value();
        ArrayRef<mlir::Value> args(expr_args);

        auto c = functionMap.lookup(callee);
        if (c.getFunctionType().getResults().size() > 0) /// Function that returns a value
        {
          auto res = builder.create<GenericCallOp>(location, c.getFunctionType().getResults()[0], callee, args);
          sumVal = res.getResults()[0];
        }
        else /// Void function
        {
          builder.create<GenericCallOp>(location, callee, args);
          sumVal = mlir::Value();
        }
      }

      /// Otherwise this is a call to a user-defined function. Calls to ser-defined
      /// functions are mapped to a custom call that takes the callee name as an
      /// attribute.
      return sumVal;
    }

    /// Emit a print expression. It emits specific operations for two builtins:
    /// transpose(x) and print(x).
    mlir::LogicalResult mlirGen(PrintExprAST &call)
    {
      auto arg = mlirGen(*call.getArg());
      if (!arg)
        return mlir::failure();

      builder.create<PrintOp>(loc(call.loc()), arg);
      return mlir::success();
    }

    /// Emit a constant for a single number (FIXME: semantic? broadcast?)
    mlir::Value mlirGen(NumberExprAST &num)
    {
      auto type = getType(1);
      std::vector<double> data;
      data.push_back(num.getValue());
      /// The type of this attribute is tensor of 64-bit floating-point with the
      /// shape of one.
      mlir::Type elementType = builder.getF64Type();
      auto dataType = mlir::RankedTensorType::get(1, elementType);

      /// This is the actual attribute that holds the list of values for this
      /// tensor to represent scalar op.
      auto dataAttribute =
          mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

      /// Build the MLIR op `ta.constant`. This invokes the `DenseConstantOp::build`
      /// method.
      return builder.create<DenseConstantOp>(loc(num.loc()), type, dataAttribute);
    }

    /// Emit a tensor expression
    /// D[a, d] = ....
    mlir::Value mlirGen(TensorOpExprAST &tensor_op)
    {
      comet_debug() << " mlirGen TensorOpExprAST\n";
      auto lhs = mlirGen(*tensor_op.getLHS());
      if (!lhs)
        return nullptr;
      comet_debug() << " get lhs\n";

      mlir::Value lhs_decl;
      std::string out_format;
      if (isa<DenseTensorDeclOp>(lhs.getDefiningOp()))
      {
        lhs_decl = lhs;
        out_format = "Dense";
      }
      else if (isa<SparseTensorDeclOp>(lhs.getDefiningOp()))
      {
        lhs_decl = lhs;
        auto lhs_decl_op = cast<SparseTensorDeclOp>(lhs_decl.getDefiningOp());
        std::string formatStr(lhs_decl_op.getFormatAttr().getValue());
        out_format = formatStr;
      }
      else
      {
        emitError(loc(tensor_op.loc()),
                  "error: Tensor Decl Not Found! '");
      }

      auto out_lbls_vec =
          cast<tensorAlgebra::LabeledTensorExprAST>(*tensor_op.getLHS())
              .getLabelNames();
      std::set<std::string> out_labels(out_lbls_vec.begin(), out_lbls_vec.end());

      comet_debug() << " out_labels: ";
      for (auto n : out_labels)
      {
        comet_debug() << n << " ";
      }
      comet_debug() << "\n";

      auto rhs = mlirGen(*tensor_op.getRHS(), out_labels, out_format);
      if (!rhs)
        return nullptr;
      comet_debug() << " get rhs\n";

      auto tens_beta = tensor_op.getBeta();

      auto ret_op = builder.create<TensorSetOp>(loc(tensor_op.loc()), rhs, lhs_decl);
      ret_op.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));

      return rhs;
    }

    /// Emit a tensor expression
    mlir::Value mlirGen(LabeledTensorExprAST &lbl_tensor)
    {
      comet_debug() << " mlirGen LabeledTensorExprAST \n";
      auto tensor_name = lbl_tensor.getTensorName();
      auto label_names = lbl_tensor.getLabelNames();

      mlir::Value tensor_op;
      std::vector<mlir::Value> labels;

      if ((tensor_op = symbolTable.lookup(tensor_name)) != NULL)
      {
        if (!isa<DenseTensorDeclOp>(tensor_op.getDefiningOp()) && !isa<SparseTensorDeclOp>(tensor_op.getDefiningOp()))
        {
          emitError(loc(lbl_tensor.loc()), "Tensor declaration required '")
              << tensor_name << "'";
        }
      }
      else
      {
        emitError(loc(lbl_tensor.loc()),
                  "Unknown variable LabeledTensorExprAST'");
      }

      return tensor_op;
    }

    /// Dispatch codegen for the right expression subclass using RTTI.
    mlir::Value mlirGen(ExprAST &expr,
                        const std::set<std::string> out_lbls = {}, std::string out_format = "")
    {
      comet_debug() << " mlirGen ExprAST " << expr.getKind() << " \n";
      switch (expr.getKind())
      {
      case tensorAlgebra::ExprAST::Expr_BinOp:
        return mlirGen(cast<BinaryExprAST>(expr), out_lbls, out_format);
      case tensorAlgebra::ExprAST::Expr_Var:
        return mlirGen(cast<VariableExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Literal:
        return mlirGen(cast<LiteralExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Call:
        return mlirGen(cast<CallExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Transpose:
        return mlirGen(cast<TransposeExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Num:
        return mlirGen(cast<NumberExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_LabeledTensor:
        comet_debug() << " Is Expr_LabeledTensor\n";
        return mlirGen(cast<LabeledTensorExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_Tensor:
        comet_debug() << " Is Expr_Tensor\n";
        return mlirGen(cast<TensorOpExprAST>(expr));
      case tensorAlgebra::ExprAST::Expr_GetTime:
        return mlirGen(cast<GetTimeExprAST>(expr));
      default:
        emitError(loc(expr.loc()))
            << "MLIR codegen encountered an unhandled expr kind '"
            << Twine(expr.getKind()) << "'";
        return nullptr;
      }
    }

    mlir::Value mlirGen(GetTimeExprAST &gettime)
    {
      return builder.create<GetTimeOp>(loc(gettime.loc()), builder.getF64Type());
    }

    /// Emit a print expression. It emits specific operations for two builtins:
    /// transpose(x) and print(x).
    mlir::LogicalResult mlirGen(PrintElapsedTimeExprAST &call)
    {
      auto start = mlirGen(*call.getStart());
      if (!start)
        return mlir::failure();

      auto end = mlirGen(*call.getEnd());
      if (!end)
        return mlir::failure();

      builder.create<PrintElapsedTimeOp>(loc(call.loc()), start, end);
      return mlir::success();
    }

    /// Handle a variable declaration, we'll codegen the expression that forms the
    /// initializer and record the value in the symbol table before returning it.
    /// Future expressions will be able to reference this variable through symbol
    /// table lookup.
    mlir::Value mlirGen(VarDeclExprAST &vardecl)
    {
      auto init = vardecl.getInitVal();
      if (!init)
      {
        emitError(loc(vardecl.loc()),
                  "missing initializer in variable declaration");
        return nullptr;
      }

      std::set<std::string> out_lbls = {};
      llvm::StringRef out_var = vardecl.getName();
      std::string out_varStr(out_var.str()); /// the info of variable on the LHS.
      mlir::Value value = mlirGen(*init, out_lbls, out_varStr);

      if (!value)
        return nullptr;

      /// Register the value in the symbol table.
      if (failed(declare(vardecl.getName(), value)))
        return nullptr;
      return value;
    }

    /// Handle index label declaration
    mlir::Value mlirGen(IndexLabelDeclExprAST &labeldecl)
    {
      mlir::Value value;
      declarelabel(labeldecl);
      value =
          builder.create<IndexLabelOp>(loc(labeldecl.loc()));
      if (failed(declare(labeldecl.getName(), value)))
        return nullptr;
      return value;
    }

    /// Handle tensor declaration
    mlir::Value mlirGen(TensorDeclExprAST &tensordecl)
    {

      auto dim_lbls = tensordecl.getDims();
      std::vector<int64_t> dims_sizes;
      std::vector<mlir::Value> labels;
      auto tensor_format = tensordecl.getFormat();
      for (const auto &lbl_str : dim_lbls)
      {
        if (lbl_str == "?")
        {
          dims_sizes.push_back(mlir::ShapedType::kDynamic);
        }
        else if (symbolTable.lookup(lbl_str))
        {
          dims_sizes.push_back(ilabel_symbolTable[lbl_str]);
          if (ilabel_symbolTable[lbl_str] == mlir::ShapedType::kDynamic)
          {
            if (labelToTensorDim.end() != labelToTensorDim.find(lbl_str))
            {
              labels.push_back(labelToTensorDim[lbl_str]);
            }
          }
        }
        else if (is_number(lbl_str))
        {
          dims_sizes.push_back(std::stoi(lbl_str));
        }
        else
        {
          emitError(loc(tensordecl.loc()), "Unknown variable TensorDeclExprAST '")
              << lbl_str << "'";
        }
      }

      auto vartype = tensordecl.getElementType();
      vartype.shape = dims_sizes;
      auto tensor_type = getTensorType(vartype);
      auto name = tensordecl.getName();

      mlir::Value value;
      std::string formats_str(tensor_format.data());
      if (isDense(formats_str, ", ") == false)
      {
        /// BoolAttr is false because there is explicit sparse densor declaration.
        /// SparseTensorDeclOp is not for temporaries in compound expression
        std::vector<TensorFormatEnum> format = mlir::tensorAlgebra::getFormats(tensor_format, dims_sizes.size(), builder.getContext());
        mlir::Type element_type;
        switch (vartype.elt_ty)
        {
          case VarType::TY_FLOAT:
            element_type =  builder.getF32Type();
            break;
          case VarType::TY_DOUBLE:
            element_type = builder.getF64Type();
            break;
          case VarType::TY_INT:
            element_type  = builder.getIntegerType(64);
            break;
        }
        auto sp_tensor_type = SparseTensorType::get(builder.getContext(), element_type, builder.getIntegerType(defaultSpTensorIndiceBitWidth), dims_sizes, format);
        value = builder.create<SparseTensorDeclOp>(loc(tensordecl.loc()),
                                                   sp_tensor_type, labels, tensor_format, false);
        comet_debug() << "MLIRGen SparseTensorDeclaration creation\n";
        comet_vdump(value);

        /// If a dynamic index label is first used in sparse tensor declaration, its size is determined by the size
        /// of the tensor's respective dimension size.
        for (size_t i = 0; i < tensordecl.getDims().size(); i++)
        {
          if (labelToTensorDim.count(tensordecl.getDims()[i]) == 0)
          {
            declarelabeltotensordim(tensordecl.getDims()[i], builder.create<TensorDimOp>(loc(tensordecl.loc()), value, i));
          }
        }
      }
      else
      {
        value = builder.create<DenseTensorDeclOp>(loc(tensordecl.loc()),
                                                  tensor_type, labels, tensor_format);
        comet_debug() << "MLIRGen DenseTensorDeclaration creation\n";
        comet_vdump(value);
      }

      if (failed(declare(name, value)))
        return nullptr;
      return value;
    }

    /// Handle B[j, i] = tranpose(A[i,j], {j, i}) in DSL
    mlir::Value mlirGen(TransposeExprAST &transpose, LabeledTensorExprAST &lhsLT)
    {
      comet_debug() << "TransposeExprAST \n";

      mlir::Value rhs_tensor = symbolTable.lookup(transpose.getName());

      comet_vdump(rhs_tensor);

      auto rhs_lbls = transpose.getSrcDims();
      auto lhs_lbls = transpose.getDstDims();

      std::vector<std::string> all_lbls = rhs_lbls;
      all_lbls.insert(all_lbls.end(), lhs_lbls.begin(), lhs_lbls.end());
      std::vector<mlir::Value> all_lbls_value;
      for (auto s : all_lbls)
      {
        all_lbls_value.push_back(symbolTable.lookup(s));
      }

      std::map<std::string, mlir::AffineExpr> expr_map;
      unsigned dim = 0;
      for (const auto &lbl : all_lbls)
      {
        if (expr_map.find(lbl) == expr_map.end())
        {
          expr_map[lbl] = getAffineDimExpr(dim++, builder.getContext());
        }
      }
      std::vector<mlir::AffineExpr> rhs_exprs;
      std::vector<mlir::AffineExpr> lhs_exprs;

      for (const auto &lbl : rhs_lbls)
      {
        rhs_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : lhs_lbls)
      {
        lhs_exprs.push_back(expr_map[lbl]);
      }

      auto context = builder.getContext();

      SmallVector<mlir::AffineMap, 8> affine_maps{
          mlir::AffineMap::get(dim, 0, rhs_exprs, context),
          mlir::AffineMap::get(dim, 0, lhs_exprs, context)};

      auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);

      SmallVector<mlir::StringRef, 8> formats;

      /// Firstly, Look at the rhs
      if (isa<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()))
      {
        comet_debug() << " is TensorDeclOp\n";
        /// infer the format
        auto rhs_format = dyn_cast<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()).getFormat();
        comet_debug() << " rhs_format: " << rhs_format << "\n";
        formats.push_back(rhs_format);
      }
      else if (isa<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()))
      {
        comet_debug() << " is TensorDeclOp\n";
        /// infer the format
        auto rhs_format = dyn_cast<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()).getFormat();
        comet_debug() << " rhs_format: " << rhs_format << "\n";
        formats.push_back(rhs_format);
      }
      else
      {
        comet_debug() << " not TensorDeclOp\n";
      }

      /// Secondly, Look at the lhs
      std::vector<LabeledTensorExprAST *> exprs{&lhsLT};
      std::vector<mlir::Value> lhs_lbls_value;

      for (auto e : exprs)
      {
        auto lhsLT_tensor_name = e->getTensorName();
        mlir::Value lhsLT_op;
        if ((lhsLT_op = symbolTable.lookup(lhsLT_tensor_name)) != NULL)
        {
          if (isa<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";
            /// infer the format
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()).getFormat();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);
          }
          else if (isa<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";
            /// infer the format
            auto lhs_format = dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()).getFormat();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);
          }
          else
          {
            comet_debug() << " not TensorDeclOp\n";
          }
        }
      }

      comet_debug() << " formats.size(): " << formats.size() << "\n";
      auto strAttr = builder.getStrArrayAttr(formats);

      auto lhs_tensor = symbolTable.lookup(lhsLT.getTensorName());

      comet_debug() << " create TransposeOp\n";
      mlir::Value t = builder.create<mlir::tensorAlgebra::TransposeOp>(loc(transpose.loc()), lhs_tensor.getType(),
                                                                       rhs_tensor, all_lbls_value, affineMapArrayAttr, strAttr);
      builder.create<TensorSetOp>(loc(transpose.loc()), t.getDefiningOp()->getResult(0), lhs_tensor);
      comet_vdump(t);

      return t;
    }

    /// Handle tranpose(A[i,j], {j, i}) in DSL, when no lhs_LabeledTensor has been created.
    mlir::Value mlirGen(TransposeExprAST &transpose)
    {
      comet_debug() << "TransposeExprAST with no lhs labeled tensor \n";

      mlir::Value rhs_tensor = symbolTable.lookup(transpose.getName());

      comet_vdump(rhs_tensor);

      auto rhs_lbls = transpose.getSrcDims();
      auto lhs_lbls = transpose.getDstDims();

      std::vector<std::string> all_lbls = rhs_lbls;
      all_lbls.insert(all_lbls.end(), lhs_lbls.begin(), lhs_lbls.end());

      std::map<std::string, mlir::AffineExpr> expr_map;
      unsigned dim = 0;
      for (const auto &lbl : all_lbls)
      {
        if (expr_map.find(lbl) == expr_map.end())
        {
          expr_map[lbl] = getAffineDimExpr(dim++, builder.getContext());
        }
      }
      std::vector<mlir::AffineExpr> rhs_exprs;
      std::vector<mlir::AffineExpr> lhs_exprs;

      for (const auto &lbl : rhs_lbls)
      {
        rhs_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : lhs_lbls)
      {
        lhs_exprs.push_back(expr_map[lbl]);
      }

      auto context = builder.getContext();

      SmallVector<mlir::AffineMap, 8> affine_maps{
          mlir::AffineMap::get(dim, 0, rhs_exprs, context),
          mlir::AffineMap::get(dim, 0, lhs_exprs, context)};

      auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);

      SmallVector<mlir::StringRef, 8> formats;

      /// Firstly, Look at the rhs
      if (isa<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()))
      {
        comet_debug() << " is TensorDeclOp\n";
        /// infer the format
        auto rhs_format = dyn_cast<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()).getFormat();
        comet_debug() << " rhs_format: " << rhs_format << "\n";
        formats.push_back(rhs_format);
      }
      else if (isa<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()))
      {
        comet_debug() << " is TensorDeclOp\n";
        /// infer the format
        auto rhs_format = dyn_cast<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()).getFormat();
        comet_debug() << " rhs_format: " << rhs_format << "\n";
        formats.push_back(rhs_format);
      }
      else
      {
        comet_debug() << " not TensorDeclOp\n";
      }

      /// Secondly, Look at the lhs
      /// Collect labels values
      std::vector<mlir::Value> lhs_labels_val;
      std::vector<mlir::Value> all_labels_val;
      for (const auto &lbl_str : rhs_lbls)
      {
        if (auto var = symbolTable.lookup(lbl_str))
        {
          if (isa<IndexLabelOp>(var.getDefiningOp()))
          {
            all_labels_val.push_back(var);
          }
          else
          {
            emitError(loc(transpose.loc()), "Index label variable required '")
                << lbl_str << "'";
          }
        }
        else
        {
          emitError(loc(transpose.loc()),
                    " Unknown variable TransposeExprAST' ")
              << lbl_str << "'";
        }
      }
      for (const auto &lbl_str : lhs_lbls)
      {
        if (auto var = symbolTable.lookup(lbl_str))
        {
          if (isa<IndexLabelOp>(var.getDefiningOp()))
          {
            all_labels_val.push_back(var);
            lhs_labels_val.push_back(var);
          }
          else
          {
            emitError(loc(transpose.loc()), "Index label variable required '")
                << lbl_str << "'";
          }
        }
        else
        {
          emitError(loc(transpose.loc()),
                    " Unknown variable TransposeExprAST' ")
              << lbl_str << "'";
        }
      }

      /// get return-type based on affine-maps
      auto res_map = affineMapArrayAttr[1].cast<mlir::AffineMapAttr>().getValue();
      std::vector<mlir::Value> indices;
      std::vector<int64_t> shape;
      for (auto v : res_map.getResults())
      {
        auto map = affineMapArrayAttr[0].cast<mlir::AffineMapAttr>().getValue();
        if (auto pos = map.getResultPosition(v))
        {
          if (isa<mlir::TensorType>(rhs_tensor.getType()) && !rhs_tensor.getType().cast<mlir::TensorType>().isDynamicDim(*pos))
          {
            shape.push_back(rhs_tensor.getType().cast<mlir::TensorType>().getDimSize(*pos));
          }
          else
          {
            indices.push_back(builder.create<TensorDimOp>(loc(transpose.loc()), rhs_tensor, *pos));
            shape.push_back(mlir::ShapedType::kDynamic);
          }
        }
      }

      mlir::Type return_type = getType(shape);

      /// Create Tensor Declarations Ops and populate formats (for lhs)
      mlir::Value lhs_tensor;
      if (isa<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()))
      {
        /// for DenseTensorDeclOp create
        mlir::StringRef format_strref = dyn_cast<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()).getFormat();
        mlir::StringAttr formatAttr = builder.getStringAttr(format_strref);
        mlir::ShapedType shapedT = mlir::cast<mlir::ShapedType>(rhs_tensor.getType());

        lhs_tensor = builder.create<DenseTensorDeclOp>(loc(transpose.loc()), mlir::RankedTensorType::get(shape, shapedT.getElementType()), indices, formatAttr);

        /// populate formats
        /// assumes lhs and rhs formats are same
        auto lhs_format = dyn_cast<DenseTensorDeclOp>(rhs_tensor.getDefiningOp()).getFormat();
        formats.push_back(lhs_format);
      }
      else if (isa<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()))
      {

        /// for SparseTensorDeclOp create
        mlir::StringRef format_strref = dyn_cast<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()).getFormat();
        mlir::StringAttr formatAttr = builder.getStringAttr(format_strref);

        std::vector<TensorFormatEnum> format = mlir::tensorAlgebra::getFormats(format_strref, shape.size(), builder.getContext());
        mlir::ShapedType shapedT = mlir::cast<mlir::ShapedType>(rhs_tensor.getType());
        mlir::Type element_type = shapedT.getElementType();
        return_type = SparseTensorType::get(builder.getContext(), element_type, builder.getIntegerType(defaultSpTensorIndiceBitWidth), shape, format);
        /// no lhs_LabeledTensor has been created. The output tensor of tranpose doesn't have explicit declaration,
        /// BoolAttr is true to speficy SparseTensorDeclOp is for temporaries
        auto sp_tensor_type = SparseTensorType::get(builder.getContext(), element_type, builder.getIntegerType(defaultSpTensorIndiceBitWidth), shape, format);

        lhs_tensor = builder.create<SparseTensorDeclOp>(loc(transpose.loc()), sp_tensor_type, indices, formatAttr, builder.getBoolAttr(true));
        comet_debug() << "MLIRGen SparseTensorDeclaration creation\n";
        comet_vdump(lhs_tensor);

        /// populate formats
        /// assumes lhs and rhs formats are same
        auto lhs_format = dyn_cast<SparseTensorDeclOp>(rhs_tensor.getDefiningOp()).getFormat();
        formats.push_back(lhs_format);
      }

      comet_debug() << " formats.size(): " << formats.size() << "\n";
      auto strAttr = builder.getStrArrayAttr(formats);

      comet_debug() << " create TransposeOp\n";
      mlir::ShapedType shapedT = mlir::cast<mlir::ShapedType>(rhs_tensor.getType());
      mlir::Value t = builder.create<mlir::tensorAlgebra::TransposeOp>(loc(transpose.loc()), mlir::RankedTensorType::get(shape, shapedT.getElementType()),
                                                                       rhs_tensor, all_labels_val, affineMapArrayAttr, strAttr);
      builder.create<TensorSetOp>(loc(transpose.loc()), t.getDefiningOp()->getResult(0), lhs_tensor);
      comet_vdump(t);

      return t;
    }

    /// Codegen for-loop
    mlir::LogicalResult mlirGen(ForLoopExprAST &forLoop)
    {
      comet_debug() << "codegen: ForLoopExprAST \n";

      mlir::Value lo = builder.create<ConstantIndexOp>(
          loc(forLoop.loc()), forLoop.getBegin());
      mlir::Value hi = builder.create<ConstantIndexOp>(loc(forLoop.loc()),
                                                       forLoop.getEnd());
      mlir::Value step = builder.create<ConstantIndexOp>(
          loc(forLoop.loc()), forLoop.getIncrement());

      builder.create<ForLoopBeginOp>(loc(forLoop.loc()), lo, hi, step, forLoop.getName());

      comet_debug() << "codegen: ForLoopExprAST done \n";
      return mlir::success();
    }

    /// Codegen for-loop
    mlir::LogicalResult mlirGen(ForLoopEndExprAST &forLoopEnd)
    {
      comet_debug() << "codegen: ForLoopEndExprAST \n";

      builder.create<ForLoopEndOp>(loc(forLoopEnd.loc()));

      comet_debug() << "codegen: ForLoopEndExprAST done \n";
      return mlir::success();
    }

    /// Codegen a list of expression, return failure if one of them hit an error.
    mlir::LogicalResult mlirGen(ExprASTList &blockAST)
    {
      ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);
      for (auto &expr : blockAST)
      {
        /// Specific handling for variable declarations, return statement, and
        /// print. These can only appear in block list and not in nested
        /// expressions.

        if (auto *labeldecl = dyn_cast<IndexLabelDeclExprAST>(expr.get()))
        {
          if (!mlirGen(*labeldecl))
            return mlir::failure();
          continue;
        }
        if (auto *labeldecl = dyn_cast<IndexLabelDeclDynamicExprAST>(expr.get()))
        {
          if (!mlirGen(*labeldecl))
            return mlir::failure();
          continue;
        }

        if (auto *tensordecl = dyn_cast<TensorDeclExprAST>(expr.get()))
        {
          if (!mlirGen(*tensordecl))
            return mlir::failure();
          continue;
        }

        if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get()))
        {
          if (!mlirGen(*vardecl))
            return mlir::failure();
          continue;
        }
        if (auto *transpose = dyn_cast<TransposeExprAST>(expr.get()))
        {
          if (!mlirGen(*transpose))
            return mlir::failure();
          continue;
        }
        if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
          return mlirGen(*ret);
        if (auto *print = dyn_cast<PrintExprAST>(expr.get()))
        {
          if (mlir::failed(mlirGen(*print)))
            return mlir::success();
          continue;
        }

        if (auto *printElapsedTime = dyn_cast<PrintElapsedTimeExprAST>(expr.get()))
        {
          if (mlir::failed(mlirGen(*printElapsedTime)))
            return mlir::success();
          continue;
        }

        if (auto *forLoopStart = dyn_cast<ForLoopExprAST>(expr.get()))
        {
          if (mlir::failed(mlirGen(*forLoopStart)))
            return mlir::success();
          continue;
        }

        if (auto *forLoopEnd = dyn_cast<ForLoopEndExprAST>(expr.get()))
        {
          if (mlir::failed(mlirGen(*forLoopEnd)))
            return mlir::success();
          continue;
        }

        if (auto *tensor_op = dyn_cast<TensorOpExprAST>(expr.get()))
        {
          comet_debug() << " generate ops for TensorOpExprAST\n";
          comet_debug() << " Right hand side " << tensor_op->getRHS()->getKind() << "\n";
          comet_debug() << " Left hand side " << tensor_op->getLHS()->getKind() << "\n";
          /// A[i,j] = ...
          if (tensor_op->getLHS()->getKind() ==
              ExprAST::ExprASTKind::Expr_LabeledTensor)
          {
            comet_debug() << " in TensorOpExprAST, lhs is labeledTensor\n";

            /// A[i,j] = B[i,k] */+/- C[k,j]
            if (tensor_op->getRHS()->getKind() ==
                    ExprAST::ExprASTKind::Expr_BinOp &&
                llvm::cast<BinaryExprAST>(tensor_op->getRHS())
                        ->getLHS()
                        ->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor &&
                llvm::cast<BinaryExprAST>(tensor_op->getRHS())
                        ->getRHS()
                        ->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is BinaryExprAST and its lhs and rhs are Expr_LabeledTensor\n";

              if (mlir::failed(mlirGenTensorOperations(*tensor_op)))
                return mlir::success();
              continue;
            }
            /// A[i,j] = 1.0
            else if (tensor_op->getRHS()->getKind() ==
                     ExprAST::ExprASTKind::Expr_Num)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is Expr_Num\n";

              auto tensor_name =
                  llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS())
                      ->getTensorName();
              auto value =
                  llvm::cast<NumberExprAST>(tensor_op->getRHS())->getValue();

              if (mlir::failed(mlirGenTensorFill(loc(tensor_op->loc()), tensor_name, value)))
                return mlir::success();
              continue;
            }

            /// A[i,j] = comet_read()
            else if (tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_FileRead)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is Expr_FileRead\n";
              auto tensor_name =
                  llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS())
                      ->getTensorName();
              auto call = llvm::cast<FileReadExprAST>(tensor_op->getRHS());
              llvm::StringRef callee = call->getCallee();

              int readModeVal = 1; /// DEFAULT, standard matrix read

              /// Builting calls have their custom operation, meaning this is a
              /// straightforward emission.
              if (callee == "comet_read")
              {
                comet_debug() << " call comet_read \n";

                ExprAST *filename = call->getFileID();
                ExprAST *readMode = call->getReadMode();
                comet_debug() << "\n";

                std::string filenamestring;
                llvm::StringRef filenamestr;
                if (filename == nullptr) /// no argument provided
                {
                  comet_debug() << __LINE__ << " Empty filename\n";
                  filenamestring = "SPARSE_FILE_NAME";
                  filenamestr = filenamestring;
                }
                else if (filename != nullptr && readMode == nullptr) /// only 1 arg provided
                {                                                    /// Not empty filename
                  comet_debug() << __LINE__ << " One argument was provided in comet_read().\n";

                  /// User will provide num arg in comet_read()
                  /// that will be used to read file based on unique env vars.
                  /// e.g., comet_read(0) --> SPARSE_FILE_NAME0
                  if (filename->getKind() == NumberExprAST::Expr_Num)
                  {
                    auto *filenameast = llvm::cast<NumberExprAST>(filename);

                    /// get arg val
                    int val = (int)cast<NumberExprAST>(filenameast)->getValue();
                    filenamestring = "SPARSE_FILE_NAME" + std::to_string(val);
                    filenamestr = filenamestring;

                    comet_debug() << " " << filenamestr << "\n";
                  }
                  else if (filename->getKind() == ExprAST::ExprASTKind::Expr_Var)
                  {
                    auto *filenameast = llvm::cast<VariableExprAST>(filename);

                    filenamestr = filenameast->getName();
                    comet_debug() << " " << filenamestr << "\n";
                  }
                  else
                  {
                    llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: unrecognized args provided to comet_read!\n";
                  }
                }
                else /// 2 args provided to comet_read
                {
                  comet_debug() << " Two arguments were provided in comet_read().\n";

                  /// check 1st arg
                  if (filename->getKind() == NumberExprAST::Expr_Num)
                  {
                    auto *filenameast = llvm::cast<NumberExprAST>(filename);

                    /// get arg val
                    int val = (int)cast<NumberExprAST>(filenameast)->getValue();
                    filenamestring = "SPARSE_FILE_NAME" + std::to_string(val);
                    filenamestr = filenamestring;

                    comet_debug() << " " << filenamestr << "\n";
                  }
                  else
                  {
                    llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: un-recognized args provided to comet_read!\n";
                  }

                  /// check 2nd arg
                  if (readMode->getKind() == NumberExprAST::Expr_Num)
                  {
                    auto *readModeAST = llvm::cast<NumberExprAST>(readMode);
                    /// get arg val
                    readModeVal = (int)cast<NumberExprAST>(readModeAST)->getValue();
                  }
                  else
                  {
                    llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: un-recognized args provided to comet_read!\n";
                  }
                }

                if (mlir::failed(mlirGenTensorFillFromFile(loc(tensor_op->loc()), tensor_name, filenamestr, readModeVal)))
                  return mlir::success();
              }
              /// TODO: put check here, if the user mis-spells something...

              continue;
            }

            /// A[i,j] = random()
            else if (tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_Call)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is Expr_Call\n";
              auto tensor_name =
                  llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS())
                      ->getTensorName();
              auto call = llvm::cast<CallExprAST>(tensor_op->getRHS());
              llvm::StringRef callee = call->getCallee();

              if (callee == "random")
              {
                comet_debug() << " call random \n";

                std::random_device os_seed;
                srand(static_cast<unsigned>(os_seed())); /// seed the random-num generator
                                                         /// something like time(0) does not work!

                if (mlir::failed(mlirGenTensorFillRandom(loc(tensor_op->loc()), tensor_name)))
                  return mlir::success();
              }
              else
              {
                LabeledTensorExprAST *lhsLabeledTensorExprAST = llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS());
                auto call_res = mlirGen(*call);
                auto lhs_tensor = symbolTable.lookup(lhsLabeledTensorExprAST->getTensorName());
                builder.create<TensorSetOp>(loc(tensor_op->loc()), call_res, lhs_tensor);
              }
              /// TODO: put check here, if the user mis-spells something...

              continue;
            }

            /// A[i,j] = transpose(B[j,i], {i,j})
            else if (tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_Transpose)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is Expr_Transpose\n";
              /// create transpose op
              LabeledTensorExprAST *lhsLabeledTensorExprAST = llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS());

              TransposeExprAST *transpose = llvm::cast<TransposeExprAST>(tensor_op->getRHS());
              mlirGen(*transpose, *lhsLabeledTensorExprAST);
              continue;
            }
            else if (tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_Call)
            {
              comet_debug() << __LINE__ << "  in TensorOpExprAST, rhs is Expr_Call\n";

              LabeledTensorExprAST *lhsLabeledTensorExprAST = llvm::cast<LabeledTensorExprAST>(tensor_op->getLHS());
              CallExprAST *call = llvm::cast<CallExprAST>(tensor_op->getRHS());
              auto call_res = mlirGen(*call);
              auto lhs_tensor = symbolTable.lookup(lhsLabeledTensorExprAST->getTensorName());
              builder.create<TensorSetOp>(loc(tensor_op->loc()), call_res, lhs_tensor);
              continue;
            }
            /// TODO(gkestor): evaluate use of Expr_LabeledTensor for slicing
            else if ((tensor_op->getRHS()->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor &&
                      tensor_op->getLHS()->getKind() == ExprAST::ExprASTKind::Expr_LabeledTensor))
            {

              if (mlir::failed(mlirGenTensorarithexprs(*tensor_op)))
                return mlir::success();
              continue;
            }
            /// there is a fall through case at the end, so don't put an else here.
          }
          else
          {
            comet_debug() << " in TensorOpExprAST, lhs is NOT labeledTensor\n";
          }
        }

        /// Generic expression dispatch codegen.
        comet_debug() << " expr->getKind(): " << expr->getKind() << "\n";

        /// If calling a void function this will return null, thus we cannot count on this for
        /// error checking
        mlirGen(*expr);
      }
      return mlir::success();
    }

    /// Build a tensor type from a list of shape dimensions.
    mlir::Type getType(ArrayRef<int64_t> shape)
    {
      /// If the shape is empty, then this type is unranked.
      if (shape.empty())
        return mlir::UnrankedTensorType::get(builder.getF64Type());

      /// Otherwise, we use the given shape.
      return mlir::RankedTensorType::get(shape, builder.getF64Type());
    }

    /// Build an MLIR type from a Tensor Algebra AST variable type (forward to the
    /// generic getType above).
    mlir::Type getType(const VarType &type, const Location &location)
    {
      return getType(type.shape);
    }

    mlir::Type getType(const VarType &type) { return getType(type.shape); }
    /// Build an MLIR type from a Tensor Algebra AST variable type with respect to
    /// the element type
    mlir::Type getTensorType(const VarType &type)
    {
      if (type.shape.empty())
        return mlir::UnrankedTensorType::get(builder.getF64Type());

      mlir::Type retType;
      switch (type.elt_ty)
      {
      case VarType::TY_FLOAT:
        retType = mlir::RankedTensorType::get(type.shape, builder.getF32Type());
        break;
      case VarType::TY_DOUBLE:
        retType = mlir::RankedTensorType::get(type.shape, builder.getF64Type());
        break;
      case VarType::TY_INT:
        retType = mlir::RankedTensorType::get(type.shape,
                                              builder.getIntegerType(64));
        break;
      }

      return retType;
    }

    ElType convertType(mlir::Type type)
    {

      if (type.isF32())
        return ElType::fp32;
      else if (type.isF64())
        return ElType::fp64;
      else if (type.isInteger(64))
        return ElType::i64;
      else
        return ElType::inv;
    }

    mlir::Type getBinOpResultType(mlir::Type lhs, mlir::Type rhs)
    {
      ElType type = lub(convertType(lhs), convertType(rhs));

      switch (type)
      {
      case ElType::i64:
        return builder.getIntegerType(64);
      case ElType::fp32:
        return builder.getF32Type();
      case ElType::fp64:
        return builder.getF64Type();
      case ElType::inv:
        return builder.getF64Type();
      default:
        return builder.getF64Type();
        break;
      }
    }

    mlir::LogicalResult mlirGenTensorarithexprs(TensorOpExprAST &tensor_op)
    {

      LabeledTensorExprAST *lhsLT =
          llvm::cast<LabeledTensorExprAST>(tensor_op.getLHS());
      LabeledTensorExprAST *rhsLT =
          llvm::cast<LabeledTensorExprAST>(tensor_op.getRHS());

      std::vector<mlir::Value> tensors;
      auto binop = tensor_op.getOp();
      SmallVector<mlir::StringRef, 8> formats;
      std::vector<LabeledTensorExprAST *> exprs{rhsLT, lhsLT};
      for (auto e : exprs)
      {
        auto lhsLT_tensor_name = e->getTensorName();
        mlir::Value lhsLT_op;
        if ((lhsLT_op = symbolTable.lookup(lhsLT_tensor_name)) != NULL)
        {
          if (isa<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            /// infer the format
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()).getFormat();
            formats.push_back(lhs_format);
            tensors.push_back(dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()));
          }
          else if (isa<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            /// infer the format
            auto lhs_format = dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()).getFormat();
            formats.push_back(lhs_format);
            tensors.push_back(dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()));
          }
          else
          {
            comet_debug() << " not TensorDeclOp\n";
          }
        }
      }

      std::vector<mlir::Operation *> lhsLabelOps, rhsLabelOps;
      if (binop == TensorOpKind::Tensor_Red_Add)
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Unsupported tensor elementwise addition\n";
        /// TODO(gkestor): look at tensor elementwise addition
        /// auto SemiringAttr = builder.getStringAttr("eltwise_add"); /// this is for standard elementwise addition
        /// auto op = builder.create<TensorAddOp>(loc(tensor_op.loc()), mlir::UnrankedTensorType::get(builder.getF64Type()),
        ///                                       tensors[1], tensors[0], builder.getStrArrayAttr(formats), );
        /// builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[1]);
      }

      else if (binop == TensorOpKind::Tensor_Red_Sub)
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Unsupported tensor elementwise substraction\n";
        /// TODO(gkestor): look at tensor elementwise subtraction
        /// auto op = builder.create<TensorSubtractOp>(loc(tensor_op.loc()), mlir::UnrankedTensorType::get(builder.getF64Type()),
        ///                                             tensors[1], tensors[0], builder.getStrArrayAttr(formats));
        /// builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[1]);
      }

      return mlir::success();
    }

    /// handle case: Expr_LabeledTensor = Expr_LabeledTensor */+/- Expr_LabeledTensor
    mlir::LogicalResult mlirGenTensorOperations(TensorOpExprAST &tensor_op)
    {
      assert(tensor_op.getLHS()->getKind() ==
             ExprAST::ExprASTKind::Expr_LabeledTensor);
      assert(tensor_op.getRHS()->getKind() == ExprAST::ExprASTKind::Expr_BinOp);

      LabeledTensorExprAST *lhsLT =
          llvm::cast<LabeledTensorExprAST>(tensor_op.getLHS());
      BinaryExprAST *rhsBinOp = llvm::cast<BinaryExprAST>(tensor_op.getRHS());
      auto tens_beta = tensor_op.getBeta();
      if (rhsBinOp->getLHS()->getKind() !=
              ExprAST::ExprASTKind::Expr_LabeledTensor ||
          rhsBinOp->getRHS()->getKind() !=
              ExprAST::ExprASTKind::Expr_LabeledTensor)
      {
        emitError(loc(tensor_op.loc()), "error: the tensor operation allows only "
                                        "two labeled tensor operands '");
      }

      LabeledTensorExprAST *rhs1LT =
          llvm::cast<LabeledTensorExprAST>(rhsBinOp->getLHS());
      LabeledTensorExprAST *rhs2LT =
          llvm::cast<LabeledTensorExprAST>(rhsBinOp->getRHS());
      auto binop = rhsBinOp->getOp();
      comet_debug() << " binop: " << binop << "\n";

      int SemiringOp1st = int(rhsBinOp->getSemiringOp1());
      int SemiringOp2nd = int(rhsBinOp->getSemiringOp2());
      comet_debug() << " semirop: 1) " << SemiringOp1st << " 2) " << SemiringOp2nd << "\n";
      std::string SemiringOp1str = getSemiringOpName(SemiringOp1st);
      comet_debug() << " SemiringOp1str " << SemiringOp1str << "\n";
      std::string SemiringOp2str = getSemiringOpName(SemiringOp2nd);
      comet_debug() << " SemiringOp2str " << SemiringOp2str << "\n";
      std::string SemiringOperators = SemiringOp1str + "_" + SemiringOp2str;
      auto SemiringAttr = builder.getStringAttr(SemiringOperators);

      /// masking support: determine type (push/pull/auto/none) and variable (DenseTensorDeclOp or SparseTensorDeclOp)
      /// NOTE: mask is optional and not required, so we will populate with default values where necessary.
      MaskExprAST *mask = nullptr;
      std::string MaskingName;
      std::string MaskingVar_name;
      mlir::Value maskVal; /// this may not be found in symbol table.
                           /// if not found, mask will not be included as an operand.

      if (tensor_op.getMask() != nullptr)
      {

        mask = llvm::cast<MaskExprAST>(tensor_op.getMask());
        MaskingName = mask->getMaskType();
        MaskingVar_name = mask->getTensorName();

        mlir::Value maskLT_op;
        /// find the variable name in symbol table
        if ((maskLT_op = symbolTable.lookup(MaskingVar_name)) != NULL)
        {
          comet_debug() << "Masking variable found!\n";
          if (isa<DenseTensorDeclOp>(maskLT_op.getDefiningOp()))
          {
            comet_debug() << " is DenseTensorDeclOp\n";
            maskVal = dyn_cast<DenseTensorDeclOp>(maskLT_op.getDefiningOp());
          }
          else if (isa<SparseTensorDeclOp>(maskLT_op.getDefiningOp()))
          {
            comet_debug() << " is SparseTensorDeclOp\n";
            maskVal = dyn_cast<SparseTensorDeclOp>(maskLT_op.getDefiningOp());
          }
          else
          {
            comet_debug() << " not TensorDeclOp\n";
          }
        }
        comet_debug() << "masking name: " << MaskingName << "\n";
        comet_debug() << "masking var: " << MaskingVar_name << "\n";
        comet_vdump(maskVal);
      }
      else
      {
        comet_debug() << "No mask input provided by user!\n";
        MaskingName = "none";
      }
      auto MaskingAttr = builder.getStringAttr(MaskingName);

      auto lhs_lbls = lhsLT->getLabelNames();
      auto rhs1_lbls = rhs1LT->getLabelNames();
      auto rhs2_lbls = rhs2LT->getLabelNames();
      auto all_lbls = rhsBinOp->getLabels();

      std::vector<mlir::Value> lhs_lbls_value;
      std::vector<mlir::Value> all_lbls_value;
      for (auto n : lhs_lbls)
      {
        lhs_lbls_value.push_back(symbolTable.lookup(n));
      }
      for (auto n : rhs1_lbls)
      {
        all_lbls_value.push_back(symbolTable.lookup(n));
      }
      for (auto n : rhs2_lbls)
      {
        all_lbls_value.push_back(symbolTable.lookup(n));
      }
      for (auto n : lhs_lbls)
      {
        all_lbls_value.push_back(symbolTable.lookup(n));
      }

      std::map<std::string, mlir::AffineExpr> expr_map;
      unsigned dim = 0;
      for (const auto &lbl : all_lbls)
      {
        expr_map[lbl] = getAffineDimExpr(dim++, builder.getContext());
      }
      std::vector<mlir::AffineExpr> rhs1_exprs;
      std::vector<mlir::AffineExpr> rhs2_exprs;
      std::vector<mlir::AffineExpr> lhs_exprs;

      for (const auto &lbl : rhs1_lbls)
      {
        rhs1_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : rhs2_lbls)
      {
        rhs2_exprs.push_back(expr_map[lbl]);
      }

      for (const auto &lbl : lhs_lbls)
      {
        lhs_exprs.push_back(expr_map[lbl]);
      }

      auto context = builder.getContext();
      SmallVector<mlir::AffineMap, 8> affine_maps{
          mlir::AffineMap::get(dim, 0, rhs1_exprs, context),
          mlir::AffineMap::get(dim, 0, rhs2_exprs, context),
          mlir::AffineMap::get(dim, 0, lhs_exprs, context)};

      auto affineMapArrayAttr = builder.getAffineMapArrayAttr(affine_maps);

      SmallVector<mlir::StringRef, 8> formats;
      std::vector<LabeledTensorExprAST *> exprs{rhs1LT, rhs2LT, lhsLT};
      std::vector<mlir::Value> tensors;
      for (auto e : exprs)
      {
        auto lhsLT_tensor_name = e->getTensorName();
        mlir::Value lhsLT_op;
        if ((lhsLT_op = symbolTable.lookup(lhsLT_tensor_name)) != NULL)
        {
          if (isa<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";

            /// infer the format
            auto lhs_format = dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()).getFormat();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<DenseTensorDeclOp>(lhsLT_op.getDefiningOp()));
          }
          else if (isa<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()))
          {
            comet_debug() << " is TensorDeclOp\n";

            /// infer the format
            auto lhs_format = dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()).getFormat();
            comet_debug() << " lhs_format: " << lhs_format << "\n";
            formats.push_back(lhs_format);

            tensors.push_back(dyn_cast<SparseTensorDeclOp>(lhsLT_op.getDefiningOp()));
          }
          else
          {
            comet_debug() << " not TensorDeclOp\n";
          }
        }
      }
      comet_debug() << " formats.size(): " << formats.size() << "\n";
      auto strAttr = builder.getStrArrayAttr(formats);

      assert(tensors.size() == 3 && "Not 3 tensors for ta.tc or ta.elews_mul\n");

      if (binop != '*' && binop != '+' && binop != '-' &&
          binop != tok_elews &&
          binop != tok_semiring && binop != tok_monoid)
      {
        emitError(loc(tensor_op.loc()),
                  "error: only tensor multiplication and elementwise multiplication operations are supported "
                  "for tensor ops'");
      }

      if (binop == '+')
      {
        comet_debug() << "creating TensorAddOp\n";
        auto op = builder.create<TensorAddOp>(loc(tensor_op.loc()),
                                              tensors[2].getType(),
                                              tensors[0], tensors[1],
                                              all_lbls_value,
                                              affineMapArrayAttr,
                                              strAttr, SemiringAttr,
                                              MaskingAttr);

        comet_vdump(op);
        /// source is 1st parameter, dest is the second
        builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[2]);
      }
      else if (binop == '-')
      {
        comet_debug() << "creating TensorSubtractOp\n";
        auto op = builder.create<TensorSubtractOp>(loc(tensor_op.loc()),
                                                   tensors[2].getType(),
                                                   tensors[0], tensors[1],
                                                   all_lbls_value,
                                                   affineMapArrayAttr,
                                                   strAttr, SemiringAttr,
                                                   MaskingAttr);
        comet_vdump(op);
        /// source is 1st parameter, dest is the second
        builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[2]);
      }
      else if (binop == '*' || binop == tok_semiring)
      {
        auto op = builder.create<TensorMultOp>(loc(tensor_op.loc()),
                                               tensors[2].getType(),
                                               tensors[0], tensors[1],
                                               all_lbls_value,
                                               affineMapArrayAttr,
                                               strAttr, SemiringAttr,
                                               MaskingAttr, maskVal);
        op.getOperation()->setAttr("__alpha__", builder.getF64FloatAttr(1.0));
        op.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));

        /// source is 1st parameter, dest is the second
        auto setop = builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[2]);
        setop.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));
      }
      else if (binop == tok_elews || binop == tok_monoid)
      {
        auto op = builder.create<TensorElewsMultOp>(loc(tensor_op.loc()), tensors[2].getType(), tensors[0], tensors[1], all_lbls_value, affineMapArrayAttr, strAttr, SemiringAttr, MaskingAttr);
        op.getOperation()->setAttr("__alpha__", builder.getF64FloatAttr(1.0));
        op.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));

        auto setop = builder.create<TensorSetOp>(loc(tensor_op.loc()), op.getOperation()->getResult(0), tensors[2]);
        setop.getOperation()->setAttr("__beta__", builder.getF64FloatAttr(tens_beta));
      }
      else
      {
        emitError(loc(tensor_op.loc()), "error: the tensor operation allows only "
                                        "specific kind of binary operators. '");
      }

      return mlir::success();
    }

    mlir::LogicalResult mlirGenTensorFill(mlir::Location loc,
                                          StringRef tensor_name, double value)
    {
      mlir::Value tensorValue = symbolTable.lookup(tensor_name);
      auto tensorType = tensorValue.getDefiningOp()->getOpResult(0).getType();
      auto tensorElType = tensorType.cast<mlir::TensorType>().getElementType();

      mlir::FloatAttr valueAttr;
      if (tensorElType.isF64())
      {
        valueAttr = builder.getF64FloatAttr(value);
      }
      else if (tensorElType.isF32())
      {
        valueAttr = builder.getF32FloatAttr(value);
      }
      else
      {
        emitError(loc,
                  "error: only float and double tensors are supported "
                  "for tensor ops'");
      }

      builder.create<TensorFillOp>(loc, tensorValue, valueAttr);

      return mlir::success();
    }

    mlir::LogicalResult mlirGenTensorFillRandom(mlir::Location loc,
                                                StringRef tensor_name)
    {
      comet_debug() << " in mlirGenTensorFillRandom\n";

      mlir::Value tensorValue = symbolTable.lookup(tensor_name);
      auto lhs_labeledtensor = tensorValue.getDefiningOp()->getOpResult(0);
      comet_debug() << "\n";
      comet_vdump(lhs_labeledtensor);
      std::vector<int64_t> result_dims;

      std::vector<mlir::Value> lhs_lbls_value;
      if (isa<DenseTensorDeclOp>(lhs_labeledtensor.getDefiningOp()))
      {
        mlir::TensorType tensor = cast<mlir::TensorType>(lhs_labeledtensor.getType());
        result_dims = tensor.getShape();
      }
      else
      {
        if (isa<SparseTensorDeclOp>(lhs_labeledtensor.getDefiningOp()))
          llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: random initialization is currently not supported for sparse tensors.\n";

        llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: Not supported format encountered during random initialization of tensor.\n";
      }

      comet_debug() << " dims size: " << result_dims.size() << "\n";

      /// The attribute is a vector with a floating point value per element
      /// (number) in the array
      std::vector<double> data;
      int dataArraySize = std::accumulate(result_dims.begin(), result_dims.end(), 1,
                                          std::multiplies<int>());

      /// fill data array with random numbers
      double upperLimit = 10.0;
      for (int i = 0; i < dataArraySize; i++)
      {
        double randNum = static_cast<double>(rand()) / (static_cast<float>(RAND_MAX / upperLimit));
        data.push_back(randNum);
      }

       auto lhs_labeledtensor_dataType = mlir::RankedTensorType::get(1, lhs_labeledtensor.getType());
      /// This is the actual attribute that holds the list of values for this
      /// tensor literal.
      auto dataAttribute =
          mlir::DenseElementsAttr::get(lhs_labeledtensor_dataType, llvm::ArrayRef(data));

      /// Build the MLIR op `ta.constant`. This invokes the `DenseConstantOp::build`
      /// method.
      auto denseConst = builder.create<DenseConstantOp>(loc, lhs_labeledtensor.getType(), dataAttribute);
      builder.create<TensorSetOp>(loc, denseConst, lhs_labeledtensor);

      return mlir::success();
    }

    mlir::LogicalResult mlirGenTensorFillFromFile(mlir::Location loc,
                                                  StringRef tensor_name, StringRef filename,
                                                  int readMode)
    {
      mlir::Value tensorValue = symbolTable.lookup(tensor_name);
      if (tensorValue == nullptr)
      {
        /// the variable was not declared by user.
        llvm::errs() << __FILE__ << ":" << __LINE__ << " ERROR: please check your variable definitions!";
      }
      mlir::StringAttr filenameAttr = builder.getStringAttr(filename);
      mlir::IntegerAttr readModeAttr = builder.getI32IntegerAttr(readMode);
      builder.create<TensorFillFromFileOp>(loc, tensorValue, filenameAttr, readModeAttr);

      return mlir::success();
    }
  };

} /// namespace

namespace tensorAlgebra
{

  /// The public API for codegen.
  mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                            ModuleAST &moduleAST)
  {
    return MLIRGenImpl(context).mlirGen(moduleAST);
  }

} /// namespace tensorAlgebra