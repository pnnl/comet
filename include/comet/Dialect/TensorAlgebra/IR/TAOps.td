//===- TAOps.td - TA dialect operation definitions ----------*- tablegen -*-===//
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
// Defines the operations of the TA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TA_OPS
#define TA_OPS


include "mlir/IR/OpBase.td" 
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Provide a definition of the 'TA' dialect in the ODS framework so that we
/// can define our operations.
def TA_Dialect : Dialect {
  let name = "ta";
  let cppNamespace = "::mlir::tensorAlgebra"; 

  /// We set this bit to generate the declarations for the dialect's type parsing
  /// and printing hooks.
  let useDefaultTypePrinterParser = 1;
  
}

/// An implementation of IndexLabelType.
def TA_IndexLabelType :
    DialectType<TA_Dialect,
                CPred<"$_self.isa<IndexLabelType>()">,
                "IndexLabelType">;


/// Whether a type is a IndexLabelType.
def TAIsIndexLabelTypePred : CPred<"$_self.isa<IndexLabelType>()">;
def IndexLabel : Type<TAIsIndexLabelTypePred, "indexlabel">;

/// Base class for ta dialect operations. This operation inherits from the base
/// `Op` class in OpBase.td, and provides:
///   * The parent dialect of the operation.
///   * The mnemonic for the operation, or the name without the dialect prefix.
///   * A list of traits for the operation.
class TA_Op<string mnemonic, list<Trait> traits = []> :
    Op<TA_Dialect, mnemonic, traits>;

/// Provide a definition for the TA SparseTensorType for use in ODS. 
/// This allows for using SparseTensorType in a similar way to Tensor or MemRef.
def SparseTensor :
    Type<CPred<"$_self.isa<SparseTensorType>()">, "TA sparse tensor type">;

/// Provide a definition of the types that are used within the TA dialect.
def TA_AnyTensor : AnyTypeOf<[TensorOf<[AnyType]>, SparseTensor]>;


//===----------------------------------------------------------------------===//
/// Tensor Algebra Operations
//===----------------------------------------------------------------------===//
def IndexLabelOp :
    TA_Op<"index_label", [Pure]>,
    Results<(outs IndexLabel)> {
  let summary = "Create an index label type value, used to create views";


  let arguments = (ins );
  let results = (outs AnyType:$output);
  let builders = [OpBuilder<(ins ), 
    [{
      auto indexLabelType = IndexLabelType::get($_builder.getContext());
      build($_builder, $_state, indexLabelType);
    }]>];
  

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
/// Tensor Algebra Operations
//===----------------------------------------------------------------------===//

def DenseTensorDeclOp : TA_Op<"dense_tensor_decl", [Pure]> {
  /// Provide a summary and description for this operation. This can be used to
  /// auto-generate documenatation of the operations within our dialect.
  let summary = "Create an output tensor declaration";
  let description = [{
    Tensor declaration operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.dense_tensor_decl"()
         { labels = vector[] : tensor<*x*xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  /// The constant operation takes an attribute as the only input.
  let arguments = (ins Variadic<Index>:$labels, StrAttr:$format);

  /// The constant operation returns a single value of TensorType.
  let results = (outs AnyTensor);

  /// Invoke a static verify method to verify this constant operation.
  
  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

def SparseTensorDeclOp : TA_Op<"spTensor_decl", [Pure]> {
  /// Provide a summary and description for this operation. This can be used to
  /// auto-generate documenatation of the operations within our dialect.
  let summary = "Create a tensor declaration";
  let description = [{
    Tensor declaration operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.spTensor_decl"()
         { labels = vector[] : tensor<*x*xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  /// The constant operation takes an attribute as the only input.
  let arguments = (ins Variadic<Index>:$labels, StrAttr:$format, BoolAttr:$temporal_tensor);

  /// The constant operation returns a single value of TensorType.
  let results = (outs TA_AnyTensor);

  let extraClassDeclaration = [{
    unsigned int getParameterCount() {
        mlir::TensorType type = getResult().getType().cast<TensorType>();
        return (type.getRank() * 6) + 1;
    }

    unsigned int getDimArrayCount() {
        mlir::TensorType type = getResult().getType().cast<TensorType>();
        return type.getRank() * 4;
    }

    unsigned int getValueArrayPos() {
        mlir::TensorType type = getResult().getType().cast<TensorType>();

        return (type.getRank() * 4) + 1;
    }

    unsigned int getTotalArrayCount() {
        mlir::TensorType type = getResult().getType().cast<TensorType>();
        return (type.getRank() * 4) + 1;
    }
  }];

  /// Invoke a static verify method to verify this constant operation.
  
  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

def SparseOutputTensorDeclOp : TA_Op<"sparse_output_tensor_decl", [Pure]> {
  /// Provide a summary and description for this operation. This can be used to
  /// auto-generate documenatation of the operations within our dialect.
  let summary = "Create a sparse output tensor declaration";
  let description = [{
    Tensor declaration operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.sparse_output_tensor_decl"()
         { labels = vector[] : tensor<*x*xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  /// The constant operation takes an attribute as the only input.
  let arguments = (ins Variadic<Index>:$labels, StrAttr:$format);

  /// The constant operation returns a single value of TensorType.
  let results = (outs AnyTensor);

  /// Invoke a static verify method to verify this constant operation.
  
  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

//TempSparseOutputTensorDeclOp operation is similar to SparseOutputTensorDeclOp.
//TempSparseOutputTensorDeclOp should be lowered before SparseOutputTensorDeclOp in the compilation pipeline to produce  
//necessary information to lower SparseOutputTensorDeclOp properly
def TempSparseOutputTensorDeclOp : TA_Op<"temp_sparse_output_tensor_decl", [Pure]> {
  /// Provide a summary and description for this operation. This can be used to
  /// auto-generate documenatation of the operations within our dialect.
  let summary = "Create a sparse output tensor declaration for temporaries generated by parsing compound operations";
  let description = [{
    Tensor declaration operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.temp_sparse_output_tensor_decl"()
         { labels = vector[] : tensor<*x*xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  /// The constant operation takes an attribute as the only input.
  let arguments = (ins Variadic<Index>:$labels, StrAttr:$format);

  /// The constant operation returns a single value of TensorType.
  let results = (outs AnyTensor);

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

/// We define a ta operation by inherting from our base 'TA_Op' class above.
/// Here we provide the mnemonic and a list of traits for the operation. The
/// constant operation is marked as 'Pure' as it is a pure operation
/// and may be removed if dead.
def DenseConstantOp : TA_Op<"constant", [Pure]> {
  /// Provide a summary and description for this operation. This can be used to
  /// auto-generate documenatation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "ta.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  /// The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  /// The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  /// Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), 
    [{
      build($_builder, $_state, value.getType(), value);
    }]>, 

    /// Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>];

  /// Invoke a static verify method to verify this constant operation.
  let hasVerifier = 1;
}

def SparseTensorConstructOp : TA_Op<"spTensor_construct", [Pure]>{

  let summary = "";
  let description = [{
  }];

  //The size of arguments depends on the rank of sparse tensors
  //For example, for 2D sparse matrix, spTensor_construct consists of 12 elements:
  //A1pos, A1crd (first dimension) - each dimension consists of pos and crd arrays 
  //A2pos, A2crd (second dimension)
  //Aval (value array, nonzero elements)
  //A1pos_size, A1crd_size, (size of each pos and crd arrays)
  //A2pos_size, A2crd_size, 
  //Aval_size (size of value array) 
  //dim1_size, dim2_size (size of each dimension in sparse tensor) 
  //TODO(gkestor): might be better to have a struct with all the data elements
  let arguments = (ins Variadic<AnyType>:$indices, I32Attr:$tensor_rank);
  let results = (outs TA_AnyTensor:$output);

  let assemblyFormat = [{
    `(` $indices `)` attr-dict `:` `(` type($indices) `)` `->` `(` type($output) `)`
  }];

  // TODO(pflynn) This may need to be adjusted
  // This works for the moment; the idea is that each rank has a block
  /// dimension per dimension.
  let extraClassDeclaration = [{
    int getTotalParamCount() {
      return (getTensorRank() * 6) + 1;
    }

    int getDimArrayCount() {
      return getTensorRank() * 4;
    }

    unsigned int getTotalDimArrayCount() {
      return (getTensorRank() * 4) + 1;
    }

    int getValueArrayPos() {
      /// Value position- technically +1, but since we start at 0,
      /// we do not need to add anything
      return getTensorRank() * 4;
    }

    int getIndexValueSize() {
      return (getTensorRank() * 8) + 1;
    }
  }];

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
  
}

def TensorFillOp : TA_Op<"fill", [Pure]>{
  
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$lhs, AnyAttr:$value);
  
  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
  
}

def TensorMultOp : TA_Op<"mul", [Pure, AttrSizedOperandSegments]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$rhs1, TA_AnyTensor:$rhs2, Variadic<IndexLabel>:$index_labels, 
                   AffineMapArrayAttr:$indexing_maps, StrArrayAttr:$formats, StrAttr:$semiring, 
                   OptionalAttr<StrAttr>:$MaskType,  /// TODO: try to use: DefaultValuedAttr
                   Optional<AnyType>:$mask);

  /// Return value
  let results = (outs TA_AnyTensor);

  let extraClassDeclaration = [{
    std::vector<mlir::Value> getRhs1IndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = 0;

      mlir::TensorType type = getRhs1().getType().cast<TensorType>();
      size_t end = type.getRank();

      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getRhs2IndexLabels()
    {
      std::vector<mlir::Value> labels;
      
      mlir::TensorType type1 = getRhs1().getType().cast<TensorType>();
      mlir::TensorType type2 = getRhs2().getType().cast<TensorType>();

      size_t start = type1.getRank();
      
      size_t end = static_cast<size_t>(type1.getRank() + type2.getRank());

      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getResultIndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t end = getIndexLabels().size();
      
      mlir::TensorType type1 = getRhs1().getType().cast<TensorType>();
      mlir::TensorType type2 = getRhs2().getType().cast<TensorType>();

      size_t start = type1.getRank() + type2.getRank();

      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;
    }

  }];
  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
  
}

def TensorElewsMultOp : TA_Op<"elews_mul", [Pure]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$rhs1, 
                   TA_AnyTensor:$rhs2, 
                   Variadic<IndexLabel>:$index_labels, 
                   AffineMapArrayAttr:$indexing_maps, 
                   StrArrayAttr:$formats, 
                   StrAttr:$semiring,
                   OptionalAttr<StrAttr>:$MaskType); 
  
  let results = (outs TA_AnyTensor);

  let extraClassDeclaration = [{

    std::vector<mlir::Value> getRhs1IndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = 0;
      mlir::TensorType type = getRhs1().getType().cast<TensorType>();
      size_t end = type.getRank();

      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getRhs2IndexLabels()
    {
      std::vector<mlir::Value> labels;
      mlir::TensorType type1 = getRhs1().getType().cast<TensorType>();
      mlir::TensorType type2 = getRhs2().getType().cast<TensorType>();
      size_t start = type1.getRank();
      
      size_t end = static_cast<size_t>(type1.getRank() + type2.getRank());
      
      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getResultIndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t end = getIndexLabels().size();
      
      mlir::TensorType type1 = getRhs1().getType().cast<TensorType>();
      mlir::TensorType type2 = getRhs2().getType().cast<TensorType>();
      size_t start = type1.getRank() + type2.getRank();
      
      
      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;
    }

  }];

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
  
}

def TensorDimOp : TA_Op<"dim", [Pure]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins  TA_AnyTensor:$tensor,
                        Index:$index);
  
  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$index)>
  ];

  let results = (outs Index);
  //let hasCanonicalizer = 1; 

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}


def TensorSetOp : TA_Op<"set_op", [Pure]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins  AnyTypeOf<[TA_AnyTensor,F64MemRef]>:$lhs,
                        AnyTypeOf<[TA_AnyTensor,F64MemRef]>:$rhs);
  
  //let hasCanonicalizer = 1; 

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

def TransposeOp : TA_Op<"transpose",[Pure]> {
  let summary = "transpose operation";

  let arguments = (ins TA_AnyTensor:$rhs, Variadic<IndexLabel>:$index_labels, AffineMapArrayAttr:$indexing_maps, StrArrayAttr:$formats);
  
  let results = (outs TA_AnyTensor); 



  let extraClassDeclaration = [{
    std::vector<mlir::Value> getRhsIndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = 0;
      size_t end = static_cast<size_t>(getRhs().getType().cast<TensorType>().getRank());
      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getResultIndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = static_cast<size_t>(getRhs().getType().cast<TensorType>().getRank());
      size_t end = getIndexLabels().size();

      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;
    }

  }];
  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

def ReduceOp : TA_Op<"reduce",[Pure]> {
  let summary = "reduction operation";

  let arguments = (ins TA_AnyTensor:$rhs); 

  let results = (outs F64:$lhs);

  let builders = [OpBuilder<
    (ins "Value":$input)>];

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

def TensorAddOp : TA_Op<"add",
    [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins TA_AnyTensor:$rhs1, 
                   TA_AnyTensor:$rhs2, 
                   Variadic<IndexLabel>:$index_labels, 
                   AffineMapArrayAttr:$indexing_maps, 
                   StrArrayAttr:$formats, 
                   StrAttr:$semiring,
                   OptionalAttr<StrAttr>:$MaskType); 
  
  let results = (outs TA_AnyTensor);

  let extraClassDeclaration = [{
    std::vector<mlir::Value> getRhs1IndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = 0;
      size_t end = static_cast<size_t>(getRhs1().getType().cast<TensorType>().getRank());
      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getRhs2IndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = static_cast<size_t>(getRhs1().getType().cast<TensorType>().getRank());
      size_t end = static_cast<size_t>(getRhs1().getType().cast<TensorType>().getRank() + getRhs2().getType().cast<TensorType>().getRank());
      
      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getResultIndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = static_cast<size_t>(getRhs1().getType().cast<TensorType>().getRank() + getRhs2().getType().cast<TensorType>().getRank());
      size_t end = getIndexLabels().size();

      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;
    }

  }];
}

def TensorSubtractOp : TA_Op<"subtract",
    [Pure]> {
  let summary = "element-wise subtract operation";
  let description = [{
    The "subtract" operation performs element-wise subtract between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins TA_AnyTensor:$rhs1, 
                   TA_AnyTensor:$rhs2, 
                   Variadic<IndexLabel>:$index_labels, 
                   AffineMapArrayAttr:$indexing_maps, 
                   StrArrayAttr:$formats, 
                   StrAttr:$semiring,
                   OptionalAttr<StrAttr>:$MaskType); 
  
  let results = (outs TA_AnyTensor);

  let extraClassDeclaration = [{
    std::vector<mlir::Value> getRhs1IndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = 0;
      size_t end = static_cast<size_t>(getRhs1().getType().cast<TensorType>().getRank());
      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getRhs2IndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = static_cast<size_t>(getRhs1().getType().cast<TensorType>().getRank());
      size_t end = static_cast<size_t>(getRhs1().getType().cast<TensorType>().getRank() + getRhs2().getType().cast<TensorType>().getRank());
      
      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;

    }

    std::vector<mlir::Value> getResultIndexLabels()
    {
      std::vector<mlir::Value> labels;
      size_t start = static_cast<size_t>(getRhs1().getType().cast<TensorType>().getRank() + getRhs2().getType().cast<TensorType>().getRank());
      size_t end = getIndexLabels().size();

      for(size_t i = start; i < end; i++ )
      {
        labels.push_back(getIndexLabels()[i]);
      }

      return labels;
    }

  }];
}

//TODO(gkestor): support other datatypes
def ScalarOp : TA_Op<"scalar",[Pure]> {
  let summary = "scalar arithmetic operation";

  let arguments = (ins AnyTypeOf<[AnyTensor,F64]>:$rhs, AnyTypeOf<[AnyTensor,F64]>:$lhs, StrAttr:$op); 

  let results = (outs AnyTypeOf<[AnyTensor,
                                 F64]>:$res);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  //TODO(gkestor): add verifier
  ///let hasVerifier = 1;
}

//TODO(gkestor): test this operator
def DivOp : TA_Op<"div",
    [Pure]> {
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, Variadic<IndexLabel>:$sum_labels);

  let results = (outs AnyTensor);

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>,
    OpBuilder<(ins "TensorType":$resultType, "Value":$lhs, "Value":$rhs)>,
  ];
}

def TAReturnOp : TA_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      ta.func @foo() -> tensor<2xf64> {
        ...
        ta.return %0 : tensor<2xf64>
      }
    ```
  }];

  /// The return operation takes an optional input operand to return. This
  /// value must match the return type of the enclosing function.
  let arguments = (ins Variadic<TA_AnyTensor>:$input);

  /// The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  /// Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  /// Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  /// Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
/// Tensor Algebra Operations for programming principles.
/// This operations might be moved to a new dialect
//===----------------------------------------------------------------------===//

def ForLoopBeginOp : TA_Op<"begin_for_loop">
{
  let summary = "identifies the start of a for-loop body";
  let description = [{
    The "begin_for_loop" identifies the start of a for-loop body
  }];

  /// the inputs are 1) index start, 2) index end, 3) index step,
  /// and 4) string-attr with name of the iterator (this info to be used).
  let arguments = (ins Index:$min, Index:$max, Index:$step, StrAttr:$iterator);
}

def ForLoopEndOp : TA_Op<"end_for_loop"> {
  let summary = "identifies the end of a for-loop body";
  let description = [{
    The "end_for_loop" identifies the end of a for-loop body
  }];
}

//===----------------------------------------------------------------------===//
/// Tensor Algebra Operations for utility functions
//===----------------------------------------------------------------------===//
def PrintOp : TA_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  /// The print operation takes an input tensor to print.
  /// We can extend the list of supported datatype for print with F64Tensor, I8MemRef, I64MemRef, F32MemRef, etc.
  let arguments = (ins AnyTypeOf<[F64,
                                  F64MemRef, 
                                  TA_AnyTensor]>:$input);
}

def GetTimeOp : TA_Op<"getTime"> {
  let summary = "getTime operation";
  let description = [{
    The "getTime" builtin operation for getting the clock time
  }];

  let results = (outs F64);
}

def PrintElapsedTimeOp : TA_Op<"print_elapsed_time"> {
  let summary = "printElapsedTime operation";
  let description = [{
    The "print_elapsed_time" builtin operation prints the time elapsed for the start
    and end times
  }];

  /// The print elapsed time gets two clock times as an input to compute and print
  /// elapsed time
  let arguments = (ins F64:$start, F64:$end);
}

//===----------------------------------------------------------------------===//
/// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : TA_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "ta.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
    ta.func @main() {
      ta.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    /// FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def GenericCallOp : TA_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = ta.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  /// The generic call operation takes a symbol reference attribute as the
  /// callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<TA_AnyTensor>:$inputs);

  /// The generic call operation returns a single value of TensorType or
  /// StructType.
  let results = (outs Optional<TA_AnyTensor>);

  /// Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  /// Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

def TensorFillFromFileOp : TA_Op<"fill_from_file", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$lhs, AnyAttr:$filename, AnyAttr:$readMode);
  
  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
  
}

def TensorCopyOp : TA_Op<"copy", [Pure]>{
  
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, AffineMapAttr:$inputPerm, AffineMapAttr:$outputPerm);

  //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

#endif /// TA_OPS




