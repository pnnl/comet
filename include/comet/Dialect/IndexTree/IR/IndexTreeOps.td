//===- IndexTreeOps.td - IT dialect operation definitions ----------*- tablegen -*-===//
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
// Defines the operations of the IndexTree dialect.
//
//===----------------------------------------------------------------------===//

#ifndef INDEXTREE_OPS
#define INDEXTREE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

include "comet/Dialect/IndexTree/IR/IndexTreeTypes.td"
include "comet/Dialect/TensorAlgebra/IR/TATypes.td"

// Base class for ta dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class IndexTree_Op<string mnemonic, list<Trait> traits = []> :
    Op<IndexTreeDialect, mnemonic, traits>;
            
//===----------------------------------------------------------------------===//
// Index Tree Operations
//===----------------------------------------------------------------------===//

def IndexTreeOp : IndexTree_Op<"itree", 
    [SingleBlockImplicitTerminator<"indexTree::YieldOp">]> {
  let summary = "Create a scope for index tree iteration";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
}

def YieldOp : IndexTree_Op<"yield", [Pure, ReturnLike, Terminator,
    HasParent<"IndexTreeOp">]> {
  let summary = "index tree yield and termination operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

def IndexTreeRootOp : IndexTree_Op<"RootOp", [
    HasParent<"IndexTreeOp">, Pure]> {
  let summary = "Create the base of the iteration tree";
  let description = [{}];

  let results = (outs IndexTree_TreeType:$output);
}

def IndexTreeIndicesOp : IndexTree_Op<"IndexOp", [Pure]>{
  let summary = "Create an index variable bound to a specific computation.";
  let description = [{
  }];

  let arguments = (ins IndexTree_NodeType:$parent, Optional<IndexTree_DomainType>:$domain, DefaultValuedAttr<BoolAttr, "false">:$IsParallel);

  let results = (outs IndexTree_IndexNodeType:$output);
}
def IndexTreeLHSOperandOp : IndexTree_Op<"LHSOperandOp", [Pure, SameVariadicOperandSize]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins 
    TA_AnyTensor:$tensor, 
    Variadic<Index>:$pos,
    Variadic<Index>:$crds);
  let results = (outs IndexTree_OperandType:$result);
}

def IndexTreeOperandOp : IndexTree_Op<"OperandOp", [Pure, SameVariadicOperandSize]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins 
    TA_AnyTensor:$tensor, 
    Variadic<Index>:$pos,
    Variadic<Index>:$crds);
  let results = (outs IndexTree_OperandType:$result);
}

def IndexTreeComputeOp : IndexTree_Op<"ComputeOp", [Pure, AttrSizedOperandSegments]>{
  let summary = "";
  let description = [{
  }];

  //TODO(gkestor): rethink the use of comp_worksp_opt, should we decouple that?
  //TODO(alokvk2): Rename semiring to operation to account for elementwise addition
  /// MaskType attribute: {push, pull, auto, none}
  let arguments = (ins 
    IndexTree_NodeType:$parent, 
    IndexTree_OperandType:$lhs, 
    Variadic<IndexTree_OperandType>:$rhs,
    Optional<IndexTree_OperandType>:$mask, 
    StrAttr:$semiring, 
    DefaultValuedAttr<BoolAttr, "false">:$compute_missing
  );

  let results = (outs TA_AnyTensor);

   //TODO(gkestor): add verifier
  //let hasVerifier = 1;
}

def IndexTreeTensorDomainOp : IndexTree_Op<"DomainOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins TA_AnyTensor:$tensor, UI32Attr:$dim, I32Attr:$format, Optional<AnyTypeOf<[Index, IndexTree_NodeType]>>:$parent);

  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeEmptyDomainOp : IndexTree_Op<"EmptyDomain", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let results = (outs IndexTree_DomainType:$domain);
}

def ConcreteDomainInterface : OpInterface<"ConcreteDomain"> {
  let description = [{
    Describes an operation which implements a concrete domain. That is
    the domain is fully-dscribed by it's arguments and does not rely on the
    the tensor definition to create the domain.
  }];
  let cppNamespace = "::mlir::indexTree";

  let methods = [
    InterfaceMethod<[{
      Method to get the dimension size of a concrete dimension.
    }],
      "mlir::Value", "getDimensionSize", (ins), /*methodBody=*/[{
        return $_op.getDimSize();
    }]>,
  ];
}

def IndexTreeSparseDomainOp : IndexTree_Op<"SparseDomainOp", 
    [Pure, ConcreteDomainInterface]>{

  let summary = "";
  let description = [{
  }];

  let arguments = (ins
      TA_AnyTensor:$tensor,
      UI32Attr:$dim, 
      I32Attr:$format, 
      AnyTensor:$pos,
      AnyTensor:$crd,
      Index:$pos_size,
      Index:$crd_size,
      Index:$dim_size,
      Optional<AnyTypeOf<[Index, IndexTree_TensorAccessType]>>:$parent);

  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeDenseDomainOp : IndexTree_Op<"DenseDomainOp", 
    [Pure, SameVariadicOperandSize, ConcreteDomainInterface]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins AnyTypeOf<[UI32,Index]>:$dim_size, Variadic<TA_AnyTensor>:$tensors, I32ArrayAttr:$dims);

  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeWorkspaceDomainOp : IndexTree_Op<"WorkspaceDomainOp", 
    [Pure, ConcreteDomainInterface]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins 
    WorkspaceTensor:$tensor,
    Index:$dim_size,
    UI32Attr:$dim, 
    Optional<AnyTypeOf<[Index, IndexTree_TensorAccessType]>>:$parent
  );
  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeDomainUnionOp : IndexTree_Op<"DomainUnionOp", 
    [Pure, UnknownDomain, ConcreteDomainInterface, AttrSizedOperandSegments]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins Variadic<IndexTree_DomainType>:$domains, Optional<Index>:$dim_size);
  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeDomainIntersectionOp : IndexTree_Op<"DomainIntersectionOp",
    [Pure, UnknownDomain, ConcreteDomainInterface, AttrSizedOperandSegments]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins Variadic<IndexTree_DomainType>:$domains, Optional<Index>:$dim_size);
  let results = (outs IndexTree_DomainType:$domain);
}

def IndexTreeNestedDomainOp : IndexTree_Op<"NestedDomainOp",
    [Pure, UnknownDomain, ConcreteDomainInterface]>{
  let summary = "";
  let description = [{
  }];
  
  let arguments = (ins Variadic<IndexTree_DomainType>:$domains, Index:$dim_size);
  let results = (outs IndexTree_DomainType:$domain);
}

def DomainGetSize : IndexTree_Op<"DomainGetSize", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins IndexTree_DomainType:$domain);
  let results = (outs Index:$result);
}

def IndexTreeIndexToTensorOp : IndexTree_Op<"IndexToTensorDim", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins 
    TA_AnyTensor:$tensor, 
    IndexTree_IndexNodeType:$index,
    UI32Attr:$dim,
    Optional<Index>:$prev_dim
  );

  let results = (outs
    Index:$crd,
    Index:$pos
  );
}

def DeclDomainOp : IndexTree_Op<"DeclDomainOp", [Pure]>{
  let summary = "";
  let description = [{
  }];

  let arguments = (ins Index:$dim_size, Index:$num_rows, OptionalAttr<BoolAttr>:$is_dynamic, OptionalAttr<I32Attr>:$indices_bitwidth);
  let results = (outs IndexTree_SymbolicDomainType);
}

def ComputeSymbolicDomainOp : IndexTree_Op<"ComputeSymbolicDomainOp", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins IndexTree_NodeType:$parent, IndexTree_SymbolicDomainType:$domain, DefaultValuedAttr<BoolAttr, "false">:$is_unique);
  let results = (outs IndexTree_SymbolicDomainType);
}

def ComputeSymbolicDomainRowOp : IndexTree_Op<"ComputeSymbolicDomainRowOp", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins IndexTree_NodeType:$parent, IndexTree_SymbolicDomainType:$domain, DefaultValuedAttr<BoolAttr, "true">:$needs_mark);
  let results = (outs IndexTree_SymbolicDomainType);
}

def SymbolicDomainInsertOp : IndexTree_Op<"SymbolicDomainInsertOp", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins IndexTree_SymbolicDomainType:$domain, Index:$crd, DefaultValuedAttr<BoolAttr, "false">:$is_unique);
  let results = (outs IndexTree_SymbolicDomainType);
}

def SymbolicDomainEndRowOp : IndexTree_Op<"SymbolicDomainEndRowOp", [Pure]>{
  let summary = "";
  let description = [{}];

  let arguments = (ins IndexTree_SymbolicDomainType:$domain, DefaultValuedAttr<BoolAttr, "true">:$needs_mark);
  let results = (outs IndexTree_SymbolicDomainType);
}

def IndexTreeSparseTensorOp : IndexTree_Op<"IndexTreeSparseTensorOp", [Pure]>{
  let summary = "Declare a sparse tensor from an index tree domain";
  let description = [{}];

  let arguments = (ins Variadic<IndexTree_AnyDomainType>:$domains);
  let results = (outs TA_AnyTensor);
}

def IndexTreeCleanWorkspaceOp : IndexTree_Op<"WorkspaceStartRowOp", [Pure]>{
  let summary = "Associate the loop of this index variable with a clean workspace";
  let description = [{}];

  let arguments = (ins IndexTree_NodeType:$parent, WorkspaceTensor:$workspace);
  let results = (outs WorkspaceTensor:$result);
}
#endif // INDEXTREE_OPS