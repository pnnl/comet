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

#ifndef INDEXTREE_TYPES
#define INDEXTREE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "comet/Dialect/IndexTree/IR/IndexTreeBase.td"

class IndexTree_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IndexTreeDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IndexTree_TreeType : IndexTree_Type<"IndexTree", "index_tree"> {
  let summary = "Operand for compute expression";
  let description = [{}];
}

def IndexTree_IndexNodeType : IndexTree_Type<"IndexNode", "index"> {
  let summary = "Iteration Tree for Index Variables";
  let description = [{
    Type for storing iteration tree of index variables.
  }];
}

def IndexTree_NodeType : AnyTypeOf<[IndexTree_TreeType, IndexTree_IndexNodeType]>;

def IndexTree_TensorAccessType : IndexTree_Type<"TensorAccess", "tensor_access"> {
  let summary = "Tensor access variabls";
  let description = [{
    Type for storing tensor access of index variables.
  }];
}

def IndexTree_OperandType : IndexTree_Type<"Operand", "operand"> {
  let summary = "Operand for compute expression";
  let description = [{}];
}

def IndexTree_SymbolicDomainType : IndexTree_Type<"SymbolicDomain", "symbolic_domain"> {
  let summary = "Type representing a computed iteration/tensor domain";
  let description = [{}];
  let parameters = (
        ins "unsigned":$indices_bitwidth
    );

    let assemblyFormat = "`<`$indices_bitwidth`>`";
}

def IndexTree_DomainType : IndexTree_Type<"Domain", "domain"> {
  let summary = "Type representing an iteration domain";
  let description = [{
    Type for storing iteration domain of an index variable.
  }];
}

def IndexTree_AnyDomainType : AnyTypeOf<[IndexTree_SymbolicDomainType, IndexTree_DomainType]>;

def IndexTree_ProvenanceGraphType : IndexTree_Type<"ProvenanceGraph", "prov_graph"> {
    let summary = "Provenance graph for index tree transformations";
    let description = [{
        Type for storing provenance graph associated with a index.
    }];
}

#endif // INDEXTREE_TYPES
