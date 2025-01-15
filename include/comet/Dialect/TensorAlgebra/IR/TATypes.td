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

#ifndef TA_TYPES
#define TA_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "comet/Dialect/TensorAlgebra/IR/TABase.td"

class TensorAlgebra_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TA_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IndexLabel : TensorAlgebra_Type<"IndexLabel","index"> {
    let summary = "Type representing the index of one dimnsion of a tensor";
    let description = [{}];
}

def Range : TensorAlgebra_Type<"Range","range"> {
    let summary = "Type representing the range of one dimnsion of a tensor";
    let description = [{}];
}

def SparseTensor : TensorAlgebra_Type<"SparseTensor", "sparse_tensor", [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
    let summary = "Sparse tensor to use in tensor algebra dialect";
    let description = [{}];

    let parameters = (ins
        "::mlir::Type":$element_type,
        "::mlir::IntegerType":$indices_type,
        ArrayRefParameter<"int64_t", "Dimensions of tensor">:$dims,
        ArrayRefParameter<"TensorFormatEnum", "Format">:$format
    );
    
    let hasCustomAssemblyFormat = 1;
    // TODO: Implement custom builder from "common" format strings into format strings
}

def WorkspaceTensor : TensorAlgebra_Type<"Workspace", "workspace", [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
    let summary = "Temporary tensor generated from a workspace transfrom";
    let description = [{
        Dense, temporary tensor generated from a workspace transformation.
        Needed to represent dense row as well as mark array.
    }];
    let parameters = (
        ins "::mlir::Type":$element_type,
        "::mlir::IntegerType":$indices_type,
        ArrayRefParameter<"int64_t", "Dimensions of workspace">:$dims
    );
    let assemblyFormat = "`<` $element_type `,` $indices_type `,` $dims `>`";
}

/// Provide a definition of the types that are used within the TA dialect.
def TA_AnyTensor : AnyTypeOf<[TensorOf<[AnyType]>, SparseTensor, WorkspaceTensor]>;
def TA_FloatTensor: TensorOf<[F32, F64]>;
#endif //TA_TYPES