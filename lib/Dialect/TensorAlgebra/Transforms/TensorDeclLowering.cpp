//===- TensorDeclLowering.cpp -- Lower dense and sparse tensor declarations for input and output tensors----===//
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
//
// This file implements a lowering of sparse and dense tensor declarations
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"

#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;

using namespace mlir::tensorAlgebra;
using namespace mlir::indexTree;

#define DEBUG_TYPE "tensor-decl-lowering"

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

//===----------------------------------------------------------------------===//
/// Lowering Passes for sparse/dense tensor declarations
//===----------------------------------------------------------------------===//
namespace
{
  void insertReadFileLibCall(int rank_size, Type floatEleType, Type indicesType, MLIRContext *ctx, ModuleOp &module, func::FuncOp function)
  {
    comet_debug() << "Inserting insertReadFileLibCall\n";
    IndexType indexType = IndexType::get(function.getContext());
    IntegerType i32Type = IntegerType::get(ctx, 32);
    auto unrankedMemref_index = mlir::UnrankedMemRefType::get(indexType, 0);
    auto unrankedMemref_element_type = mlir::UnrankedMemRefType::get(floatEleType, 0);
    auto unrankedMemref_indices_type = mlir::UnrankedMemRefType::get(indicesType, 0);
    llvm::SmallVector<Type, 21> inputFuncArgTypes;
    llvm::SmallVector<Type, 10> inputSizeFuncArgTypes;
    inputFuncArgTypes.push_back(i32Type);
    inputSizeFuncArgTypes.push_back(i32Type);
    for(int i = 0; i < rank_size * 2; i++)
    {
      inputFuncArgTypes.push_back(indexType);
      inputSizeFuncArgTypes.push_back(indexType);
    }
    for(int i = 0; i < rank_size * 4; i++)
    {
      inputFuncArgTypes.push_back(unrankedMemref_indices_type);
    }
    inputSizeFuncArgTypes.push_back(unrankedMemref_index);
    inputFuncArgTypes.push_back(unrankedMemref_element_type);
    inputSizeFuncArgTypes.push_back(i32Type);
    inputFuncArgTypes.push_back(i32Type);

    auto readInpuFunc = FunctionType::get(ctx, TypeRange(inputFuncArgTypes), {});
    assert(rank_size <=3 && rank_size >=2);

    std::string func_name = "read_input_"+std::to_string(rank_size)+"D";
    if (floatEleType.isF32())
    {
      func_name += "_f32";
    }
    else if (floatEleType.isF64())
    {
      func_name += "_f64";
    }
    else 
    {
      assert(false && "Unexpected type");
    }

    if(indicesType.isIndex())
    {
      func_name += "_64";
    }
    else if(indicesType.isInteger(32))
    {
      func_name += "_i32";
    }
    else if(indicesType.isInteger(64))
    {
      func_name += "_i64";
    }
    else 
    {
      assert(false && "Unexpected type");
    }

    if (!hasFuncDeclaration(module, func_name))
    {
      comet_debug() << "Adding " << func_name <<" to the module\n";
      func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                readInpuFunc, ArrayRef<NamedAttribute>{});
      func1.setPrivate();
      module.push_back(func1);
    }


    auto readInputSizesFunc = FunctionType::get(ctx, TypeRange(inputSizeFuncArgTypes), {}); /// last arg (i32Type): readMode

    std::string input_size_func_name = "read_input_sizes_"+std::to_string(rank_size)+"D";
    if (floatEleType.isF32())
    {
      input_size_func_name+="_f32";
    }
    else if (floatEleType.isF64())
    {
      input_size_func_name+="_f64";
    }
    else 
    {
      assert(false && "Unsupported type");
    }


    if (!hasFuncDeclaration(module, input_size_func_name))
    {
      comet_debug() << "Adding read_input_sizes_2D_f32 to the module\n";
      func::FuncOp func1 = func::FuncOp::create(function.getLoc(), input_size_func_name,
                                                readInputSizesFunc, ArrayRef<NamedAttribute>{});
      func1.setPrivate();
      module.push_back(func1);
    }
  }

Value insertSparseTensorDeclOp(PatternRewriter & rewriter,
                               MLIRContext* ctx,
                               Location loc,
                               unsigned rank_size, 
                               std::vector<Value>& tensorload_sizes_vec,
                               std::vector<Value>& array_sizes_vec,
                               std::vector<std::vector<int64_t>>& allPerms,
                               std::vector<Value>& dimSizes,
                               std::string formats_str,
                               Type ty)
  {
    comet_debug() << " Get users after ";
    /// create sparse tensor construct after lowering each sparse tensor output users
    comet_debug() << " tensorload_sizes_vec.size(): " << tensorload_sizes_vec.size() << ", rank_size: " << rank_size << "\n";
    /// create sptensor_construct
    
    std::vector<TensorFormatEnum> dim_formats = mlir::tensorAlgebra::getFormats(formats_str, rank_size, ctx);
    llvm::SmallVector<Attribute, 4> dim_formats_attr;
    for(TensorFormatEnum& format: dim_formats)
    {
      dim_formats_attr.push_back(TensorFormatEnumAttr::get(ctx,format));
    }

    Value sptensor;
    if (rank_size == 2)
    {
      Value dims = rewriter.create<tensor::FromElementsOp>(loc, ValueRange{dimSizes[0], dimSizes[1]});
      sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty,
                                                                          dims,
                                                                          ValueRange{
                                                                              tensorload_sizes_vec[0], /// A1pos (each dimension consists of pos and crd arrays)
                                                                              tensorload_sizes_vec[4], /// A2pos
                                                                          },
                                                                          ValueRange{
                                                                              tensorload_sizes_vec[1], /// A1crd
                                                                              tensorload_sizes_vec[5], /// A2crd
                                                                          },
                                                                          ValueRange {
                                                                              tensorload_sizes_vec[2], /// A1tile_pos
                                                                              tensorload_sizes_vec[6], /// A2tile_pos
                                                                          },
                                                                          ValueRange {
                                                                              tensorload_sizes_vec[3], /// A1tile_crd
                                                                              tensorload_sizes_vec[7], /// A2tile_crd
                                                                          },
                                                                          tensorload_sizes_vec[8], /// Aval
                                                                           2, ArrayAttr::get(ctx, dim_formats_attr));
    }
    else if (rank_size == 3)
    {
      Value dims = rewriter.create<tensor::FromElementsOp>(loc, ValueRange{dimSizes[0], dimSizes[1], dimSizes[2]});

      sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, dims,
                                                                          ValueRange{
                                                                              tensorload_sizes_vec[0],  /// A1pos (each dimension consists of pos and crd arrays)
                                                                              tensorload_sizes_vec[4],  /// A2pos
                                                                              tensorload_sizes_vec[8],  /// A3pos
                                                                          },
                                                                          ValueRange{
                                                                              tensorload_sizes_vec[1],  /// A1crd
                                                                              tensorload_sizes_vec[5],  /// A2crd
                                                                              tensorload_sizes_vec[9],  /// A3crd
                                                                          },
                                                                          ValueRange{
                                                                              tensorload_sizes_vec[2],  /// A1tile_pos
                                                                              tensorload_sizes_vec[6],  /// A2tile_pos
                                                                              tensorload_sizes_vec[10], /// A3tile_pos
                                                                          },
                                                                          ValueRange{
                                                                              tensorload_sizes_vec[3],  /// A1tile_crd
                                                                              tensorload_sizes_vec[7],  /// A2tile_crd
                                                                              tensorload_sizes_vec[11], /// A3tile_crd
                                                                          },
                                                                          // ValueRange{
                                                                          tensorload_sizes_vec[12], /// Aval
                                                                          3, ArrayAttr::get(ctx, dim_formats_attr));
    }
    else
    {
      llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: Not supported format (Tensors of dimensions greater than 3 are currently not supported).\n";
    }

    comet_debug() << "SparseTensorConstructOp generated for sparse output tensor:\n";
    comet_vdump(sptensor);

    return sptensor;
  }

  /// This a common lowering function used to lower SparseOutputTensorDeclOp and TempSparseOutputTensorDeclOp
  template <typename T>
  void lowerSparseOutputTensorDec(T op, PatternRewriter &rewriter)
  {
    if (isa<SparseOutputTensorDeclOp>(op))
    {
      comet_debug() << "lowerSparseOutputTensorDec::SparseOutputTensorDeclOp lowering\n";
    }
    else if (isa<TempSparseOutputTensorDeclOp>(op))
    {
      comet_debug() << "lowerSparseOutputTensorDec::TempSparseOutputTensorDeclOp lowering\n";
    }
    else 
    {
      assert(false && "Op should be either SparseOutputTensorDeclOp or TempSparseOutputTensorDeclOp");
    }

    SparseTensorType spType = mlir::cast<SparseTensorType>(op->getResultTypes()[0]);


    comet_vdump(op);
    auto loc = op.getLoc();
    StringRef formatsAttr = op.getFormat();
    std::string formats_str(formatsAttr.data());
    comet_debug() << " --- " << formats_str << "\n";

    comet_debug() << " " << op.getNumOperands() << "\n";
    auto rank_size = mlir::cast<SparseTensorType>(op.getResult().getType()).getRank();

    IndexType indexType = IndexType::get(op.getContext());
    Type valsType = spType.getElementType();
    Type indicesType = spType.getIndicesType();

    /// A1_pos ... A_value
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); /// memref<?xindex>
    auto dynamicmemTy_1d_vals_type = MemRefType::get({ShapedType::kDynamic}, valsType);     /// memref<?xf64>
    auto dynamicmemTy_1d_indices_type = MemRefType::get({ShapedType::kDynamic}, indicesType); 

    comet_debug() << " " << formats_str << " isDense: " << isDense(formats_str, ", ") << "\n";

    Value new_tensor;

    /// sparse output
    if (isDense(formats_str, ", ") == false)
    {
      /// search read_from_file function call to get the input file name
      /// Currently, has no filename

      std::vector<Value> array_sizes_vec; /// Store the size of C1pos, C1crd,..., Cval,C_dim1_size, C_dim2_size....

      /// No need to read from file
      /// We need to fill this tensorload_sizes_vec operations with new ones.....
      /// Some should get from sparse input, some are dense
      std::string input_filename;
      std::vector<std::vector<int64_t>> allPerms;

      std::vector<Value> dimSizes; /// for dimSizes in sptensor_construct

      for (auto u : op.getOperation()->getUsers())
      {
        comet_debug() << " Users:\n";
        comet_pdump(u);

        if (isa<tensorAlgebra::TransposeOp>(u) ||
            (isa<tensorAlgebra::TensorSetOp>(u) &&
             isa<tensorAlgebra::TransposeOp>(cast<tensorAlgebra::TensorSetOp>(u).getOperand(0).getDefiningOp())))
        {
          if (!isa<tensorAlgebra::TransposeOp>(u))
          {
            comet_debug() << "User of sparse tensor is a set Operation. Src of setOp is transpose\n";
            /// Set the insertion point before its user
            rewriter.setInsertionPoint(cast<tensorAlgebra::TensorSetOp>(u).getOperand(0).getDefiningOp());
          }
          else
          {
            comet_debug() << "User of sparse tensor is transpose operation\n";
            /// Set the insertion point before its user
            rewriter.setInsertionPoint(u);
          }

          /// Get the freeIndices of the sparse input tensor
          /// Check the dimension size, if it is integer, format is dense and get dim_size
          /// If it is ?, get the sparse input and get the definition, and the freeindex,
          /// tensorAlgebra::TransposeOp transpose_op = cast<tensorAlgebra::TransposeOp>(u);
          tensorAlgebra::TransposeOp transpose_op;
          if (isa<tensorAlgebra::TransposeOp>(u))
          {
            transpose_op = cast<tensorAlgebra::TransposeOp>(u);
          }
          else
          {
            transpose_op = cast<tensorAlgebra::TransposeOp>(cast<tensorAlgebra::TensorSetOp>(u).getOperand(0).getDefiningOp());
          }

          ArrayAttr indexMaps = transpose_op.getIndexingMaps();
          comet_debug() << " we get the indexMaps\n";
          allPerms = getAllPerms(indexMaps);
          comet_debug() << " we get the permutations\n";

          mlir::Value src_input = transpose_op.getRhs();
          comet_debug() << " ";
          comet_vdump(src_input);
          mlir::Value dst_input;
          for (auto u : op.getOperation()->getResult(0).getUsers())
          {
            comet_debug() << " ";
            comet_pdump(u);
            if (isa<tensorAlgebra::TensorSetOp>(u))
            {
              dst_input = u->getOperand(1); /// dest tensor is the 2nd
              comet_vdump(dst_input);
            }
          }

          /// If in COO format, for every dimension, different dimensions are
          std::vector<unsigned int> dstIndexLocInSrcVec;
          for (auto n : allPerms[1])
          { /// In dst index
            unsigned int dstIndexLocInSrc = findIndexInVector(allPerms[0], n);
            assert(dstIndexLocInSrc < allPerms[0].size() && " the index in dest is not found in src for transpose op\n");
            dstIndexLocInSrcVec.push_back(dstIndexLocInSrc);
          }

          ArrayAttr allFormats = transpose_op.getFormats();
          std::vector<std::string> allFormatsStr;
          for (unsigned int i = 0; i < allFormats.size(); i++)
          {
            std::string formats_str(allFormats[i].cast<mlir::StringAttr>().getValue());
            allFormatsStr.push_back(formats_str);
          }
          std::string src_format = allFormatsStr[0];
          std::string dst_format = allFormatsStr[1];

          /// If in COO format, then the sizes are the same as the input
          /// for A and B: 2x+1 + 2x+1 + x = 5x+2
          /// for ith index in B: pos is 2*i, crd is 2*i + 1
          ///                     pos_size is (2*rank+1) + 2*i, crd_size is (2*rank+1) + 2*i+1
          comet_debug() << " ";
          comet_vdump(dst_input);
          comet_debug() << " ";
          comet_pdump(dst_input.getDefiningOp());
          mlir::tensorAlgebra::SparseTensorType type;
          auto res = dst_input.getDefiningOp()->getResult(0);
          if (res.getType().isa<tensorAlgebra::SparseTensorType>())
          {
            type = res.getType().cast<tensorAlgebra::SparseTensorType>();
          }
          else
          {
            assert(false && "Expected SparseTensorType");
          }
          // unsigned int dst_rank = dst_input.getDefiningOp()->getNumOperands();
          unsigned int dst_rank = type.getRank();
          for (unsigned int i = 0; i < dst_rank; i++)
          {
            /// 4*rank+2 + i
            dimSizes.push_back(rewriter.create<SpTensorGetDimSize>(loc, src_input, rewriter.getI32IntegerAttr(i)));
          }

          Value cst_index_0 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(0));
          comet_vdump(cst_index_0);
          Value cst_index_1 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(1));
          comet_vdump(cst_index_1);
          Value cst_index_2 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(2));
          comet_vdump(cst_index_2);

          /// For COO format, 2D and 3D are the same
          /// if src format is in COO format,
          if (src_format.compare("COO") == 0)
          {
            for (unsigned int i = 0; i < dst_rank; i++)
            {
              /// 2*dst_rank+1
              unsigned int dstIndexLocInSrc = dstIndexLocInSrcVec[i];
              /// src_rank = dst_rank
              unsigned int posLocInSrc = (4 * dst_rank + 1) + 4 * dstIndexLocInSrc;
              unsigned int crdLocInSrc = posLocInSrc + 1;

              unsigned int posLocInSrc2 = posLocInSrc + 2;
              unsigned int crdLocInSrc2 = crdLocInSrc + 2;

              array_sizes_vec.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetDimPos>(loc, src_input, rewriter.getI32IntegerAttr(0)), 0));
              array_sizes_vec.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetDimCrd>(loc, src_input, rewriter.getI32IntegerAttr(0)), 0));
              array_sizes_vec.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetDimPos>(loc, src_input, rewriter.getI32IntegerAttr(1)), 0));
              array_sizes_vec.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetDimCrd>(loc, src_input, rewriter.getI32IntegerAttr(1)), 0));
            }
            /// val array size
            array_sizes_vec.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetVals>(loc, RankedTensorType::get({ShapedType::kDynamic}, src_input.getType().cast<SparseTensorType>().getElementType()), src_input), 0));

            /// set the pos array size, 1st dim as 2, all others as 1.
            for (unsigned int i = 0; i < dst_rank * 2; i++)
            {
              if (i == 0)
              {
                array_sizes_vec[2 * i] = cst_index_2;
              }
              else
              {
                array_sizes_vec[2 * i] = cst_index_1;
              }
            }
          }
          /// For 2D, consider CSR
          else if (dst_rank == 2)
          {
            if (src_format.compare("CSR") == 0)
            {
              comet_debug() << " 2D CSR transpose to 2D CSR\n";
              /// A1
              array_sizes_vec.push_back(cst_index_1);
              array_sizes_vec.push_back(cst_index_1);

              /// A1_tile
              array_sizes_vec.push_back(cst_index_0);
              array_sizes_vec.push_back(cst_index_0);

              mlir::Value crd_size = rewriter.create<AddIOp>(loc, dimSizes[0], cst_index_1);
              comet_debug() << "AddIOp generated for crd_size for CSR:\n";
              comet_vdump(crd_size);
              array_sizes_vec.push_back(crd_size);
              /// B2pos, Bval are the same size with A2pos, Aval
              mlir::Value vals_size = rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetVals>(loc, RankedTensorType::get({ShapedType::kDynamic}, src_input.getType().cast<SparseTensorType>().getElementType()), src_input), 0);
              array_sizes_vec.push_back(vals_size);

              /// A2tile
              array_sizes_vec.push_back(cst_index_0);
              array_sizes_vec.push_back(cst_index_0);

              /// Aval
              array_sizes_vec.push_back(vals_size);
            }
            else if (src_format.compare("ELL") == 0)
            {
              comet_debug() << " 2D ELL transpose to 2D ELL\n";
              comet_pdump(src_input.getDefiningOp());
              /// A1
              array_sizes_vec.push_back(cst_index_1);
              array_sizes_vec.push_back(cst_index_1);

              /// A1_tile
              array_sizes_vec.push_back(cst_index_1);
              array_sizes_vec.push_back(cst_index_1);

              /// A2
              array_sizes_vec.push_back(cst_index_1);
              /// TODO(PT): Verify this
              array_sizes_vec.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetDimCrd>(loc, src_input, rewriter.getI32IntegerAttr(1)), 0));

              /// A2tile
              array_sizes_vec.push_back(cst_index_0);
              array_sizes_vec.push_back(cst_index_0);

              /// Aval
              mlir::Value vals_size = rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetVals>(loc, src_input), 0);
            }
          }
          /// For 3D, consider CSF
          else if (dst_rank == 3)
          {
            if (src_format.compare("CSF") == 0) /// [PT] Code for CSF does not work properly
            {
              comet_debug() << " 3D CSF transpose to 3D CSF\n";
              array_sizes_vec.push_back(cst_index_2);
              mlir::Value vals_size = rewriter.create<tensor::DimOp>(loc, rewriter.create<SpTensorGetVals>(loc, src_input), 0);

              mlir::Value src_nnz = vals_size; 
              mlir::Value src_nnz_add1 = rewriter.create<AddIOp>(loc, src_nnz, cst_index_1);
              comet_debug() << "AddIOp generated for nnz for CSF:\n";
              comet_vdump(src_nnz_add1);
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);

              /// For the tiling dimensions
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);
              array_sizes_vec.push_back(src_nnz);
              array_sizes_vec.push_back(src_nnz_add1);
            }
          }

          comet_debug() << " array_sizes_vec.size(): " << array_sizes_vec.size() << "\n";
          comet_debug() << " dst_rank: " << dst_rank << "\n";
          std::vector<Value> tensorload_sizes_vec;
          for (unsigned int i = 0; i < 4 * dst_rank + 1; i++)
          {
            Value alloc_sizes;
            if (i < 4 * dst_rank)
            {
              alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_indices_type, ValueRange{array_sizes_vec[i]}, rewriter);
              comet_debug() << " AllocOp: ";
              comet_vdump(alloc_sizes);
            }
            else
            {
              alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_vals_type, ValueRange{array_sizes_vec[i]}, rewriter);
              comet_debug() << " AllocOp: ";
              comet_vdump(alloc_sizes);
            }
            Value tensorload_sizes = rewriter.create<ToTensorOp>(loc, alloc_sizes, rewriter.getUnitAttr(), rewriter.getUnitAttr());
            tensorload_sizes_vec.push_back(tensorload_sizes);
          }
          new_tensor = insertSparseTensorDeclOp(rewriter, op.getContext(), loc, rank_size, tensorload_sizes_vec, array_sizes_vec, allPerms, dimSizes, formats_str, op.getResult().getType());
          break;
        }
        else if (isa<indexTree::IndexTreeLHSOperandOp>(u))
        {
          comet_debug() << " Sparse output is used in it.LHSOperandOp\n";
          // Tensor is created as the output of a sparse tensor operation
          // For now we defer to the index tree dialect by inserting a tensor decl
          // that just contains empty domains.
          auto lhs_op = llvm::cast<indexTree::IndexTreeLHSOperandOp>(u);
          rank_size = lhs_op.getCrds().size();
          indexTree::DomainType domain_type = indexTree::DomainType::get(op.getContext()); 
          rewriter.setInsertionPoint(op);
          Value empty_domain = rewriter.create<indexTree::IndexTreeEmptyDomainOp>(loc, domain_type);
          llvm::SmallVector<Value> args = llvm::SmallVector<Value>(rank_size, empty_domain);

          new_tensor = rewriter.create<indexTree::IndexTreeSparseTensorOp>(loc, op.getResult().getType(), args);


          // Eventually, there are 2 cases:
          // Case 1: We can determine apriori the dimension of the sparse tensor
          //         This is the case if none of the index variables in the output
          //         tensor are used in a union or a insersect op. In this case we use
          //         the sparse tensor decleration of the input in order to determine
          //          the output tensor. We allocate arrays of the same size and then
          //         insert a ta.SpTensorDeclOp.
          // Case 2: We can't determine the dimension of the sparse tensor.
          //         This happens in all other cases. Here we insert a tensor
          //         that is defined with an (at least one) empty domain. In 
          //         the lowering process we can either use the symbolic phase
          //         to determine the allocations needed, or we can perform the
          //         allocations during the computational phase
          break;
        }
      }
      op.replaceAllUsesWith(new_tensor);
      rewriter.replaceOp(op, {new_tensor});
    }
    else
    { /// format == "Dense"

      auto resultTensorType = op.getResult().getType().template cast<mlir::TensorType>();
      std::vector<Value> cur_indices;
      std::vector<int64_t> cur_memref;
      auto resultMemTy = convertTensorToMemRef(resultTensorType);
      int j = 0;
      for (int i = 0; i < resultMemTy.getRank(); i++)
      {
        if (resultMemTy.isDynamicDim(i))
          cur_memref.push_back(ShapedType::kDynamic);
        else /// The constant dim size must NOT comes from the sparse matrix
          cur_memref.push_back(resultMemTy.getDimSize(i));

        if (resultMemTy.isDynamicDim(i))
          cur_indices.push_back(op.getLabels()[j++]); /// IndexCastOp
      }
      llvm::ArrayRef<int64_t> cur_memref_arrayref = llvm::ArrayRef<int64_t>(cur_memref);

      MemRefType memrefType2 = MemRefType::get(cur_memref_arrayref, valsType);
      Value alloc_sizes1 = insertAllocAndInitialize(loc, memrefType2, ValueRange(cur_indices), rewriter);
      comet_debug() << " AllocOp: ";
      comet_vdump(alloc_sizes1);

      Value tensorLoad = rewriter.create<ToTensorOp>(loc, alloc_sizes1, rewriter.getUnitAttr(), rewriter.getUnitAttr());
      comet_vdump(tensorLoad);

      op.replaceAllUsesWith(tensorLoad);
      rewriter.replaceOp(op, tensorLoad);
    }
  }

  struct DenseTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::DenseTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::DenseTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions for dense tensor declaration
     * Step 3: Remove the DenseTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::DenseTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {

      comet_debug() << "--------------DenseTensorDeclarationLowering in format begin\n";
      [[maybe_unused]] auto module = op->getParentOfType<ModuleOp>(); /// debuging purposes
      comet_debug() << "\n\n";

      comet_vdump(op);
      auto loc = op.getLoc();

      auto tensor_decl_value = cast<tensorAlgebra::DenseTensorDeclOp>(op);

      auto resultTensorType = op.getResult().getType();
      std::vector<Value> cur_indices;
      std::vector<int64_t> cur_memref;
      auto resultMemTy = convertTensorToMemRef(resultTensorType.cast<TensorType>());

      int j = 0;
      for (int i = 0; i < resultMemTy.getRank(); i++)
      {
        if (resultMemTy.isDynamicDim(i))
        {
          cur_indices.push_back(tensor_decl_value.getLabels()[j++]);
        }
      }

      /// Check if this tensor is explicitly initialized with ta.fill operation
      bool is_filled = false;
      for (auto u : op->getUsers())
      {
        comet_pdump(u);
        if (isa<tensorAlgebra::TensorFillOp>(u) || isa<tensorAlgebra::TensorSetOp>(u))
          is_filled = true;
      }

      comet_debug() << " AllocOp for initialization is_filled: " << is_filled << " \n";
      Value init_alloc;
      if (is_filled)
      {
        /// if is_filled is true, only allocate memory and let ta.fill initializes tensors
        init_alloc = rewriter.create<memref::AllocOp>(loc, resultMemTy, ValueRange(cur_indices));
        comet_vdump(init_alloc);
      }
      else
      {
        /// if is_filled is false, allocate memory and initialize it
        init_alloc = insertAllocAndInitialize(loc, resultMemTy, ValueRange(cur_indices), rewriter);
        comet_vdump(init_alloc);
      }

      cast<memref::AllocOp>(init_alloc.getDefiningOp()).setAlignmentAttr(rewriter.getI64IntegerAttr(32));

      Value tensorLoad = rewriter.create<ToTensorOp>(loc, init_alloc, rewriter.getUnitAttr(), rewriter.getUnitAttr());
      comet_debug() << " TensorLoad:\n";
      comet_vdump(tensorLoad);

      op.replaceAllUsesWith(tensorLoad);
      /// rewriter.replaceOp(op, tensorLoad);
      rewriter.eraseOp(op);

      comet_debug() << "--------------DenseTensorDeclarationLowering in format end\n";
      comet_debug() << "\n\n";

      /// module->dump(); //debugging purposes
      return success();
    }
  };

  struct SparseInputTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::SparseTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::SparseTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions and ta.sptensor_construct operation.
     * Step 3: Remove the SparseTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::SparseTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::SparseTensorDeclOp>(op));
      auto sp_decl = cast<tensorAlgebra::SparseTensorDeclOp>(op);
      comet_debug() << " SparseInputTensorDeclOpLowering in format begin\n";
      comet_vdump(op);
      mlir::MLIRContext *ctx = rewriter.getContext();
      auto function = cast<func::FuncOp>(op->getParentOp());
      auto module = function.getOperation()->getParentOfType<ModuleOp>();

      bool isOutputTensor = false;

      auto loc = op.getLoc();
      StringRef formatsAttr = op.getFormat();
      std::string formats_str(formatsAttr.data());
      comet_debug() << " --- " << formats_str << "\n";

      comet_debug() << " " << op.getNumOperands() << "\n";
      auto res = op.getResult();
      mlir::tensorAlgebra::SparseTensorType type;
      if(res.getType().isa<SparseTensorType>())
      {
        type = res.getType().cast<SparseTensorType>();
      }
      else
      {
        assert(false && "Expected TensorType");
      }
      auto rank_size = type.getRank();
      // auto rank_size = op.getResult().getType().cast<TensorType>().getRank();
      // auto rank_size = op.getNumOperands();

      IndexType indexType = IndexType::get(op.getContext());
      Type floatEleType = type.getElementType();
      IntegerType indicesType = type.getIndicesType();

      for (auto u1 : op.getOperation()->getUsers())
      {
        comet_debug() << "\nCheck the tensor is input or output\n";
        comet_pdump(u1);
        if (isa<tensorAlgebra::TensorMultOp, tensorAlgebra::TensorElewsMultOp, tensorAlgebra::TransposeOp>(u1))
        {
          comet_debug() << " used in ta.tc op\n";
          auto p = u1->getOperand(2);
          if(p == op)
          {
            isOutputTensor = true;
          }
        }
        else if (isa<tensorAlgebra::TensorSetOp>(u1))
        {
          comet_debug() << " used in ta.set op\n";
          auto p = u1->getOperand(1);
          if(p == op)
          {
            isOutputTensor = true;
          }
        }
        else if (isa<indexTree::IndexTreeLHSOperandOp>(u1))
        {
          comet_debug() << " used in it.LHSOperand op\n";
          isOutputTensor = true;
        }
        else if (isa<indexTree::IndexTreeOperandOp>(u1))
        {
          comet_debug() << " used in it.Operand op\n";
        }
        else if (isa<indexTree::IndexTreeIndexToTensorOp>(u1))
        {
          comet_debug() << " used in it.TensorAccess op\n";
        }
        else if (isa<indexTree::IndexTreeTensorDomainOp>(u1))
        {
          comet_debug() << " used in it.Domain op\n";
        }
        else if (isa<tensorAlgebra::TensorFillFromFileOp>(u1))
        {
          /// do nothing
          comet_debug() << " the tensor is in fill_from_file op\n";
        }
        else if (isa<tensorAlgebra::PrintOp>(u1))
        {
          comet_debug() << " the tensor is in PrintOp\n";
        }
        else if (isa<tensorAlgebra::ReduceOp>(u1))
        {
          comet_debug() << " the tensor is in ReduceOp\n";
        }
        else if (isa<tensorAlgebra::TensorElewsMultOp>(u1))
        {
          comet_debug() << " the tensor is in Elementwise multiplication\n";
        }
        else if (isa<tensorAlgebra::TensorFillOp>(u1))
        {
          /// TODO(gkestor): should we add this warning for user?
          llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: the sparse input tensor is using fill-op. Please use read_from_file() for sparse tensor inputs.\n";
        }
        else if (isa<tensorAlgebra::TensorDimOp>(u1))
        {
          comet_debug() << " the tensor has use in TensorDimOp and this use will be ignored!\n";
        }
        else if (isa<tensorAlgebra::AllocWorkspaceOp>(u1))
        {
          /// do nothing!
          comet_debug() << " the tensor has use in AllocWorkspaceOp\n";
        }
        else if(isa<func::CallOp>(u1))
        {
          /// do nothing!
          comet_debug() << " the tensor has use in func::CallOp\n";
        }
        else if(isa<func::ReturnOp>(u1))
        {
          comet_debug() << " the tensor has use in func::ReturnOp\n";
        }
        else if (isa<indexTree::IndexTreeOp>(u1))
        {
          /// do nothing!
          comet_debug() << " the tensor has use in a Index Tree\n";
        }
        else if (isa<tensorAlgebra::SpTensorAliasOp>(u1))
        {
          /// do nothing!
          comet_debug() << " the tensor has use in alias op\n";
        }
        else
        {
          u1->dump();
          llvm::errs() << __FILE__ << ":" << __LINE__ << "The tensor is in not supported operation\n";
        }
      }

      comet_debug() << " isOutputTensor: " << isOutputTensor << "\n";

      /// A1_pos ... A_value
      auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); /// memref<?xindex>
      auto dynamicmemTy_1d_float = MemRefType::get({ShapedType::kDynamic}, floatEleType);     /// memref<?xfloat>

      Type unrankedMemTy_index = UnrankedMemRefType::get(indexType, 0);
      Type unrankedMemTy_float = UnrankedMemRefType::get(floatEleType, 0);

      comet_debug() << " " << formats_str << " isDense: " << isDense(formats_str, ", ") << "\n";

      /// tensor is sparse and input.
      if (isDense(formats_str, ", ") == false && isOutputTensor == false)
      {
        comet_debug() << " Sparse input tensor \n";

        /// search read_from_file function call to get the input file name
        /// Currently, has no filename
        std::string input_filename;
        int readModeVal = -1;
        for (auto u : op.getOperation()->getUsers())
        {

          /// Used in TensorFillFromFileOp
          if (isa<tensorAlgebra::TensorFillFromFileOp>(u))
          {
            auto fillfromfileop = cast<tensorAlgebra::TensorFillFromFileOp>(u);
            /// Can get filename, from "filename" attribute of fillfromfileop
            StringAttr filename = fillfromfileop.getFilename().cast<StringAttr>();
            IntegerAttr readModeAttr = fillfromfileop.getReadMode().cast<IntegerAttr>();
            rewriter.eraseOp(fillfromfileop);

            comet_debug() << " filename: " << filename.getValue() << "\n";

            std::string filename_str(filename.getValue());
            input_filename = filename_str;
            comet_debug() << " " << input_filename << "\n";

            readModeVal = readModeAttr.getInt();
            comet_debug() << " readMode: " << readModeVal << "\n";
          }
        }

        comet_debug() << "sp_decl.getParameterCount(): " << sp_decl.getParameterCount() << "\n";
        MemRefType memTy_alloc_sizes = MemRefType::get({sp_decl.getParameterCount()}, indexType);
        Value alloc_sizes = rewriter.create<memref::AllocOp>(loc, memTy_alloc_sizes);
        comet_debug() << " ";
        comet_vdump(alloc_sizes);

        Value alloc_sizes_cast = rewriter.create<memref::CastOp>(loc, unrankedMemTy_index, alloc_sizes);

        std::vector<Value> dim_format = mlir::tensorAlgebra::getFormatsValue(formats_str, rank_size, rewriter, loc, indexType);
        std::vector<TensorFormatEnum> dim_format_int = mlir::tensorAlgebra::getFormats(formats_str, rank_size, ctx);
        llvm::SmallVector<Attribute, 4> dim_format_attr;
        for(TensorFormatEnum& format: dim_format_int)
        {
          dim_format_attr.push_back(TensorFormatEnumAttr::get(ctx, format));
        }
        auto dim_format_attrs = ArrayAttr::get(ctx, ArrayRef(dim_format_attr));
        comet_debug() << " Get the dim_format\n";

        /// inform the runtime of what env var to use for parsing input file
        IntegerType i32Type = IntegerType::get(op.getContext(), 32);
        Value sparseFileID;
        std::size_t pos = input_filename.find("SPARSE_FILE_NAME");
        if (pos == std::string::npos) /// not found
        {
          /// currently, reading of file when path of file is provided as arg is not supported at runtime.
          sparseFileID = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, -1));
        }
        /// 16 is the length of SPARSE_FILE_NAME
        std::string fileID = input_filename.substr(pos + 16, 1); /// this will only catch 0..9
        if (fileID.empty())
        { /// SPARSE_FILE_NAME
          sparseFileID = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, 9999));
        }
        else
        { /// SPARSE_FILE_NAME{int}
          comet_debug() << " Parsed fileID: " << fileID << "\n";
          int intFileID = std::stoi(fileID);
          sparseFileID = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, intFileID));
        }

        Value readModeConst;
        if (readModeVal == -1) /// none specified
        {                      /// 1, Default: standard matrix read
          readModeConst = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, 1));
        }
        else
        { /// readMode specified by user
          readModeConst = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getIntegerAttr(i32Type, readModeVal));
        }

        ///  Now, setup the runtime calls to read sizes related to the input matrices (e.g., read_input_sizes_2D_f32)
        if (rank_size == 2)
        { /// 2D
          comet_debug() << " 2D\n";
          /// Add function definition to the module
          insertReadFileLibCall(rank_size, floatEleType, indicesType, ctx, module, function);

          std::string read_input_sizes_str;
          if (floatEleType.isF32())
          {
            read_input_sizes_str = "read_input_sizes_2D_f32";
          }
          else if(floatEleType.isF64())
          {
            read_input_sizes_str = "read_input_sizes_2D_f64";
          }
          else 
          {
            assert(false && "Unexpected data type");
          }
          auto read_input_sizes_Call = rewriter.create<func::CallOp>(loc, read_input_sizes_str, SmallVector<Type, 2>{},
                                                                     ValueRange{sparseFileID,
                                                                                dim_format[0], dim_format[1], dim_format[2], dim_format[3],
                                                                                alloc_sizes_cast, readModeConst});
          read_input_sizes_Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
        }
        else if (rank_size == 3)
        { /// 3D
          comet_debug() << " 3D\n";
          /// Add function definition to the module
          insertReadFileLibCall(rank_size, floatEleType, indicesType, ctx, module, function);


          std::string read_input_sizes_str;
          if (floatEleType.isF32())
          {
            read_input_sizes_str = "read_input_sizes_3D_f32";
          }
          else if(floatEleType.isF64())
          {
            read_input_sizes_str = "read_input_sizes_3D_f64";
          }
          else 
          {
            assert(false && "Unexpected data type");
          }
          auto read_input_sizes_3D_Call = rewriter.create<func::CallOp>(loc, read_input_sizes_str, SmallVector<Type, 2>{},
                                                                        ValueRange{sparseFileID,
                                                                                   dim_format[0], dim_format[1], /// A1, A1_tile
                                                                                   dim_format[2], dim_format[3], /// A2, A2_tile
                                                                                   dim_format[4], dim_format[5], /// A3, A3_tile
                                                                                   alloc_sizes_cast, readModeConst});
          read_input_sizes_3D_Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
          comet_debug() << "\n";
        }
        else
        {
          assert(false && " Utility functions to read sparse tensors are supported up to 3 dimensions\n");
        }

        std::vector<Value> array_sizes;
        for (unsigned int i = 0; i < sp_decl.getParameterCount(); i++)
        { /// 2*rank_size + 1 + rank_size
          Value idx = rewriter.create<ConstantIndexOp>(loc, i);
          Value cor = rewriter.create<memref::LoadOp>(loc, alloc_sizes, idx);
          comet_debug() << " ";
          comet_vdump(cor);
          array_sizes.push_back(cor);
        }

        std::vector<Value> alloc_sizes_cast_vec;
        std::vector<Value> alloc_sizes_vec;

        /// A1_pos ... A_value
        auto dynamicmemTy_1d_indices_type = MemRefType::get({ShapedType::kDynamic}, type.getIndicesType()); /// memref<?xindex>
        Type unrankedMemTy_indices_type = UnrankedMemRefType::get(type.getIndicesType(), 0);
      
        for (unsigned int i = 0; i < sp_decl.getDimArrayCount(); i++)
        {
          std::vector<Value> idxes;
          idxes.push_back(array_sizes[i]);
          comet_vdump(array_sizes[i]);
          Value alloc_size = insertAllocAndInitialize(loc, dynamicmemTy_1d_indices_type, ValueRange{idxes}, rewriter);
          comet_debug() << " ";
          comet_vdump(alloc_size);

          alloc_sizes_vec.push_back(alloc_size);
          Value alloc_size_cast = rewriter.create<memref::CastOp>(loc, unrankedMemTy_indices_type, alloc_size);
          alloc_sizes_cast_vec.push_back(alloc_size_cast);
        }

        for (unsigned int i = sp_decl.getDimArrayCount(); i < sp_decl.getValueArrayPos(); i++)
        {
          std::vector<Value> idxes;
          idxes.push_back(array_sizes[i]);
          Value alloc_size = insertAllocAndInitialize(loc, dynamicmemTy_1d_float, ValueRange{idxes}, rewriter);
          comet_debug() << " ";
          comet_vdump(alloc_size);
          alloc_sizes_vec.push_back(alloc_size);
          Value alloc_size_cast = rewriter.create<memref::CastOp>(loc, unrankedMemTy_float, alloc_size);
          alloc_sizes_cast_vec.push_back(alloc_size_cast);
        }

        /// Now, setup the runtime calls to read the input matrices (e.g., read_input_3D_f64)
        if (rank_size == 2)
        { /// 2D
          std::string read_input_str;
          
          if (floatEleType.isF32())
          {
            read_input_str = "read_input_2D_f32";
          }
          else if (floatEleType.isF64())
          {
            read_input_str = "read_input_2D_f64";
          }
          else 
          {
            assert(false && "Unexpected type");
          }

          read_input_str += "_i"+std::to_string(indicesType.getWidth());

          auto read_input_f64Call = rewriter.create<func::CallOp>(loc, read_input_str, SmallVector<Type, 2>{},
                                                                  ValueRange{sparseFileID,
                                                                             dim_format[0], dim_format[1], /// A1_format, A1_tile_format
                                                                             dim_format[2], dim_format[3], /// A2_format, A2_tile_format
                                                                             alloc_sizes_cast_vec[0],      /// A1_pos
                                                                             alloc_sizes_cast_vec[1],      /// A1_crd
                                                                             alloc_sizes_cast_vec[2],      /// A1_tile_pos
                                                                             alloc_sizes_cast_vec[3],      /// A1_tile_crd
                                                                             alloc_sizes_cast_vec[4],      /// A2_pos
                                                                             alloc_sizes_cast_vec[5],      /// A2_crd
                                                                             alloc_sizes_cast_vec[6],      /// A2_tile_pos
                                                                             alloc_sizes_cast_vec[7],      /// A2_tile_crd
                                                                             alloc_sizes_cast_vec[8], readModeConst});
          read_input_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
        }
        else if (rank_size == 3)
        { /// 3D
          std::string read_input_str;
          if (floatEleType.isF32())
          {
            read_input_str = "read_input_3D_f32";
          }
          else if (floatEleType.isF64())
          {
            read_input_str = "read_input_3D_f64";
          }

          read_input_str += "_i"+std::to_string(indicesType.getWidth());

          auto read_input_f64Call = rewriter.create<func::CallOp>(loc, read_input_str, SmallVector<Type, 2>{},
                                                                  ValueRange{sparseFileID,
                                                                             dim_format[0], dim_format[1],                       /// A1, A1_tile
                                                                             dim_format[2], dim_format[3],                       /// A2, A2_tile
                                                                             dim_format[4], dim_format[5],                       /// A3, A3_tile
                                                                             alloc_sizes_cast_vec[0], alloc_sizes_cast_vec[1],   /// A1
                                                                             alloc_sizes_cast_vec[2], alloc_sizes_cast_vec[3],   /// A1_tile
                                                                             alloc_sizes_cast_vec[4], alloc_sizes_cast_vec[5],   /// A2
                                                                             alloc_sizes_cast_vec[6], alloc_sizes_cast_vec[7],   /// A2_tile
                                                                             alloc_sizes_cast_vec[8], alloc_sizes_cast_vec[9],   /// A3
                                                                             alloc_sizes_cast_vec[10], alloc_sizes_cast_vec[11], /// A3_tile
                                                                             alloc_sizes_cast_vec[12], readModeConst});
          read_input_f64Call.getOperation()->setAttr("filename", rewriter.getStringAttr(input_filename));
        }
        else
        {
          llvm::errs() << __LINE__ << " more than 3D, not supported\n";
        }

        comet_debug() << " Generate read_input_2D or read_input_3D functions\n";
        std::vector<Value> alloc_tensor_vec;
        for (unsigned int i = 0; i < sp_decl.getTotalArrayCount(); i++)
        {
          Value tensorLoad = rewriter.create<ToTensorOp>(loc, alloc_sizes_vec[i], rewriter.getUnitAttr(), rewriter.getUnitAttr());
          alloc_tensor_vec.push_back(tensorLoad);
        }

        llvm::SmallVector<int64_t> dim_sizes(rank_size, ShapedType::kDynamic); // TODO: Determine sizes!!!!
        auto ty = op.getResult().getType();

        Value sptensor;
        if (rank_size == 2)
        {
          Value dims = rewriter.create<tensor::FromElementsOp>(loc, ValueRange{array_sizes[9], array_sizes[10]});
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, 
                                                                                                dims, /// Dim sizes
                                                                                                ValueRange{
                                                                                                  alloc_tensor_vec[0], // A1_pos
                                                                                                  alloc_tensor_vec[4], /// A2_pos
                                                                                                },
                                                                                                ValueRange{
                                                                                                  alloc_tensor_vec[1], /// A1_crd
                                                                                                  alloc_tensor_vec[5], /// A2_crd
                                                                                                },
                                                                                                ValueRange{
                                                                                                  alloc_tensor_vec[2], /// A1_tile_pos
                                                                                                  alloc_tensor_vec[6], /// A2_tile_pos
                                                                                                },
                                                                                                ValueRange{
                                                                                                  alloc_tensor_vec[3], /// A1_tile_crd
                                                                                                  alloc_tensor_vec[7], /// A2_tile_crd
                                                                                                },
                                                                                                alloc_tensor_vec[8], /// Avals
                                                                             2, dim_format_attrs);
        }
        else if (rank_size == 3)
        {
          Value dims = rewriter.create<tensor::FromElementsOp>(loc, ValueRange{array_sizes[16], array_sizes[17], array_sizes[18]});

          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, dims,
                                                                                      ValueRange {
                                                                                        alloc_tensor_vec[0], /// A1_pos
                                                                                        alloc_tensor_vec[4],   /// A2_pos
                                                                                        alloc_tensor_vec[8],   /// A3_pos
                                                                                      },
                                                                                      ValueRange {
                                                                                        alloc_tensor_vec[1],   /// A1_crd
                                                                                        alloc_tensor_vec[5],   /// A2_crd
                                                                                        alloc_tensor_vec[9],   /// A3_crd
                                                                                      },
                                                                                      ValueRange {
                                                                                        alloc_tensor_vec[2],   /// A1_tile_pos
                                                                                        alloc_tensor_vec[6],   /// A2_tile_pos
                                                                                        alloc_tensor_vec[10], /// A3_tile_pos
                                                                                      },
                                                                                      ValueRange {
                                                                                        alloc_tensor_vec[3],   /// A1_tile_crd
                                                                                        alloc_tensor_vec[7],   /// A2_tile_crd
                                                                                        alloc_tensor_vec[11], /// A3_tile_crd
                                                                                      },
                                                                                      alloc_tensor_vec[12], /// Avals
                                                                             3, dim_format_attrs);
        }
        else
        {
          llvm::errs() << __LINE__ << " more than 3D, not supported\n";
        }

        comet_debug() << "SparseTensorConstructOp generated for input sparse tensor:\n";
        comet_vdump(sptensor);

        op.replaceAllUsesWith(sptensor);
        rewriter.replaceOp(op, sptensor);
      }

      /// The tensor is sparse output
      else if (isDense(formats_str, ", ") == false && isOutputTensor == true)
      {
        /// Is sparse output ,lower to ta.output_tensor_decl
        auto tensor_decl_value = cast<tensorAlgebra::SparseTensorDeclOp>(op);
        auto labels = tensor_decl_value.getLabels();
        auto tensor_format = tensor_decl_value.getFormat();
        auto tensor_type = tensor_decl_value.getType();
        auto is_temporal_tensor = tensor_decl_value.getTemporalTensor();

        mlir::Value outputtensordecl;
        if (is_temporal_tensor)
        {
          /// TempSparseOutputTensorDeclOp should be lowered before SparseOutputTensorDeclOp
          outputtensordecl = rewriter.create<TempSparseOutputTensorDeclOp>(loc,
                                                                           tensor_type, labels, tensor_format);
          comet_debug() << "Gokcen\n";
          comet_vdump(outputtensordecl);
        }
        else
          outputtensordecl = rewriter.create<SparseOutputTensorDeclOp>(loc,
                                                                       tensor_type, labels, tensor_format);
        comet_debug() << "SparseOutputTensorDecl or TempSparseOutputTensorDeclOp Operation is generated\n";
        comet_vdump(outputtensordecl);
        op.replaceAllUsesWith(outputtensordecl);
        rewriter.replaceOp(op, outputtensordecl);
      }
      /**** The tensor is dense *****/
      else
      { /// format == "Dense"
        comet_debug() << " it is dense tensor\n";
      }

      comet_debug() << " SparseInputTensorDeclOpLowering in format end\n";
      return success();
    }
  };

  struct SparseOutputTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::SparseOutputTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::SparseOutputTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions and ta.sptensor_construct operation.
     * Step 3: Remove the SparseOutputTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::SparseOutputTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {
      /// Sparse output tensor declaration happens after lowering to index tree dialect
      assert(isa<tensorAlgebra::SparseOutputTensorDeclOp>(op));
      comet_debug() << "SparseOutputTensorDeclOpLowering in format begin\n";
      comet_vdump(op);

      lowerSparseOutputTensorDec<tensorAlgebra::SparseOutputTensorDeclOp>(op, rewriter);

      comet_debug() << "--------------SparseOutputTensorDeclOpLowering in format end\n";
      return success();
    }
  };

  struct TempSparseOutputTensorDeclOpLowering : public OpRewritePattern<tensorAlgebra::TempSparseOutputTensorDeclOp>
  {
    using OpRewritePattern<tensorAlgebra::TempSparseOutputTensorDeclOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Get format and dims
     * Step 2: Emit alloc() instructions and ta.sptensor_construct operation.
     * Step 3: Remove the TempSparseOutputTensorDeclOp
     */
    LogicalResult matchAndRewrite(tensorAlgebra::TempSparseOutputTensorDeclOp op,
                                  PatternRewriter &rewriter) const final
    {
      /// Sparse output tensor declaration happens after lowering to index tree dialect
      assert(isa<tensorAlgebra::TempSparseOutputTensorDeclOp>(op));

      comet_debug() << "TempSparseOutputTensorDeclOpLowering in format begins\n";
      lowerSparseOutputTensorDec<tensorAlgebra::TempSparseOutputTensorDeclOp>(op, rewriter);
      comet_debug() << "TempSparseOutputTensorDeclOpLowering in format ends\n";

      return success();
    }
  };

  class DenseTensorDeclLoweringPass
      : public PassWrapper<DenseTensorDeclLoweringPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DenseTensorDeclLoweringPass)
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      patterns.insert<DenseTensorDeclOpLowering>(patterns.getContext());

      func::FuncOp function = getOperation();
      ConversionTarget target(getContext());
      target.addLegalDialect<ArithDialect,
                             memref::MemRefDialect,
                             scf::SCFDialect,
                             bufferization::BufferizationDialect,
                             linalg::LinalgDialect,
                             IndexTreeDialect>();

      target.addLegalOp<tensorAlgebra::PrintOp,
                        tensorAlgebra::ReduceOp,
                        tensorAlgebra::TransposeOp,
                        tensorAlgebra::TensorFillOp,
                        tensorAlgebra::GetTimeOp,
                        tensorAlgebra::PrintElapsedTimeOp,
                        tensorAlgebra::TensorSetOp,
                        tensorAlgebra::SparseOutputTensorDeclOp,
                        tensorAlgebra::TempSparseOutputTensorDeclOp,
                        tensorAlgebra::TensorMultOp,
                        tensorAlgebra::IndexLabelOp,
                        tensorAlgebra::ScalarOp,
                        tensorAlgebra::SparseTensorConstructOp,
                        tensorAlgebra::SpTensorAliasOp,
                        tensorAlgebra::SpTensorGetDimPos,
                        tensorAlgebra::SpTensorGetDimCrd,
                        tensorAlgebra::SpTensorGetVals,
                        tensorAlgebra::SpTensorGetDimSize>();

      if (failed(applyPartialConversion(function, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to applyPartialConversion in DenseTensorDeclLoweringPass\n";
        signalPassFailure();
      }
    }
  };

  class SparseTensorDeclLoweringPass
      : public PassWrapper<SparseTensorDeclLoweringPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseTensorDeclLoweringPass)
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      patterns.insert<SparseInputTensorDeclOpLowering>(&getContext());

      func::FuncOp function = getOperation();
      ConversionTarget target(getContext());

      target.addLegalDialect<LinalgDialect,
                             ArithDialect,
                             scf::SCFDialect,
                             mlir::memref::MemRefDialect,
                             IndexTreeDialect,
                             tensor::TensorDialect,
                             bufferization::BufferizationDialect>();

      target.addLegalOp<tensorAlgebra::PrintOp,
                        tensorAlgebra::GetTimeOp,
                        tensorAlgebra::PrintElapsedTimeOp,
                        tensorAlgebra::ReduceOp,
                        tensorAlgebra::TransposeOp,
                        tensorAlgebra::TensorFillOp,
                        tensorAlgebra::SparseTensorConstructOp,
                        tensorAlgebra::SparseOutputTensorDeclOp,
                        tensorAlgebra::TempSparseOutputTensorDeclOp,
                        tensorAlgebra::TensorSetOp,
                        tensorAlgebra::DenseTensorDeclOp,
                        tensorAlgebra::IndexLabelOp,
                        // tensorAlgebra::IndexLabelDynamicOp,
                        func::CallOp>();

      if (failed(applyPartialConversion(function, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to Lower SparseTensorDeclLoweringPass\n";
        signalPassFailure();
      }
      comet_debug() << "---------------SparseTensorDeclLoweringPass end\n";
    }
  };

  class SparseTempOutputTensorDeclLoweringPass
      : public PassWrapper<SparseTempOutputTensorDeclLoweringPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseTensorDeclLoweringPass)
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);
      patterns.add<TempSparseOutputTensorDeclOpLowering>(patterns.getContext());

      func::FuncOp function = getOperation();
      ConversionTarget target(getContext());
      target.markUnknownOpDynamicallyLegal([](Operation *)
                                           { return true; });

      target.addLegalDialect<LinalgDialect,
                             ArithDialect,
                             scf::SCFDialect,
                             mlir::memref::MemRefDialect,
                             IndexTreeDialect,
                             tensor::TensorDialect,
                             bufferization::BufferizationDialect>();

      target.addIllegalDialect<TADialect>();
      target.addLegalOp<tensorAlgebra::PrintOp,
                        tensorAlgebra::GetTimeOp,
                        tensorAlgebra::PrintElapsedTimeOp,
                        tensorAlgebra::ReduceOp,
                        tensorAlgebra::TransposeOp,
                        tensorAlgebra::TensorFillOp,
                        tensorAlgebra::SparseTensorConstructOp,
                        tensorAlgebra::TensorSetOp,
                        tensorAlgebra::DenseTensorDeclOp,
                        tensorAlgebra::SparseOutputTensorDeclOp,
                        tensorAlgebra::IndexLabelOp,
                        tensorAlgebra::DenseConstantOp,
                        tensorAlgebra::TensorDimOp,
                        tensorAlgebra::SpTensorAliasOp,
                        tensorAlgebra::SpTensorGetDimCrd,
                        tensorAlgebra::SpTensorGetDimPos,
                        tensorAlgebra::SpTensorGetDimSize,
                        tensorAlgebra::SpTensorGetVals,
                        tensorAlgebra::ScalarOp,
                        tensorAlgebra::AllocWorkspaceOp,
                        tensorAlgebra::TensorMultOp, // Should this be dynamically legal to only work with dense tensors?
                        func::CallOp>();

      if (failed(applyPartialConversion(function, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to Lower SparseOutputTensorDeclLoweringPass\n";
        signalPassFailure();
      }
      comet_debug() << "---------------SparseOutputTensorDeclLoweringPass end\n";
    }
  };

  class SparseOutputTensorDeclLoweringPass
      : public PassWrapper<SparseOutputTensorDeclLoweringPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseTensorDeclLoweringPass)
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);

      patterns.add<SparseOutputTensorDeclOpLowering>(patterns.getContext());

      func::FuncOp function = getOperation();
      ConversionTarget target(getContext());

      target.markUnknownOpDynamicallyLegal([](Operation *)
                                           { return true; });

      target.addLegalDialect<LinalgDialect,
                             ArithDialect,
                             scf::SCFDialect,
                             mlir::memref::MemRefDialect,
                             IndexTreeDialect,
                             tensor::TensorDialect,
                             bufferization::BufferizationDialect>();

      target.addIllegalDialect<TADialect>();
      target.addLegalOp<tensorAlgebra::PrintOp,
                        tensorAlgebra::GetTimeOp,
                        tensorAlgebra::PrintElapsedTimeOp,
                        tensorAlgebra::ReduceOp,
                        tensorAlgebra::TransposeOp,
                        tensorAlgebra::TensorFillOp,
                        tensorAlgebra::SparseTensorConstructOp,
                        tensorAlgebra::TensorSetOp,
                        tensorAlgebra::DenseTensorDeclOp,
                        tensorAlgebra::IndexLabelOp,
                        tensorAlgebra::DenseConstantOp,
                        tensorAlgebra::TensorDimOp,
                        tensorAlgebra::ScalarOp,
                        tensorAlgebra::AllocWorkspaceOp,
                        tensorAlgebra::SpTensorAliasOp,
                        tensorAlgebra::SpTensorGetDimCrd,
                        tensorAlgebra::SpTensorGetDimPos,
                        tensorAlgebra::SpTensorGetDimSize,
                        tensorAlgebra::SpTensorGetVals,
                        tensorAlgebra::TensorMultOp, // Should this be dynamically legal to only work with dense tensors?
                        func::CallOp>();

      if (failed(applyPartialConversion(function, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to Lower SparseOutputTensorDeclLoweringPass\n";
        signalPassFailure();
      }
      comet_debug() << "---------------SparseOutputTensorDeclLoweringPass end\n";
    }
  };
}
//===----------------------------------------------------------------------===//
/// Early Lowering Passes end
//===----------------------------------------------------------------------===//
/// TODO(gkestor): could it be possible to merge some of the tensor declaration lowerings?
std::unique_ptr<Pass> mlir::comet::createDenseTensorDeclLoweringPass()
{
  return std::make_unique<DenseTensorDeclLoweringPass>();
}

std::unique_ptr<Pass> mlir::comet::createSparseTensorDeclLoweringPass()
{
  return std::make_unique<SparseTensorDeclLoweringPass>();
}

std::unique_ptr<Pass> mlir::comet::createSparseTempOutputTensorDeclLoweringPass()
{
  return std::make_unique<SparseTempOutputTensorDeclLoweringPass>();
}

std::unique_ptr<Pass> mlir::comet::createSparseOutputTensorDeclLoweringPass()
{
  return std::make_unique<SparseOutputTensorDeclLoweringPass>();
}
