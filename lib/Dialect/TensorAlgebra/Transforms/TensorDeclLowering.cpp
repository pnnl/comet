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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include <limits>
#include <map>
#include <set>
#include <unordered_map>

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
  void mixModeEltWiseMultSparseTensorOutputLowering(Value computeOp, Location loc,
                                                    std::vector<std::vector<int>> rshPerms,
                                                    std::vector<Value> &dimSizes,
                                                    std::vector<Value> &tensorload_sizes_vec,
                                                    std::vector<Value> &array_sizes_vec,
                                                    PatternRewriter &rewriter)
  {

    IndexType indexType = IndexType::get(computeOp.getContext());
    FloatType f64Type = FloatType::getF64(computeOp.getContext());
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); /// memref<?xindex>
    auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamic}, f64Type);     /// memref<?xf64>

    comet_debug() << "mixModeEltWiseMultSparseTensorOutputLowering computeOp\n";
    comet_vdump(computeOp);

    ///  elementwise mul op in mix sparse dense case
    ///  If elementwise, copy sparse input arrays for elementwise mul
    int sparse_inputtensor_id = -1;
    auto rhsComputeOp = computeOp.getDefiningOp()->getOperand(0).getDefiningOp();

    auto first_operand = rhsComputeOp->getOperand(0).getDefiningOp();
    auto second_operand = rhsComputeOp->getOperand(1).getDefiningOp();
    comet_debug() << "EltWiseMult Operands:\n";
    comet_pdump(first_operand);
    comet_pdump(second_operand);

    if (isa<tensorAlgebra::SparseTensorConstructOp>(first_operand))
    {
      sparse_inputtensor_id = 0;
    }
    else if (isa<tensorAlgebra::SparseTensorConstructOp>(second_operand))
    {
      sparse_inputtensor_id = 1;
    }
    else
    {
      llvm::errs() << "ERROR: SparseTensorConstructOp was not found as one of the operands for itCompute\n";
    }

    comet_debug() << " SparseTensorConstructOp for computeOp: \n";
    comet_pdump(rhsComputeOp->getOperand(sparse_inputtensor_id).getDefiningOp());
    auto sptensor_construct_op = cast<tensorAlgebra::SparseTensorConstructOp>(rhsComputeOp->getOperand(sparse_inputtensor_id).getDefiningOp());

    for (unsigned int i = 0; i < 4 * (rshPerms[sparse_inputtensor_id].size()) + 1; i++)
    {
      comet_debug() << " in for loop\n";
      Value intput_tensorload_op = cast<ToTensorOp>(sptensor_construct_op.getOperand(i).getDefiningOp());
      Value input_alloc_op = cast<memref::AllocOp>(intput_tensorload_op.getDefiningOp()->getOperand(0).getDefiningOp());
      comet_debug() << " AllocOp: ";
      comet_vdump(input_alloc_op);

      comet_debug() << " ";
      Value input_alloc_op_param = input_alloc_op.getDefiningOp()->getOperand(0);
      comet_debug() << " ";

      Value output_alloc_op;
      if (i < 4 * (rshPerms[sparse_inputtensor_id].size()))
      {
        /// Memory allocation for position and coordinate arrays in sparse tensor contractions
        output_alloc_op = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{input_alloc_op_param}, rewriter);
      }
      else
      {
        /// Memory allocation for value array in sparse tensor contractions
        output_alloc_op = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{input_alloc_op_param}, rewriter); /// Cval array
        comet_debug() << " AllocOp: ";
        comet_vdump(output_alloc_op);
      }

      Value output_tensorload_op = rewriter.create<ToTensorOp>(loc, output_alloc_op);
      tensorload_sizes_vec.push_back(output_tensorload_op);
    }
    comet_debug() << " ";

    /// [0...2d, 2d+1...4d+1, 4d+2...5d+1]
    for (unsigned int i = 0; i < 4 * (rshPerms[sparse_inputtensor_id].size()) + 1; i++)
    {
      int sizes_i = i + 4 * (rshPerms[sparse_inputtensor_id].size()) + 1;
      comet_debug() << " ";
      comet_pdump(sptensor_construct_op.getOperand(sizes_i).getDefiningOp());

      Value input_load_op = sptensor_construct_op.getOperand(sizes_i);
      comet_debug() << "Ops push_back for Sparse Tensor Construct Op for MixedMode elementwise multiplication (array_sizes_vec):\n";
      comet_vdump(input_load_op);
      array_sizes_vec.push_back(input_load_op);
    }

    for (unsigned int i = 0; i < rshPerms[sparse_inputtensor_id].size(); i++)
    {
      int sizes_i = i + 2 * (2 * (rshPerms[sparse_inputtensor_id].size()) + 1);
      comet_debug() << " ";
      comet_pdump(sptensor_construct_op.getOperand(sizes_i).getDefiningOp());

      Value input_load_op = sptensor_construct_op.getOperand(sizes_i);
      comet_debug() << "Ops push_back for Sparse Tensor Construct Op for MixedMode elementwise multiplication (dimSizes):\n";
      comet_vdump(input_load_op);
      dimSizes.push_back(input_load_op);
    }
  }

  template <typename T>
  void pureSparseMultSparseTensorOutputLowering(T op,
                                                Location loc,
                                                std::string sparseOutputFormat,
                                                std::vector<Value> &dimSizes,
                                                std::vector<Value> &tensorload_sizes_vec,
                                                std::vector<Value> &array_sizes_vec,
                                                PatternRewriter &rewriter)
  {
    comet_debug() << " sparse output is used in itComputeOp op\n";
    comet_debug() << " sparseOutputFormat: " << sparseOutputFormat << "\n";

    comet_vdump(op);

    IndexType indexType = IndexType::get(op.getContext());
    FloatType f64Type = FloatType::getF64(op.getContext());
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); /// memref<?xindex>
    auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamic}, f64Type);     /// memref<?xf64>

    Value cst_index_0 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(0));
    comet_vdump(cst_index_0);
    Value cst_index_1 = rewriter.create<ConstantOp>(loc, IndexType::get(op.getContext()), rewriter.getIndexAttr(1));
    comet_vdump(cst_index_1);
    comet_vdump(op);
    comet_pdump(op.getOperation());
    
    unsigned int tensor_rank = op.getResult().getType().getRank();

    std::vector<mlir::Value> array_sizes;
    std::vector<mlir::Value> array_sizes_alloc_vec;
    std::vector<mlir::Value> initial_array_sizes;

    if (sparseOutputFormat.compare("CSR") == 0)
    { /// CSR format
      comet_debug() << " 2D CSR format in sparse output decl op\n";
      /// AllocOp, storeOp, LoadOp
      initial_array_sizes.push_back(cst_index_1);
      initial_array_sizes.push_back(cst_index_1);

      /// A1tile
      initial_array_sizes.push_back(cst_index_0);
      initial_array_sizes.push_back(cst_index_0);

      /// The other three size information size..
      /// get the dimension size from operand or type

      TensorType t = op.getResult().getType();
      for(int i = 0; i < t.getRank(); i++)
      {
        if(t.isDynamicDim(i))
        {
          dimSizes.push_back(op.getOperation()->getOperand(i));
        }
        else
        {
          dimSizes.push_back(rewriter.create<ConstantIndexOp>(loc, t.getShape()[i] ));
        }
      }

      /// The dim size is the second parameter of the
      Value dim2_posSize = rewriter.create<AddIOp>(loc, dimSizes[0], cst_index_1);
      comet_debug() << "AddIOp generated for dim2_posSize:\n";
      comet_vdump(dim2_posSize);
      initial_array_sizes.push_back(dim2_posSize);

      Value dim2_crdSize = rewriter.create<MulIOp>(loc, dimSizes[0], dimSizes[1]);
      initial_array_sizes.push_back(dim2_crdSize);

      /// A2tile
      initial_array_sizes.push_back(cst_index_0);
      initial_array_sizes.push_back(cst_index_0);

      /// Aval
      initial_array_sizes.push_back(dim2_crdSize);
      comet_debug() << " ";
      comet_vdump(dim2_crdSize);
    }
    else
    {
      llvm::errs() << __FILE__ << ":" << __LINE__ << "Not supported format\n";
    }

    /// same with transpose case
    comet_debug() << " initial_array_sizes.size(): " << initial_array_sizes.size() << "\n";
    comet_debug() << " tensor_rank: " << tensor_rank << "\n";
    std::vector<Value> array_alloc_vec;
    for (unsigned int i = 0; i < 4 * tensor_rank + 1; i++)
    {
      Value alloc_sizes;
      if (i < 4 * tensor_rank)
      {
        comet_debug() << " Inserting AllocOp: ";
        alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{initial_array_sizes[i]}, rewriter);
        comet_debug() << " AllocOp: ";
        comet_vdump(alloc_sizes);
      }
      else
      {
        alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{initial_array_sizes[i]}, rewriter);
        comet_debug() << " AllocOp: ";
        comet_vdump(alloc_sizes);
      }
      Value tensorload_sizes = rewriter.create<ToTensorOp>(loc, alloc_sizes);
      tensorload_sizes_vec.push_back(tensorload_sizes);
      array_alloc_vec.push_back(alloc_sizes);
    }

    /// Initialize the sizes of pos/crd/val arrays
    array_sizes.push_back(cst_index_1); /// A1pos_size
    array_sizes.push_back(cst_index_1); /// A1crd_size
    array_sizes.push_back(cst_index_1); /// A1tile_pos_size
    array_sizes.push_back(cst_index_0); /// A1tile_crd_size
    array_sizes.push_back(cst_index_1); /// A2pos_size
    array_sizes.push_back(cst_index_0); /// A2crd_size
    array_sizes.push_back(cst_index_0); /// A2tile_pos_size
    array_sizes.push_back(cst_index_1); /// A2tile_crd_size
    array_sizes.push_back(cst_index_0); /// Aval_size
    /// put the array sizes into alloc/store/loadOp
    for (auto size : array_sizes)
    {
      MemRefType memTy_alloc_sizes = MemRefType::get({1}, IndexType::get(op.getContext()));
      Value allocop = rewriter.create<memref::AllocOp>(loc, memTy_alloc_sizes);
      rewriter.create<memref::StoreOp>(loc, size, allocop, ValueRange{cst_index_0});
      Value loadop = rewriter.create<memref::LoadOp>(loc, allocop, ValueRange{cst_index_0});
      array_sizes_vec.push_back(loadop);
      array_sizes_alloc_vec.push_back(allocop);
    }

    rewriter.create<memref::StoreOp>(loc, dimSizes[0], array_alloc_vec[0], ValueRange{cst_index_0});
  }

  void insertReadFileLibCall(int rank_size, MLIRContext *ctx, ModuleOp &module, func::FuncOp function)
  {
    comet_debug() << "Inserting insertReadFileLibCall\n";
    FloatType f32Type, f64Type;
    if (VALUETYPE.compare("f32") == 0)
    {
      f32Type = FloatType::getF32(ctx);
    }
    else
    {
      f64Type = FloatType::getF64(ctx);
    }

    IndexType indexType = IndexType::get(function.getContext());
    IntegerType i32Type = IntegerType::get(ctx, 32);
    auto unrankedMemref_f64 = mlir::UnrankedMemRefType::get(f64Type, 0);
    /// TODO(gkestor): there is an issue with F32 UnrankedMemRefType
    auto unrankedMemref_f32 = mlir::UnrankedMemRefType::get(f64Type, 0);
    auto unrankedMemref_index = mlir::UnrankedMemRefType::get(indexType, 0);

    if (rank_size == 2)
    {
      comet_debug() << " Rank Size is 2\n";
      auto readInput2DF32Func = FunctionType::get(ctx, {i32Type, indexType, indexType,              /// A1_format, A1_tile_format
                                                        indexType, indexType,                       /// A2_format, A2_tile_format
                                                        unrankedMemref_index, unrankedMemref_index, /// A1_pos, A1_crd
                                                        unrankedMemref_index, unrankedMemref_index, /// A1_tile_pos, A1_tile_crd
                                                        unrankedMemref_index, unrankedMemref_index, /// A2_pos, A2_crd
                                                        unrankedMemref_index, unrankedMemref_index, /// A2_tile_pos, A2_tile_crd
                                                        unrankedMemref_f32, i32Type},
                                                  {});                                              /// last arg (i32Type): readMode
      auto readInput2DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType,              /// A1_format, A1_tile_format
                                                        indexType, indexType,                       /// A2_format, A2_tile_format
                                                        unrankedMemref_index, unrankedMemref_index, /// A1_pos, A1_crd
                                                        unrankedMemref_index, unrankedMemref_index, /// A1_tile_pos, A1_tile_crd
                                                        unrankedMemref_index, unrankedMemref_index, /// A2_pos, A2_crd
                                                        unrankedMemref_index, unrankedMemref_index, /// A2_tile_pos, A2_tile_crd
                                                        unrankedMemref_f64, i32Type},
                                                  {});

      if (VALUETYPE.compare("f32") == 0)
      {
        std::string func_name = "read_input_2D_f32";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << "Adding read_input_2D_f32 to the module\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInput2DF32Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else /// f64
      {
        std::string func_name = "read_input_2D_f64";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << "Adding read_input_2D_f64 to the module\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInput2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }

      auto readInputSizes2DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, indexType, unrankedMemref_index, i32Type}, {}); /// last arg (i32Type): readMode

      if (VALUETYPE.compare("f32") == 0)
      {
        std::string func_name = "read_input_sizes_2D_f32";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << "Adding read_input_sizes_2D_f32 to the module\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInputSizes2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        std::string func_name = "read_input_sizes_2D_f64";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << "Adding read_input_sizes_2D_f64 to the module\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInputSizes2DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
    }
    /// 3D tensor
    else if (rank_size == 3)
    {
      auto readInput3DF32Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, indexType, indexType, indexType, /// Dimensions
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A1
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A1_tile
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A2
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A2_tile
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A3
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A3_tile
                                                        unrankedMemref_f32, i32Type},
                                                  {});                                                                             /// last arg (i32Type): readMode
      auto readInput3DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, indexType, indexType, indexType, /// Dimensions
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A1
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A1_tile
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A2
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A2_tile
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A3
                                                        unrankedMemref_index, unrankedMemref_index,                                /// A3_tile
                                                        unrankedMemref_f64, i32Type},
                                                  {});

      if (VALUETYPE.compare("f32") == 0)
      {
        std::string func_name = "read_input_3D_f32";
        if (!hasFuncDeclaration(module, func_name))
        {
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInput3DF32Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        std::string func_name = "read_input_3D_f64";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << " Insert read_input_3D_f64 decl\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInput3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }

      auto readInputSizes3DF64Func = FunctionType::get(ctx, {i32Type, indexType, indexType, indexType, indexType, indexType, indexType, unrankedMemref_index, i32Type}, {}); /// last arg (i32Type): readMode

      if (VALUETYPE.compare("f32") == 0)
      {

        std::string func_name = "read_input_sizes_3D_f32";
        if (!hasFuncDeclaration(module, func_name))
        {
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInputSizes3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
      else
      {
        std::string func_name = "read_input_sizes_3D_f64";
        if (!hasFuncDeclaration(module, func_name))
        {
          comet_debug() << " Insert read_input_sizes_3D_f64 decl\n";
          func::FuncOp func1 = func::FuncOp::create(function.getLoc(), func_name,
                                                    readInputSizes3DF64Func, ArrayRef<NamedAttribute>{});
          func1.setPrivate();
          module.push_back(func1);
        }
      }
    }
    else
    {
      llvm::errs() << __LINE__ << "Not supported dims\n";
    }
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

    assert(isa<SparseOutputTensorDeclOp>(op) || (isa<TempSparseOutputTensorDeclOp>(op) &&
                                                    "Op should be either SparseOutputTensorDeclOp or TempSparseOutputTensorDeclOp"));

    comet_vdump(op);
    auto loc = op.getLoc();
    StringRef formatsAttr = op.getFormat();
    std::string formats_str(formatsAttr.data());
    comet_debug() << " --- " << formats_str << "\n";

    comet_debug() << " " << op.getNumOperands() << "\n";
    auto rank_size = op.getResult().getType().getRank();

    IndexType indexType = IndexType::get(op.getContext());
    FloatType f64Type = FloatType::getF64(op.getContext());
    if (VALUETYPE.compare(0, 3, "f32") == 0)
      f64Type = FloatType::getF32(op.getContext());

    /// A1_pos ... A_value
    auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); /// memref<?xindex>
    auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamic}, f64Type);     /// memref<?xf64>

    comet_debug() << " " << formats_str << " isDense: " << isDense(formats_str, ", ") << "\n";

    /// sparse output
    if (isDense(formats_str, ", ") == false)
    {
      /// search read_from_file function call to get the input file name
      /// Currently, has no filename

      std::vector<Value> tensorload_sizes_vec;
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
          mlir::TensorType type;
          auto res = dst_input.getDefiningOp()->getResult(0);
          if (res.getType().isa<TensorType>())
          {
            type = res.getType().cast<TensorType>();
          }
          else
          {
            assert(false && "Expected TensorType");
          }
          // unsigned int dst_rank = dst_input.getDefiningOp()->getNumOperands();
          unsigned int dst_rank = type.getRank();
          for (unsigned int i = 0; i < dst_rank; i++)
          {
            /// 4*rank+2 + i
            dimSizes.push_back(src_input.getDefiningOp()->getOperand(8 * dst_rank + 2 + dstIndexLocInSrcVec[i]));
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

              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(posLocInSrc));
              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(crdLocInSrc));
              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(posLocInSrc2));
              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(crdLocInSrc2));
            }
            /// val array size
            array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(8 * dst_rank + 1));

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
              /// TODO(gkestor): Do not hardcode
              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(17));

              /// A2tile
              array_sizes_vec.push_back(cst_index_0);
              array_sizes_vec.push_back(cst_index_0);

              /// Aval
              /// TODO(gkestor): Do not hardcode
              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(17));
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
              /// TODO(gkestor): Do not hardcode
              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(14));

              /// A2tile
              array_sizes_vec.push_back(cst_index_0);
              array_sizes_vec.push_back(cst_index_0);

              /// Aval
              /// TODO(gkestor): Do not hardcode
              array_sizes_vec.push_back(src_input.getDefiningOp()->getOperand(17));
            }
          }
          /// For 3D, consider CSF
          else if (dst_rank == 3)
          {
            if (src_format.compare("CSF") == 0) /// [PT] Code for CSF does not work properly
            {
              comet_debug() << " 3D CSF transpose to 3D CSF\n";
              array_sizes_vec.push_back(cst_index_2);
              mlir::Value src_nnz = src_input.getDefiningOp()->getOperand(25); 
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
          for (unsigned int i = 0; i < 4 * dst_rank + 1; i++)
          {
            Value alloc_sizes;
            if (i < 4 * dst_rank)
            {
              alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{array_sizes_vec[i]}, rewriter);
              comet_debug() << " AllocOp: ";
              comet_vdump(alloc_sizes);
            }
            else
            {
              alloc_sizes = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{array_sizes_vec[i]}, rewriter);
              comet_debug() << " AllocOp: ";
              comet_vdump(alloc_sizes);
            }
            Value tensorload_sizes = rewriter.create<ToTensorOp>(loc, alloc_sizes);
            tensorload_sizes_vec.push_back(tensorload_sizes);
          }
        }
        else if (isa<indexTree::IndexTreeComputeLHSOp>(u))
        {
          comet_debug() << " sparse output is used in itComputeOp op\n";

          /// Set the insertion point before its user
          rewriter.setInsertionPoint(u);

          indexTree::IndexTreeComputeLHSOp lhsOp = cast<indexTree::IndexTreeComputeLHSOp>(u);
          comet_debug() << " formats_str: " << formats_str << "\n";
          comet_debug() << " current Op: ";
          comet_vdump(lhsOp);

          for (auto uLHS : lhsOp.getOperation()->getUsers())
          {
            assert(isa<indexTree::IndexTreeComputeOp>(uLHS) && "User of IndexTreeComputeLHSOp can only be IndexTreeComputeOp");

            comet_debug() << " lhsOp user: ";
            comet_pdump(uLHS);

            auto computeOp = cast<indexTree::IndexTreeComputeOp>(uLHS);
            comet_debug() << " Get RHS op: ";
            comet_vdump(computeOp);

            std::vector<std::vector<int>> rhsPerms;
            getRHSPermsOfComputeOp(computeOp, rhsPerms);

            std::vector<std::vector<std::string>> rhsFormats;
            getRHSFormatsOfComputeOp(computeOp, rhsFormats);

            comet_debug() << " rhsPerms: \n";
            for (auto m : rhsPerms)
            {
              comet_debug() << " \n";
              for (auto n : m)
              {
                comet_debug() << n << " \n";
              }
              comet_debug() << "\n";
            }

            comet_debug() << " rhsFormats: \n";
            for (auto m : rhsFormats)
            {
              comet_debug() << " \n";
              for (auto n : m)
              {
                comet_debug() << n << " \n";
              }
              comet_debug() << "\n";
            }

            bool isElementwise = checkIsElementwise(rhsPerms);

            comet_debug() << "Checking if it is mixed mode\n";
            bool isMixedMode = checkIsMixedMode(rhsFormats);

            comet_debug() << "IsElementWise: " << isElementwise << " isMixedMode: " << isMixedMode << "\n";
            if (isElementwise && isMixedMode)
            {
              comet_debug() << "It is an elementwise multiplication in mixed Mode sparse = sparse * dense\n";
              if (isMixedMode)
              {
                comet_debug() << "It is an mix-mode elementwise multiplication in Mix Mode\n";
                mixModeEltWiseMultSparseTensorOutputLowering(computeOp,
                                                             loc,
                                                             rhsPerms,
                                                             dimSizes,
                                                             tensorload_sizes_vec,
                                                             array_sizes_vec, rewriter);
              }
              else
              {
                comet_debug() << "It is an pure-sparse elementwise multiplication\n";
                pureSparseMultSparseTensorOutputLowering<>(op,
                                                           loc,
                                                           formats_str,
                                                           dimSizes,
                                                           tensorload_sizes_vec,
                                                           array_sizes_vec,
                                                           rewriter);
              }
            }
            else
            {
              if (!isMixedMode)
              {
                comet_debug() << "It is an pure-sparse multiplication or assigment from dense to sparse (produced after workspace transformations)\n";
                pureSparseMultSparseTensorOutputLowering(op,
                                                         loc,
                                                         formats_str,
                                                         dimSizes,
                                                         tensorload_sizes_vec,
                                                         array_sizes_vec,
                                                         rewriter);
              }
              else
              {
                /// TODO(gkestor) Mix-mode sparse computation with sparse output not yet supported such as TTM (tensor times matrix)
                /// TODO(gkestor): if the sparsity patterns is known
                comet_debug() << "It is an mix mode element-wise multiplication\n";
                mixModeEltWiseMultSparseTensorOutputLowering(computeOp,
                                                             loc,
                                                             rhsPerms,
                                                             dimSizes,
                                                             tensorload_sizes_vec,
                                                             array_sizes_vec, rewriter);
              }
            }
          }
        }
        else if (isa<tensorAlgebra::TensorFillFromFileOp>(u))
        {
          comet_debug() << " Sparse output is used in TensorFillFromFileOp\n";
          auto fillfromfileop = cast<tensorAlgebra::TensorFillFromFileOp>(u);
          /// Can get filename, from "filename" attribute of fillfromfileop
          rewriter.eraseOp(fillfromfileop);
        }
        else if (isa<indexTree::IndexTreeComputeRHSOp>(u))
        {
          comet_debug() << "The tensor is in IndexTreeComputeRHSOp, no action taken\n";
          continue;
        }
        else if (isa<tensorAlgebra::PrintOp>(u))
        {
          comet_debug() << "The tensor is in print op,  no action taken\n";
          continue;
        }
        else if (isa<tensorAlgebra::ReduceOp>(u))
        {
          comet_debug() << "The tensor is in sum op,  no action taken\n";
          continue;
        }
        else if (isa<tensorAlgebra::TensorDimOp>(u))
        {
          comet_debug() << "The tensor is in dim op,  no action taken\n";
          continue;
        }
        else
        {
          comet_pdump(u);
          llvm::errs() << __FILE__ << __LINE__ << " tensor is used in the following unsupported op\n";
        }

        comet_debug() << " Get users after ";
        /// create sparse tensor construct after lowering each sparse tensor output users
        comet_debug() << " tensorload_sizes_vec.size(): " << tensorload_sizes_vec.size() << ", rank_size: " << rank_size << "\n";
        /// create sptensor_construct
        SmallVector<mlir::Type, 1> elementTypes;
        for (unsigned int i = 0; i < 4 * rank_size + 1; i++)
        {
          assert(tensorload_sizes_vec.size() > 0 && "ERROR: Please report this error to the developers!");
          comet_debug() << " " << i << " ";
          comet_vdump(tensorload_sizes_vec[i]);
          elementTypes.push_back(tensorload_sizes_vec[i].getType());
        }
        comet_debug() << "\n ";
        /// [0 ... 2*rank_size, 2*rank_size+1 ... 4*rank_size+1, 4*rank_size+2 ... 5*rank_size + 1]
        /// 2d+1 + 2d+1 + d => 5d+2
        for (unsigned int i = 0; i < 4 * rank_size + 1; i++)
        {
          assert(array_sizes_vec.size() > 0 && "ERROR: Please report this error to the developers!");
          comet_debug() << " " << i << " ";
          comet_vdump(array_sizes_vec[i]);
          elementTypes.push_back(array_sizes_vec[i].getType());
        }
        comet_debug() << "\n ";
        for (unsigned int i = 0; i < rank_size; i++)
        {
          assert(dimSizes.size() > 0 && "ERROR: Please report this error to the developers!");
          elementTypes.push_back(dimSizes[i].getType());
        }
        comet_debug() << "\n ";

        auto ty = tensorAlgebra::SparseTensorType::get(elementTypes);

        Value sptensor;
        if (rank_size == 2)
        {
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty,
                                                                             ValueRange{
                                                                                 tensorload_sizes_vec[0], /// A1pos (each dimension consists of pos and crd arrays)
                                                                                 tensorload_sizes_vec[1], /// A1crd
                                                                                 tensorload_sizes_vec[2], /// A1tile_pos
                                                                                 tensorload_sizes_vec[3], /// A1tile_crd
                                                                                 tensorload_sizes_vec[4], /// A2pos
                                                                                 tensorload_sizes_vec[5], /// A2crd
                                                                                 tensorload_sizes_vec[6], /// A2tile_pos
                                                                                 tensorload_sizes_vec[7], /// A2tile_crd
                                                                                 tensorload_sizes_vec[8], /// Aval
                                                                                 array_sizes_vec[0],      /// A1pos_size (size of each pos and crd arrays)
                                                                                 array_sizes_vec[1],      /// A1crd_size
                                                                                 array_sizes_vec[2],      /// A1tile_pos_size
                                                                                 array_sizes_vec[3],      /// A1tile_crd_size
                                                                                 array_sizes_vec[4],      /// A2pos_size
                                                                                 array_sizes_vec[5],      /// A2crd_size
                                                                                 array_sizes_vec[6],      /// A2tile_pos_size
                                                                                 array_sizes_vec[7],      /// A2tile_crd_size
                                                                                 array_sizes_vec[8],      /// Aval_size (size of value array)
                                                                                 dimSizes[0],             /// dim1_size(size of each dimension in sparse tensor)
                                                                                 dimSizes[1]              /// dim2_size (size of each dimension in sparse tensor)
                                                                             },
                                                                             2);
        }
        else if (rank_size == 3)
        {
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty,
                                                                             ValueRange{
                                                                                 tensorload_sizes_vec[0],  /// A1pos (each dimension consists of pos and crd arrays)
                                                                                 tensorload_sizes_vec[1],  /// A1crd
                                                                                 tensorload_sizes_vec[2],  /// A1tile_pos
                                                                                 tensorload_sizes_vec[3],  /// A1tile_crd
                                                                                 tensorload_sizes_vec[4],  /// A2pos
                                                                                 tensorload_sizes_vec[5],  /// A2crd
                                                                                 tensorload_sizes_vec[6],  /// A2tile_pos
                                                                                 tensorload_sizes_vec[7],  /// A2tile_crd
                                                                                 tensorload_sizes_vec[8],  /// A3pos
                                                                                 tensorload_sizes_vec[9],  /// A3crd
                                                                                 tensorload_sizes_vec[10], /// A3tile_pos
                                                                                 tensorload_sizes_vec[11], /// A3tile_crd
                                                                                 tensorload_sizes_vec[12], /// Aval
                                                                                 array_sizes_vec[0],       /// A1pos_size (size of each pos and crd arrays)
                                                                                 array_sizes_vec[1],       /// A1crd_size
                                                                                 array_sizes_vec[2],       /// A1tile_pos_size
                                                                                 array_sizes_vec[3],       /// A1tile_crd_size
                                                                                 array_sizes_vec[4],       /// A2pos_size
                                                                                 array_sizes_vec[5],       /// A2crd_size
                                                                                 array_sizes_vec[6],       /// A2tile_pos_size
                                                                                 array_sizes_vec[7],       /// A2tile_crd_size
                                                                                 array_sizes_vec[8],       /// A3pos_size
                                                                                 array_sizes_vec[9],       /// A3crd_size
                                                                                 array_sizes_vec[10],      /// A3tile_pos_size
                                                                                 array_sizes_vec[11],      /// A3tile_crd_size
                                                                                 array_sizes_vec[12],      /// Aval_size (size of value array)
                                                                                 dimSizes[0],              /// dim1_size (size of each dimension in sparse tensor)
                                                                                 dimSizes[1],              /// dim2_size (size of each dimension in sparse tensor)
                                                                                 dimSizes[2]               /// dim3_size
                                                                             },
                                                                             3);
        }
        else
        {
          llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: Not supported format (Tensors of dimensions greater than 3 are currently not supported).\n";
        }

        comet_debug() << "SparseTensorConstructOp generated for sparse output tensor:\n";
        comet_vdump(sptensor);

        /// create ta.index_label operation.
        comet_vdump(op);

        op.replaceAllUsesWith(sptensor);
        rewriter.replaceOp(op, sptensor);
      } /// for (auto u : op.getOperation()->getUsers())
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

      MemRefType memrefType2 = MemRefType::get(cur_memref_arrayref, f64Type);
      Value alloc_sizes1 = insertAllocAndInitialize(loc, memrefType2, ValueRange(cur_indices), rewriter);
      comet_debug() << " AllocOp: ";
      comet_vdump(alloc_sizes1);

      Value tensorLoad = rewriter.create<ToTensorOp>(loc, alloc_sizes1);
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

      Value tensorLoad = rewriter.create<ToTensorOp>(loc, init_alloc);
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

      std::string op_str = dump2str(op);
      bool isOutputTensor = false;

      auto loc = op.getLoc();
      StringRef formatsAttr = op.getFormat();
      std::string formats_str(formatsAttr.data());
      comet_debug() << " --- " << formats_str << "\n";

      comet_debug() << " " << op.getNumOperands() << "\n";
      auto res = op.getResult();
      mlir::TensorType type;
      if(res.getType().isa<TensorType>())
      {
        type = res.getType().cast<TensorType>();
      }
      else
      {
        assert(false && "Expected TensorType");
      }
      auto rank_size = type.getRank();
      // auto rank_size = op.getResult().getType().cast<TensorType>().getRank();
      // auto rank_size = op.getNumOperands();

      IndexType indexType = IndexType::get(op.getContext());
      FloatType f64Type = FloatType::getF64(op.getContext());
      if (VALUETYPE.compare(0, 3, "f32") == 0)
        f64Type = FloatType::getF32(op.getContext());

      for (auto u1 : op.getOperation()->getUsers())
      {
        comet_debug() << "\nCheck the tensor is input or output\n";
        comet_pdump(u1);
        if (isa<tensorAlgebra::TensorMultOp>(u1))
        {
          comet_debug() << " used in ta.tc op\n";
          auto p = cast<tensorAlgebra::TensorMultOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            /// comet_vdump(n);
            std::string n_str = dump2str(p->getOperand(i));
            comet_debug() << "the operands: " << n_str << "\n";
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_debug() << " FIND IT: " << i << "\n";
              if (i == 2)
              {
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<tensorAlgebra::TensorElewsMultOp>(u1))
        {
          comet_debug() << " used in ta.elews_mul op\n";
          auto p = cast<tensorAlgebra::TensorElewsMultOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_debug() << " FIND IT: " << i << "\n";
              if (i == 2)
              {
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<tensorAlgebra::TensorSetOp>(u1))
        {
          comet_debug() << " used in ta.set op\n";
          auto p = cast<tensorAlgebra::TensorSetOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            comet_debug() << " the " << i << "th operand\n";
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_debug() << " FIND IT: " << i << "\n";
              if (i == 1)
              {
                /// The source tensor of the set op
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<tensorAlgebra::TransposeOp>(u1))
        {
          comet_debug() << " used in transpose op\n";
          auto p = cast<tensorAlgebra::TransposeOp>(u1).getOperation();
          for (unsigned int i = 0; i < p->getNumOperands(); i++)
          {
            std::string n_str = dump2str(p->getOperand(i));
            if (n_str.compare(0, op_str.size(), op_str) == 0)
            {
              comet_debug() << " FIND IT: " << i << "\n";
              if (i == 2)
              {
                /// output of ta.elews_mul
                isOutputTensor = true;
              }
            }
          }
        }
        else if (isa<indexTree::IndexTreeComputeRHSOp>(u1))
        {
          comet_debug() << " used in ta.itComputeRHS op\n";
          isOutputTensor = false;
        }
        else if (isa<indexTree::IndexTreeComputeLHSOp>(u1))
        {
          comet_debug() << " used in ta.itComputeLHS op\n";
          isOutputTensor = true;
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
        else
        {
          u1->dump();
          llvm::errs() << __FILE__ << ":" << __LINE__ << "The tensor is in not supported operation\n";
        }
      }

      comet_debug() << " isOutputTensor: " << isOutputTensor << "\n";

      /// A1_pos ... A_value
      auto dynamicmemTy_1d_index = MemRefType::get({ShapedType::kDynamic}, indexType); /// memref<?xindex>
      auto dynamicmemTy_1d_f64 = MemRefType::get({ShapedType::kDynamic}, f64Type);     /// memref<?xf64>

      Type unrankedMemTy_index = UnrankedMemRefType::get(indexType, 0);
      Type unrankedMemTy_f64 = UnrankedMemRefType::get(f64Type, 0);

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
          insertReadFileLibCall(rank_size, ctx, module, function);

          std::string read_input_sizes_str;
          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            read_input_sizes_str = "read_input_sizes_2D_f32";
          }
          else
          {
            read_input_sizes_str = "read_input_sizes_2D_f64";
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
          insertReadFileLibCall(rank_size, ctx, module, function);

          std::string read_input_sizes_str;
          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            read_input_sizes_str = "read_input_sizes_3D_f32";
          }
          else
          { /// default f64
            read_input_sizes_str = "read_input_sizes_3D_f64";
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
        for (unsigned int i = 0; i < sp_decl.getDimArrayCount(); i++)
        {
          std::vector<Value> idxes;
          idxes.push_back(array_sizes[i]);
          comet_vdump(array_sizes[i]);
          Value alloc_size = insertAllocAndInitialize(loc, dynamicmemTy_1d_index, ValueRange{idxes}, rewriter);
          comet_debug() << " ";
          comet_vdump(alloc_size);

          alloc_sizes_vec.push_back(alloc_size);
          Value alloc_size_cast = rewriter.create<memref::CastOp>(loc, unrankedMemTy_index, alloc_size);
          alloc_sizes_cast_vec.push_back(alloc_size_cast);
        }

        for (unsigned int i = sp_decl.getDimArrayCount(); i < sp_decl.getValueArrayPos(); i++)
        {
          std::vector<Value> idxes;
          idxes.push_back(array_sizes[i]);
          Value alloc_size = insertAllocAndInitialize(loc, dynamicmemTy_1d_f64, ValueRange{idxes}, rewriter);
          comet_debug() << " ";
          comet_vdump(alloc_size);
          alloc_sizes_vec.push_back(alloc_size);
          Value alloc_size_cast = rewriter.create<memref::CastOp>(loc, unrankedMemTy_f64, alloc_size);
          alloc_sizes_cast_vec.push_back(alloc_size_cast);
        }

        /// Now, setup the runtime calls to read the input matrices (e.g., read_input_3D_f64)
        if (rank_size == 2)
        { /// 2D
          std::string read_input_str;
          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            read_input_str = "read_input_2D_f32";
          }
          else
          {
            read_input_str = "read_input_2D_f64";
          }
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
          if (VALUETYPE.compare(0, 3, "f32") == 0)
          {
            read_input_str = "read_input_3D_f32";
          }
          else
          {
            read_input_str = "read_input_3D_f64";
          }
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
          Value tensorLoad = rewriter.create<ToTensorOp>(loc, alloc_sizes_vec[i]);
          alloc_tensor_vec.push_back(tensorLoad);
        }

        /// create sptensor_construct
        SmallVector<mlir::Type, 1> elementTypes;
        for (unsigned int i = 0; i < sp_decl.getTotalArrayCount(); i++)
        {
          elementTypes.push_back(alloc_tensor_vec[i].getType());
        }
        for (unsigned int i = 0; i < 5 * rank_size + 1; i++)
        {
          elementTypes.push_back(array_sizes[i].getType());
        }

        auto ty = tensorAlgebra::SparseTensorType::get(elementTypes);

        Value sptensor;
        if (rank_size == 2)
        {
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, ValueRange{alloc_tensor_vec[0], alloc_tensor_vec[1], /// A1
                                                                                                 alloc_tensor_vec[2], alloc_tensor_vec[3], /// A1_tile
                                                                                                 alloc_tensor_vec[4], alloc_tensor_vec[5], /// A2
                                                                                                 alloc_tensor_vec[6], alloc_tensor_vec[7], /// A2_tile
                                                                                                 alloc_tensor_vec[8], array_sizes[0], array_sizes[1], array_sizes[2], array_sizes[3], array_sizes[4], array_sizes[5], array_sizes[6], array_sizes[7], array_sizes[8], array_sizes[9], array_sizes[10]},
                                                                             2);
        }
        else if (rank_size == 3)
        {
          sptensor = rewriter.create<tensorAlgebra::SparseTensorConstructOp>(loc, ty, ValueRange{alloc_tensor_vec[0], alloc_tensor_vec[1],   /// A1
                                                                                                 alloc_tensor_vec[2], alloc_tensor_vec[3],   /// A1_tile
                                                                                                 alloc_tensor_vec[4], alloc_tensor_vec[5],   /// A2
                                                                                                 alloc_tensor_vec[6], alloc_tensor_vec[7],   /// A2_tile
                                                                                                 alloc_tensor_vec[8], alloc_tensor_vec[9],   /// A3
                                                                                                 alloc_tensor_vec[10], alloc_tensor_vec[11], /// A3_tile
                                                                                                 alloc_tensor_vec[12], array_sizes[0], array_sizes[1], array_sizes[2], array_sizes[3], array_sizes[4], array_sizes[5], array_sizes[6], array_sizes[7], array_sizes[8], array_sizes[9], array_sizes[10], array_sizes[11], array_sizes[12], array_sizes[13], array_sizes[14], array_sizes[15], array_sizes[16], array_sizes[17], array_sizes[18]},
                                                                             3);
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
                        tensorAlgebra::IndexLabelOp,
                        tensorAlgebra::ScalarOp,
                        tensorAlgebra::SparseTensorConstructOp>();

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
                        tensorAlgebra::ScalarOp,
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
