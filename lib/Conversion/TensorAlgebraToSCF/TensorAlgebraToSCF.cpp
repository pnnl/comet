//===- TensorOpsLowering.cpp------===//
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
// This file implements the lowering of some TA operations such as elementwise tensor addition, subtract, reduce, etc.
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Conversion/TensorAlgebraToSCF/TensorAlgebraToSCF.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::tensorAlgebra;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

int64_t perm2num(std::vector<int64_t> vec)
{
  int64_t psum = 0;
  int64_t sz = vec.size();
  int64_t decimal = pow(10, (sz - 1));

  for (auto n : vec)
  {
    psum += n * decimal;
    decimal /= 10;
  }

  return psum;
}

//===----------------------------------------------------------------------===//
/// TensorOps lowering RewritePatterns
//===----------------------------------------------------------------------===//
namespace
{
  struct ConstantOpLowering : public OpRewritePattern<tensorAlgebra::DenseConstantOp>
  {
    using OpRewritePattern<tensorAlgebra::DenseConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensorAlgebra::DenseConstantOp op,
                                  PatternRewriter &rewriter) const final
    {
      comet_debug() << "ConstantOpLowering starts\n";
      DenseElementsAttr constantValue = op.getValue();
      Location loc = op.getLoc();

      tensorAlgebra::TensorSetOp setnewop;
      bool user_setOp = false;
      for (auto u : op.getOperation()->getResult(0).getUsers())
      {
        if (isa<tensorAlgebra::TensorSetOp>(u))
        {
          setnewop = cast<tensorAlgebra::TensorSetOp>(u);
          user_setOp = true;
        }
      }

      /// When lowering the constant operation, we allocate and assign the constant
      /// values to a corresponding memref allocation.
      auto tensorType = cast<TensorType>(op.getType());
      auto memRefType = convertTensorToMemRef(tensorType);

      comet_debug() << "User_setop: " << user_setOp << "/n";
      Value alloc;
      if (user_setOp)
      {
        if (isa<ToTensorOp>(setnewop.getOperand(1).getDefiningOp()))
        {
          Operation *tensorload = cast<ToTensorOp>(setnewop.getOperand(1).getDefiningOp());
          auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
          comet_vdump(alloc_op);
          alloc = alloc_op;
        }
        else {
          alloc = rewriter.create<memref::AllocOp>(loc, memRefType);
        }
      }
      else
      {
        alloc = rewriter.create<memref::AllocOp>(loc, memRefType);
      }

      /// We will be generating constant indices up-to the largest dimension.
      /// Create these constants up-front to avoid large amounts of redundant
      /// operations.
      auto valueShape = memRefType.getShape();
      auto constTensor = op.getValue().getType().cast<mlir::TensorType>();
      if(constTensor.getRank() == 1 && constTensor.getDimSize(0) == 1)
      {
        auto float_attr = *constantValue.getValues<FloatAttr>().begin();
        auto f_val = float_attr.getValue();
        auto val = rewriter.create<ConstantFloatOp>(op->getLoc(), f_val, rewriter.getF64Type());
        rewriter.create<linalg::FillOp>(op->getLoc(), ValueRange(val), ValueRange(alloc));
      }
      else 
      {
        SmallVector<Value, 8> constantIndices;

        if (!valueShape.empty())
        {
          for (auto i : llvm::seq<int64_t>(
                  0, *std::max_element(valueShape.begin(), valueShape.end())))
            constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
        }
        else
        {
          /// This is the case of a tensor of rank 0.
          constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        }
        /// The constant operation represents a multi-dimensional constant, so we
        /// will need to generate a store for each of the elements. The following
        /// functor recursively walks the dimensions of the constant shape,
        /// generating a store when the recursion hits the base case.
        SmallVector<Value, 2> indices;
        auto valueIt = constantValue.getValues<FloatAttr>().begin();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension)
        {
          /// The last dimension is the base case of the recursion, at this point
          /// we store the element at the given index.
          if (dimension == valueShape.size())
          {
            rewriter.create<memref::StoreOp>(
                loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
                llvm::ArrayRef(indices));
            return;
          }

          /// Otherwise, iterate over the current dimension and add the indices to
          /// the list.
          for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i)
          {
            indices.push_back(constantIndices[i]);
            storeElements(dimension + 1);
            indices.pop_back();
          }
        };

        /// Start the element storing recursion from the first dimension.
        storeElements(/*dimension=*/0);
      }

      /// Replace this operation with the generated alloc.
      op->replaceAllUsesWith(rewriter.create<ToTensorOp>(op->getLoc(), alloc, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
      rewriter.eraseOp(op);
      comet_debug() << "ConstantOpLowering ends\n";
      return success();
    }
  };

  //===----------------------------------------------------------------------===//
  /// Lowering dense and sparse tensor transpose
  //===----------------------------------------------------------------------===//
  struct TensorTransposeLowering : public OpRewritePattern<tensorAlgebra::TransposeOp>
  {
    using OpRewritePattern<tensorAlgebra::TransposeOp>::OpRewritePattern;
    /**
     * @brief :
     * Step 1: Identify the tensor sparsity property - dense or sparse
     * Step 2: Write auto gen mlir code algorithm
     *        - If it is dense, lower transposeOp to SCF
     *        - If input sparse tensor format CSR and the output tensor format CSC - generate code in SCF
     *        - For the rest of the cases, we need a sorting algorithm to reorder coordinate after transpose.
     *          In this case, make a call to utility library for sorting.
     */
    LogicalResult matchAndRewrite(tensorAlgebra::TransposeOp op,
                                  PatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::TransposeOp>(op));

      auto module = op->getParentOfType<ModuleOp>();
      Location loc = op.getLoc();
      comet_vdump(op);
      auto *ctx = op->getContext();
      auto inputType = op->getOperand(0).getType();
      auto  inputTensor = op->getOperand(0);

      /// Get tensor contraction expression through analyzing the index map
      ArrayAttr indexMaps = op.getIndexingMaps();
      std::vector<std::vector<int64_t>> allPerms = getAllPerms(indexMaps);

      /// There are tensors for transpose operation: input and output tensors
      unsigned int tensors_num = 2;
      tensorAlgebra::TensorSetOp setOp;
      Value lhs;

      if (inputType.isa<TensorType>())
      { /// for dense
        comet_debug() << "Dense transpose\n";

        
        // auto inputTensorLoadOp = cast<ToTensorOp>(op->getOperand(0).getDefiningOp());
        // auto inputMemref = inputTensorLoadOp.getMemref();

        for (auto u : op.getOperation()->getResult(0).getUsers())
        {
          if (isa<tensorAlgebra::TensorSetOp>(u))
          {
            setOp = cast<tensorAlgebra::TensorSetOp>(u);
            Value dstTensor = u->getOperand(1);
            lhs = dstTensor;
          }
        }

        comet_vdump(lhs);
        // auto outputMemref = lhs.getDefiningOp()->getOperand(0);
        auto la_transpose = rewriter.create<linalg::TransposeOp>(loc, inputTensor, lhs, llvm::ArrayRef<int64_t>(allPerms[1]));
        // Value res_value = rewriter.create<ToTensorOp>(loc, outputMemref, rewriter.getUnitAttr(), rewriter.getUnitAttr());
        rewriter.replaceOp(op, la_transpose.getResults());
        return success();
      }
      else if(auto spType = mlir::dyn_cast<SparseTensorType>(inputType))
      { /// for sparse tensors
        int64_t pnum[2];
        /// print allPerms
        int i = 0;
        for (auto perm : allPerms)
        { /// lhs, rhs: from left to right order
          pnum[i] = perm2num(perm);
          i++;
        }

        ArrayAttr opFormatsArrayAttr = op.getFormats();
        std::string formats_strIn(opFormatsArrayAttr[0].cast<mlir::StringAttr>().getValue());
        std::string formats_strOut(opFormatsArrayAttr[1].cast<mlir::StringAttr>().getValue());
        IntegerType i32Type = IntegerType::get(ctx, 32);
        IndexType indexType = IndexType::get(ctx);

        Value input_perm_num = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(pnum[0]));
        Value output_perm_num = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(pnum[1]));

        UnrankedMemRefType unrankedMemrefType_float = UnrankedMemRefType::get(spType.getElementType(), 0);
        Type unrankedMemrefType_index = UnrankedMemRefType::get(indexType, 0);
        Type unrankedMemrefType_indices_type = UnrankedMemRefType::get(spType.getIndicesType(), 0);

        mlir::func::FuncOp transpose_func; /// runtime call

        std::vector<std::vector<mlir::Value>> alloc_sizes_cast_vecs{tensors_num};
        std::vector<std::vector<mlir::Value>> allocs_for_sparse_tensors{tensors_num};
        std::vector<mlir::Value> tensors = {op.getOperation()->getOperand(0)};

        for (auto u : op.getOperation()->getResult(0).getUsers())
        {
          if (isa<tensorAlgebra::TensorSetOp>(u))
          {
            setOp = cast<tensorAlgebra::TensorSetOp>(u);
            mlir::Value lhs = setOp->getOperand(1); /// dest tensor is the 2nd
            tensors.push_back(lhs);
          }
        }

        for (unsigned int n = 0; n < tensors_num; n++)
        {
          auto tensor_rank = tensors[n].getType().cast<ShapedType>().getRank();
          comet_debug() << "ATTR_Val: " << tensor_rank << "\n";

          comet_debug() << " tensor_rank: " << tensor_rank << "\n";
          comet_debug() << " tensor[n]: "
                        << "\n";
          comet_pdump(tensors[n].getDefiningOp());
          auto tensor = tensors[n];
          auto tensor_type = tensors[n].getType().cast<SparseTensorType>();
          for(int i = 0; i < tensor_rank; i++)
          {
            Value pos = rewriter.create<SpTensorGetDimPos>(loc, tensor, rewriter.getI32IntegerAttr(i));
            ShapedType shape = pos.getType().cast<ShapedType>();
            Value pos_memref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(shape.getShape(), shape.getElementType()), pos);
            Value pos_v = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_indices_type, pos_memref);
            alloc_sizes_cast_vecs[n].push_back(pos_v);

            mlir::Value crd = rewriter.create<SpTensorGetDimCrd>(loc, tensor, rewriter.getI32IntegerAttr(i));
            mlir::Value crd_memref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(shape.getShape(), shape.getElementType()), crd);
            mlir::Value crd_v = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_indices_type, crd_memref);
            alloc_sizes_cast_vecs[n].push_back(crd_v);
            
            Value block_pos = rewriter.create<SpTensorGetDimBlockPos>(loc, tensor, rewriter.getI32IntegerAttr(i));
            Value block_pos_memref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(shape.getShape(), shape.getElementType()), block_pos);
            Value block_pos_v = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_indices_type, block_pos_memref);
            alloc_sizes_cast_vecs[n].push_back(block_pos_v);
            
            mlir::Value block_crd = rewriter.create<SpTensorGetDimBlockCrd>(loc, tensor, rewriter.getI32IntegerAttr(i));
            mlir::Value block_crd_memref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get(shape.getShape(), shape.getElementType()), block_crd);
            mlir::Value block_crd_v = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_indices_type, block_crd_memref);
            alloc_sizes_cast_vecs[n].push_back(block_crd_v);
          }
          Value vals = rewriter.create<SpTensorGetVals>(loc, tensor);
          Value vals_memref = rewriter.create<bufferization::ToMemrefOp>(loc, MemRefType::get({ShapedType::kDynamic}, tensor_type.getElementType()), vals);
          Value vals_v = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_float, vals_memref);
          alloc_sizes_cast_vecs[n].push_back(vals_v);

          auto dims_tensor = mlir::cast<SparseTensorConstructOp>(tensors[n].getDefiningOp()).getDims();
          auto dims_memref = rewriter.create<ToMemrefOp>(loc, MemRefType::get(dims_tensor.getType().getShape(), dims_tensor.getType().getElementType()), dims_tensor);
          allocs_for_sparse_tensors[n].push_back(dims_memref);
          comet_debug() << " memrefload_op "
                        << "\n";
          comet_vdump(memrefload_op);
        }

        Value last_dim_size_alloc = allocs_for_sparse_tensors[0][0];
        comet_debug() << "Alloc for last dim size:\n";
        comet_vdump(last_dim_size_alloc);

        mlir::Value sparse_tensor_desc = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_index, last_dim_size_alloc);
        comet_debug() << "Sparse tensor descriptive to extract row/col values:\n";
        comet_vdump(sparse_tensor_desc);

        auto tensor_rank_attr = tensors[0].getDefiningOp()->getAttr("tensor_rank");
        auto tensor_rank_int_attr = cast<IntegerAttr>(tensor_rank_attr);
        unsigned int rank_size = tensor_rank_int_attr.getValue().getLimitedValue();
        comet_debug() << "ATTR_Val: Rank_size: " << rank_size << "\n";
        assert(rank_size <= 3 && rank_size >=2 && "Rank size not supported");

        /// dim format of input tensor
        std::vector<Value>
            dim_formatIn = mlir::tensorAlgebra::getFormatsValueInt(formats_strIn, rank_size, rewriter, loc, i32Type);

        /// dim format of output tensor
        std::vector<Value>
            dim_formatOut = mlir::tensorAlgebra::getFormatsValueInt(formats_strOut, rank_size, rewriter, loc, i32Type);
        std::string func_name = "transpose_" + std::to_string(rank_size)+ "D" + "_" + "f" + std::to_string(unrankedMemrefType_float.getElementType().getIntOrFloatBitWidth()) + "_i"+std::to_string(spType.getIndicesType().getWidth());
        llvm::SmallVector<Type, 28> funcArgTypes;
        if( rank_size == 3)
        {
          funcArgTypes.push_back(i32Type);
          funcArgTypes.push_back(i32Type);
        }
        for(int k = 0; k < 2; k++)
        {
          for(unsigned i = 0; i < rank_size * 2; i++)
          { 
            funcArgTypes.push_back(i32Type);
          }
          for(unsigned i = 0; i < rank_size * 4; i++)
          {
            funcArgTypes.push_back(unrankedMemrefType_indices_type);
          }
          funcArgTypes.push_back(unrankedMemrefType_float);
        }
        funcArgTypes.push_back(unrankedMemrefType_index);
        auto transposeFunc = FunctionType::get(ctx, TypeRange(funcArgTypes), {});
        if (!hasFuncDeclaration(module, func_name))
        {
          transpose_func = mlir::func::FuncOp::create(loc, func_name, transposeFunc, ArrayRef<NamedAttribute>{});
          transpose_func.setPrivate();
          module.push_back(transpose_func);
        }
        
        std::vector<Value> allInputs;
        if(rank_size == 3)
        {
          allInputs.push_back(input_perm_num);
          allInputs.push_back(output_perm_num);
        }
        allInputs.insert(allInputs.end(), dim_formatIn.begin(), dim_formatIn.end());
        allInputs.insert(allInputs.end(), alloc_sizes_cast_vecs[0].begin(), alloc_sizes_cast_vecs[0].end());
        allInputs.insert(allInputs.end(), dim_formatOut.begin(), dim_formatOut.end());
        allInputs.insert(allInputs.end(), alloc_sizes_cast_vecs[1].begin(), alloc_sizes_cast_vecs[1].end());
        allInputs.push_back(sparse_tensor_desc);
        
        rewriter.create<func::CallOp>(loc, func_name, SmallVector<Type, 2>{}, ValueRange(allInputs) );

        rewriter.eraseOp(setOp);
        rewriter.eraseOp(op);
        return success();

      } /// end else sparse tensor
      else 
      {
        return failure();
      }
    }   /// Tensor TransposeLowering
  };

  //===----------------------------------------------------------------------===//
  /// ReduceOptoSCF RewritePatterns: Reduction operation lowering for sparse and dense tensors
  //===----------------------------------------------------------------------===//
  struct ReduceOpLowering : public OpRewritePattern<tensorAlgebra::ReduceOp>
  {
    using OpRewritePattern<tensorAlgebra::ReduceOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tensorAlgebra::ReduceOp op,
                                  PatternRewriter &rewriter) const final
    {

      assert(isa<tensorAlgebra::ReduceOp>(op));
      comet_debug() << "Lowering Reduce operation to SCF\n";

      Location loc = op.getLoc();
      // auto f64Type = rewriter.getF64Type();
      auto inputType = op->getOperand(0).getType();

      /// Allocate memory for the result and initialized it
      auto cst_zero = rewriter.create<ConstantIndexOp>(loc, 0); /// need to access res alloc
      ShapedType shapeT = mlir::cast<ShapedType>(inputType);
      MemRefType memTy_alloc_res = MemRefType::get({1}, shapeT.getElementType());

      Value res = rewriter.create<memref::AllocOp>(loc, memTy_alloc_res);
      FloatAttr zero;
      if(shapeT.getElementType().isF32())
      {
        zero = rewriter.getF32FloatAttr(0);
      }
      else if(shapeT.getElementType().isF64())
      {
        zero = rewriter.getF64FloatAttr(0);
      }
      else 
      {
        assert(false && "Unexpected type");
      }
      Value const_float_0 = rewriter.create<ConstantOp>(loc, shapeT.getElementType(), zero);
      std::vector<Value> alloc_zero_loc = {cst_zero};
      rewriter.create<memref::StoreOp>(loc, const_float_0,
                                       res, alloc_zero_loc);

      comet_vdump(res);

      if (inputType.isa<TensorType>())
      { /// tensor is dense
        comet_debug() << "Input Tensor is dense\n";
        std::vector<Value> indices;

        comet_vdump(alloc_op);
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);


        for (unsigned rank = 0; rank < inputType.cast<mlir::TensorType>().getRank(); rank++)
        {
          Value upperBound = rewriter.create<tensor::DimOp>(loc, op->getOperand(0), rank);
          
          /// create for loops
          auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
          indices.push_back(loop.getInductionVar());
          rewriter.setInsertionPointToStart(loop.getBody());
        }

        /// Build loop body
        auto load_rhs = rewriter.create<tensor::ExtractOp>(loc, op->getOperand(0), indices);
        auto res_load = rewriter.create<memref::LoadOp>(loc, res, alloc_zero_loc);
        auto reduced = rewriter.create<AddFOp>(loc, load_rhs, res_load);
        rewriter.create<memref::StoreOp>(loc, reduced, res, alloc_zero_loc);
      }
      else
      { /// sparse tensor type
        SparseTensorType sp_tensor_type = inputType.cast<SparseTensorType>();
        comet_debug() << "Input Tensor is sparse\n";

        comet_pdump(op);

        int tensorRanks = sp_tensor_type.getRank();
        comet_debug() << " tensorRank: " << tensorRanks << " \n";
        comet_debug() << "Tensor to reduce:\n";
        comet_pdump(op->getOperand(0).getDefiningOp());
        Value sp_tensor_values = rewriter.create<tensorAlgebra::SpTensorGetVals>(loc, RankedTensorType::get({ShapedType::kDynamic,}, sp_tensor_type.getElementType()), op->getOperand(0));
        Value upperBound = rewriter.create<tensor::DimOp>(loc, sp_tensor_values, 0);

        comet_debug() << "Upper Bound:\n";
        comet_vdump(upperBound);
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);

        /// create for loops
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        auto insertPt = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(loop.getBody());

        /// Build loop body
        std::vector<Value> indices = {loop.getInductionVar()};
        auto load_rhs = rewriter.create<tensor::ExtractOp>(loc, sp_tensor_values, indices);
        auto res_load = rewriter.create<memref::LoadOp>(loc, res, alloc_zero_loc);
        auto reduce = rewriter.create<AddFOp>(loc, load_rhs, res_load);
        rewriter.create<memref::StoreOp>(loc, reduce, res, alloc_zero_loc);

        /// need to restore the insertion point to the previous point
        rewriter.restoreInsertionPoint(insertPt);
        comet_vdump(loop);
      }
      rewriter.setInsertionPoint(op);
      /// Important to replace all uses of this operation with the new one, otherwise, the current op won't be lowered.
      memref::LoadOp load = rewriter.create<memref::LoadOp>(op->getLoc(), res, ValueRange(cst_zero));
      rewriter.replaceOp(op, load);

      return success();
    }
  }; /// ReduceOpLowering

  struct ScalarOpsLowering : public OpRewritePattern<tensorAlgebra::ScalarOp>
  {
    using OpRewritePattern<tensorAlgebra::ScalarOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tensorAlgebra::ScalarOp op,
                                  PatternRewriter &rewriter) const final
    {
      assert(isa<tensorAlgebra::ScalarOp>(op));
      comet_debug() << "ScalarOpsLowering starts\n";

      /// Scalar operation could be between
      ///  1. two tensors (size of 1) and the output with a tensor of size 1
      ///  2. two F64 values and the output will be F64
      Location loc = op.getLoc();
      Value rhs = op.getRhs();
      Value lhs = op.getLhs();

      comet_vdump(rhs);
      comet_vdump(lhs);

      // auto rh = op->getOperand(0).getType();
      // auto lh = op->getOperand(1).getType();

      [[maybe_unused]] auto f64Type = rewriter.getF64Type();
      Value const_index_0 = rewriter.create<ConstantIndexOp>(loc, 0);
      comet_vdump(const_index_0);
      std::vector<Value> alloc_zero_loc = {const_index_0};

      Value res;
      bool res_comes_from_setop = false;
      for (auto u : op.getOperation()->getResult(0).getUsers())
      {
        comet_debug() << "Users:\n";
        comet_pdump(u);
        if (isa<tensorAlgebra::TensorSetOp>(u))
        {
          // u->dump();
          // u->getBlock()->dump();
          res = cast<tensorAlgebra::TensorSetOp>(u).getOperation()->getOperand(1);
          // (++res.getUsers().begin())->dump();
          if(!res.getUsers().empty() &&  isa<TensorSetOp>(*(++res.getUsers().begin())))
          {
            res = cast<tensorAlgebra::TensorSetOp>(*(++res.getUsers().begin())).getRhs();
          }
          res_comes_from_setop = true;
          break;
        }
      }

      assert(res_comes_from_setop && "SetOp is needed to assign the scalar operation result to final variable");

      comet_debug() << "Scalar lowering Final step\n";
      comet_debug() << "RHS, LHS and result:\n";
      comet_vdump(rhs);
      comet_vdump(lhs);
      comet_vdump(res);

      /// Op rhs and lhs
      auto arith_op_attr = op.getOpAttr();
      std::string op_attr(arith_op_attr.getValue());
      comet_debug() << "aritmetic op: " << op_attr << "\n";
      Value res_val;
      if (op_attr.compare("+") == 0)
      {
        res_val = rewriter.create<linalg::AddOp>(loc, ValueRange{lhs, rhs}, ValueRange(res)).getResultTensors()[0];
      }
      else if (op_attr.compare("-") == 0)
      {
        res_val = rewriter.create<linalg::SubOp>(loc, ValueRange{lhs, rhs}, ValueRange(res)).getResultTensors()[0];
      }
      else if (op_attr.compare("*") == 0)
      {
        res_val = rewriter.create<linalg::MulOp>(loc, ValueRange{lhs, rhs}, ValueRange(res)).getResultTensors()[0];
      }
      else if (op_attr.compare("/") == 0)
      {
        res_val = rewriter.create<linalg::DivOp>(loc, ValueRange{lhs, rhs}, ValueRange(res)).getResultTensors()[0];
      }
      else
      {
        llvm::errs() << "ERROR: Unsuported Operation\n";
      }
      
      op.replaceAllUsesWith(res_val);
      rewriter.eraseOp(op);
      return success();
    }
  }; /// ScalarOpsLowering

class ConvertSetOp : public OpConversionPattern<TensorSetOp> {
  using OpConversionPattern<TensorSetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto opAdaptor = llvm::cast<TensorSetOpAdaptor>(adaptor);
    Value lhs = opAdaptor.getLhs();
    Value rhs = opAdaptor.getRhs();
    rewriter.replaceUsesWithIf(rhs, lhs, [&](OpOperand& use) { 
      auto user = use.getOwner();
      auto ancestor = op->getBlock()->findAncestorOpInBlock(*user);
      return (ancestor && op->isBeforeInBlock(ancestor)); 
    });
    rewriter.eraseOp(op);
    return success();
  }
};

} /// end anonymous namespace.

/// This is a partial lowering to linear algebra of the tensor algebra operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the TA dialect.
namespace
{
  struct LowerTensorAlgebraToSCFPass
      : public PassWrapper<LowerTensorAlgebraToSCFPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorAlgebraToSCFPass)
    void runOnOperation() override;
  };
} /// end anonymous namespace.

void LowerTensorAlgebraToSCFPass::runOnOperation()
{
  mlir::func::FuncOp function = getOperation();

  if (function.getName() == "main")
  {
    ///  Verify that the given main has no inputs and results.
    if (function.getNumArguments() || function.getFunctionType().getNumResults())
    {
      function.emitError("expected 'main' to have 0 inputs and 0 results");
      return signalPassFailure();
    }
  }

  /// The first thing to define is the conversion target. This will define the
  /// final target for this lowering.
  ConversionTarget target(getContext());

  /// We define the specific operations, or dialects, that are legal targets for
  /// this lowering. In our case, we are lowering to a combination of the
  /// `LinAlg` and `Standard` dialects.
  target.addLegalDialect<LinalgDialect,
                         scf::SCFDialect,
                         ArithDialect,
                         memref::MemRefDialect,
                         func::FuncDialect,
                         bufferization::BufferizationDialect>();

  target.addLegalDialect<tensorAlgebra::TADialect, indexTree::IndexTreeDialect, tensor::TensorDialect>();
  target.addIllegalOp<tensorAlgebra::TransposeOp, 
                      tensorAlgebra::ReduceOp,
                      tensorAlgebra::ScalarOp,
                      tensorAlgebra::DenseConstantOp, 
                      tensorAlgebra::TensorSetOp>();
  /// Now that the conversion target has been defined, we just need to provide
  /// the set of patterns that will lower the TA operations.

  RewritePatternSet patterns(&getContext());
  patterns.insert<TensorTransposeLowering,
                  ReduceOpLowering,
                  ScalarOpsLowering,
                  ConstantOpLowering,
                  ConvertSetOp>(&getContext());
  /// With the target and rewrite patterns defined, we can now attempt the
  /// conversion. The conversion will signal failure if any of our `illegal`
  /// operations were not converted successfully.

  /// function.dump();
  if (failed(applyPartialConversion(function, target, std::move(patterns))))
  {
    signalPassFailure();
  }
}

/// Create a pass for lowering tensor operations in the TensorAlgebra dialect to other lower level dialects
std::unique_ptr<Pass> mlir::comet::createLowerTensorAlgebraToSCFPass()
{
  return std::make_unique<LowerTensorAlgebraToSCFPass>();
}
