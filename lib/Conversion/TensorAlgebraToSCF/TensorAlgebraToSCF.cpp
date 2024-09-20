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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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
      op->replaceAllUsesWith(rewriter.create<ToTensorOp>(op->getLoc(),alloc));
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

        auto inputTensorLoadOp = cast<ToTensorOp>(op->getOperand(0).getDefiningOp());
        auto inputMemref = inputTensorLoadOp.getMemref();

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
        auto outputMemref = lhs.getDefiningOp()->getOperand(0);
        rewriter.create<linalg::TransposeOp>(loc, inputMemref, outputMemref, llvm::ArrayRef<int64_t>(allPerms[1]));
        Value res_value = rewriter.create<ToTensorOp>(loc, outputMemref);

        op.replaceAllUsesWith(res_value);
        rewriter.eraseOp(op);

        return success();
      }
      else
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
        FloatType f64Type = FloatType::getF64(ctx);

        Value input_perm_num = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(pnum[0]));
        Value output_perm_num = rewriter.create<ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(pnum[1]));

        Type unrankedMemrefType_f64 = UnrankedMemRefType::get(f64Type, 0);
        Type unrankedMemrefType_index = UnrankedMemRefType::get(indexType, 0);

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
          auto tensor_rank_attr = tensors[n].getDefiningOp()->getAttr("tensor_rank");
          auto tensor_rank_int_attr = cast<IntegerAttr>(tensor_rank_attr);
          unsigned int tensor_rank = tensor_rank_int_attr.getValue().getLimitedValue();
          comet_debug() << "ATTR_Val: " << tensor_rank << "\n";

          comet_debug() << " tensor_rank: " << tensor_rank << "\n";
          comet_debug() << " tensor[n]: "
                        << "\n";
          comet_pdump(tensors[n].getDefiningOp());

          for (unsigned int i = 0; i < 4 * tensor_rank + 1; i++)
          {
            auto tensorload_op = tensors[n].getDefiningOp()->getOperand(i);
            comet_debug() << " tensorload_op "
                          << "\n";
            comet_vdump(tensorload_op);

            auto alloc_op = tensorload_op.getDefiningOp()->getOperand(0);
            comet_debug() << " alloc_op "
                          << "\n";
            comet_vdump(alloc_op);

            if (i < 4 * tensor_rank)
            {
              /// indexes crd's
              mlir::Value v = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_index, alloc_op);
              alloc_sizes_cast_vecs[n].push_back(v);
            }
            else
            {
              /// NNZ vals
              mlir::Value v = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_f64, alloc_op);
              alloc_sizes_cast_vecs[n].push_back(v);
            }
          }

          auto memrefload_op = tensors[n].getDefiningOp()->getOperand(tensors[n].getDefiningOp()->getNumOperands() - 1);
          allocs_for_sparse_tensors[n].push_back(memrefload_op);
          comet_debug() << " memrefload_op "
                        << "\n";
          comet_vdump(memrefload_op);
        }

        Value last_dim_size_alloc = allocs_for_sparse_tensors[0][0].getDefiningOp()->getOperand(0);
        comet_debug() << "Alloc for last dim size:\n";
        comet_vdump(last_dim_size_alloc);

        mlir::Value sparse_tensor_desc = rewriter.create<memref::CastOp>(loc, unrankedMemrefType_index, last_dim_size_alloc);
        comet_debug() << "Sparse tensor descriptive to extract row/col values:\n";
        comet_vdump(sparse_tensor_desc);

        auto tensor_rank_attr = tensors[0].getDefiningOp()->getAttr("tensor_rank");
        auto tensor_rank_int_attr = cast<IntegerAttr>(tensor_rank_attr);
        unsigned int rank_size = tensor_rank_int_attr.getValue().getLimitedValue();
        comet_debug() << "ATTR_Val: Rank_size: " << rank_size << "\n";

        /// dim format of input tensor
        std::vector<Value>
            dim_formatIn = mlir::tensorAlgebra::getFormatsValueInt(formats_strIn, rank_size, rewriter, loc, i32Type);

        /// dim format of output tensor
        std::vector<Value>
            dim_formatOut = mlir::tensorAlgebra::getFormatsValueInt(formats_strOut, rank_size, rewriter, loc, i32Type);

        if (rank_size == 2)
        { /// 2D
          auto transpose2DF64Func = FunctionType::get(ctx,
                                                      {i32Type, i32Type, i32Type, i32Type,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_f64,
                                                       i32Type, i32Type, i32Type, i32Type,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_f64,
                                                       unrankedMemrefType_index},
                                                      {});

          std::string func_name = "transpose_2D_f64";
          if (!hasFuncDeclaration(module, func_name))
          {
            transpose_func = mlir::func::FuncOp::create(loc, func_name, transpose2DF64Func, ArrayRef<NamedAttribute>{});
            transpose_func.setPrivate();
            module.push_back(transpose_func);
          }
          comet_debug() << "alloc_sizes_vec: " << alloc_sizes_cast_vecs.size() << "\n";
          comet_debug() << "alloc_sizes_vec[0]: " << alloc_sizes_cast_vecs[0].size() << "\n";
          comet_debug() << "alloc_sizes_vec[1]: " << alloc_sizes_cast_vecs[1].size() << "\n";
          comet_debug() << "dim_formatIn: " << dim_formatIn.size() << "\n";
          comet_debug() << "dim_formatOut: " << dim_formatOut.size() << "\n";

          rewriter.create<func::CallOp>(loc, func_name, SmallVector<Type, 2>{},
                                        ValueRange{dim_formatIn[0], dim_formatIn[1], dim_formatIn[2], dim_formatIn[3],
                                                   alloc_sizes_cast_vecs[0][0], alloc_sizes_cast_vecs[0][1],
                                                   alloc_sizes_cast_vecs[0][2], alloc_sizes_cast_vecs[0][3],
                                                   alloc_sizes_cast_vecs[0][4], alloc_sizes_cast_vecs[0][5],
                                                   alloc_sizes_cast_vecs[0][6], alloc_sizes_cast_vecs[0][7],
                                                   alloc_sizes_cast_vecs[0][8],
                                                   dim_formatOut[0], dim_formatOut[1], dim_formatOut[2], dim_formatOut[3],
                                                   alloc_sizes_cast_vecs[1][0], alloc_sizes_cast_vecs[1][1],
                                                   alloc_sizes_cast_vecs[1][2], alloc_sizes_cast_vecs[1][3],
                                                   alloc_sizes_cast_vecs[1][4], alloc_sizes_cast_vecs[1][5],
                                                   alloc_sizes_cast_vecs[1][6], alloc_sizes_cast_vecs[1][7],
                                                   alloc_sizes_cast_vecs[1][8],
                                                   sparse_tensor_desc});
        }
        else if (rank_size == 3)
        { /// 3D
          auto transpose3DF64Func = FunctionType::get(ctx,
                                                      {i32Type, i32Type,
                                                       i32Type, i32Type,
                                                       i32Type, i32Type,
                                                       i32Type, i32Type,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_f64,
                                                       i32Type, i32Type,
                                                       i32Type, i32Type,
                                                       i32Type, i32Type,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_index, unrankedMemrefType_index,
                                                       unrankedMemrefType_f64,
                                                       unrankedMemrefType_index},
                                                      {});

          std::string func_name = "transpose_3D_f64";
          if (!hasFuncDeclaration(module, func_name))
          {
            transpose_func = mlir::func::FuncOp::create(loc, func_name, transpose3DF64Func, ArrayRef<NamedAttribute>{});
            transpose_func.setPrivate();
            module.push_back(transpose_func);
          }

          rewriter.create<func::CallOp>(loc, func_name, SmallVector<Type, 2>{},
                                        ValueRange{input_perm_num, output_perm_num,
                                                   dim_formatIn[0], dim_formatIn[1], dim_formatIn[2],
                                                   dim_formatIn[3], dim_formatIn[4], dim_formatIn[5],
                                                   alloc_sizes_cast_vecs[0][0], alloc_sizes_cast_vecs[0][1],
                                                   alloc_sizes_cast_vecs[0][2], alloc_sizes_cast_vecs[0][3],
                                                   alloc_sizes_cast_vecs[0][4], alloc_sizes_cast_vecs[0][5],
                                                   alloc_sizes_cast_vecs[0][6], alloc_sizes_cast_vecs[0][7],
                                                   alloc_sizes_cast_vecs[0][8], alloc_sizes_cast_vecs[0][9],
                                                   alloc_sizes_cast_vecs[0][10], alloc_sizes_cast_vecs[0][11],
                                                   alloc_sizes_cast_vecs[0][12],
                                                   dim_formatOut[0], dim_formatOut[1], dim_formatOut[2],
                                                   dim_formatOut[3], dim_formatOut[4], dim_formatOut[5],
                                                   alloc_sizes_cast_vecs[1][0], alloc_sizes_cast_vecs[1][1],
                                                   alloc_sizes_cast_vecs[1][2], alloc_sizes_cast_vecs[1][3],
                                                   alloc_sizes_cast_vecs[1][4], alloc_sizes_cast_vecs[1][5],
                                                   alloc_sizes_cast_vecs[1][6], alloc_sizes_cast_vecs[1][7],
                                                   alloc_sizes_cast_vecs[1][8], alloc_sizes_cast_vecs[1][9],
                                                   alloc_sizes_cast_vecs[1][10], alloc_sizes_cast_vecs[1][11],
                                                   alloc_sizes_cast_vecs[1][12],
                                                   sparse_tensor_desc});
        }
        else
        {
          llvm::errs() << "ERROR: Tensors greater than 3 are not currently supported.\n";
        }

        rewriter.eraseOp(setOp);
        rewriter.eraseOp(op);
        return success();

      } /// end else sparse tensor
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
      auto f64Type = rewriter.getF64Type();
      auto inputType = op->getOperand(0).getType();

      /// Allocate memory for the result and initialized it
      auto cst_zero = rewriter.create<ConstantIndexOp>(loc, 0); /// need to access res alloc
      MemRefType memTy_alloc_res = MemRefType::get({1}, f64Type);
      Value res = rewriter.create<memref::AllocOp>(loc, memTy_alloc_res);
      Value const_f64_0 = rewriter.create<ConstantOp>(loc, f64Type, rewriter.getF64FloatAttr(0));
      std::vector<Value> alloc_zero_loc = {cst_zero};
      rewriter.create<memref::StoreOp>(loc, const_f64_0,
                                       res, alloc_zero_loc);

      comet_vdump(res);

      if (inputType.isa<TensorType>())
      { /// tensor is dense
        comet_debug() << "Input Tensor is dense\n";
        std::vector<Value> indices;
        auto alloc_op = op->getOperand(0).getDefiningOp()->getOperand(0);

        comet_vdump(alloc_op);

        for (unsigned rank = 0; rank < inputType.cast<mlir::TensorType>().getRank(); rank++)
        {
          auto dimSize = inputType.cast<mlir::TensorType>().getDimSize(rank);
          Value upperBound;
          if (dimSize == ShapedType::kDynamic)
          {
            comet_debug() << " This dimension is a dynamic size\n";

            comet_vdump(alloc_op);
            auto memRefType = alloc_op.getType().dyn_cast<MemRefType>();
            unsigned dynamicDimPos = memRefType.getDynamicDimIndex(rank);
            comet_debug() << " dynamicDimPos: " << dynamicDimPos << "\n";
            upperBound = alloc_op.getDefiningOp()->getOperand(dynamicDimPos);
          }
          else
          {
            upperBound = rewriter.create<ConstantIndexOp>(loc, dimSize);
          }
          auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
          auto step = rewriter.create<ConstantIndexOp>(loc, 1);
          /// create for loops
          auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
          indices.push_back(loop.getInductionVar());
          rewriter.setInsertionPointToStart(loop.getBody());
        }

        /// Build loop body
        auto load_rhs = rewriter.create<memref::LoadOp>(loc, alloc_op, indices);
        auto res_load = rewriter.create<memref::LoadOp>(loc, res, alloc_zero_loc);
        auto reduced = rewriter.create<AddFOp>(loc, load_rhs, res_load);
        rewriter.create<memref::StoreOp>(loc, reduced, res, alloc_zero_loc);
      }
      else
      { /// sparse tensor type
        assert(inputType.isa<SparseTensorType>());
        comet_debug() << "Input Tensor is sparse\n";

        comet_pdump(op);
        assert(isa<tensorAlgebra::SparseTensorConstructOp>(op->getOperand(0).getDefiningOp()));
        tensorAlgebra::SparseTensorConstructOp sp_op = cast<tensorAlgebra::SparseTensorConstructOp>(op->getOperand(0).getDefiningOp());

        int tensorRanks = sp_op.getTensorRank();
        comet_debug() << " tensorRank: " << tensorRanks << " \n";
        comet_debug() << "Tensor to reduce:\n";
        comet_pdump(op->getOperand(0).getDefiningOp());

        ///  create the lowerBound, upperbound and step for loop
        int indexValueSize = sp_op.getIndexValueSize();
        comet_debug() << "indexValueSize in SparseTensorConstructOp:" << indexValueSize << "\n";

        auto loadOpForNNZ = op->getOperand(0).getDefiningOp()->getOperand(indexValueSize);
        comet_debug() << "Corresponding AllocOp from SparseTensorConstructOp:\n";
        comet_vdump(loadOpForNNZ);
        auto memAllocForNNZ = loadOpForNNZ.getDefiningOp()->getOperand(0);
        comet_debug() << "Corresponding MemAllocOp for NNZ:\n";
        comet_vdump(memAllocForNNZ);

        MemRefType resultMemTy = memAllocForNNZ.getDefiningOp()->getResult(0).getType().cast<MemRefType>();
        auto memRefRank = resultMemTy.getRank();
        comet_debug() << "memRefRank for alloc: " << memRefRank << "\n";
        assert(memRefRank == 1); /// Memref rank should be 1

        auto memRefDimSize = resultMemTy.getDimSize(memRefRank - 1);
        comet_debug() << "memRefDimSize for alloc: " << memRefDimSize << "\n";

        Value upperBound;
        if (memRefDimSize == 1) /// size of value array comes from temporary sparse tensor and Dimsize of alloc is one
        {
          upperBound = rewriter.create<memref::LoadOp>(loc, memAllocForNNZ, alloc_zero_loc);
        }
        else
        {
          /// size of value array comes from read_input_sizes_2D_f64, and alloc dimsize can be only expected size
          auto expectedMemRefSize = sp_op.getTotalParamCount();
          comet_debug() << "tensorRanks: " << tensorRanks << "\n";
          comet_debug() << "expectedMemRefSize: " << expectedMemRefSize << "\n";
          assert(memRefDimSize == expectedMemRefSize);
          upperBound = op->getOperand(0).getDefiningOp()->getOperand(indexValueSize);
        }
        comet_debug() << "Upper Bound:\n";
        comet_vdump(upperBound);
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);

        /// create for loops
        auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

        auto insertPt = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(loop.getBody());

        /// Build loop body
        int indexValuePtr = (tensorRanks * 4); /// 4 corresponding to pos, crd
        auto alloc_op = op->getOperand(0).getDefiningOp()->getOperand(indexValuePtr).getDefiningOp()->getOperand(0);
        comet_debug() << " ValueAllocOp";
        comet_vdump(alloc_op);
        std::vector<Value> indices = {loop.getInductionVar()};
        auto load_rhs = rewriter.create<memref::LoadOp>(loc, alloc_op, indices);
        auto res_load = rewriter.create<memref::LoadOp>(loc, res, alloc_zero_loc);
        auto reduce = rewriter.create<AddFOp>(loc, load_rhs, res_load);
        rewriter.create<memref::StoreOp>(loc, reduce, res, alloc_zero_loc);

        /// need to restore the insertion point to the previous point
        rewriter.restoreInsertionPoint(insertPt);
        comet_vdump(loop);
      }

      /// Important to replace all uses of this operation with the new one, otherwise, the current op won't be lowered.
      op.replaceAllUsesWith(res);
      rewriter.eraseOp(op);

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

      if (auto toTensorOp = llvm::dyn_cast_if_present<ToTensorOp>(rhs.getDefiningOp()))
      {
        rhs = toTensorOp.getMemref();
        // comet_debug() << "RHS is a tensor\n";
        // rhs = rewriter.create<memref::LoadOp>(loc, rhs, alloc_zero_loc);
        // comet_vdump(rhs);
      }
      if (auto toTensorOp = llvm::dyn_cast_if_present<ToTensorOp>(lhs.getDefiningOp()))
      {
        lhs = toTensorOp.getMemref();
        // comet_debug() << "RHS is a tensor\n";
        // rhs = rewriter.create<memref::LoadOp>(loc, rhs, alloc_zero_loc);
        // comet_vdump(rhs);
      }
      // if (lhsType.isa<MemRefType>())
      // {
      //   comet_debug() << "LHS is a tensor\n";
      //   lhs = rewriter.create<memref::LoadOp>(loc, lhs, alloc_zero_loc);
      // }

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
          if(auto toTensor = mlir::dyn_cast_or_null<ToTensorOp>(res.getDefiningOp()))
          {
            res = toTensor.getMemref();
          }
          comet_debug() << "Result from SetOp:\n";
          comet_vdump(res);
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
//      Value res_val;
      if (op_attr.compare("+") == 0)
      {
        rewriter.create<linalg::AddOp>(loc, ValueRange{lhs, rhs}, ValueRange(res));
        // res_val = rewriter.create<AddFOp>(loc, lhs, rhs);
      }
      else if (op_attr.compare("-") == 0)
      {
        rewriter.create<linalg::SubOp>(loc, ValueRange{lhs, rhs}, ValueRange(res));
        // res_val = rewriter.create<SubFOp>(loc, lhs, rhs);
      }
      else if (op_attr.compare("*") == 0)
      {
        rewriter.create<linalg::MulOp>(loc, ValueRange{lhs, rhs}, ValueRange(res));
        // res_val = rewriter.create<MulFOp>(loc, lhs, rhs);
      }
      else if (op_attr.compare("/") == 0)
      {
        rewriter.create<linalg::DivOp>(loc, ValueRange{lhs, rhs}, ValueRange(res));
        // res_val = rewriter.create<DivFOp>(loc, lhs, rhs);
      }
      else
      {
        llvm::errs() << "ERROR: Unsuported Operation\n";
      }

      comet_vdump(res_val);
      /// store res_val to res
      // rewriter.create<linalg::CopyOp>(loc, res_val, res);
      // [[maybe_unused]] auto storeOp = rewriter.create<memref::StoreOp>(loc, res_val, res, alloc_zero_loc);
      comet_vdump(storeOp);

      op.replaceAllUsesWith(res);
      rewriter.eraseOp(op);
      return success();
    }
  }; /// ScalarOpsLowering

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
                         bufferization::BufferizationDialect>();

  target.addLegalOp<func::CallOp>();

  /// Now that the conversion target has been defined, we just need to provide
  /// the set of patterns that will lower the TA operations.

  RewritePatternSet patterns(&getContext());
  patterns.insert<TensorTransposeLowering,
                  ReduceOpLowering,
                  ScalarOpsLowering,
                  ConstantOpLowering>(&getContext());
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
