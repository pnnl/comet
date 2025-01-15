//===- LowerIndexTreeIRToSCF.cpp  ------===//
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
// This file implements a lowering of index tree dialect to SCF dialect
//===----------------------------------------------------------------------===//

#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Conversion/IndexTreeToSCF/AbstractLoopOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/SliceAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <iostream>
#include <algorithm>
#include <ostream>
#include <vector>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <queue>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

using llvm::SmallVector;
using llvm::StringRef;
using llvm::SmallDenseMap;

#define DEBUG_TYPE "lowering-it-to-scf"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

namespace comet
{
#define GEN_PASS_DEF_CONVERTINDEXTREETOSCF
#include "comet/Conversion/Passes.h.inc"
} /// namespace comet

namespace
{

  /// Valid semiring operators.
  static const llvm::StringSet<> Semiring_ops{
      "atan2", "div", "eq", "first", "ge", "gt", "hypot",
      "land", "le", "lor", "lt", "max", "minxy", "minus",
      "ne", "pairxy", "plusxy", "pow", "rdiv", "second", "times"};

  /// List of valid semiring operands for reduce op
  static const llvm::StringSet<> Semiring_reduceOps{"any", "land", "lor", "max",
                                                    "minxy", "plusxy", "times",
                                                    "noop"}; /// noop is for monoid op support

  /// Operands' indices in the sptensor_construct function argument list.
  /// ta.sptensor_construct is called to declare a sparse tensor.
  /// The information of the output matrix C
  /// %55 = ta.sptensor_construct(%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %8, %24) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
  /// %91 = ta.sptensor_construct(%73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %11, %12) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
  /// The 20-argument one is the current version
  /**
      sptensor_construct(
          0) A1pos,  /// number of rows
          1) A1crd,  /// discard
          2) A2pos,  /// rowptr array
          3) A2crd,  /// col_id array
          4) Aval, /// data array
          5) A1pos_size,
          6) A1crd_size,
          7) A2pos_size,
          8) A2crd_size,
          9) Aval_size,
          10) dim1_size,
          11) dim2_size,
          ------------------------------
          0) A1pos,  /// number of rows
          1) A1crd,  /// discard
          2) A1tile_pos,
          3) A1tile_crd,
          4) A2pos,  /// rowptr array
          5) A2crd,  /// col_id array
          6) A2tile_pos,
          7) A2tile_crd,
          8) Aval, /// data array
          9) A1pos_size,
          10) A1crd_size,
          11) A1tile_pos_size,
          12) A1tile_crd_size,
          13) A2pos_size,
          14) A2crd_size,
          15) A2tile_pos_size,
          16) A2tile_crd_size,
          17) Aval_size,
          18) dim1_size,
          19) dim2_size,
      )
  */
  enum CSR_sptensor_construct_arguments_indices
  {
    CSR_A1POS = 0, /// number of rows
    CSR_A1CRD = 1, /// discard for CSR
    CSR_A1TILE_POS = 2,
    CSR_A1TILE_CRD = 3,
    CSR_A2POS = 4, /// rowptr array
    CSR_A2CRD = 5, /// col_id array
    CSR_A2TILE_POS = 6,
    CSR_A2TILE_CRD = 7,
    CSR_AVAL = 8, /// data array
    CSR_A1POS_SIZE = 9,
    CSR_A1CRD_SIZE = 10,
    CSR_A1TILE_POS_SIZE = 11,
    CSR_A1TILE_CRD_SIZE = 12,
    CSR_A2POS_SIZE = 13,
    CSR_A2CRD_SIZE = 14,
    CSR_A2TILE_POS_SIZE = 15,
    CSR_A2TILE_CRD_SIZE = 16,
    CSR_AVAL_SIZE = 17,
    CSR_DIM1_SIZE = 18,
    CSR_DIM2_SIZE = 19
  };

  /// MASKING_TYPE to indicate what type of masking is used.
  enum MASKING_TYPE
  {
    NO_MASKING = 0,
    PUSH_BASED_MASKING = 1,
    PULL_BASED_MASKING = 2
  };

  /// class MaksingInfo, passed as a parameter to the formSemiringLoopBody() to indicate if using masking or not.
  struct MaskingInfo
  {
  public:
    MASKING_TYPE mask_type;

    mlir::Value mask_tensor;
    mlir::Value mask_rowptr;
    mlir::Value mask_col;
    mlir::Value mask_val;

    /// TODO(zhen.peng): Pull-based mask info and auxiliary variables.

  public:
    MaskingInfo() : mask_type(NO_MASKING) {}

    ///  MaskingInfo(MASKING_TYPE type_, mlir::Value states_) : maskType(type_), states(states_) { }

    void dump()
    {
      switch (mask_type)
      {
      case NO_MASKING:
        std::cout << "maskType: NO_MASKING\n";
        break;
      case PUSH_BASED_MASKING:
        std::cout << "maskType: PUSH_BASED_MASKING "
                  << "mask_tensor: ";
        mask_tensor.dump();
        ///        std::cout << "maskType: PUSH_BASED_MASKING " << "states: ";
        ///        states.dump();
        break;
      case PULL_BASED_MASKING:
        std::cout << "maskType: PULL_BASED_MASKING ... Not supported";
        break;
      }
    }
  };

   /// ----------------- ///
  /// Add declaration of the function comet_index_func;
  /// ----------------- ///
  void declareSortFunc(ModuleOp &module,
                       MLIRContext *ctx,
                       Location loc)
  {
    IndexType indexType = IndexType::get(ctx);

    /// Declare comet_sort_index()
    auto sort_index_func = FunctionType::get(ctx,
                                             {UnrankedMemRefType::get(indexType, 0), indexType, indexType} /* inputs */, {} /* return */);
    std::string func_name = "comet_sort_index";
    if (!hasFuncDeclaration(module, func_name /* func name */))
    {
      func::FuncOp func_declare = func::FuncOp::create(loc,
                                                       func_name,
                                                       sort_index_func,
                                                       ArrayRef<NamedAttribute>{});
      func_declare.setPrivate();
      module.push_back(func_declare);
    }
  }

  Value getSemiringSecondVal(OpBuilder &builder, Location &loc,
                             llvm::StringRef &semiringSecond, Value &Input0, Value &Input1)
  {

    Value elementWiseResult;
    if (semiringSecond == "times")
    {
      elementWiseResult = builder.create<MulFOp>(loc, Input0, Input1);
    }
    else if (semiringSecond == "first")
    {
      elementWiseResult = Input0;
    }
    else if (semiringSecond == "second")
    {
      elementWiseResult = Input1;
    }
    else if (semiringSecond == "atan2")
    {
      elementWiseResult = builder.create<math::Atan2Op>(loc, Input0, Input1);
    }
    else if (semiringSecond == "div")
    {
      elementWiseResult = builder.create<DivFOp>(loc, Input0, Input1);
    }
    else if (semiringSecond == "eq")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OEQ, Input0, Input1);
    }
    else if (semiringSecond == "ge")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OGE, Input0, Input1);
    }
    else if (semiringSecond == "gt")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
    }
    else if (semiringSecond == "le")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OLE, Input0, Input1);
    }
    else if (semiringSecond == "lt")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
    }
    else if (semiringSecond == "land")
    {
      /// land requires integer type input
      llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                   << "land"
                   << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
      exit(1);
    }
    else if (semiringSecond == "lor")
    {
      /// lor requires integer type input
      llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                   << "lor"
                   << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
      exit(1);
    }
    else if (semiringSecond == "lxor")
    {
      /// lxor requires integer type input
      llvm::errs() << "Not supported semiring operator: "
                   << "lxor"
                   << "\n";
    }
    else if (semiringSecond == "minxy")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
      elementWiseResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringSecond == "max")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
      elementWiseResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringSecond == "ne")
    {
      elementWiseResult = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, Input0, Input1);
    }
    else if (semiringSecond == "minus")
    {
      elementWiseResult = builder.create<SubFOp>(loc, Input0, Input1);
    }
    else if (semiringSecond == "plusxy")
    {
      elementWiseResult = builder.create<AddFOp>(loc, Input0, Input1);
    }
    else if (semiringSecond == "pairxy")
    {
      elementWiseResult = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(1));
    }
    else if (semiringSecond == "pow")
    {
      elementWiseResult = builder.create<math::PowFOp>(loc, Input0, Input1);
    }
    else
    {
      llvm::errs() << "Not supported semiring operator: " << semiringSecond << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
    }

    return elementWiseResult;
  }

  Value getSemiringFirstVal(OpBuilder &builder, Location &loc,
                            llvm::StringRef &semiringFirst, Value &Input0, Value &Input1)
  {

    Value reduceResult;
    if (semiringFirst == "times")
    {
      reduceResult = builder.create<MulFOp>(loc, Input0, Input1);
    }
    else if (semiringFirst == "plusxy")
    {
      reduceResult = builder.create<AddFOp>(loc, Input0, Input1);
    }
    else if (semiringFirst == "minxy")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
      reduceResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringFirst == "max")
    {
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Input0, Input1);
      reduceResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringFirst == "land")
    {
      /// land requires integer type input
      llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                   << "land"
                   << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
    }
    else if (semiringFirst == "lor")
    {
      /// lor requires integer type input
      llvm::errs() << "Not supported semiring operator (only works for int datatypes): "
                   << "lor"
                   << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
    }
    else if (semiringFirst == "any")
    {
      reduceResult = Input1;
    }
    else if (semiringFirst == "noop")
    {
      reduceResult = Input1;
    }
    else
    {
      llvm::errs() << "Not supported semiring operator: " << semiringFirst << "\n";
      /// we should not proceed forward from this point to avoid faulty behavior.
    }

    return reduceResult;
  }

  struct TensorSubsetInfo {
    int64_t dim;
    int64_t tiles;
    int64_t tile_size;
  };


  class IndexTreeInferOutputSets {
    SmallDenseMap<IndexTreeIndicesOp, SmallDenseMap<Value, TensorSubsetInfo>> output_sets;

    public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexTreeInferOutputSets)

    IndexTreeInferOutputSets(Operation* op)
    {
      // TODO: Will not work with workspace op or with sparse tensors!
      // Also will not work with indices that don't align to tensor dims!

      IndexTreeOp tree = llvm::cast<IndexTreeOp>(op);
      for(Value input : tree.getInputs()) {
        for(auto user : input.getUsers()) {
          if(auto lhs = llvm::dyn_cast<IndexTreeLHSOperandOp>(user)) {
            int64_t dim = 0;
            for(Value crd : lhs.getCrds()) {
              if(auto access = crd.getDefiningOp<IndexTreeIndexToTensorOp>()) {
                auto node = access.getIndex().getDefiningOp<IndexTreeIndicesOp>();
                TensorSubsetInfo slice = {dim, -1, -1};
                output_sets[node].insert(std::make_pair(input, slice));
              }
              dim += 1; 
            }
          }
        }
      }
    }

    SmallDenseMap<Value, TensorSubsetInfo> getOutputSets(IndexTreeIndicesOp op)
    {
      if(output_sets.contains(op)){
        return output_sets[op];
      }

      // This also will cause errors!!!
      return SmallDenseMap<Value, TensorSubsetInfo>();
    }

  };

  class IndexVar {
    public:
      virtual Value getCrd(IRRewriter& rewriter) = 0;
      virtual Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) = 0;
      virtual ~IndexVar(){};
  };

  class LoopInfo : public IndexVar {
    protected:
      SmallVector<Value> currentInputs;
      ValueRange results;

    public:
      Operation* loopBody;
      IRMapping map; // Maps values from outer scope (i.e. TreeRegion) to loop

      LoopInfo(ValueRange inputs, ValueRange outputs, Operation* body, IRMapping& ir_map):
        currentInputs(inputs), results(outputs), loopBody(body), map(ir_map) {}

      virtual Value getCrd(IRRewriter& rewriter) = 0;
      virtual Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) = 0;
      virtual void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) = 0;
      virtual ~LoopInfo(){};

      ValueRange getInputs(){return currentInputs;}
      Value getInput(uint32_t idx) {return currentInputs[idx];}
      virtual Value getInput(IRRewriter& rewriter, uint32_t idx, SmallVector<int> dims) {
        return getInput(idx);
      }
      virtual ValueRange getResults() {return results;}
  };

  class SentinelLoopInfo : LoopInfo {
    private:
      indexTree::YieldOp terminator;
    public:
      SentinelLoopInfo(ValueRange inputs, ValueRange outputs, Operation* body, IRMapping& ir_map, indexTree::YieldOp yield_op):
        LoopInfo(inputs, outputs, body, ir_map), terminator(yield_op) {}

      static LoopInfo* build(ValueRange inputs, ValueRange outputs, Operation* body, IRMapping& ir_map, indexTree::YieldOp yield_op) {
        return new SentinelLoopInfo(inputs, outputs, body, ir_map, yield_op);
      }

      virtual Value getCrd(IRRewriter& rewriter) override {return nullptr;}
      virtual Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {return nullptr;}
      virtual void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.updateRootInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class DenseLoopInfo : LoopInfo {
    private:
      Value inductionVar;
      scf::YieldOp terminator;

    public:
      DenseLoopInfo(ValueRange inputs, 
                    ResultRange outputs,
                    Operation* body,
                    IRMapping ir_map,
                    Value i, 
                    scf::YieldOp yield): 
        LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), terminator(yield){}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        Value inductionVar;
        Value lb = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
        Value ub = domain_op->getOperand(0);
        Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(1));
        scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, inputs);
        rewriter.setInsertionPointToStart(for_loop.getBody());
        scf::YieldOp terminator = rewriter.create<scf::YieldOp>(loc, for_loop.getRegionIterArgs());
        rewriter.setInsertionPointAfter(for_loop);
        IRMapping map;
        return new DenseLoopInfo(for_loop.getRegionIterArgs(), for_loop->getResults(), terminator, map, for_loop.getInductionVar(), terminator);
      }

      Value getCrd(IRRewriter& rewriter) override {return inductionVar;}

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(dyn_cast<TensorType>(tensor.getType())){
          return inductionVar;
        } else if(SparseTensorType tt = dyn_cast<SparseTensorType>(tensor.getType())){
          if((TensorFormatEnum)(tt.getFormat()[2 * dim]) == TensorFormatEnum::D) {
            return inductionVar;
          }
          assert(false && "Invalid type passed to DenseLoopInfo getPos");
          return nullptr;
         }
        assert(false && "Invalid type passed to DenseLoopInfo getPos");
        return nullptr;
      }

      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
       rewriter.updateRootInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class DenseParallelLoopInfo : public LoopInfo {
    private:
      Value inductionVar;
      SmallVector<Operation*> terminator_ops;
      SmallDenseMap<Value, TensorSubsetInfo> output_sets;

    public:
      DenseParallelLoopInfo(ValueRange inputs, 
                            ResultRange outputs,
                            Operation* body,
                            IRMapping ir_map,
                            Value i,
                            SmallVector<Operation*>& terminator_ops,
                            SmallDenseMap<Value, TensorSubsetInfo> output_sets): 
        LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), terminator_ops(terminator_ops), output_sets(output_sets) {}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs, SmallDenseMap<Value, TensorSubsetInfo> output_sets)
      {
        auto loc = domain_op->getLoc();
        Value ub = domain_op->getOperand(0);
        scf::ForallOp for_loop = rewriter.create<scf::ForallOp>(
          loc,
          ArrayRef<OpFoldResult>(ub),
          inputs,
          std::nullopt,
          nullptr);
        Value inductionVar = for_loop.getInductionVar(0);
        rewriter.setInsertionPointToStart(for_loop.getBody());

        SmallVector<Value> input_slices;        
        for(auto& input : for_loop.getOutputsMutable())
        {
          auto& os = output_sets.at(input.get());
          RankedTensorType tt = llvm::cast<RankedTensorType>(input.get().getType());
          int64_t nDims = tt.getRank();
          SmallVector<Value> sizes;
          for(int i = 0; i < nDims; i++)
          {
            if(i != os.dim && tt.getDimSize(i) == ShapedType::kDynamic) {
              Value idx = rewriter.create<index::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(i));
              sizes.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.getIndexType(), input.get(), idx));
            }
          }

          SmallVector<int64_t> static_offsets(nDims, 0);
          SmallVector<int64_t> static_sizes(tt.getShape());
          static_offsets[os.dim] = ShapedType::kDynamic;
          static_sizes[os.dim] = 1;

          auto slice = rewriter.create<tensor::ExtractSliceOp>(
            loc,
            RankedTensorType::get(static_sizes, tt.getElementType()),
            for_loop.getTiedBlockArgument(&input),
            ValueRange(inductionVar),
            sizes,
            ValueRange(),
            rewriter.getDenseI64ArrayAttr(static_offsets),
            rewriter.getDenseI64ArrayAttr(static_sizes),
            rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(nDims, 1))
          );
          input_slices.push_back(slice->getResult(0));
          output_sets.insert(std::make_pair(slice->getResult(0), os));
        }

        SmallVector<Operation*> terminator_ops;
        auto par_op = for_loop.getTerminator();
        auto slice = input_slices.begin();
        for(auto& input : for_loop.getOutputsMutable()) {
          rewriter.setInsertionPoint(par_op);
          auto& os = output_sets.at(input.get());
          auto tt = cast<RankedTensorType>(slice->getType());
          int64_t nDims = tt.getRank();
          SmallVector<Value> sizes;
          for(int i = 0; i < nDims; i++)
          {
            if(tt.getDimSize(i) == ShapedType::kDynamic) {
              Value idx = rewriter.create<index::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(i));
              sizes.push_back(rewriter.create<tensor::DimOp>(loc, rewriter.getIndexType(), input.get(), idx));
            }
          }

          rewriter.setInsertionPointToEnd(par_op.getBody());
          SmallVector<int64_t> static_offsets(nDims, 0);
          static_offsets[os.dim] = ShapedType::kDynamic;

          Operation* insert = rewriter.create<tensor::ParallelInsertSliceOp>(
            loc, 
            *slice,
            for_loop.getTiedBlockArgument(&input),
            inductionVar,
            sizes,
            ValueRange(),
            rewriter.getDenseI64ArrayAttr(static_offsets),
            rewriter.getDenseI64ArrayAttr(tt.getShape()),
            rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(nDims, 1))
          );
          terminator_ops.push_back(insert);
          slice++;
        }
        rewriter.setInsertionPointAfter(for_loop);
        
        IRMapping map;
        return new DenseParallelLoopInfo(input_slices, for_loop.getResults(), par_op, map, inductionVar, terminator_ops, output_sets);
      }

      Value getCrd(IRRewriter& rewriter) override {return inductionVar;}

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(dyn_cast<TensorType>(tensor.getType())){
          // TODO: Fix me. Right now, getCrd on the output tensor is just zero becuase 
          // we have already extracted the slice that we want. This will only work 
          // in very limited scenarios
          if(std::find(currentInputs.begin(), currentInputs.end(), tensor) != currentInputs.end()){
            Value zero = rewriter.create<index::ConstantOp>(tensor.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
            return zero;
          } else {
            return inductionVar;
          }
          
        } else if(SparseTensorType tt = dyn_cast<SparseTensorType>(tensor.getType())){
          if((TensorFormatEnum)(tt.getFormat()[2 * dim]) == TensorFormatEnum::D) {
            return inductionVar;
          }
          assert(false && "Invalid type passed to DenseLoopInfo getPos");
          return nullptr;
        }
        assert(false && "Invalid type passed to DenseLoopInfo getPos");
        return nullptr;
      }

      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        Operation* terminator = terminator_ops[idx];
        rewriter.updateRootInPlace(terminator, [&](){terminator->setOperand(0, newOutput);});
      }
  };

  class SparseLoopInfo : LoopInfo {
    private:
      Value inductionVar;
      scf::YieldOp terminator;
      Value controlTensor;
      uint32_t dim;
      Value crd;

    public:
      SparseLoopInfo(ValueRange inputs, ResultRange outputs, Operation* body, IRMapping ir_map, Value i, scf::YieldOp yield, Value tensor, uint32_t d): 
          LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), terminator(yield), controlTensor(tensor), dim(d), crd(nullptr) {}
        
      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        auto sparse_domain = cast<IndexTreeSparseDomainOp>(domain_op);
        auto index_type = rewriter.getIndexType();

        Value start_idx = sparse_domain.getParent();
        if(!start_idx){
          start_idx = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
        }

        Value inc = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
        Value end_idx = rewriter.create<arith::AddIOp>(loc, index_type, start_idx, inc);
        Value lb = rewriter.create<tensor::ExtractOp>(loc, sparse_domain.getPos(), start_idx);
        lb = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), lb);
        Value ub = rewriter.create<tensor::ExtractOp>(loc, sparse_domain.getPos(), end_idx);
        ub = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), ub);
        Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), 
                                                        rewriter.getIndexAttr(1));
        scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, inputs);

        IRMapping map;
        unsigned init_arg_idx = 0;
        for(Value init_arg : inputs){
          map.map(init_arg, for_loop.getRegionIterArg(init_arg_idx));
          init_arg_idx += 1;
        }
        
        rewriter.setInsertionPointToStart(for_loop.getBody());
        auto yield_op = rewriter.create<scf::YieldOp>(loc, for_loop.getRegionIterArgs());
        rewriter.setInsertionPointAfter(for_loop);

        return new SparseLoopInfo(ValueRange(for_loop.getRegionIterArgs()), for_loop.getResults(), yield_op, map, for_loop.getInductionVar(), yield_op, sparse_domain.getTensor(), sparse_domain.getDim());
      }

      Value getCrd(IRRewriter& rewriter) override {
        if(crd != nullptr) return crd;
        auto loc = controlTensor.getLoc();
        // SparseTensorType tt = cast<SparseTensorType>(controlTensor.getType());
        crd = rewriter.create<SpTensorGetCrd>(loc, controlTensor, inductionVar, rewriter.getI32IntegerAttr(dim));
        crd = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), crd);
        return crd;
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(tensor == controlTensor && dim == this->dim){
          return inductionVar;
        }

        if(dyn_cast<TensorType>(tensor.getType()) || dyn_cast<WorkspaceType>(tensor.getType())){
          return crd;
        } else if(SparseTensorType tt = dyn_cast<SparseTensorType>(tensor.getType())){
          if((TensorFormatEnum)(tt.getFormat()[2 * dim]) == TensorFormatEnum::D) {
            return crd;
          }
        }
        auto loc = controlTensor.getLoc();
        Value pos = rewriter.create<tensorAlgebra::TensorFindPos>(loc, rewriter.getIndexType(), tensor, crd, rewriter.getI32IntegerAttr(dim), rewriter.getBoolAttr(true));
        return pos;
      }

      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.updateRootInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class SingletonLoopInfo : LoopInfo {
    private:
      LoopInfo* parent;
      Value inductionVar;
      Value tensor;
      int32_t dim;

    public:
      SingletonLoopInfo(ValueRange inputs, Operation* body, IRMapping ir_map, LoopInfo* parent_info, Value i, Value sparse_tensor, int32_t dim) : 
        LoopInfo(inputs, ValueRange(inputs), body, ir_map), parent(parent_info), inductionVar(i), tensor(sparse_tensor), dim(dim)
        {}
      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs, LoopInfo* parent_info)
      {
        auto sparse_domain = cast<IndexTreeSparseDomainOp>(domain_op);
        Value inductionVar = sparse_domain.getParent();
        IRMapping map;
        return new SingletonLoopInfo(inputs, &(*(rewriter.saveInsertionPoint().getPoint())), map, parent_info, inductionVar, sparse_domain.getTensor(), sparse_domain.getDim());
      }

      Value getCrd(IRRewriter& rewriter) override {
        auto loc = rewriter.getUnknownLoc(); // This is not correct
        // auto tt = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getIndexType());
        Value crd_tensor = rewriter.create<SpTensorGetDimCrd>(loc, tensor, rewriter.getI32IntegerAttr(dim));
        Value crd = rewriter.create<tensor::ExtractOp>(loc, crd_tensor, inductionVar);
        crd = rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), crd);
        return crd;
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(llvm::isa<TensorType>(tensor.getType())) {
          // For dense tensors, positions and crd should be the same.
          return getCrd(rewriter);
        }
        return inductionVar;
      }


      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        parent->updateOutput(rewriter, idx, newOutput);
      }

      ValueRange getResults() override {
        return ValueRange(currentInputs);
      }
      
  };

  class WorkspaceLoopInfo : LoopInfo {
    private:
      Value inductionVar;
      scf::YieldOp terminator;
      Value workspaceTensor;
      Value crd;

    public:
      WorkspaceLoopInfo(ValueRange inputs, ResultRange outputs, Operation* body, IRMapping ir_map, Value i, scf::YieldOp yield, Value workspace) : 
        LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), terminator(yield), workspaceTensor(workspace)  {}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        auto workspace_domain_op = llvm::cast<IndexTreeWorkspaceDomainOp>(domain_op);
        auto index_type = rewriter.getIndexType();
        
        /** TODO: Sort crd array? **/
        Value lb = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
        Value ub = rewriter.create<tensorAlgebra::SpTensorGetNNZ>(loc, index_type, workspace_domain_op.getTensor());
        Value step = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(), 
                                                        rewriter.getIndexAttr(1));
        scf::ForOp for_loop = rewriter.create<scf::ForOp>(loc, lb, ub, step, inputs);
        Block* loop_body = for_loop.getBody();

        rewriter.setInsertionPointToStart(loop_body);
        IRMapping map;
        unsigned init_arg_idx = 0;
        for(Value init_arg : inputs){
          map.map(init_arg, for_loop.getRegionIterArg(init_arg_idx));
          init_arg_idx += 1;
        }
        auto yield_op = rewriter.create<scf::YieldOp>(loc, for_loop.getRegionIterArgs());
        rewriter.setInsertionPointAfter(for_loop);
        return new WorkspaceLoopInfo(ValueRange(for_loop.getRegionIterArgs()), for_loop.getResults(), yield_op, map, for_loop.getInductionVar(), yield_op, workspace_domain_op.getTensor());
      }

      Value getCrd(IRRewriter& rewriter) override {
        if(crd != nullptr) return crd;
        auto loc = workspaceTensor.getLoc();
        WorkspaceType tt = cast<WorkspaceType>(workspaceTensor.getType());
        crd = rewriter.create<SpTensorGetCrd>(loc, workspaceTensor, inductionVar, rewriter.getI32IntegerAttr(0));
        crd = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), crd);

        return crd;
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        if(llvm::isa<SparseTensorType>(tensor.getType()))
        {
          auto loc = rewriter.getUnknownLoc();
          return rewriter.create<TensorFindPos>(loc, rewriter.getIndexType(), tensor, nullptr, (int32_t)dim, true);
        }
        return inductionVar;
      }


      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.updateRootInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  class IntersectionLoopInfo : LoopInfo {
    private:
      Value inductionVar;
      Value crd;
      scf::YieldOp terminator;
      llvm::SmallDenseMap<std::pair<Value, uint32_t>, std::pair<Value, Value>> controlVars;
    public:
      IntersectionLoopInfo(ValueRange inputs, ResultRange outputs, Operation* body, IRMapping ir_map, Value i, Value c, scf::YieldOp yield, SmallDenseMap<std::pair<Value, uint32_t>, std::pair<Value, Value>> controls) : 
          LoopInfo(inputs, outputs, body, ir_map), inductionVar(i), crd(c), terminator(yield), controlVars(controls){}

      static LoopInfo* build(Operation* domain_op, IRRewriter& rewriter, ValueRange inputs)
      {
        auto loc = domain_op->getLoc();
        auto intersection_domain_op = llvm::cast<IndexTreeDomainIntersectionOp>(domain_op);
        auto index_type = rewriter.getIndexType();
        auto context = rewriter.getContext();

        SmallVector<Value> loop_args = SmallVector<Value>(inputs);

        SmallDenseMap<std::pair<Value, unsigned>, std::pair<Value, Value>> tensor_access_map;

        // Intersection between sparse domains
        auto domains = intersection_domain_op.getDomains();
        Value inc = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
        SmallVector<Value> loop_conditions;
        SmallVector<Value> array_crds;

        Block* cond_block = new Block();
        Block* body_block = new Block();
        
        // Create loop carried arguments for output tensors and iteration counter
        IRMapping map;
        for(Value init_arg : loop_args){
          cond_block->addArgument(init_arg.getType(), loc);
          BlockArgument body_arg = body_block->addArgument(init_arg.getType(), loc);
          map.map(init_arg, body_arg);
        }
        
        Value loop_ctr_init = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
        loop_args.push_back(loop_ctr_init);
        cond_block->addArgument(index_type, loc);
        body_block->addArgument(index_type, loc);
        unsigned loop_carry_args = loop_args.size();

        // Create control iterators for each of the tensors
        OpBuilder::InsertPoint before;
        for(Value domain : domains)
        {
          IndexTreeSparseDomainOp sparse_domain = llvm::cast<IndexTreeSparseDomainOp>(domain.getDefiningOp());
          TensorFormatEnum format = (TensorFormatEnum)sparse_domain.getFormat();
          switch(format)
          {
            case TensorFormatEnum::CN:
            case TensorFormatEnum::S:
            {
              // Not yet supported!!!
              break;
            }
            case TensorFormatEnum::CU:
            {
              Value start_idx = sparse_domain.getParent();
              if(!start_idx){
                start_idx = rewriter.create<arith::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
              }
              Value end_idx = rewriter.create<arith::AddIOp>(loc, index_type, start_idx, inc);
              Value start = rewriter.create<tensor::ExtractOp>(loc, sparse_domain.getPos(), start_idx);
              start = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), start);
              Value end = rewriter.create<tensor::ExtractOp>(loc, sparse_domain.getPos(), end_idx);
              end = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), end);
              loop_args.push_back(start);
              before = rewriter.saveInsertionPoint();

              Value crd_idx = cond_block->addArgument(start.getType(), loc);
              rewriter.setInsertionPointToStart(cond_block);
              Value cnd = rewriter.create<arith::CmpIOp>(
                loc, rewriter.getI1Type(),
                arith::CmpIPredicateAttr::get(context, arith::CmpIPredicate::ult), 
                crd_idx, end
              );
              loop_conditions.push_back(cnd);

              crd_idx = body_block->addArgument(start.getType(), loc);
              rewriter.setInsertionPointToStart(body_block);
              auto dim = rewriter.getI32IntegerAttr(sparse_domain.getDim());
              Value array_crd = rewriter.create<SpTensorGetCrd>(loc, sparse_domain.getTensor(), crd_idx, dim);
              array_crd = rewriter.createOrFold<IndexCastOp>(loc, rewriter.getIndexType(), array_crd);
              array_crds.push_back(array_crd);

              tensor_access_map.insert(std::make_pair(
                std::make_pair(sparse_domain.getTensor(), sparse_domain.getDim()), 
                std::make_pair(crd_idx, array_crd)
              ));
            }
          }
          rewriter.restoreInsertionPoint(before);
        }

        // Create while loop
        scf::WhileOp while_loop = rewriter.create<scf::WhileOp>(loc, cond_block->getArgumentTypes(), loop_args);
        while_loop.getBefore().push_front(cond_block);

        rewriter.setInsertionPointToEnd(cond_block);
        Value loop_condition = nullptr;
        for(Value cnd : loop_conditions)
        {
          if(loop_condition == nullptr)
          {
            loop_condition = cnd;
          } else {
            loop_condition = rewriter.create<arith::AndIOp>(loc, rewriter.getI1Type(), loop_condition, cnd);
          }
        }
        rewriter.create<scf::ConditionOp>(loc, loop_condition, cond_block->getArguments());

        while_loop.getAfter().push_front(body_block);
        // Create intersection
        rewriter.setInsertionPointToEnd(body_block);
        Value crd = nullptr;
        for(Value array_crd : array_crds){
          if(crd == nullptr)
          {
            crd = array_crd;
          } else {
            crd = rewriter.create<arith::MinUIOp>(loc, index_type, crd, array_crd);
          }
        }

        Value intersection_cnd = nullptr;
        SmallVector<Value> intersections;
        for(Value array_crd : array_crds)
        {
          Value is_intersect = rewriter.create<arith::CmpIOp>(
            loc, rewriter.getI1Type(),
            arith::CmpIPredicateAttr::get(context, arith::CmpIPredicate::eq),
            crd, array_crd
          );
          if(intersection_cnd == nullptr)
          {
            intersection_cnd = is_intersect;
          } else {
            intersection_cnd = rewriter.create<arith::AndIOp>(loc, rewriter.getI1Type(), intersection_cnd, is_intersect);
          }
          intersections.push_back(is_intersect);
        }

        SmallVector<Type> if_types;
        for(unsigned i = 0; i < loop_carry_args; i++)
        {
          if_types.push_back(loop_args[i].getType());
        }

        scf::IfOp if_op = rewriter.create<scf::IfOp>(loc, if_types, intersection_cnd, true);
        rewriter.setInsertionPointToStart(if_op.elseBlock());
        rewriter.create<scf::YieldOp>(
          loc, 
          std::vector<Value>(
            body_block->args_begin(),
            body_block->args_begin() + if_op->getNumResults())
        );
        rewriter.setInsertionPointToStart(if_op.thenBlock());
        // OpBuilder::InsertPoint loop_end = rewriter.saveInsertionPoint();
        Value induction_var = body_block->getArgument(loop_carry_args - 1);
        auto step = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
        auto loop_ctr = rewriter.create<index::AddOp>(loc, index_type, induction_var, step);
        auto yield_op = rewriter.create<scf::YieldOp>(
          loc, 
          std::vector<Value>(
            body_block->args_begin(),
            body_block->args_begin() + if_op->getNumResults())
        );
        yield_op->setOperand(loop_carry_args - 1, loop_ctr.getResult());

        // Increment each argument
        rewriter.setInsertionPointAfter(if_op);
        SmallVector<Value> yield_args;
        for(auto result : if_op.getResults()) {
          yield_args.push_back(result);
        }
        auto cntrl_arg = body_block->args_begin() + loop_carry_args;
        for(Value cnd : intersections)
        {
          Value inc = rewriter.create<index::CastUOp>(loc, index_type, cnd);
          yield_args.push_back(rewriter.create<index::AddOp>(loc, index_type, *cntrl_arg, inc));
          cntrl_arg += 1;
        }

        // Create YieldOp
        rewriter.create<scf::YieldOp>(loc, yield_args);
        rewriter.setInsertionPointAfter(while_loop);

        ValueRange inner_inputs = body_block->getArguments().drop_back(loop_args.size() - loop_carry_args - 1);
        ResultRange outputs = ResultRange(while_loop->result_begin(), while_loop->result_begin() + (loop_carry_args - 1));
        return new IntersectionLoopInfo(inner_inputs, outputs, loop_ctr, map, induction_var, crd, yield_op, tensor_access_map);
      }

      Value getCrd(IRRewriter& rewriter) override {
        return crd;
      }

      Value getPos(IRRewriter& rewriter, Value tensor, uint32_t dim) override {
        auto control = controlVars.find(std::make_pair(tensor, dim));
        if(control != controlVars.end()) 
          return control->getSecond().first;
        auto loc = tensor.getLoc();
        Value pos = rewriter.create<tensorAlgebra::TensorFindPos>(loc, rewriter.getIndexType(), tensor, crd, rewriter.getI32IntegerAttr(dim), rewriter.getBoolAttr(true));
        return pos;
      }


      void updateOutput(IRRewriter& rewriter, uint32_t idx, Value newOutput) override {
        currentInputs[idx] = newOutput;
        rewriter.updateRootInPlace(terminator, [&](){terminator.setOperand(idx, newOutput);});
      }
  };

  struct LowerIndexTreeToSCFPass
      : public PassWrapper<LowerIndexTreeToSCFPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIndexTreeToSCFPass)

    SmallDenseMap<Value, LoopInfo*> nodeMap;
    SmallDenseMap<Value, uint32_t> leafMap;
    llvm::ScopedPrinter logger{llvm::dbgs()};

    Value mapInputIntoLoop(Value input, LoopInfo* loop_info)
    {
      auto input_idx_iter = leafMap.find(input);
      if(input_idx_iter != leafMap.end()){
        uint32_t input_idx = input_idx_iter->getSecond();
        return loop_info->getInput(input_idx);
      }
      return input;
    }

    void updateOutput(Value input, Value old_output, Value new_output, LoopInfo* loop_info, IRRewriter& rewriter)
    {
      uint32_t input_idx = leafMap.find(input)->getSecond();
      leafMap.insert(std::pair<Value, uint32_t>(old_output, input_idx));
      loop_info->updateOutput(rewriter, input_idx, new_output);
    }

    Value convertOperand(LoopInfo* loop_info, IndexTreeLHSOperandOp op, StringRef semiring, IRRewriter &rewriter)
    {
      Location loc = op->getLoc();
      Value tensor = mapInputIntoLoop(op.getTensor(), loop_info);
      auto crds = op.getCrds();
      auto positions = op.getPos();

      TensorType tensor_type;
      if((tensor_type = llvm::dyn_cast<mlir::TensorType>(tensor.getType()))){
        return rewriter.create<tensor::ExtractOp>(loc, tensor_type.getElementType(), tensor, positions);
      } else {
        Type element_type;
        if(llvm::isa<SparseTensorType>(tensor.getType()))
        {
          element_type = llvm::cast<SparseTensorType>(tensor.getType()).getElementType();
        } else if(llvm::isa<WorkspaceType>(tensor.getType()))
        {
          element_type = llvm::cast<WorkspaceType>(tensor.getType()).getElementType();
        }
        Value pos = positions[positions.size() - 1];
        double zero = 0;
        if(semiring == "minxy"){
          zero = INFINITY;
        }
        return rewriter.create<TensorExtractOp>(loc, element_type, tensor, pos, crds, rewriter.getF64FloatAttr(zero));
      }
    }

    Value convertOperand(LoopInfo* loop_info, IndexTreeOperandOp op, StringRef semiring, IRRewriter &rewriter)
    {
      Location loc = op->getLoc();
      Value tensor = mapInputIntoLoop(op.getTensor(), loop_info);
      auto crds = op.getCrds();
      auto positions = op.getPos();

      TensorType tensor_type;
      if((tensor_type = llvm::dyn_cast<mlir::TensorType>(tensor.getType()))){
        return rewriter.create<tensor::ExtractOp>(loc, tensor_type.getElementType(), tensor, crds);
      } else {
        // LHS may not be constant (i.e. if we are inserting into a tensor that we need to resize), 
        // so cannot directly lower like we can the RHS
        Value pos = positions[positions.size() - 1];
        double zero = 0;
        if(semiring == "minxy"){
          zero = INFINITY;
        }

        return rewriter.create<TensorExtractOp>(loc, rewriter.getF64Type(), tensor, pos, crds, rewriter.getF64FloatAttr(zero));
      }
    }

    mlir::LogicalResult convertCompute(IndexTreeComputeOp compute_op, IRRewriter &rewriter)
    {
      auto loc = compute_op->getLoc();
      LoopInfo* parent_info = nodeMap.find(compute_op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      
      auto semiringParts = compute_op.getSemiring().split('_');
      Value elementwise_result;
      for(auto rhs = compute_op.getRhs().begin(); rhs != compute_op.getRhs().end(); rhs++)
      {
        Value rhs_value = convertOperand(parent_info, cast<IndexTreeOperandOp>((*rhs).getDefiningOp()), semiringParts.first, rewriter);
        if(rhs == compute_op.getRhs().begin()){
          elementwise_result = rhs_value;
        } else {
          elementwise_result = getSemiringSecondVal(rewriter, loc, semiringParts.second, 
                                                    elementwise_result, rhs_value);
        }
      }

      IndexTreeLHSOperandOp lhs = llvm::cast<IndexTreeLHSOperandOp>(compute_op.getLhs().getDefiningOp());
      Value reduce_result = convertOperand(parent_info, lhs, semiringParts.first, rewriter);
      reduce_result = getSemiringFirstVal(rewriter, loc, semiringParts.first, 
                                          reduce_result, elementwise_result);

      Value old_tensor = mapInputIntoLoop(lhs.getTensor(), parent_info);
      Value output_tensor;
      if(llvm::isa<mlir::TensorType>(old_tensor.getType()))
      {
        output_tensor = rewriter.create<tensor::InsertOp>(loc, old_tensor.getType(), reduce_result, old_tensor, lhs.getPos());
      } else {
        output_tensor = rewriter.create<tensorAlgebra::TensorInsertOp>(loc, old_tensor.getType(), old_tensor, lhs.getPos(), lhs.getCrds(), reduce_result);
      }
      updateOutput(lhs.getTensor(), compute_op.getResult(), output_tensor, parent_info, rewriter);
      return success();
    }

    mlir::LogicalResult convertClearWorkspaceOp(IndexTreeCleanWorkspaceOp op, IRRewriter &rewriter)
    {
      Location loc = op->getLoc();
      LoopInfo* parent_info = nodeMap.find(op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      Value input = mapInputIntoLoop(op.getWorkspace(), parent_info);
      Value workspace = rewriter.create<WorkspaceClearOp>(loc, op->getResultTypes(), input);
      updateOutput(op.getWorkspace(), op.getResult(), workspace, parent_info, rewriter);
      return success();
    }

    mlir::LogicalResult convertSymbolicDomainOp(ComputeSymbolicDomainOp op, IRRewriter &rewriter)
    {
      
      Location loc = op->getLoc();
      LoopInfo* parent_info = nodeMap.find(op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      Value symbolic_domain = mapInputIntoLoop(op.getDomain(), parent_info);
      Value new_domain = rewriter.create<SymbolicDomainInsertOp>(loc, 
                                                                 symbolic_domain.getType(),
                                                                 symbolic_domain,
                                                                 parent_info->getCrd(rewriter),
                                                                 op.getIsUniqueAttr());
      updateOutput(op.getDomain(), op.getResult(), new_domain, parent_info, rewriter);
      return success();
    }

    mlir::LogicalResult convertSymbolicDomainEndRowOp(ComputeSymbolicDomainRowOp op, IRRewriter &rewriter)
    {
      Location loc = op->getLoc();
      LoopInfo* parent_info = nodeMap.find(op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);
      Value symbolic_domain = mapInputIntoLoop(op.getDomain(), parent_info);
      Value new_domain = rewriter.create<SymbolicDomainEndRowOp>(loc, 
                                                                 symbolic_domain.getType(),
                                                                 symbolic_domain,
                                                                 op.getNeedsMarkAttr());
      updateOutput(op.getDomain(), op.getResult(), new_domain, parent_info, rewriter);
      return success();
    }

    mlir::LogicalResult convertTensorAccessOp(IndexTreeIndexToTensorOp access_op, IRRewriter &rewriter)
    {
      // TODO: Figure out how to find the right position to insert these operations!!!!
      LoopInfo* parent_info = nodeMap.find(access_op.getIndex())->getSecond();
      Value tensor = mapInputIntoLoop(access_op.getTensor(), parent_info);
      auto dim = access_op.getDim();

      Value access_crd = parent_info->getCrd(rewriter);
      Value access_pos = parent_info->getPos(rewriter, tensor, dim);
      if(auto tensor_type = dyn_cast<tensorAlgebra::SparseTensorType>(tensor.getType()))
      {
        TensorFormatEnum format = (TensorFormatEnum) tensor_type.getFormat()[2 * dim];
        if(format == TensorFormatEnum::D)
        {
          // TODO: This is incorrect, deal with reordering!!!!
          if(access_op.getPrevDim()) {
            auto loc = access_op.getLoc();
            auto index_type = rewriter.getIndexType();
            Value dim_size = rewriter.create<tensorAlgebra::SpTensorGetDimSize>(loc, index_type, tensor, rewriter.getI32IntegerAttr(dim));
            Value pos_start = rewriter.create<arith::MulIOp>(loc, index_type, dim_size, access_op.getPrevDim());
            access_pos = rewriter.create<arith::AddIOp>(loc, index_type, pos_start, access_pos);
          }
        }
      } 
      rewriter.replaceAllUsesWith(access_op.getPos(), access_pos);
      rewriter.replaceAllUsesWith(access_op.getCrd(), access_crd);
      return success();
    }

    mlir::LogicalResult convertRoot(IndexTreeRootOp root_op, IRRewriter &rewriter) {
      IndexTreeOp tree = root_op->getParentOfType<IndexTreeOp>();
      indexTree::YieldOp yield = cast<indexTree::YieldOp>(tree.getRegion().getBlocks().front().getTerminator());
      IRMapping map;
      LoopInfo* sentinel_info = SentinelLoopInfo::build(tree.getOperands(), tree.getResults(), root_op, map, yield);
      nodeMap.insert(std::make_pair(root_op.getResult(), sentinel_info));
      
      uint32_t i = 0;
      for(Value v: tree.getOperands()){
        leafMap.insert(std::make_pair(v, i));
        i++;
      }
      return success();
    }

    mlir::LogicalResult convertIndexNode(IndexTreeIndicesOp index_node_op, IRRewriter &rewriter)
    {
      IndexTreeOp tree = index_node_op->getParentOfType<IndexTreeOp>();
      Operation* domain_op = index_node_op.getDomain().getDefiningOp();
      Value index_node = index_node_op->getResult(0);
      LoopInfo* parent_info = nodeMap.find(index_node_op.getParent())->getSecond();
      rewriter.setInsertionPoint(parent_info->loopBody);

      LoopInfo* loop_info = llvm::TypeSwitch<Operation*, LoopInfo*>(domain_op) 
        .Case<IndexTreeDenseDomainOp>([&](IndexTreeDenseDomainOp op) {
          if(index_node_op.getIsParallel()){
            ValueRange inputs = parent_info->getInputs();
            auto output_analysis = Pass::getChildAnalysis<IndexTreeInferOutputSets>(index_node_op->getParentOp());
            auto output_sets = output_analysis.getOutputSets(index_node_op);
            uint32_t i = 0;
            for(Value v: inputs){
              output_sets.insert(std::make_pair(v, output_sets[tree.getOperand(i)]));
              i++;
            }
            return DenseParallelLoopInfo::build(op, rewriter, parent_info->getInputs(), output_sets);
          }
          return DenseLoopInfo::build(op, rewriter, parent_info->getInputs());
        })
        .Case<IndexTreeSparseDomainOp>([&](IndexTreeSparseDomainOp op) {
          switch((TensorFormatEnum)op.getFormat()){
            case TensorFormatEnum::CN:
            case TensorFormatEnum::CU:
              return SparseLoopInfo::build(op, rewriter, parent_info->getInputs());
            case TensorFormatEnum::S:
              return SingletonLoopInfo::build(op, rewriter, parent_info->getInputs(), parent_info);
          }
        })
        .Case<IndexTreeWorkspaceDomainOp>([&](IndexTreeWorkspaceDomainOp op) {
          op.getTensorMutable().assign(mapInputIntoLoop(op.getTensor(), parent_info));
          return WorkspaceLoopInfo::build(op, rewriter, parent_info->getInputs());
        })
        .Case<IndexTreeDomainIntersectionOp>([&](IndexTreeDomainIntersectionOp op) {
          return IntersectionLoopInfo::build(op, rewriter, parent_info->getInputs());
        })
        .Default([](Operation *op) {
          assert(false && "IndexNode not given a valid domain");
          return nullptr;
        });

      ValueRange loop_outputs = loop_info->getResults();
      uint32_t i = 0;
      for(Value v : loop_outputs)
      {
        parent_info->updateOutput(rewriter, i, v);
        i++;
      }
      nodeMap.insert(std::make_pair(index_node, loop_info));
      rewriter.setInsertionPoint(loop_info->loopBody);
      return success();
    }

    mlir::LogicalResult convertTree(IndexTreeOp treeOp, IRRewriter &rewriter)
    {
      // Walk the Block of the IndexTreeOp and Convert Operands in order to maintain
      // order in generated code
      SmallVector<Operation*> toDelete;
      LLVM_DEBUG({logger.startLine() << "Current Tree: \n" << treeOp << "\n";});
      for (Operation &op : treeOp.getRegion().front()){
        // A node in the tree should be able to use the loopMaps map to find where it 
        // should be rewritten to
        LogicalResult result = llvm::TypeSwitch<Operation*, LogicalResult>(&op) 
          .Case<IndexTreeRootOp>([&](IndexTreeRootOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertRoot(op, rewriter);
          })
          .Case<IndexTreeComputeOp>([&](IndexTreeComputeOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertCompute(op, rewriter);
          })
          .Case<IndexTreeCleanWorkspaceOp>([&](IndexTreeCleanWorkspaceOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertClearWorkspaceOp(op, rewriter);
          })
          .Case<ComputeSymbolicDomainOp>([&](ComputeSymbolicDomainOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertSymbolicDomainOp(op, rewriter);
          })
          .Case<ComputeSymbolicDomainRowOp>([&](ComputeSymbolicDomainRowOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertSymbolicDomainEndRowOp(op, rewriter);
          })
          .Case<IndexTreeIndexToTensorOp>([&](IndexTreeIndexToTensorOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            return convertTensorAccessOp(op, rewriter);
          })
          .Case<IndexTreeIndicesOp>([&](IndexTreeIndicesOp op) {
            LLVM_DEBUG({
              logger.startLine() << "Converting: " << op <<  "\n";
            });
            LogicalResult result = convertIndexNode(op, rewriter);
            // LLVM_DEBUG({logger.startLine() << "Current Tree: \n" << treeOp << "\n";});
            return result;

          })
          .Default([](Operation *op) {
            return success();
          });

          if(result.failed()){
            LLVM_DEBUG({
              logger.startLine() << "Conversion failed!" <<  "\n";
            });
            return failure();
          }

          if(isa<indexTree::YieldOp>(op)){
            break;
          }
          toDelete.push_back(&op);
      }
      for (auto op = toDelete.rbegin(); op != toDelete.rend(); ++op){
        // Erase all the old ops in the region
        LLVM_DEBUG({(*op)->emitOpError() <<  "Removing \n";});
        rewriter.eraseOp(*op);
      }

      return success();
    }

    void runOnOperation() override {
      std::vector<IndexTreeOp> iTrees;
      func::FuncOp funcOp = getOperation();
      funcOp.walk([&](IndexTreeOp op){ iTrees.push_back(op); });
      
      for(auto op : iTrees)
      {
        OpBuilder builder(op);
        IRRewriter rewriter(builder);
        convertTree(op, rewriter);
      }
  
    }
  };
}

/// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createLowerIndexTreeToSCFPass()
{
  return std::make_unique<LowerIndexTreeToSCFPass>();
}
