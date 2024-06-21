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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringSet.h"
#include <iostream>
#include <algorithm>
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

#define DEBUG_TYPE "lowering-it-to-scf"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
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

  class OpsTree
  {
    /// private:
  public:
    std::vector<scf::ForOp> forOps;         /// The (nested) for loops
    std::vector<Value> accessIdx;           /// The coordinate of accessing that dimension
    std::vector<scf::ForOp> symbolicForOps; /// For-loops in symbolic phase (if necessary)
    std::vector<Value> symbolicAccessIdx;   /// The accessing index for that for-loop in symbolic phase (if necessary)
                                            ///  std::vector<Value> cmptOps;     /// The computation ops (no used?)
    std::vector<OpsTree *> children;
    OpsTree *parent;
    int id; /// the index in the ws_op array. The order is the DFS order.

    std::vector<scf::ForOp> symbolicForOps_debug;
    std::vector<Value> symbolicAccessIdx_debug;

  public:
    OpsTree() {}

    OpsTree(std::vector<scf::ForOp> &forOps, std::vector<Value> &accessIdx,
            OpsTree *parent, int id) : forOps(forOps), accessIdx(accessIdx), parent(parent), id(id)
    {
    }

    OpsTree(std::vector<scf::ForOp> &forOps, std::vector<Value> &accessIdx,
            OpsTree *parent) : forOps(forOps), accessIdx(accessIdx), parent(parent)
    {
    }

    ~OpsTree() {}

    void addChild(OpsTree *tree)
    { /// const T& node
      this->children.push_back(tree);
    }

    std::vector<scf::ForOp> &getForOps()
    {
      return this->forOps;
    }

    OpsTree *getParent()
    {
      return this->parent;
    }

    void setForOps(std::vector<scf::ForOp> &forOps)
    {
      this->forOps = forOps;
    }

    std::vector<OpsTree *> &getChildren()
    {
      return this->children;
    }
  };

  /// ----------------- ///
  /// struct to pass symbolic phase information to the numeric phase
  /// ----------------- ///
  struct SymbolicInfo
  {
    bool are_inputs_sparse = false; /// If both inputs are sparse. It is true for SpGEMM and sparse elementwise operations.
                                    /// All other members are only used when are_inputs_sparse is true.

    bool has_symbolic_phase = false; /// If current generated code should have a symbolic phase.
                                     /// Currently, if are_inputs_parse == true; then has_symbolic_phase = true;

    Value mtxC_num_rows = nullptr;
    Value mtxC_num_cols = nullptr;

    Value mtxC_rowptr = nullptr;   /// Output C's rowptr array when do C = A * B and they are all sparse
                                   /// %alloc_100 = memref.alloc(%43) : memref<?xindex>
    Value mtxC_col = nullptr;      /// Output C's col array when do C = A * B and they are all sparse
                                   /// %alloc_104 = memref.alloc(%44) : memref<?xindex>
    Value mtxC_val = nullptr;      /// Output C's val array when do C = A * B and they are all sparse
                                   /// %alloc_108 = memref.alloc(%44) : memref<?xf64>
    Value mtxC_val_size = nullptr; /// Output C' correct number of non-zeros or C_val_size (ready after symbolic phase)

    Value mtxC_rowptr_size = nullptr; /// rowptr array's size, which is number of columns plus one (num_col + 1).

    Value row_offset = nullptr; /// In Numeric Phase, row_offset is the insertion location in the C_col and C_val.

    Value mtxC = nullptr; /// The sparse tensor
                          /// It is %55 below.
  };

  /// ----------------- ///
  /// Auxiliary structures for the numeric phase
  /// ----------------- ///
  struct NumericInfo
  {
    Value ws_bitmap = nullptr;                /// workspace's bitmap to tell if a column ID is visited.
    Value ws_bitmap_valueAccessIdx = nullptr; /// value access index for the workspace bitmap.

    Value mask_array = nullptr; /// the intermediate dense vector for a row of the mask.
  };

  /// ----------------- ///
  /// Remove an operantion's user who is a memref.store
  /// This is very ad-hoc, just to avoid segmentation fault for old very large C.val array and C.col array.
  /// ----------------- ///
  void removeMemrefStoreUser(Value &opd)
  {
    {
      comet_vdump(opd);
    }
    std::vector<Operation *> users;
    for (Operation *user : opd.getUsers())
    {
      if (isa<memref::StoreOp>(user))
      {
        users.push_back(user);
        {
          comet_pdump(user);
        }
      }
    }
    for (Operation *user : users)
    {
      user->erase();
    }
  }

  /// ----------------- ///
  /// Find all users of the old_Value, and replace those users' corresponding operand to new_Value. For example,
  /// "ta.print"(%old_Value)  =>  "ta.print"(%new_Value)
  /// ----------------- ///
  void replaceOldValueToNewValue(Value &old_Value,
                                 Value &new_Value)
  {
    {
      comet_vdump(old_Value);
      comet_vdump(new_Value);
    }

    /// Traverse each user of new_Value
    std::vector<Operation *> users;
    for (Operation *user : old_Value.getUsers())
    {
      users.push_back(user);
    }
    DominanceInfo domInfo(new_Value.getDefiningOp()); /// To check dominance
    for (Operation *user : users)
    {
      {
        comet_debug() << "before replace operand.\n";
        comet_pdump(user);
      }
      /// Check if new_Value dominates the user
      if (!domInfo.dominates(new_Value, user))
      {
        continue;
      }
      uint64_t op_i = 0;
      for (Value op : user->getOperands())
      {
        /// Find the mtxC in the user's operands
        if (op.getDefiningOp() == old_Value.getDefiningOp())
        {
          /// Replace the old sparse tensor to the new one
          user->setOperand(op_i, new_Value);
          {
            comet_debug() << "after replace operand.\n";
            comet_pdump(user);
          }
        }
        ++op_i;
      }
    }
  }

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

  /// Get mask_rowptr, mask_col, and mask_val arrays.
  /// ----------------- ///
  /// mask_tensor = %50
  /// mask_rowptr = %alloc_99
  /// mask_col = %alloc_104
  /// mask_val = %alloc_109
  /// ----------------- ///
  /// %45 = bufferization.to_tensor %alloc_99 : memref<?xindex>
  /// %46 = bufferization.to_tensor %alloc_104 : memref<?xindex>
  /// %49 = bufferization.to_tensor %alloc_109 : memref<?xf64>
  /// %50 = ta.sptensor_construct(%41, %42, %43, %44, %45, %46, %47, %48, %49, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
  /// ----------------- ///
  void getMaskSparseTensorInfo(MaskingInfo &maskingInfo /* contents updated after call*/)
  {
    Value &mask_tensor = maskingInfo.mask_tensor;

    /// A2pos
    Value mask_rowtpr_buff = mask_tensor.getDefiningOp()->getOperand(CSR_A2POS); /// 2
    maskingInfo.mask_rowptr = mask_rowtpr_buff.getDefiningOp()->getOperand(0);

    /// A2crd
    Value mask_col_buff = mask_tensor.getDefiningOp()->getOperand(CSR_A2CRD); /// 3
    maskingInfo.mask_col = mask_col_buff.getDefiningOp()->getOperand(0);

    /// Aval
    Value mask_val_buff = mask_tensor.getDefiningOp()->getOperand(CSR_AVAL); /// 4
    maskingInfo.mask_val = mask_val_buff.getDefiningOp()->getOperand(0);

    {
      comet_vdump(mask_tensor);
      comet_vdump(maskingInfo.mask_rowptr);
      comet_vdump(maskingInfo.mask_col);
      comet_vdump(maskingInfo.mask_val);
    }
  }

  unsigned int findIndexInVector_OpsTree(std::vector<OpsTree *> vec, OpsTree *e)
  {
    /// Check if element e exists in vector
    auto it = std::find(vec.begin(), vec.end(), e);

    /// It accepts a range and an element to search in the given range. If element is found then it returns an iterator to the first element in the given range that’s equal to given element, else it returns an end of the list.
    unsigned int ret = 0;
    if (it != vec.end())
    {
      /// Get index of element from iterator
      ret = std::distance(vec.begin(), it);
    }
    else
    {
      ret = vec.size();
    }
    return ret;
  }

  // Value findCorrespondingAlloc(Value &iOp)
  // {
  //   comet_debug() << "findCorrespondingAlloc for loop upper bound\n";
  //   comet_vdump(iOp);
  //   auto init_alloc = iOp.getDefiningOp()->getOperand(0);
  //   comet_vdump(init_alloc);

  //   while (true)
  //   {
  //     if (isa<memref::AllocOp>(init_alloc.getDefiningOp()))
  //     {
  //       if (init_alloc.getType().dyn_cast<MemRefType>().getDimSize(0) != ShapedType::kDynamic)
  //       {
  //         return init_alloc;
  //       }
  //     }
  //     if (init_alloc.getDefiningOp()->getNumOperands() > 0)
  //     {
  //       init_alloc = init_alloc.getDefiningOp()->getOperand(0);
  //     }
  //     else
  //     {
  //       /// Alloc related to another sparse tensor construct such as coming from sparse transpose
  //       comet_debug() << "Return alloc op - comes from sptensor_construct\n";
  //       comet_vdump(init_alloc);
  //       return init_alloc;
  //     }
  //   }
  // }

  /// Get allocs for a tensor (sparse or dense)
  std::vector<Value> getAllocs(Value &tensor)
  {
    comet_vdump(tensor);
    std::vector<Value> allocs;
    if (tensor.getType().isa<mlir::TensorType>())
    { /// Dense tensor
      comet_debug() << " getAllocs() -  it is dense\n";
      if (isa<ToTensorOp>(tensor.getDefiningOp()))
      {
        Operation *tensorload = cast<ToTensorOp>(tensor.getDefiningOp());
        auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
        comet_vdump(alloc_op);
        allocs.push_back(alloc_op);
      }
      else
      {
        for (unsigned int i = 0; i < tensor.getDefiningOp()->getNumOperands(); i++)
        {
          if (isa<ToTensorOp>(tensor.getDefiningOp()->getOperand(i).getDefiningOp()))
          {
            Operation *tensorload = cast<ToTensorOp>(tensor.getDefiningOp()->getOperand(i).getDefiningOp());
            auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
            comet_vdump(alloc_op);
            allocs.push_back(alloc_op);
          }
        }
      }
    }
    else if (tensor.getType().isa<tensorAlgebra::SparseTensorType>())
    { /// nSparse tensor
      comet_debug() << " getAllocs() -  it is sparse\n";
      auto defop = tensor.getDefiningOp<tensorAlgebra::SparseTensorConstructOp>();

      for (unsigned int n = 0; n < defop.getTotalDimArrayCount(); n++)
      {
        comet_vdump(defop.getIndices()[n]);
        Operation *tensorload = defop.getIndices()[n].getDefiningOp<ToTensorOp>();
        auto alloc_op = cast<memref::AllocOp>(tensorload->getOperand(0).getDefiningOp());
        allocs.push_back(alloc_op);
        comet_vdump(alloc_op);
      }
    }
    else if (dyn_cast<ConstantOp>(tensor.getDefiningOp()))
    { /// ConstantOp
      allocs.push_back(tensor);
    }
    return allocs;
  }

  std::vector<std::vector<Value>> getAllAllocs(std::vector<Value> &tensors)
  {
    std::vector<std::vector<Value>> allAllocs(tensors.size());
    for (unsigned int i = 0; i < tensors.size(); i++)
    {
      allAllocs[i] = getAllocs(tensors[i]);
    }
    return allAllocs;
  }

  /// while until parent == null
  void getAncestorsOps(OpsTree *opstree, std::vector<OpsTree *> &ret)
  {

    while (opstree->parent != nullptr)
    {
      ret.push_back(opstree->parent);
      opstree = opstree->parent;
    }
  }

  /// In genForOps, set Insertion Point for numeric loops.
  void setInsertionPointInNumericLoops(OpBuilder &builder,
                                       std::vector<OpsTree *> &ancestorsOps,
                                       OpsTree *opstree)
  {
    /// If parent is for loop, insert into the body, How to get end of body?
    if (ancestorsOps.size() > 0)
    {
      /// ancestorsOps[0] stores the closest parent
      scf::ForOp parent_forop = nullptr;
      comet_debug() << "\n";
      std::vector<scf::ForOp> parent_forops = ancestorsOps[0]->forOps;
      comet_debug() << " parent_forops.size(): " << parent_forops.size() << " \n";

      parent_forop = parent_forops[parent_forops.size() - 1];

      comet_debug() << " reset the insertion point\n";
      comet_vdump(parent_forop);

      unsigned int order = findIndexInVector_OpsTree(ancestorsOps[0]->getChildren(), opstree);
      comet_debug() << " order: " << order << "\n";
      if (order == ancestorsOps[0]->getChildren().size())
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: Not belong to parent's children\n";
      }
      else
      {
        /// Get the children of the parent_forop
        comet_debug() << " number of children: " << parent_forops.size() << "\n";
        if (order == 0)
        {
          /// builder.setInsertionPointToStart(parent_forop.getBody());
          comet_debug() << "Insertion point order == 0\n";
          builder.setInsertionPoint(parent_forop.getBody()->getTerminator());
        }
        else
        {
          comet_debug() << "\n";
          std::vector<scf::ForOp> brother_forops = ancestorsOps[0]->getChildren()[order - 1]->forOps;
          if (brother_forops.size() > 0)
          {
            comet_debug() << " brother_forops.size(): " << brother_forops.size() << "\n";
            if (opstree->forOps.size() == 0)
            {
              comet_debug() << "\n";
              comet_vdump(brother_forops[0]);
              comet_debug() << "Insertion point (brother_forops.size() > 0 &&  opstree->forOps.size() == 0)\n";
              builder.setInsertionPointAfter(brother_forops[0]);
            }
            else
            { /// current opstree contains loops, insert in the body of the loops
              comet_debug() << " -------- current opstree contain loops --- impossible\n";
              comet_debug() << "Insertion point (brother_forops.size() > 0 &&  opstree->forOps.size() != 0)\n";
              builder.setInsertionPoint(opstree->forOps[opstree->forOps.size() - 1].getBody()->getTerminator());
            }
          }
        }
      }
      comet_debug() << " reset the insertion point\n";
    }
  }

  /// ----------------- ///
  /// In genForOps, generate for-loop for a indexOp node if the index is corresponding to Format "D"
  /// ----------------- ///
  void genForOpFormat_D(OpBuilder &builder,
                        Location &loc,
                        Value &tensor,
                        unsigned int id,
                        unsigned int i,
                        std::vector<std::vector<Value>> &allAllocs,
                        scf::ForOp &forLoop /* output */,
                        Value &accessIndex /* output */)
  {
    ///  Value upperBound;
    ///  Value lowerBound;
    /// Check which tensor is sparse, which is dense;
    /// Since this function only handles mixed sparse/dense, then "D" only occurs in one tensor
    /// Both the dense and sparse tensor contain the dim size; But they are different. Use one.
    comet_debug() << " ";
    comet_vdump(tensor);

    Value lowerBound = builder.create<ConstantIndexOp>(loc, 0);
    auto step = builder.create<ConstantIndexOp>(loc, 1);

    if (tensor.getType().isa<mlir::RankedTensorType>())
    { /// Dense tensor
      Value upperBound;
      auto dim = builder.create<tensor::DimOp>(loc, tensor, id);
      upperBound = dim;

      auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);

      comet_debug() << " D Loop\n";
      comet_vdump(loop);

      ///    opstree->forOps.push_back(loop);
      ///    opstree->accessIdx.push_back(loop.getInductionVar());
      forLoop = loop;
      accessIndex = loop.getInductionVar();
    }
    else if (tensor.getType().isa<mlir::UnrankedTensorType>())
    {
      comet_debug() << " \n";
      comet_pdump(tensor.getDefiningOp());
      if (indexTree::IndexTreeComputeRHSOp rhsop = dyn_cast<indexTree::IndexTreeComputeRHSOp>(
              tensor.getDefiningOp()))
      {
        comet_debug() << " \n";
      }
    }
    else if (tensor.getType().cast<tensorAlgebra::SparseTensorType>())
    {
      comet_debug() << "cur_idx is in tensor " << i << "\n";

      auto index_0 = builder.create<ConstantIndexOp>(loc, 0);
      std::vector<Value> upper_indices = {index_0};
      Value upperBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);
      comet_vdump(allAllocs[i][4 * id]);
      auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);

      comet_debug() << " D Loop\n";
      comet_vdump(loop);

      forLoop = loop;
      accessIndex = loop.getInductionVar();
    }
    /// }
  }

  /// ----------------- ///
  /// In genForOps, generate for-loop for a indexOp node if the index is corresponding to Format "CU"
  /// ----------------- ///
  void genForOpFormat_CU(OpBuilder &builder,
                         Location &loc,
                         OpsTree *opstree,
                         Value &tensor,
                         unsigned int id,
                         unsigned int i,
                         std::vector<std::vector<Value>> &allAllocs,
                         scf::ForOp &parent_forop,
                         Value &parent_accessIdx,
                         scf::ForOp &forLoop /* output */,
                         Value &accessIndex /* output */)
  {
    /// Generate for(int m = pos[0]; m < pos[1]; m++){int i = crd[m];}
    /// if i = 0, index is [0,1]
    /// if parent loop and child loop is accessing the same sparse tensor (CSF), index is [m, m+1], m is the nearest loop induction variable
    /// Otherwise, the m comes from load operation of the input sparse tensor such as
    /// j = crd[i];
    /// for (int m = pos[j]; m < pos[j+1]; m++)

    comet_debug() << " format is CU id: " << id << "\n";
    comet_debug() << " Tensor: \n";
    comet_vdump(tensor);
    Value index_lower;
    Value index_upper;
    if (tensor.getType().cast<tensorAlgebra::SparseTensorType>())
    {
      comet_debug() << " Tensor type is sparse\n";
      if (id == 0)
      { /// The first index in the tensor
        index_lower = builder.create<ConstantIndexOp>(loc, 0);
        comet_vdump(index_lower);
      }
      else
      {
        if (opstree->parent != nullptr)
        {
          comet_debug() << " opstree->parent is not NULL\n";
          comet_debug() << " parent forop\n";
          comet_vdump(parent_forop);

          /// TODO (PT) Not sure why this (now commented out) code was needed but it breaks spgemm for cases like C = A * A
          ///  check if parent's and child's upper bounds come from the same sparse tensor
          /// auto parent_UpperBound = parent_forop.getUpperBound();
          /// comet_debug() << " parent upperBound:\n";
          /// comet_vdump(parent_UpperBound);
          /// auto alloc_parent_bounds = findCorrespondingAlloc(parent_UpperBound);
          /// comet_debug() << " parent upperBound alloc\n";
          /// comet_vdump(alloc_parent_bounds);

          /// comet_debug() << " child upperBound:\n";
          /// comet_vdump(allAllocs[i][4 * id]);
          /// auto alloc_child_bounds = findCorrespondingAlloc(allAllocs[i][4 * id]);
          /// comet_debug() << " child upperBound alloc\n";
          /// comet_vdump(alloc_child_bounds);

          // if (alloc_child_bounds == alloc_parent_bounds) /// m is the nearest loop induction variable
          // {
          //   comet_debug() << " THESAME: Parent and Child has the same alloc\n";
          //   index_lower = parent_forop.getInductionVar();
          // }
          // else
          { /// m comes from the load
            comet_debug() << " DIFFERENT:Parent and Child has the different alloc\n";
            // comet_vdump(alloc_parent_bounds);
            // comet_vdump(alloc_child_bounds);
            index_lower = parent_accessIdx;
          }
        }
        else
          llvm::errs() << "ERROR: Unexpected condition\n";
      }

      comet_debug() << " index_lower:";
      comet_vdump(index_lower);
      Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
      comet_vdump(const_index_1);
      index_upper = builder.create<AddIOp>(loc, index_lower, const_index_1);
      comet_debug() << " AddIOps (index_upper):";
      comet_vdump(index_upper);

      std::vector<Value> lower_indices = {index_lower};
      Value lowerBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], lower_indices); /// 2 * id

      std::vector<Value> upper_indices = {index_upper};
      Value upperBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices); /// 2 * id
      auto step = builder.create<ConstantIndexOp>(loc, 1);
      auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);

      comet_debug() << " CU Loop\n";
      comet_vdump(loop);

      builder.setInsertionPoint(loop.getBody()->getTerminator());

      std::vector<Value> crd_indices = {loop.getInductionVar()};
      auto get_index = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

      comet_debug() << "CU loop generated\n";
      comet_vdump(loop);
      forLoop = loop;
      accessIndex = get_index;
    }
  }

  /// ----------------- ///
  /// In genForOps, generate for-loop for a indexOp node if the index is corresponding to Format "CN"
  /// ----------------- ///
  void genForOpFormat_CN(OpBuilder &builder,
                         Location &loc,
                         Value &tensor,
                         unsigned int id,
                         unsigned int i,
                         std::vector<std::vector<Value>> &allAllocs,
                         scf::ForOp &forLoop /* output */,
                         Value &accessIndex /* output */)
  {
    /// Generate for(int m = pos[0]; m < pos[1]; m++){int i = crd[m];}
    if (tensor.getType().cast<tensorAlgebra::SparseTensorType>())
    {
      auto index_0 = builder.create<ConstantIndexOp>(loc, 0);
      std::vector<Value> lower_indices = {index_0};
      Value lowerBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], lower_indices);

      auto index_1 = builder.create<ConstantIndexOp>(loc, 1);
      std::vector<Value> upper_indices = {index_1};
      Value upperBound = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id], upper_indices);
      auto step = builder.create<ConstantIndexOp>(loc, 1);
      auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);

      comet_debug() << " CN Loop\n";
      comet_vdump(loop);

      builder.setInsertionPoint(loop.getBody()->getTerminator());

      std::vector<Value> crd_indices = {loop.getInductionVar()};
      auto get_index = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

      forLoop = loop;
      accessIndex = get_index;
    }
  }

  /// ----------------- ///
  /// In genForOps, generate for-loop for a indexOp node if the index is corresponding to Format "S"
  /// ----------------- ///
  void genForOpFormat_S(OpBuilder &builder,
                        Location &loc,
                        OpsTree *opstree,
                        Value &tensor,
                        unsigned int id,
                        unsigned int i,
                        std::vector<std::vector<Value>> &allAllocs,
                        std::vector<scf::ForOp> &opstree_forops,
                        scf::ForOp &parent_forop,
                        scf::ForOp &forLoop /* output */,
                        Value &accessIndex /* output */)
  {
    /// Currently supported formats, Singleton is not the format of first dimension
    /// and it doesn't produce a loop
    /// Generate: int j = A2crd[m];

    if (tensor.getType().cast<tensorAlgebra::SparseTensorType>())
    {
      comet_debug() << "cur_idx is in tensor " << i << "\n";
      /// Accesing the last level loop info
      scf::ForOp last_forop;
      if (opstree_forops.size() > 0)
      { /// current node contain at least 1 level loop
        last_forop = opstree_forops.back();
      }
      else
      {
        if (opstree->parent != nullptr)
          last_forop = parent_forop;
      }

      std::vector<Value> crd_indices = {last_forop.getInductionVar()};
      auto get_index = builder.create<memref::LoadOp>(loc, allAllocs[i][4 * id + 1], crd_indices);

      /// Adding one iteration loop to provide consistency with the corresponding index tree.
      /// Index tree includes an index node for the dimension but "S" format for this dimension
      /// doesn't produce a loop.
      Value lowerBound = builder.create<ConstantIndexOp>(loc, 0);
      Value upperBound = builder.create<ConstantIndexOp>(loc, 1);
      auto step = builder.create<ConstantIndexOp>(loc, 1);
      auto loop = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      comet_debug() << " S Loop\n";
      comet_vdump(loop);
      forLoop = loop;
      accessIndex = get_index;
    }
    else
    {
      llvm::errs() << "Not supported tensor type\n";
    }
  }

  /// In genForOps, set Insertion Point for symbolic loops.
  void setInsertionPointInSymbolicLoops(OpBuilder &builder,
                                        std::vector<OpsTree *> &ancestorsOps,
                                        OpsTree *opstree)
  {
    /// If parent is for loop, insert into the body, How to get end of body?
    if (ancestorsOps.size() > 0)
    {
      /// ancestorsOps[0] stores the closest parent
      scf::ForOp parent_forop = nullptr;
      comet_debug() << "\n";
      std::vector<scf::ForOp> parent_forops = ancestorsOps[0]->symbolicForOps;
      comet_debug() << " parent_forops.size(): " << parent_forops.size() << " \n";

      parent_forop = parent_forops.back();

      comet_debug() << "symbolic: reset the insertion point\n";
      comet_vdump(parent_forop);

      unsigned int order = findIndexInVector_OpsTree(ancestorsOps[0]->getChildren(), opstree);
      comet_debug() << " order: " << order << "\n";
      if (order == ancestorsOps[0]->getChildren().size())
      {
        llvm::errs() << __LINE__ << "Not belong to parent's children\n";
      }
      else
      {
        /// Get the children of the parent_forop
        comet_debug() << " number of children: " << parent_forops.size() << "\n";
        if (order == 0)
        {
          /// builder.setInsertionPointToStart(parent_forop.getBody());
          comet_debug() << "Insertion point order == 0\n";
          builder.setInsertionPoint(parent_forop.getBody()->getTerminator());
        }
        else
        {
          comet_debug() << "\n";
          std::vector<scf::ForOp> brother_forops = ancestorsOps[0]->getChildren()[order - 1]->symbolicForOps;
          if (brother_forops.size() > 0)
          {
            comet_debug() << " brother_forops.size(): " << brother_forops.size() << "\n";
            if (opstree->symbolicForOps.size() == 0)
            {
              comet_debug() << "\n";
              comet_vdump(brother_forops[0]);
              comet_debug() << "Insertion point (brother_forops.size() > 0 &&  opstree->symbolicForOps.size() == 0)\n";
              builder.setInsertionPointAfter(brother_forops[0]);
            }
            else
            { /// current opstree contains loops, insert in the body of the loops
              comet_debug() << " -------- current opstree contain loops --- impossible\n";
              comet_debug() << "Insertion point (brother_forops.size() > 0 &&  opstree->symbolicForOps.size() != 0)\n";
              builder.setInsertionPoint(opstree->symbolicForOps.back().getBody()->getTerminator());
            }
          }
          else
          {
            comet_debug() << "brothers have no for-loops. Insert at the end of parent's for-loop body.\n";
            ///          builder.setInsertionPointToEnd(parent_forop.getBody());  /// This doesn't work because it inserts even after the scf.yield, which is wrong.
            builder.setInsertionPoint(parent_forop.getBody()->getTerminator());
          }
        }
      }
      comet_debug() << " reset the insertion point\n";
    }
  }

  /// In genCmptOps, generate code for a compute node with workspace transformation.
  /// For example, A = 0.0 . A could be scalar or vector.
  void genWorkspaceCmptOpInitialAssignment(OpBuilder &builder,
                                           Location &loc,
                                           int lhs_loc,
                                           ConstantOp &cstop,
                                           std::vector<scf::ForOp> &nested_forops,
                                           std::vector<std::vector<Value>> &tensors_lhs_Allocs,
                                           std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                           bool use_dynamic_init,
                                           SymbolicInfo &symbolicInfo)
  {

    /// Generate Store 1.0, A[...]  this op
    /// this case: allPerms[0] is empty, allFormats[0] is empty
    comet_vdump(cstop);
    comet_debug() << " cstop.getValue(): " << cstop.getValue() << "\n";
    comet_vdump(main_tensors_all_Allocs[lhs_loc].back());
    comet_debug() << " tensors_lhs_Allocs.size(): " << tensors_lhs_Allocs.size() << "\n";
    {
      comet_vdump(nested_forops[0]);
    }
    Value local_accessIdx = nested_forops[0].getInductionVar();
    insertInitialize(loc,
                     cstop,
                     main_tensors_all_Allocs[lhs_loc].back(),
                     local_accessIdx,
                     builder,
                     use_dynamic_init,
                     symbolicInfo.mtxC_rowptr /* dynamic_init */);
  }

  /// In genCmptOps, generate code for a compute node that copy a sparse input row into a dense vector.
  void genWorkspaceCmptOpScatterInputToWorkspace(OpBuilder &builder,
                                                 Location &loc,
                                                 int main_tensor_nums,
                                                 std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                                 std::vector<std::vector<Value>> &allValueAccessIdx)
  {

    std::vector<Value> allLoads(main_tensor_nums);
    for (auto m = 0; m < main_tensor_nums; m++)
    {
      Value s = builder.create<memref::LoadOp>(loc,
                                               main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1],
                                               allValueAccessIdx[m]);
      allLoads[m] = s;
      comet_debug() << " ";
      comet_vdump(s);
    }
    comet_debug() << " allLoads.size(): " << allLoads.size() << "\n";

    builder.create<memref::StoreOp>(loc, allLoads[0],
                                    main_tensors_all_Allocs[1][main_tensors_all_Allocs[1].size() - 1],
                                    allValueAccessIdx[1]);
  }

  /// Generate scf.for op for indices
  /// The index is the "idx"th index of "tensor"
  void genForOps(std::vector<Value> &tensors,
                 std::vector<unsigned int> &ids,
                 std::vector<std::string> &formats,
                 indexTree::IndexTreeOp rootOp,
                 OpBuilder &builder,
                 OpsTree *opstree,
                 SymbolicInfo &symbolicInfo)
  {
    comet_debug() << " genForOps indexTreeOp\n";
    comet_vdump(rootOp);
    Location loc = rootOp.getLoc();
    /// The insertion location should be "the end of the body of parent loop"
    std::vector<OpsTree *> ancestorsOps;
    getAncestorsOps(opstree, ancestorsOps);
    comet_debug() << " genForOps ancestorsOps.size(): " << ancestorsOps.size() << "\n";
    for (unsigned int i = 0; i < ancestorsOps.size(); i++)
    {
      comet_debug() << " ancestorsOps[" << i << "]->forOps.size(): " << ancestorsOps[i]->forOps.size()
                    << ", ancestorsOps->id: "
                    << ancestorsOps[i]->id << "\n";
    }
    comet_debug() << "Tensor size: " << tensors.size() << "\n";
    std::vector<std::vector<Value>> allAllocs = getAllAllocs(tensors);

    comet_debug() << "Tensors:\n";
    for (unsigned int i = 0; i < tensors.size(); i++)
    {
      comet_vdump(tensors[i]);
    }

    /// ----------------- ///
    /// Set insertion point
    /// ----------------- ///
    setInsertionPointInNumericLoops(builder,
                                    ancestorsOps,
                                    opstree);

    for (unsigned int i = 0; i < tensors.size(); i++)
    {
      if (i > 0)
      {
        /// insertion point: the body of the previous i's loop body
        comet_debug() << " -------- current opstree contain loops\n";
        builder.setInsertionPoint(opstree->forOps.back().getBody()->getTerminator());
      }

      Value &tensor = tensors[i];
      std::string format = formats[i];
      unsigned int id = ids[i];

      comet_debug() << " current index format: " << format << "\n";
      if (format.compare(0, 1, "D") == 0)
      {
        /// Symbolic Phase
        if (symbolicInfo.has_symbolic_phase)
        {
          /// Store the insertion point
          auto last_insertion_point = builder.saveInsertionPoint();

          /// Set the insertions point
          setInsertionPointInSymbolicLoops(builder,
                                           ancestorsOps,
                                           opstree);

          scf::ForOp forLoop;
          Value accessIndex;
          genForOpFormat_D(builder,
                           loc,
                           tensor,
                           id,
                           i,
                           allAllocs,
                           forLoop /* output */,
                           accessIndex /* output */);
          opstree->symbolicForOps.push_back(forLoop);
          opstree->symbolicAccessIdx.push_back(accessIndex);

          /// Restore the insertion point
          builder.restoreInsertionPoint(last_insertion_point);
        }
        /// Check which tensor is sparse, which is dense;
        /// Since this function only handles mixed sparse/dense, then "D" only occurs in one tensor
        /// Both the dense and sparse tensor contain the dim size; But they are different. Use one.
        scf::ForOp forLoop;
        Value accessIndex;
        genForOpFormat_D(builder,
                         loc,
                         tensor,
                         id,
                         i,
                         allAllocs,
                         forLoop /* output */,
                         accessIndex /* output */);
        opstree->forOps.push_back(forLoop);
        opstree->accessIdx.push_back(accessIndex);
      }
      /// mix sparse dense tensor contraction, only one sparse tensor
      else if (format.compare(0, 2, "CU") == 0)
      {
        /// Symbolic Phase
        if (symbolicInfo.has_symbolic_phase)
        {
          /// Store the insertion point
          auto last_insertion_point = builder.saveInsertionPoint();

          /// Set the insertions point
          setInsertionPointInSymbolicLoops(builder,
                                           ancestorsOps,
                                           opstree);

          scf::ForOp forLoop;
          Value accessIndex;
          scf::ForOp parent_forop;
          Value parent_accessIdx;
          if (nullptr != opstree->parent)
          {
            parent_forop = opstree->parent->symbolicForOps.back();
            parent_accessIdx = opstree->parent->symbolicAccessIdx.back();
          }
          genForOpFormat_CU(builder,
                            loc,
                            opstree,
                            tensor,
                            id,
                            i,
                            allAllocs,
                            parent_forop,
                            parent_accessIdx,
                            forLoop /* output */,
                            accessIndex /* output */);
          opstree->symbolicForOps.push_back(forLoop);
          opstree->symbolicAccessIdx.push_back(accessIndex);

          /// Restore the insertion point
          builder.restoreInsertionPoint(last_insertion_point);
        }
        /// Generate for(int m = pos[0]; m < pos[1]; m++){int i = crd[m];}
        /// if i = 0, index is [0,1]
        /// if parent loop and child loop is accessing the same sparse tensor (CSF), index is [m, m+1], m is the nearest loop induction variable
        /// Otherwise, the m comes from load operation of the input sparse tensor such as
        /// j = crd[i];
        /// for (int m = pos[j]; m < pos[j+1]; m++)

        scf::ForOp forLoop;
        Value accessIndex;
        scf::ForOp parent_forop;
        Value parent_accessIdx;
        if (nullptr != opstree->parent)
        {
          parent_forop = opstree->parent->forOps.back();
          parent_accessIdx = opstree->parent->accessIdx.back();
        }
        genForOpFormat_CU(builder,
                          loc,
                          opstree,
                          tensor,
                          id,
                          i,
                          allAllocs,
                          parent_forop,
                          parent_accessIdx,
                          forLoop /* output */,
                          accessIndex /* output */);
        opstree->forOps.push_back(forLoop);
        opstree->accessIdx.push_back(accessIndex);
      }
      else if (format.compare(0, 2, "CN") == 0)
      {
        /// Generate for(int m = pos[0]; m < pos[1]; m++){int i = crd[m];}
        scf::ForOp forLoop;
        Value accessIndex;
        genForOpFormat_CN(builder,
                          loc,
                          tensor,
                          id,
                          i,
                          allAllocs,
                          forLoop /* output */,
                          accessIndex /* output */);
        opstree->forOps.push_back(forLoop);
        opstree->accessIdx.push_back(accessIndex);
      }
      else if (format.compare(0, 1, "S") == 0)
      {
        /// Currently supported formats, Singleton is not the format of first dimension
        /// and it doesn't produce a loop
        /// Generate: int j = A2crd[m];
        scf::ForOp forLoop;
        Value accessIndex;
        std::vector<scf::ForOp> &opstree_forops = opstree->forOps;
        scf::ForOp parent_forop;
        if (nullptr != opstree->parent)
        {
          parent_forop = opstree->parent->forOps.back();
        }
        genForOpFormat_S(builder,
                         loc,
                         opstree,
                         tensor,
                         id,
                         i,
                         allAllocs,
                         opstree_forops,
                         parent_forop,
                         forLoop /* output */,
                         accessIndex /* output */);
        opstree->forOps.push_back(forLoop);
        opstree->accessIdx.push_back(accessIndex);
      }
      else
      {
        llvm::errs() << "Not supported format: " << format << "\n";
      }

      comet_debug() << "finish generate loops for current index format: " << format << "\n";
    }
  }

  Value getSemiringSecondVal(OpBuilder &builder, Location &loc,
                             llvm::StringRef &semiringSecond, Value &Input0, Value &Input1,
                             bool compressedWorkspace)
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
                            llvm::StringRef &semiringFirst, Value &Input0, Value &Input1,
                            bool compressedWorkspace)
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
      if (!compressedWorkspace)
      {
        llvm::errs() << "Not supported semiring operator "
                        "(please use compressed workspace optimization or opt-comp-workspace "
                        "where this operation is known to work): "
                     << "min"
                     << "\n";
        /// we should not proceed forward from this point to avoid in-correct results from generated code.
      }
      Value cmp = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Input0, Input1);
      reduceResult = builder.create<SelectOp>(loc, cmp, Input0, Input1);
    }
    else if (semiringFirst == "max")
    {
      if (!compressedWorkspace)
      {
        llvm::errs() << "Not supported semiring operator "
                        "(please use compressed workspace optimization or opt-comp-workspace "
                        "where this operation is known to work): "
                     << "max"
                     << "\n";
        /// we should not proceed forward from this point to avoid in-correct results from generated code.
      }
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

  /// Generate numeric semiring kernel if statement condition
  void genCmptOpKernelIfStatementCondition(OpBuilder &builder,
                                           Location &loc,
                                           NumericInfo &numericInfo,
                                           MaskingInfo &maskingInfo,
                                           scf::IfOp &if_notAlreadySet /* output */)
  {
    Value &is_visited_alloc = numericInfo.ws_bitmap;
    Value &valueAccessIdx = numericInfo.ws_bitmap_valueAccessIdx;

    Value const_i1_false = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(0));
    Value const_i1_true = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(1));
    if (PUSH_BASED_MASKING == maskingInfo.mask_type)
    {
      ///    if (mask_array[j] == true) {  /// C[i,k] is allowed by the mask and has not been seen yet
      ///       if (ws_bitmap[j] != true) {
      Value &mask_array = numericInfo.mask_array;
      Value ele_mask_array = builder.create<memref::LoadOp>(loc, mask_array, ValueRange{valueAccessIdx});
      Value compare_true = builder.create<CmpIOp>(loc, CmpIPredicate::eq, ele_mask_array, const_i1_true);
      auto if_mask_set = builder.create<scf::IfOp>(loc, compare_true, false /* no else region */);
      builder.setInsertionPointToStart(&if_mask_set.getThenRegion().front());
      Value ele_bitmap = builder.create<memref::LoadOp>(loc, is_visited_alloc, ValueRange{valueAccessIdx});
      Value compare_false = builder.create<CmpIOp>(loc, CmpIPredicate::eq, ele_bitmap, const_i1_false);
      if_notAlreadySet = builder.create<scf::IfOp>(loc, compare_false, /*WithElseRigion*/ true);
      {
        comet_vdump(ele_mask_array);
        comet_vdump(if_mask_set);
        comet_vdump(if_notAlreadySet);
      }
    }
    else if (NO_MASKING == maskingInfo.mask_type)
    {
      ///    if (ws_bitmap[j] != true) {
      /// Workspace tensors are on the lhs
      Value checkAlreadySet = builder.create<memref::LoadOp>(loc, is_visited_alloc, ValueRange{valueAccessIdx});
      Value notAlreadySet = builder.create<CmpIOp>(loc, CmpIPredicate::eq, checkAlreadySet, const_i1_false);
      if_notAlreadySet = builder.create<scf::IfOp>(loc, notAlreadySet, /*WithElseRegion*/ true);
      {
        comet_vdump(checkAlreadySet);
        comet_vdump(notAlreadySet);
        comet_vdump(if_notAlreadySet);
      }
    }
    else
    {
      llvm::errs() << "Error: mask_type " << maskingInfo.mask_type << " is not supported.\n";
    }
  }

  /// Generate numeric semiring kernel if statement then region
  void genCmptOpKernelIfStatementThenRegion(OpBuilder &builder,
                                            Location &loc,
                                            int lhs_loc,
                                            int main_tensor_nums,
                                            scf::IfOp &if_notAlreadySet,
                                            bool compressedWorkspace,
                                            llvm::StringRef &semiringSecond,
                                            std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                            std::vector<std::vector<Value>> &tensors_lhs_Allocs,
                                            std::vector<std::vector<Value>> &allValueAccessIdx,
                                            SymbolicInfo &symbolicInfo,
                                            NumericInfo &numericInfo)
  {
    Value &ws_bitmap = numericInfo.ws_bitmap;
    Value &ws_bitmap_valueAccessIdx = numericInfo.ws_bitmap_valueAccessIdx;
    Value &W_id_list_size = tensors_lhs_Allocs[3][0];
    Value &mtxC_col = symbolicInfo.mtxC_col;
    Value &W_data = main_tensors_all_Allocs[lhs_loc].back();
    Value &W_data_valueAccessIdx = ws_bitmap_valueAccessIdx;

    builder.setInsertionPointToStart(&if_notAlreadySet.getThenRegion().front());

    /// Wj = Aik * Bkj          /// computation wj, outer has k, so +=/= need if/else
    /// W_already_set[j] = 1
    /// W_index_list[W_index_list_size] = j
    /// W_index_list_size++

    std::vector<Value> allLoadsIf(main_tensor_nums);
    for (int m = 0; m < main_tensor_nums; m++)
    {
      Value s = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1], allValueAccessIdx[m]);
      allLoadsIf[m] = s;
      comet_debug() << " ";
      comet_vdump(s);
    }
    comet_debug() << " allLoadsIf.size(): " << allLoadsIf.size() << "\n";

    comet_debug() << "calculate elementWise operation only\n";
    /// val = A[j_idx] * B[j_idx];
    /// W_data[j_idx] = val;
    Value elementWiseResult = getSemiringSecondVal(builder, loc, semiringSecond, allLoadsIf[0], allLoadsIf[1], compressedWorkspace);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    auto store_sum = builder.create<memref::StoreOp>(loc,
                                                     elementWiseResult,
                                                     W_data,
                                                     W_data_valueAccessIdx);
    comet_vdump(elementWiseResult);
    comet_vdump(store_sum);
#else
    builder.create<memref::StoreOp>(loc,
                                    elementWiseResult,
                                    W_data,
                                    W_data_valueAccessIdx);
#endif
    Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
    Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
    Value const_i1_true = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(1));

    /// ws_bitmap[j_idx] = true;
    builder.create<memref::StoreOp>(loc, const_i1_true, ws_bitmap, ws_bitmap_valueAccessIdx);

    Value W_id_list_size_old = builder.create<memref::LoadOp>(loc, W_id_list_size, ValueRange{const_index_0});

    assert(allValueAccessIdx[lhs_loc].size() == 1 && " more than one access id for auxiliary array\n");

    /// C.col[W_id_list_size] = j_idx;
    builder.create<memref::StoreOp>(loc,
                                    ws_bitmap_valueAccessIdx,
                                    mtxC_col,
                                    ValueRange{W_id_list_size_old});

    /// W_id_list_size += 1
    Value W_id_list_size_new = builder.create<AddIOp>(loc, W_id_list_size_old, const_index_1);
    comet_debug() << " AddIOps (W_index_list_size_new)";
    comet_vdump(W_id_list_size_new);

    builder.create<memref::StoreOp>(loc, W_id_list_size_new, W_id_list_size, ValueRange{const_index_0});
    {
      comet_vdump(if_notAlreadySet);
    }
  }

  /// Generate numeric semiring kernel if statement else region
  void genCmptOpKernelIfStatementElseRegion(OpBuilder &builder,
                                            Location &loc,
                                            int lhs_loc,
                                            int main_tensor_nums,
                                            scf::IfOp &if_notAlreadySet,
                                            bool compressedWorkspace,
                                            llvm::StringRef &semiringFirst,
                                            llvm::StringRef &semiringSecond,
                                            std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                            std::vector<std::vector<Value>> &allValueAccessIdx)
  {

    Value &W_data = main_tensors_all_Allocs[lhs_loc].back();
    Value &W_data_valueAccessIdx = allValueAccessIdx[lhs_loc][0];

    builder.setInsertionPointToStart(&if_notAlreadySet.getElseRegion().front());

    std::vector<Value> allLoadsElse(main_tensor_nums);
    for (auto m = 0; m < main_tensor_nums; m++)
    {
      Value s = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1], allValueAccessIdx[m]);
      allLoadsElse[m] = s;
      comet_vdump(s);
    }
    comet_debug() << " allLoadsElse.size(): " << allLoadsElse.size() << "\n";

    comet_debug() << "calculate elementWise operation and reduction\n";
    Value elementWiseResult = getSemiringSecondVal(builder, loc, semiringSecond, allLoadsElse[0], allLoadsElse[1], compressedWorkspace);
    Value reduceResult = getSemiringFirstVal(builder, loc, semiringFirst, allLoadsElse[lhs_loc], elementWiseResult, compressedWorkspace);
    builder.create<memref::StoreOp>(loc, reduceResult, W_data, W_data_valueAccessIdx);
    {
      comet_vdump(if_notAlreadySet);
    }
  }

  /// Generate the numeric bitmap
  /// It should be deprecated in the future, as the bitmap would be lowered from the Index Tree dialect.
  void genNumericBitmap(OpBuilder &builder,
                        Location &loc,
                        scf::ForOp &symbolic_outermost_forLoop,
                        SymbolicInfo &symbolicInfo,
                        Value &bitmap_alloc)
  {
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Jump Insertion Point to the front of the 2nd outermost for-loop
    builder.setInsertionPoint(symbolic_outermost_forLoop);

    Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
    Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
    Value &mtxC_dim2_size = symbolicInfo.mtxC_num_cols;

    MemRefType memTy_dynamic_1i = MemRefType::get({ShapedType::kDynamic}, builder.getI1Type());
    bitmap_alloc = builder.create<memref::AllocOp>(loc,
                                                   memTy_dynamic_1i,
                                                   ValueRange{mtxC_dim2_size},
                                                   builder.getI64IntegerAttr(8) /* alignment bytes */);
    Value const_i1_0 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(0));
    scf::ForOp init_forLoop = builder.create<scf::ForOp>(loc,
                                                         const_index_0 /* lowerBound */,
                                                         mtxC_dim2_size /* upperBound */,
                                                         const_index_1 /* step */);
    builder.setInsertionPointToStart(init_forLoop.getBody());
    Value i_idx = init_forLoop.getInductionVar();
    builder.create<memref::StoreOp>(loc,
                                    const_i1_0,
                                    bitmap_alloc,
                                    ValueRange{i_idx});
    {
      comet_vdump(bitmap_alloc);
      comet_vdump(init_forLoop);
    }

    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// Generate numeric mask-array before the numeric outermost for-loop.
  /// Please don't confuse with mark-array.
  void genNumericMaskArray(OpBuilder &builder,
                           Location &loc,
                           scf::ForOp &numeric_outermost_forLoop,
                           SymbolicInfo &symbolicInfo,
                           NumericInfo &numericInfo /* output */)
  {
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Set the insertion Point before the numeric outermost for-loop
    builder.setInsertionPoint(numeric_outermost_forLoop);

    /// Generate the mask-array
    Value &dim2_size = symbolicInfo.mtxC_num_cols;
    MemRefType memTy_dynamic_i1 = MemRefType::get({ShapedType::kDynamic}, builder.getI1Type());
    Value mask_array_alloc = builder.create<memref::AllocOp>(loc,
                                                             memTy_dynamic_i1,
                                                             ValueRange{dim2_size},
                                                             builder.getI64IntegerAttr(8) /* alignment bytes */);

    /// Initialize the mask-array
    Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
    Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
    scf::ForOp mask_array_init_loop = builder.create<scf::ForOp>(loc,
                                                                 const_index_0 /* lowerBound */,
                                                                 dim2_size /* upperBound */,
                                                                 const_index_1 /* step */);
    builder.setInsertionPointToStart(mask_array_init_loop.getBody());
    Value j_idx = mask_array_init_loop.getInductionVar();
    Value const_i1_false = builder.create<ConstantOp>(loc,
                                                      builder.getI1Type(),
                                                      builder.getBoolAttr(false));
    builder.create<memref::StoreOp>(loc,
                                    const_i1_false,
                                    mask_array_alloc,
                                    ValueRange{j_idx});

    numericInfo.mask_array = mask_array_alloc;

    {
      comet_vdump(mask_array_alloc);
      comet_vdump(mask_array_init_loop);
    }

    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// Generate setting the mask-array at the begining of the numeric outermost for-loop,
  /// and resetting at the end of the outermost for-loop.
  /// ----------------- ///
  ///      %j_loc_start = memref.load %mask_rowptr[%i_idx] : memref<?xindex>  /// alloc_16 = mask.rowptr
  ///      %j_loc_bound = memref.load %mask_rowptr[%i_idx_plus_one] : memref<?xindex>
  ///      scf.for %arg1 = %j_loc_start to %j_loc_bound step %c1 {
  ///        %val = memref.load %mask_val[%arg1] : memref<?xf64>
  ///        %54 = arith.cmpf une, %val, %cst : f64
  ///        scf.if %54 {
  ///          %j_idx = memref.load %mask_col[%arg1] : memref<?xindex>
  ///          memref.store %true, %mask_array[%j_idx] : memref<?xi1>
  ///        }
  ///      }
  /// ----------------- ///
  /// Reset mask_array at the end of numeric outermost for-loop
  /// ----------------- ///
  ///      scf.for %arg1 = %j_loc_start to %j_loc_bound step %c1 {
  ///        %j_idx = memref.load %mask_col[%arg1] : memref<?xindex>
  ///        memref.store %false, %array_mask[%j_idx] : memref<?xi1>
  ///      }
  void genNumericSetAndResetMaskArray(OpBuilder &builder,
                                      Location &loc,
                                      scf::ForOp &numeric_outermost_forLoop,
                                      Value &outermost_forLoop_valueAccessIdx,
                                      NumericInfo &numericInfo,
                                      MaskingInfo &maskingInfo)
  {
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Set the insertion Point before the numeric semiring for-loop
    builder.setInsertionPointToStart(numeric_outermost_forLoop.getBody());

    /// Generate the setting for-loop entry
    Value &mask_array = numericInfo.mask_array;
    Value &mask_rowptr = maskingInfo.mask_rowptr;
    Value &mask_col = maskingInfo.mask_col;
    Value &mask_val = maskingInfo.mask_val;
    Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
    Value &i_idx = outermost_forLoop_valueAccessIdx;
    Value i_idx_plus_one = builder.create<AddIOp>(loc, i_idx, const_index_1);
    Value j_loc_start = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx});
    Value j_loc_bound = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx_plus_one});
    scf::ForOp init_for_loop = builder.create<scf::ForOp>(loc,
                                                          j_loc_start /* lower_bound */,
                                                          j_loc_bound /* upper_bound*/,
                                                          const_index_1 /* step */);

    /// Generate the setting for-loop body
    builder.setInsertionPointToStart(init_for_loop.getBody());
    Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0));
    Value j_loc = init_for_loop.getInductionVar();
    Value val = builder.create<memref::LoadOp>(loc, mask_val, ValueRange{j_loc});
    Value not_zero = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, val, const_f64_0);
    auto if_not_zero = builder.create<scf::IfOp>(loc, not_zero, false /*NoElseRegion*/);
    builder.setInsertionPointToStart(&if_not_zero.getThenRegion().front());
    Value j_idx = builder.create<memref::LoadOp>(loc, mask_col, ValueRange{j_loc});
    Value const_i1_1 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
    builder.create<memref::StoreOp>(loc,
                                    const_i1_1,
                                    mask_array,
                                    ValueRange{j_idx});
    {
      comet_vdump(val);
      comet_vdump(if_not_zero);
      comet_vdump(init_for_loop);
    }

    /// Generate the resetting for-loop entry after the semiring for-loop
    builder.setInsertionPoint(numeric_outermost_forLoop.getBody()->getTerminator());
    scf::ForOp reset_for_loop = builder.create<scf::ForOp>(loc,
                                                           j_loc_start /* lower_bound */,
                                                           j_loc_bound /* upper_bound*/,
                                                           const_index_1 /* step */);

    /// Generate the resetting for-loop body
    builder.setInsertionPointToStart(reset_for_loop.getBody());
    j_loc = reset_for_loop.getInductionVar();
    j_idx = builder.create<memref::LoadOp>(loc, mask_col, ValueRange{j_loc});
    Value const_i1_0 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
    builder.create<memref::StoreOp>(loc,
                                    const_i1_0,
                                    mask_array,
                                    ValueRange{j_idx});
    {
      comet_vdump(reset_for_loop);
      comet_vdump(numeric_outermost_forLoop);
    }

    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  void formSemiringLoopBody(indexTree::IndexTreeComputeOp &cur_op,
                            bool comp_worksp_opt,
                            llvm::StringRef &semiringFirst,
                            llvm::StringRef &semiringSecond,
                            OpBuilder &builder, Location &loc, int lhs_loc,
                            std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                            std::vector<std::vector<Value>> &tensors_lhs_Allocs,
                            std::vector<std::vector<Value>> &tensors_rhs_Allocs,
                            std::vector<std::vector<Value>> &allValueAccessIdx,
                            std::vector<std::vector<Value>> &allAccessIdx,
                            std::vector<scf::ForOp> &forLoops /* numeric for-loop statements, from innermost to outermost*/,
                            std::vector<Value> &numeric_nested_forLoop_AccessIdx,
                            std::vector<scf::ForOp> &symbolic_nested_forops /* symbolic for-loops from innermost to outermost */,
                            std::vector<std::vector<int>> &rhsPerms,
                            SymbolicInfo &symbolicInfo,
                            NumericInfo &numericInfo,
                            MaskingInfo &maskingInfo)
  {
    std::vector<std::vector<std::string>> rhsFormats;
    getRHSFormatsOfComputeOp(cur_op.getOperation()->getResult(0), rhsFormats);
    std::vector<std::vector<std::string>> lhsFormats;
    getLHSFormatsOfComputeOp(cur_op.getOperation()->getResult(0), lhsFormats);
    bool isMixedMode = checkIsMixedMode(rhsFormats);
    bool isElementwise = checkIsElementwise(rhsPerms);
    comet_debug() << " isElementwise:" << isElementwise << " isMixedMode: " << isMixedMode << "\n";
    auto ctx = builder.getContext();
    IndexType indexType = IndexType::get(ctx);

    if ((semiringFirst.size() == 0) || (semiringSecond.size() == 0))
      llvm::errs() << "Error during semiring parsing!"
                   << "\n";

    if (main_tensors_all_Allocs.size() != allValueAccessIdx.size())
      llvm::errs() << "DEBUG ONLY: issue with main_tensor_nums size"
                   << "\n";

    auto f64Type = builder.getF64Type();
    auto const_f64_0 = builder.create<ConstantOp>(loc, f64Type, builder.getF64FloatAttr(0));

    int main_tensor_nums = main_tensors_all_Allocs.size();
    bool compressedWorkspace = false;

    if (comp_worksp_opt) /// always lhs is dense after workspace transformations
    {
      compressedWorkspace = true;

      /// Generate the numeric bitmap
      if (numericInfo.ws_bitmap == nullptr)
      {
        Value bitmap_alloc;
        genNumericBitmap(builder,
                         loc,
                         symbolic_nested_forops.back(),
                         symbolicInfo,
                         bitmap_alloc);
        /// TODO(zpeng): numericInfo.ws_bitmap should be lowered from Index Tree dialect.
        numericInfo.ws_bitmap = bitmap_alloc;
        numericInfo.ws_bitmap_valueAccessIdx = allValueAccessIdx[lhs_loc][0];
      }

      /// Generate the mask-array (please not confuse with mark-array)
      if (PUSH_BASED_MASKING == maskingInfo.mask_type)
      {
        /// Generate numeric mask-array before the numeric outermost for-loop.
        /// Please don't confuse with mark-array.
        genNumericMaskArray(builder,
                            loc,
                            forLoops.back() /* numeric_outermost_forLoop= */,
                            symbolicInfo,
                            numericInfo /* output */);

        /// Generate setting the mask-array before the numeric semiring for-loop and resetting after the semiring for-loop.
        genNumericSetAndResetMaskArray(builder,
                                       loc,
                                       forLoops.back() /* numeric_outermost_forLoop */,
                                       numeric_nested_forLoop_AccessIdx.back() /* outermost_forLoop_valueAccessIdx */,
                                       numericInfo,
                                       maskingInfo);
      }

      /// Value &is_visited_alloc = tensors_lhs_Allocs[1][0];
      /// Value &is_visited_alloc_valAccessIdx = allValueAccessIdx[lhs_loc][0];
      scf::IfOp if_notAlreadySet;
      genCmptOpKernelIfStatementCondition(builder,
                                          loc,
                                          numericInfo,
                                          maskingInfo,
                                          if_notAlreadySet /* output */);

      /// if-then region corresponding to if_notAlreadySet instruction.
      /// if (&if_notAlreadySet. getThenRegion())
      if (!if_notAlreadySet.getThenRegion().empty())
      {
        genCmptOpKernelIfStatementThenRegion(builder,
                                             loc,
                                             lhs_loc,
                                             main_tensor_nums,
                                             if_notAlreadySet,
                                             compressedWorkspace,
                                             semiringSecond,
                                             main_tensors_all_Allocs,
                                             tensors_lhs_Allocs,
                                             allValueAccessIdx,
                                             symbolicInfo,
                                             numericInfo);
      }

      /// if-else region corresponding to if_notAlreadySet instruction.
      /// if (&if_notAlreadySet.getElseRegion())
      if (!if_notAlreadySet.getElseRegion().empty())
      {
        genCmptOpKernelIfStatementElseRegion(builder,
                                             loc,
                                             lhs_loc,
                                             main_tensor_nums,
                                             if_notAlreadySet,
                                             compressedWorkspace,
                                             semiringFirst,
                                             semiringSecond,
                                             main_tensors_all_Allocs,
                                             allValueAccessIdx);
      }
    }
    else
    { /// general dense or mixed mode computation, no need workspace transformations
      std::vector<Value> allLoads(main_tensor_nums);
      for (auto m = 0; m < main_tensor_nums; m++)
      {
        Value load_op = builder.create<memref::LoadOp>(loc,
                                                       main_tensors_all_Allocs[m][main_tensors_all_Allocs[m].size() - 1], allValueAccessIdx[m]);
        allLoads[m] = load_op;
        comet_debug() << " ";
        comet_vdump(load_op);
      }
      comet_debug() << " allLoads.size(): " << allLoads.size() << "\n";

      /// if computeOp is elementwise mixed mode operation, the output is sparse
      if (isMixedMode && isElementwise && !checkIsDense(lhsFormats[0]))
      {

        int dense_inputtensor_id = 0;
        for (unsigned int i = 0; i < rhsFormats.size(); i++)
        {
          if (checkIsDense(rhsFormats[i]))
          {
            dense_inputtensor_id = i;
            break;
          }
        }

        int sparse_inputtensor_id = dense_inputtensor_id ? 0 : 1;
        std::string sparse_format = getTensorFormat(rhsFormats, sparse_inputtensor_id);

        auto last_insertionPoint = builder.saveInsertionPoint();

        /// Need to initialize some memory accesses outside the nested loop
        /// Reset the insertion point: the body of the innermost loop
        comet_debug() << "LoopSize: " << forLoops.size() << " Loop:\n";
        comet_vdump(forLoops[forLoops.size() - 1]);
        builder.setInsertionPoint(forLoops[forLoops.size() - 1]);

        Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
        MemRefType memTy_alloc_Cnnz = MemRefType::get({1}, indexType);
        Value alloc_Cnnz = builder.create<memref::AllocOp>(loc, memTy_alloc_Cnnz);
        comet_debug() << " AllocOp for Cnnz: ";
        comet_vdump(alloc_Cnnz);

        std::vector<Value> alloc_Cnnz_insert_loc = {const_index_0};
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
        auto store_Cnnz = builder.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz, alloc_Cnnz_insert_loc);
        comet_debug() << " StoreOp: ";
        comet_vdump(store_Cnnz);
#else
        builder.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz, alloc_Cnnz_insert_loc);
#endif

        /// The following code block is needed to update Update C2pos in the case of output tensor is in DCSR
        Value Cnnz_index_old;
        Value alloc_Cnnz_row;
        if (sparse_format.compare("DCSR") == 0)
        {
          alloc_Cnnz_row = builder.create<memref::AllocOp>(loc, memTy_alloc_Cnnz);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
          auto store_Cnnz_row = builder.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz_row,
                                                                alloc_Cnnz_insert_loc);
          comet_debug() << " StoreOp DCSR: ";
          comet_vdump(store_Cnnz_row);
#else
          builder.create<memref::StoreOp>(loc, const_index_0, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
#endif
          /// Get Cnnz_old
          Cnnz_index_old = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
        }

        builder.restoreInsertionPoint(last_insertionPoint);

        comet_debug() << " dense_inputtensor_id: " << dense_inputtensor_id << "\n";
        comet_debug() << " sparse_inputtensor_id: " << sparse_inputtensor_id << "\n";
        Value denseInput_is_nonzero = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, allLoads[dense_inputtensor_id],
                                                             const_f64_0);
        auto if_nonzero = builder.create<scf::IfOp>(loc, denseInput_is_nonzero, /*WithElseRegion*/ false);
        comet_debug() << " If branch:\n";
        comet_vdump(if_nonzero);

        if (!if_nonzero.getThenRegion().empty())
        {

          builder.setInsertionPointToStart(&if_nonzero.getThenRegion().front());

          comet_debug() << "calculate product and sum in \n";
          Value elementWiseResult = getSemiringSecondVal(builder, loc, semiringSecond, allLoads[0], allLoads[1],
                                                         compressedWorkspace);

          /// Get Cnnz
          Value Cnnz_index = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);

/// Store product to Cval
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
          comet_debug() << "Store product to Cval\n";
          auto store_Cval = builder.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], Cnnz_index);
          comet_debug() << " StoreOp: ";
          comet_vdump(store_Cval);

          /// Update C1crd, C2crd
          comet_debug() << "Getting A1crd\n";
          comet_debug() << "allValueAccessIdx[" << sparse_inputtensor_id << "].size(): "
                        << allAccessIdx[sparse_inputtensor_id].size() << "\n";
          comet_vdump(allAccessIdx[sparse_inputtensor_id][0]);

          for (unsigned int i = 0; i < allAccessIdx.size(); i++)
          {
            comet_debug() << "allAccessIdx[" << i << "].size(): " << allAccessIdx[i].size() << "\n";
            for (auto n : allAccessIdx[i])
            {
              comet_vdump(n);
            }
          }
#else
          builder.create<memref::StoreOp>(loc, elementWiseResult, main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1], Cnnz_index);
#endif

          comet_debug() << "Store C1crd\n";
          /// Branch out COO... CSR... DCSR...
          if (sparse_format.compare("COO") == 0)
          {
            comet_debug() << "COO format for Elementwise MulOp, update all coordinates\n";
            for (unsigned d = 0; d < rhsPerms[sparse_inputtensor_id].size(); d++)
            {
              Value crd = allAccessIdx[sparse_inputtensor_id][d];
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
              auto store_coo_crd = builder.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1],
                                                                   Cnnz_index);
              comet_debug() << " COO StoreOp: ";
              comet_vdump(store_coo_crd);
#else
              builder.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1], Cnnz_index);
#endif
            }
          }
          else if (sparse_format.compare("CSR") == 0 || sparse_format.compare("DCSR") == 0)
          {
            for (unsigned int d = forLoops.size() - 1; d < rhsPerms[sparse_inputtensor_id].size(); d++)
            {
              Value crd = allAccessIdx[sparse_inputtensor_id][d];
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
              auto store_csr_crd = builder.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1],
                                                                   Cnnz_index);
              comet_debug() << " CSR or DCSR StoreOp: ";
              comet_vdump(store_csr_crd);
#else
              builder.create<memref::StoreOp>(loc, crd, main_tensors_all_Allocs[2][4 * d + 1], Cnnz_index);
#endif
            }
          }

          /// Update Cnnz
          comet_debug() << "Update Cnnz\n";
          Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
          Value new_Cnnz_index = builder.create<AddIOp>(loc, Cnnz_index, const_index_1);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
          comet_debug() << "AddIOps (new_Cnnz_index): ";
          comet_vdump(new_Cnnz_index);
          auto store_updated_cnnz = builder.create<memref::StoreOp>(loc, new_Cnnz_index, alloc_Cnnz,
                                                                    alloc_Cnnz_insert_loc);
          comet_debug() << " Update Cnnz (store new value) StoreOp: ";
          comet_vdump(store_updated_cnnz);
#else
          builder.create<memref::StoreOp>(loc, new_Cnnz_index, alloc_Cnnz, alloc_Cnnz_insert_loc);
#endif
        }

        /// Need to identify dense tensor upperbound to be able to update Cpos and Csize arrays
        std::vector<Value> denseAllocs = tensors_rhs_Allocs[dense_inputtensor_id];
        assert(denseAllocs.size() == 1);

        comet_debug() << " DenseAllocs: ";
        auto inputType = denseAllocs[0].getType();
        std::vector<Value> denseDimsSize;
        for (unsigned rank = 0; rank < inputType.cast<mlir::MemRefType>().getRank(); rank++)
        {
          auto dimSize = inputType.cast<mlir::MemRefType>().getDimSize(rank);
          Value upperBound;
          if (dimSize == ShapedType::kDynamic)
          {
            comet_debug() << " This dimension is a dynamic size:\n";
            unsigned dynamicDimPos = inputType.dyn_cast<MemRefType>().getDynamicDimIndex(rank);
            comet_debug() << " DynamicDimPos: " << dynamicDimPos << "\n";
            upperBound = denseAllocs[0].getDefiningOp()->getOperand(dynamicDimPos);
            comet_vdump(upperBound);
          }
          else
          {
            builder.setInsertionPoint(forLoops[1]); /// [PT] This is needed to make sure we can access the values created outside of the loop
            comet_debug() << " This dimension is a static size\n";
            upperBound = builder.create<ConstantIndexOp>(loc, dimSize);
            comet_vdump(upperBound);
          }

          denseDimsSize.push_back(upperBound);
        }
        builder.setInsertionPointAfter(forLoops[0]);

        /// To update Cpos
        if (sparse_format.compare("CSR") == 0)
        {
          builder.setInsertionPointAfter(forLoops[0]);
          Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
          Value arg0_next = builder.create<AddIOp>(loc, forLoops[1].getInductionVar(), const_index_1);
          comet_debug() << "AddIOp (arg0_next): ";
          comet_vdump(arg0_next);

          Value Cnnz_index_final = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
          builder.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][4], arg0_next); /// 2

          builder.setInsertionPointAfter(forLoops[1]);
          /// Update C2pos[0]
          comet_debug() << "Update C2pos[0]\n";
          std::vector<Value> insert_loc_0 = {const_index_0};
          builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][4], insert_loc_0); /// 2

          /// Update C1pos[0]
          comet_debug() << "Update C1pos[0]\n";
          Value dim0_index = denseDimsSize[0];
          builder.create<memref::StoreOp>(loc, dim0_index, main_tensors_all_Allocs[2][0], insert_loc_0);
        }
        else
        {
          if (sparse_format.compare("DCSR") == 0)
          {
            /// Update C2pos
            comet_debug() << "Update DCSR C2pos\n";
            builder.setInsertionPointAfter(forLoops[0]);
            auto Cnnz_index_new = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
            auto has_nnz_row = builder.create<CmpIOp>(loc, CmpIPredicate::ne, Cnnz_index_new, Cnnz_index_old);
            auto has_nnz_row_ifOp = builder.create<scf::IfOp>(loc, has_nnz_row, /*WithElseRegion*/ false);
            comet_debug() << " If branch:\n";
            comet_vdump(has_nnz_row_ifOp);

            if (!has_nnz_row_ifOp.getThenRegion().empty())
            {
              builder.setInsertionPointToStart(&has_nnz_row_ifOp.getThenRegion().front());

              Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
              Value arg0_next = builder.create<AddIOp>(loc, forLoops[1].getInductionVar(), const_index_1);
              comet_debug() << "AddIOp (arg0_next): ";
              comet_vdump(arg0_next);

              Value Cnnz_index_final = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
              builder.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][4], arg0_next); /// C2pos //2
              Value Cnnz_row_index = builder.create<memref::LoadOp>(loc, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
              Value idx_i = allAccessIdx[sparse_inputtensor_id][0];
              builder.create<memref::StoreOp>(loc, /*i*/ idx_i, main_tensors_all_Allocs[2][1], Cnnz_row_index); /// C1crd
              Value Cnnz_row_index_new = builder.create<AddIOp>(loc, Cnnz_row_index, const_index_1);
              comet_debug() << "AddIOp (Cnnz_row_index_new): ";
              comet_vdump(Cnnz_row_index_new);
              builder.create<memref::StoreOp>(loc, Cnnz_row_index_new, alloc_Cnnz_row,
                                              alloc_Cnnz_insert_loc); /// Update Cnnz_row
            }

            builder.setInsertionPointAfter(forLoops[1]);
            Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
            std::vector<Value> insert_loc_1 = {const_index_1};

            /// Update C2pos[0]
            std::vector<Value> insert_loc_0 = {const_index_0};
            builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][4], insert_loc_0); /// 2

            /// Update C1pos[0], C1pos[1]
            Value Cnnz_row_index = builder.create<memref::LoadOp>(loc, alloc_Cnnz_row, alloc_Cnnz_insert_loc);
            builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][0], insert_loc_0);
            builder.create<memref::StoreOp>(loc, Cnnz_row_index, main_tensors_all_Allocs[2][0], insert_loc_1);
          }
          else
          {
            if (sparse_format.compare("COO") == 0)
            {
              /// Finally, Update C1pos
              comet_debug() << "Update C1pos\n";
              builder.setInsertionPointAfter(forLoops[0]);
              Value Cnnz_index_final = builder.create<memref::LoadOp>(loc, alloc_Cnnz, alloc_Cnnz_insert_loc);
              Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
              builder.create<memref::StoreOp>(loc, const_index_0, main_tensors_all_Allocs[2][0], const_index_0);
              builder.create<memref::StoreOp>(loc, Cnnz_index_final, main_tensors_all_Allocs[2][0], const_index_1);
            }
            else
              llvm::errs() << "/// Coordinate values are not updated for output sparse tensor in " << sparse_format
                           << " format\n";
          }
        }

      } /// end if (isMixedMode && isElementwise)
      else
      {
        /// calculate elementWise operation and reduction for general dense or mix mode computation (which has dense output)
        comet_debug()
            << "calculate elementWise operation and reduction for general dense or mix mode computation (which has dense output)\n";
        Value elementWiseResult = getSemiringSecondVal(builder, loc, semiringSecond, allLoads[0], allLoads[1],
                                                       compressedWorkspace);
        Value reduceResult = getSemiringFirstVal(builder, loc, semiringFirst, allLoads[2], elementWiseResult,
                                                 compressedWorkspace);
        builder.create<memref::StoreOp>(loc, reduceResult,
                                        main_tensors_all_Allocs[2][main_tensors_all_Allocs[2].size() - 1],
                                        allValueAccessIdx[2]);
      }
    }
  }

  /// ----------------- ///
  /// Generate Cij = Wj node, gathering the results in the workspace to the sparse output C.val.
  /// Called by genCmptOps().
  /// ----------------- ///
  ///     sort(C.col, C.rowptr[i_idx], C.rowptr[i_idx + 1]);
  ///     for (int j_loc = C.rowptr[i_idx]; j_loc < C.rowptr[i_idx + 1]; ++j_loc) {
  ///       int j_idx = C.col[j_loc];
  ///       C.val[j_idx] = W_data[j_idx];
  ///       is_visited[j_idx] = false;
  ///     }
  /// ----------------- ///
  ///      %rowptr_bound = memref.load %rowptr[%c0] : memref<1xindex>
  ///      %C_col_ptr = memref.cast %C_col : memref<?xindex> to memref<*xindex>
  ///      func.call @comet_sort_index(%C_col_ptr, %rowptr_start, %rowptr_bound) : (memref<*xindex>, index, index) -> ()
  ///
  ///      scf.for %ptr = %rowptr_start to %rowptr_bound step %c1 {
  ///        %c_col_id = memref.load %C_col[%ptr] : memref<?xindex>       /// c_col_id = C_col[ptr]
  ///        %data = memref.load %ws_data[%c_col_id] : memref<?xf64>      /// data = ws_data[c_col_id]
  ///        memref.store %data, %C_val[%ptr] : memref<?xf64>             /// C_val[ptr] = data
  ///        memref.store %false, %ws_bitmap[%c_col_id] : memref<?xi1>    /// ws_bitmap[c_col_id] = false
  ///      }
  void genWorkspaceCmptOpGatherFromWorkspaceToOutput(OpBuilder &builder,
                                                     Location &loc,
                                                     std::vector<std::vector<Value>> &tensors_rhs_Allocs,
                                                     std::vector<scf::ForOp> &nested_forops,
                                                     std::vector<Value> &nested_AccessIdx,
                                                     SymbolicInfo &symbolicInfo,
                                                     NumericInfo &numericInfo)
  {
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    assert(nested_forops.size() >= 2 && nested_AccessIdx.size() >= 2 && "Error: should be at least 2 levels of for-loop.\n");
    scf::ForOp &curr_for_loop = nested_forops[0];
    scf::ForOp parent_for_loop = nested_forops[1];

    /// Set the insertion point before the innermost for-loop
    builder.setInsertionPoint(curr_for_loop);

    /// Get and set the boundary of current for-loop
    /// %rowptr_start = memref::LoadOp %C_rowptr[%i_idx] : memref<?xindex>
    /// %id_idx_plus_one = arith.addi %rowptr_start, %c1 : index
    /// %rowptr_bound = memref::LoadOp %C_rowptr[%i_idx_plus_one] : memref<?xindex>
    Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
    Value i_idx = nested_AccessIdx[1];
    Value i_idx_plus_one = builder.create<AddIOp>(loc, i_idx, const_index_1);
    Value &mtxC_rowptr = symbolicInfo.mtxC_rowptr;
    Value rowptr_start = builder.create<memref::LoadOp>(loc, mtxC_rowptr, ValueRange{i_idx});
    Value rowptr_bound = builder.create<memref::LoadOp>(loc, mtxC_rowptr, ValueRange{i_idx_plus_one});
    {
      comet_vdump(parent_for_loop);
      comet_vdump(i_idx);
      comet_vdump(rowptr_start);
      comet_vdump(rowptr_bound);
    }

    /// Generate calling comet_sort_index
    /// %C_col_ptr = memref.cast %C_col : memref<?xindex> to memref<*xindex>
    /// func.call @comet_sort_index(%C_col_ptr, %rowptr_start, %rowptr_bound) : (memref<*xindex>, index, index) -> ()
    std::string func_name = "comet_sort_index";
    Value &mtxC_col = symbolicInfo.mtxC_col;
    IndexType indexType = IndexType::get(builder.getContext());
    Value C_col_cast = builder.create<memref::CastOp>(loc,
                                                      UnrankedMemRefType::get(indexType, 0),
                                                      mtxC_col);
    builder.create<func::CallOp>(loc,
                                 func_name,
                                 SmallVector<Type, 3>{},
                                 ValueRange{C_col_cast, rowptr_start, rowptr_bound});

    /// Change current for-loop boundaries
    curr_for_loop.setLowerBound(rowptr_start);
    curr_for_loop.setUpperBound(rowptr_bound);

    /// Generate current for-loop body
    Value &mtxC_val = symbolicInfo.mtxC_val;
    Value &ws_data = tensors_rhs_Allocs[0][0];
    Value &ws_bitmap = numericInfo.ws_bitmap;
    Value rowptr = curr_for_loop.getInductionVar();
    builder.setInsertionPointToStart(curr_for_loop.getBody());
    Value c_col_id = builder.create<memref::LoadOp>(loc, mtxC_col, ValueRange{rowptr});
    Value data = builder.create<memref::LoadOp>(loc, ws_data, ValueRange{c_col_id});
    builder.create<memref::StoreOp>(loc,
                                    data,
                                    mtxC_val,
                                    ValueRange{rowptr});
    Value const_i1_0 = builder.create<ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
    builder.create<memref::StoreOp>(loc,
                                    const_i1_0,
                                    ws_bitmap,
                                    ValueRange{c_col_id});
    {
      comet_vdump(c_col_id);
      comet_vdump(data);
      comet_vdump(curr_for_loop);
    }

    /// Free up ws_data and ws_bitmap after
    builder.setInsertionPointAfter(parent_for_loop);
    builder.create<memref::DeallocOp>(loc, ws_data);
    builder.create<memref::DeallocOp>(loc, ws_bitmap);

    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// In genCmptOps, get current compute node's numeric nested for-loop and access indices.
  void getNumericNestedForOpsAndAccessIdx(std::vector<Value> &ancestorsWps,
                                          std::vector<OpsTree *> &ancestorsOps,
                                          std::vector<scf::ForOp> &nested_forops /* output */,
                                          std::vector<Value> &nested_AccessIdx /* output */,
                                          std::vector<int64_t> &nested_forops_indices /* output */)
  {

    for (unsigned int i = 0; i < ancestorsOps.size(); i++)
    {
      comet_debug() << " ancestorsOps[" << i << "]->forOps.size(): " << ancestorsOps[i]->forOps.size()
                    << ", ancestorsOps->id: "
                    << ancestorsOps[i]->id << "\n";
      if (!ancestorsOps[i]->forOps.empty())
      { /// for loops OpsTree node
        for (int j = ancestorsOps[i]->forOps.size() - 1; j >= 0; j--)
        {
          comet_debug() << " j: " << j << "\n";
          nested_forops.push_back(ancestorsOps[i]->forOps[j]);
          comet_debug() << "AccessIdx: " << ancestorsOps[i]->accessIdx[j] << "\n";
          nested_AccessIdx.push_back(ancestorsOps[i]->accessIdx[j]);
        }
      }
    }
    comet_debug() << " nested_forops.size(): " << nested_forops.size() << "\n";
    for (unsigned int i = 0; i < ancestorsWps.size(); i++)
    {
      comet_debug() << " ";
      comet_vdump(ancestorsWps[i]);

      if (indexTree::IndexTreeIndicesOp cur_op = dyn_cast<mlir::indexTree::IndexTreeIndicesOp>(
              ancestorsWps[i].getDefiningOp()))
      {
        /// Get indices
        ArrayAttr op_indices = cur_op.getIndices();

        if (op_indices.size() > 0)
        { /// for loops OpsTree node
          for (int j = op_indices.size() - 1; j >= 0; j--)
          {
            /// Get the indices;
            int64_t idx = op_indices[j].cast<IntegerAttr>().getInt();
            nested_forops_indices.push_back(idx);
          }
        }
      }
    }
  }

  /// In genCmptOps, get current compute node's RHS, LHS, tensors, formats, perms, etc.
  void getNumericTensors(indexTree::IndexTreeComputeOp &cur_op,
                         std::vector<Value> &tensors_rhs /* output */,
                         std::vector<std::vector<Value>> &tensors_lhs_Allocs /* output */,
                         std::vector<std::vector<Value>> &tensors_rhs_Allocs /* output */,
                         std::vector<std::vector<std::string>> &allFormats /*output*/,
                         std::vector<std::vector<int>> &allPerms /* output */,
                         std::vector<std::vector<int>> &allPerms_rhs /* output */,
                         std::vector<Value> &main_tensors_all /* output */,
                         std::vector<Value> &main_tensors_rhs /* output */)
  {
    comet_vdump(cur_op);
    for (auto n : cur_op.getRhs())
    {
      comet_debug() << " ";
      comet_vdump(n);
      for (unsigned i = 0; i < n.getDefiningOp()->getNumOperands(); i++)
      {
        comet_debug() << " ";
        comet_vdump(n.getDefiningOp()->getOperand(i));
        tensors_rhs.push_back(n.getDefiningOp()->getOperand(i));
      }
    }

    std::vector<Value> tensors_lhs; /// inner
    for (unsigned i = 0; i < cur_op.getLhs().getDefiningOp()->getNumOperands(); i++)
    {
      comet_debug() << " ";
      comet_vdump(cur_op.getLhs().getDefiningOp()->getOperand(i));
      tensors_lhs.push_back(cur_op.getLhs().getDefiningOp()->getOperand(i));
    }

    /// Currently, only one case, the rhs is constant. Wj = 0.0;
    tensors_lhs_Allocs = getAllAllocs(tensors_lhs); /// output
    comet_debug() << " tensors_lhs_Allocs.size(): " << tensors_lhs_Allocs.size() << "\n";
    tensors_rhs_Allocs = getAllAllocs(tensors_rhs); /// output
    comet_debug() << " tensors_rhs_Allocs.size(): " << tensors_rhs_Allocs.size() << "\n";

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    comet_debug() << " tensors_rhs_Allocs: \n";
    for (auto m : tensors_rhs_Allocs)
    {
      comet_debug() << " ";
      for (auto n : m)
      {
        comet_vdump(n);
      }
      comet_debug() << "\n";
    }
#endif

    getPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms);

    comet_debug() << " allPerms: \n";
    for (auto m : allPerms)
    {
      comet_debug() << " "; /// print_vector(m);
      for (auto n : m)
      {
        comet_debug() << n << " ";
      }
      comet_debug() << "\n";
    }

    getFormatsOfComputeOp(cur_op.getOperation()->getResult(0), allFormats);
    comet_debug() << " allFormats: \n";
    for (auto m : allFormats)
    {
      comet_debug() << " ";
      for (auto n : m)
      {
        comet_debug() << n << " ";
      }
      comet_debug() << "\n";
    }

    comet_debug() << " ";
    comet_vdump(cur_op);

    assert(allPerms.size() == allFormats.size() && "allPerms.size() != allFormats.size()\n");
    for (unsigned int m = 0; m < allPerms.size(); m++)
    {
      assert(allPerms[m].size() == allFormats[m].size() && "allPerms[m].size() != allFormats[m].size()\n");
    }
    comet_debug() << " allPerms.size(): " << allPerms.size() << "\n";
    /// tensor_nums means the actual tensors except the auxiliary tensors
    /// Suppose for LHSOp, there are "n" real tensors, then allPerms[m].size()

    getRHSPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms_rhs);
    comet_debug() << " allPerms_rhs.size(): " << allPerms_rhs.size() << "\n";
    std::vector<std::vector<int>> allPerms_lhs; /// inner
    getLHSPermsOfComputeOp(cur_op.getOperation()->getResult(0), allPerms_lhs);

    comet_debug() << " allPerms_lhs.size(): " << allPerms_lhs.size() << "\n";
    std::vector<Value> main_tensors_lhs; /// inner
    if (tensors_rhs.size() == allPerms_rhs.size())
    { /// all are "main" tensors
      main_tensors_rhs.insert(main_tensors_rhs.end(), tensors_rhs.begin(), tensors_rhs.end());
    }
    else
    { /// the rhs contains the auxiliary tensors
      assert(allPerms_rhs.size() == 1 &&
             " rhs contains auxiliary tensors and main tensors at the same time, not support currently\n"); /// only 1 main tensor on rhs
      main_tensors_rhs.push_back(tensors_rhs[0]);
    }
    comet_debug() << " main_tensors_rhs.size(): " << main_tensors_rhs.size() << "\n";

    if (tensors_lhs.size() == allPerms_lhs.size())
    { /// all are "main" tensors
      main_tensors_lhs.insert(main_tensors_lhs.end(), tensors_lhs.begin(), tensors_lhs.end());
    }
    else
    { /// the lhs contains the auxiliary tensors
      assert(allPerms_lhs.size() == 1 &&
             " lhs contains auxiliary tensors and main tensors at the same time, not support currently\n"); /// only 1 main tensor on lhs
      main_tensors_lhs.push_back(tensors_lhs[0]);
    }
    comet_debug() << " main_tensors_lhs.size(): " << main_tensors_lhs.size() << "\n";

    main_tensors_all = main_tensors_rhs;
    main_tensors_all.insert(main_tensors_all.end(), main_tensors_lhs.begin(), main_tensors_lhs.end());
    comet_debug() << " main_tensors_all.size(): " << main_tensors_all.size() << "\n";
  }

  /// In genCmptOps, get for-loops' value access indices.
  /// A value access index is not necessarily the for-loop's induction variable.
  /// For example, To access sparse matrix C.val, we need to get rowptr = C.col[idx], then rowptr is the access index.
  /// This function is used both by Numeric Phase and Symbolic Phase.
  void getForLoopsValueAccessIdx(OpBuilder &builder,
                                 Location &loc,
                                 int main_tensor_nums,
                                 std::vector<std::vector<int>> &allPerms,
                                 std::vector<std::vector<std::string>> &allFormats,
                                 std::vector<Value> &main_tensors_all,
                                 std::vector<scf::ForOp> &nested_forops,
                                 std::vector<Value> &nested_AccessIdx,
                                 std::vector<int64_t> &nested_forops_indices,
                                 std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                 std::vector<std::vector<Value>> &allAccessIdx /* output */,
                                 std::vector<std::vector<Value>> &allValueAccessIdx /* output */)
  {

    std::vector<std::vector<Value>> allLoopsArg(main_tensor_nums); /// inner
                                                                   ///  std::vector<std::vector<Value>> allAccessIdx(main_tensor_nums);  /// output
    for (unsigned int i = 0; i < main_tensors_all.size(); i++)
    {
      for (unsigned int j = 0; j < allPerms[i].size(); j++)
      {
        unsigned int index_loc = findIndexInVector<int64_t>(nested_forops_indices, allPerms[i][j]);
        comet_debug() << " index_loc " << index_loc << "\n";
        comet_debug() << " Perm: " << allPerms[i][j] << "\n";
        comet_debug() << " Format: " << allFormats[i][j] << "\n";
        assert(index_loc < nested_forops.size() &&
               "index_loc < nested_forops.size(), i.e. the index not exist in nested for loop\n");
        allLoopsArg[i].push_back(nested_forops[index_loc].getInductionVar());
        allAccessIdx[i].push_back(nested_AccessIdx[index_loc]);
      }
      /// Consider for the case w_index_list_size
      /// if allPerms[i].size() == 0
    }

    ///  std::vector<std::vector<Value>> allValueAccessIdx(main_tensor_nums);  /// output
    for (int i = 0; i < main_tensor_nums; i++)
    { /// If constantOp, do not consider it
      comet_debug() << " ";
      comet_vdump(main_tensors_all[i]);
      if (main_tensors_all[i].getType().isa<tensorAlgebra::SparseTensorType>())
      { /// sparse tensor

        /// Find the last sparse index m, then loop_arg * all dense loop args
        unsigned lastSparseIndexLoc = allPerms[i].size();
        for (int d = (int)allPerms[i].size() - 1; d >= 0; d--)
        {
          if (allFormats[i][d].compare(0, 1, "D") != 0 &&
              allFormats[i][d].compare(0, 1, "S") != 0)
          { /// sparse dimension and has a loop, i.e. "CU" or "CN"
            lastSparseIndexLoc = d;
            break;
          }
        }
        /// Calculate for ModeGeneric style format: [CN, S, D (, ... ) ]
        auto valueAccessIdx_part = allLoopsArg[i][lastSparseIndexLoc];
        if (lastSparseIndexLoc < allPerms[i].size() - 1)
        { /// There is dense index after the sparse index
          unsigned int last_d = lastSparseIndexLoc + 1;
          for (unsigned int d = lastSparseIndexLoc + 1; d < allPerms[i].size(); d++)
          { /// i=0
            if (allFormats[i][d].compare(0, 1, "D") == 0)
            {
              /// Get dense dim size
              auto index_0 = builder.create<ConstantIndexOp>(loc, 0);
              std::vector<Value> upper_indices = {index_0};
              auto upperBound = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[i][4 * d], upper_indices);
              comet_vdump(upperBound);
              valueAccessIdx_part = builder.create<MulIOp>(loc, upperBound, valueAccessIdx_part);
              last_d = d;
            }
          }
          if (allFormats[i][last_d].compare(0, 1, "D") == 0)
          {
            comet_debug() << " ";
            comet_vdump(allLoopsArg[i][allLoopsArg[i].size() - 1]);
            comet_vdump(valueAccessIdx_part);
            valueAccessIdx_part = builder.create<AddIOp>(loc, allLoopsArg[i][allLoopsArg[i].size() - 1],
                                                         valueAccessIdx_part);
            comet_debug() << " AddIOps (valueAccessIdx_part): ";
            comet_vdump(valueAccessIdx_part);
          }
        }

        allValueAccessIdx[i].push_back(valueAccessIdx_part);
      }
      else if (main_tensors_all[i].getType().isa<TensorType>())
      { /// dense tensor
        allValueAccessIdx[i] = allAccessIdx[i];
      }
    }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    for (unsigned int i = 0; i < allValueAccessIdx.size(); i++)
    {
      comet_debug() << "allValueAccessIdx[" << i << "].size(): " << allValueAccessIdx[i].size()
                    << ", main_tensors_all_Allocs[" << i << "].size()-1: " << main_tensors_all_Allocs[i].size() - 1
                    << "\n";
    }
#endif
  }

  /// In genCmptOps, get current compute node's symbolic nested for-loop and access indices.
  void getSymbolicNestedForOpsAndAccessIdx(std::vector<Value> &ancestorsWps,
                                           std::vector<OpsTree *> &ancestorsOps,
                                           std::vector<scf::ForOp> &nested_forops /* output */,
                                           std::vector<Value> &nested_AccessIdx /* output */,
                                           std::vector<int64_t> &nested_forops_indices /* output */)
  {

    for (unsigned int i = 0; i < ancestorsOps.size(); i++)
    {
      comet_debug() << " ancestorsOps[" << i << "]->forOps.size(): " << ancestorsOps[i]->symbolicForOps.size()
                    << ", ancestorsOps->id: "
                    << ancestorsOps[i]->id << "\n";
      if (!ancestorsOps[i]->symbolicForOps.empty())
      { /// for loops OpsTree node
        for (int j = ancestorsOps[i]->symbolicForOps.size() - 1; j >= 0; j--)
        {
          comet_debug() << " j: " << j << "\n";
          nested_forops.push_back(ancestorsOps[i]->symbolicForOps[j]);
          comet_debug() << "AccessIdx: " << ancestorsOps[i]->symbolicAccessIdx[j] << "\n";
          nested_AccessIdx.push_back(ancestorsOps[i]->symbolicAccessIdx[j]);
        }
      }
    }
    comet_debug() << " nested_forops.size(): " << nested_forops.size() << "\n";
    ///  std::vector<int64_t> nested_forops_indices;
    for (unsigned int i = 0; i < ancestorsWps.size(); i++)
    {
      comet_debug() << " ";
      comet_vdump(ancestorsWps[i]);

      if (indexTree::IndexTreeIndicesOp cur_op = dyn_cast<mlir::indexTree::IndexTreeIndicesOp>(
              ancestorsWps[i].getDefiningOp()))
      {
        /// Get indices
        ArrayAttr op_indices = cur_op.getIndices();

        if (op_indices.size() > 0)
        { /// for loops OpsTree node
          for (int j = op_indices.size() - 1; j >= 0; j--)
          {
            /// Get the indices;
            int64_t idx = op_indices[j].cast<IntegerAttr>().getInt();
            nested_forops_indices.push_back(idx);
          }
        }
      }
    }
  }

  /// In genCmptOps, generate code for a compute node that does general A = 0.0 but without workspace transformation.
  void genCmptOpGeneralInitialAssignment(OpBuilder &builder,
                                         Location &loc,
                                         int lhs_loc,
                                         ConstantOp &cstop,
                                         std::vector<scf::ForOp> &nested_forops,
                                         std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                         std::vector<std::vector<Value>> &allValueAccessIdx)
  {
    /// Generate Store 1.0, A[...]  this op
    /// this case: allPerms[0] is empty, allFormats[0] is empty
    comet_debug() << " cstop.getValue(): " << cstop.getValue() << "\n";
    comet_debug() << " ";
    comet_vdump(main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1]);

    if (allValueAccessIdx[lhs_loc].size() > 0)
    {
      builder.create<memref::StoreOp>(loc, cstop,
                                      main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() -
                                                                       1],
                                      allValueAccessIdx[lhs_loc]);
    }
    else
    {
      Value local_accessIdx = nested_forops[0].getInductionVar();
      insertInitialize(loc,
                       cstop,
                       main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 1],
                       local_accessIdx,
                       builder,
                       false /* use_dynamic_init */,
                       nullptr /* dynamic_init */);
    }
  }

  /// In genCmptOps, get LHS nnz value and data array before gathering results from the workspace.
  void getLHSBeforeGatherFromWorkspace(OpBuilder &builder,
                                       Location &loc,
                                       int lhs_loc,
                                       Value lhs,
                                       std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                       unsigned int &lhs_2crd_size_loc /* output */,
                                       unsigned int &lhs_2pos_size_loc /* output */,
                                       Value &lhs_nnz /* output */,
                                       Value &lhs_nnz_alloc /* output */,
                                       Value &lhs_val /* output */)
  {
    /// Get tensor ranks
    auto sp_op = cast<tensorAlgebra::SparseTensorConstructOp>(lhs.getDefiningOp());
    int lhs_ranks = sp_op.getTensorRank();

    //[0...2d,2d+1...4d+1,4d+2...5d+1]
    unsigned int lhs_val_size_loc = 8 * lhs_ranks + 1; /// 17 (2d)  /// 15
    lhs_2crd_size_loc = 7 * lhs_ranks;                 /// 14 (2d)  /// 12  /// output
    lhs_2pos_size_loc = 7 * lhs_ranks - 1;             /// 13 (2d)  /// 11  /// output

    /// [0...2d, 2d+1...4d+1, 4d+2...5d+1]
    comet_pdump(lhs.getDefiningOp());
    comet_pdump(lhs.getDefiningOp()->getParentOp());
    comet_vdump(lhs.getDefiningOp()->getOperand(lhs_val_size_loc));

    Value lhs_nnz_operand = lhs.getDefiningOp()->getOperand(lhs_val_size_loc);
    Value lhs_nnz_op;
    comet_vdump(lhs_nnz_operand);
    if (isa<IndexCastOp>(lhs_nnz_operand.getDefiningOp()))
    {
      lhs_nnz_op = lhs_nnz_operand.getDefiningOp()->getOperand(0);
    }
    else
    {
      lhs_nnz_op = lhs_nnz_operand;
    }
    comet_vdump(lhs_nnz_op);
    auto lhs_nnz_load = cast<memref::LoadOp>(lhs_nnz_op.getDefiningOp());            /// index
    lhs_nnz_alloc = cast<memref::AllocOp>(lhs_nnz_load.getMemRef().getDefiningOp()); /// index  /// output

    Value cst_0_index = builder.create<ConstantIndexOp>(loc, 0);
    lhs_nnz = builder.create<memref::LoadOp>(loc, lhs_nnz_alloc, ValueRange{cst_0_index}); /// output

    lhs_val = main_tensors_all_Allocs[lhs_loc].back(); /// output
    comet_vdump(lhs_val);
  }

  /// In genCmptOps, generate code for Cij = Wj when they are both dense.
  void genCmptOpGatherFromDenseToDense(OpBuilder &builder,
                                       Location &loc,
                                       int rhs_loc,
                                       int lhs_loc,
                                       std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                       std::vector<std::vector<Value>> &allValueAccessIdx)
  {
    /// %1 = load b[...]
    /// store %1, a[...]
    comet_debug() << " main_tensors_all_Allocs[" << rhs_loc << "].size(): "
                  << main_tensors_all_Allocs[rhs_loc].size() << ", allValueAccessIdx[" << rhs_loc << "].size(): "
                  << allValueAccessIdx[rhs_loc].size() << "\n";

    Value rhs_value = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[rhs_loc].back(), allValueAccessIdx[rhs_loc]);
    comet_vdump(rhs_value);

    comet_vdump(main_tensors_all_Allocs[lhs_loc].back());
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    auto s1 = builder.create<memref::StoreOp>(loc, rhs_value, main_tensors_all_Allocs[lhs_loc].back(), allValueAccessIdx[lhs_loc]);
    comet_vdump(s1);
#else
    builder.create<memref::StoreOp>(loc, rhs_value, main_tensors_all_Allocs[lhs_loc].back(), allValueAccessIdx[lhs_loc]);
#endif
  }

  /// Used by genCmptOps, for Cij = Wj without Workspace Transformation
  void genCmptOpGatherFromDenseToOutput(OpBuilder &builder,
                                        Location &loc,
                                        int rhs_loc,
                                        int lhs_loc,
                                        unsigned int lhs_2crd_size_loc,
                                        unsigned int lhs_2pos_size_loc,
                                        Value lhs,
                                        Value lhs_nnz,
                                        Value lhs_nnz_alloc,
                                        Value lhs_val,
                                        std::vector<std::vector<std::string>> &allFormats,
                                        std::vector<std::vector<Value>> &main_tensors_all_Allocs,
                                        std::vector<std::vector<Value>> &allAccessIdx,
                                        std::vector<std::vector<Value>> &allValueAccessIdx,
                                        std::vector<scf::ForOp> &nested_forops)
  {

    /// %1 = load b[...]
    /// if(%1 != 0) {
    ///    Cnnz = load Cop.operand(4d+1)
    ///    store %1, cval[Cnnz]
    ///    store Cnnz+1, Cop.operand(4d+1)
    /// }
    comet_debug() << " main_tensors_all_Allocs[" << rhs_loc << "].size(): "
                  << main_tensors_all_Allocs[rhs_loc].size() << ", allValueAccessIdx[" << rhs_loc
                  << "].size(): " << allValueAccessIdx[rhs_loc].size() << "\n";
    Value rhs_value = builder.create<memref::LoadOp>(loc, main_tensors_all_Allocs[rhs_loc][main_tensors_all_Allocs[rhs_loc].size() - 1], allValueAccessIdx[rhs_loc]);
    comet_debug() << " ";
    comet_vdump(rhs_value);
    auto f64Type = builder.getF64Type();
    Value const_f64_0 = builder.create<ConstantOp>(loc, f64Type, builder.getF64FloatAttr(0));
    Value isNonzero = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, rhs_value, const_f64_0);
    comet_debug() << " ";
    comet_vdump(isNonzero);
    auto if_nonzero = builder.create<scf::IfOp>(loc, isNonzero, /*WithElseRegion*/ false);
    comet_debug() << " If branch:\n";
    comet_vdump(if_nonzero);

    if (!if_nonzero.getThenRegion().empty())
    {
      auto last_insertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&if_nonzero.getThenRegion().front());

      builder.create<memref::StoreOp>(loc, rhs_value, lhs_val, ValueRange{lhs_nnz});

      /// update pos/crd arrays
      /// Fill C2crd in CSR format, parent loop's accessIdx
      /// Check format j in the output
      if (allFormats[lhs_loc][allFormats[lhs_loc].size() - 1].compare(0, 2, "CU") == 0)
      {
        Value crd_index = allAccessIdx[allAccessIdx.size() - 1][allAccessIdx[allAccessIdx.size() - 1].size() -
                                                                1];
        comet_debug() << " ";
        comet_vdump(crd_index);
        Value lhs_2crd = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 4]; //-2
        comet_debug() << " ";
        comet_vdump(lhs_2crd);

        builder.create<memref::StoreOp>(loc, crd_index, lhs_2crd, ValueRange{lhs_nnz});
      }

      comet_debug() << "\n";
      Value cst_1_index = builder.create<ConstantIndexOp>(loc, 1);
      comet_debug() << " ";
      comet_vdump(lhs_nnz);
      Value lhs_nnz_new = builder.create<AddIOp>(loc, lhs_nnz, cst_1_index);
      comet_debug() << " AddIOps: (lhs_nnz_new)";
      comet_vdump(lhs_nnz_new);
      comet_debug() << " ";
      comet_vdump(lhs_nnz_alloc);

      Value cst_0_index = builder.create<ConstantIndexOp>(loc, 0);
      builder.create<memref::StoreOp>(loc, lhs_nnz_new, lhs_nnz_alloc, ValueRange{cst_0_index});

      comet_debug() << "\n";
      Value lhs_2crd = lhs.getDefiningOp()->getOperand(lhs_2crd_size_loc);
      Value lhs_2crd_op;
      comet_vdump(lhs_2crd);
      if (isa<IndexCastOp>(lhs_2crd.getDefiningOp()))
      {
        lhs_2crd_op = lhs_2crd.getDefiningOp()->getOperand(0);
      }
      else
      {
        lhs_2crd_op = lhs_2crd;
      }
      comet_debug() << " ";
      comet_vdump(lhs_2crd_op);
      auto c2crd_size_load = cast<memref::LoadOp>(lhs_2crd_op.getDefiningOp());                    /// index
      Value c2crd_size_alloc = cast<memref::AllocOp>(c2crd_size_load.getMemRef().getDefiningOp()); /// index
      comet_debug() << " ";
      comet_vdump(c2crd_size_alloc);

      builder.create<memref::StoreOp>(loc, lhs_nnz_new, c2crd_size_alloc, ValueRange{cst_0_index});

      comet_debug() << " \n";
      builder.restoreInsertionPoint(last_insertionPoint);
    }

    comet_debug() << " \n";
    auto prev_forop = nested_forops[nested_forops.size() - 1 - 1];
    builder.setInsertionPointAfter(prev_forop);

    comet_debug() << " ";
    comet_vdump(lhs.getDefiningOp()->getOperand(lhs_2pos_size_loc));
    Value lhs_2pos_0 = lhs.getDefiningOp()->getOperand(lhs_2pos_size_loc);
    Value lhs_2pos_op;
    comet_vdump(lhs_2pos_0);
    if (isa<IndexCastOp>(lhs_2pos_0.getDefiningOp()))
    {
      lhs_2pos_op = lhs_2pos_0.getDefiningOp()->getOperand(0);
    }
    else
    {
      lhs_2pos_op = lhs_2pos_0;
    }
    comet_debug() << " ";
    comet_vdump(lhs_2pos_op);
    auto c2pos_size_load = cast<memref::LoadOp>(lhs_2pos_op.getDefiningOp());                    /// index
    Value c2pos_size_alloc = cast<memref::AllocOp>(c2pos_size_load.getMemRef().getDefiningOp()); /// index
    Value cst_0_index = builder.create<ConstantIndexOp>(loc, 0);
    Value c2pos_size_value = builder.create<memref::LoadOp>(loc, c2pos_size_alloc, ValueRange{cst_0_index});

    Value lhs_2crd = lhs.getDefiningOp()->getOperand(lhs_2crd_size_loc);
    Value lhs_2crd_op;
    comet_vdump(lhs_2crd);
    if (isa<IndexCastOp>(lhs_2crd.getDefiningOp()))
    {
      lhs_2crd_op = lhs_2crd.getDefiningOp()->getOperand(0);
    }
    else
    {
      lhs_2crd_op = lhs_2crd;
    }
    comet_debug() << " ";
    comet_vdump(lhs_2crd_op);
    auto c2crd_size_load = cast<memref::LoadOp>(lhs_2crd_op.getDefiningOp());                    /// index
    Value c2crd_size_alloc = cast<memref::AllocOp>(c2crd_size_load.getMemRef().getDefiningOp()); /// index
    Value c2crd_size_nnz = builder.create<memref::LoadOp>(loc, c2crd_size_alloc, ValueRange{cst_0_index});

    /// store crd_size into pos
    Value lhs_2pos = main_tensors_all_Allocs[lhs_loc][main_tensors_all_Allocs[lhs_loc].size() - 5]; /// -3
    comet_debug() << " ";
    comet_vdump(lhs_2pos);

    builder.create<memref::StoreOp>(loc, c2crd_size_nnz, lhs_2pos, ValueRange{c2pos_size_value});

    Value cst_1_index = builder.create<ConstantIndexOp>(loc, 1);
    comet_debug() << " ";
    comet_vdump(c2pos_size_value);
    Value c2pos_size_value_new = builder.create<AddIOp>(loc, c2pos_size_value, cst_1_index);
    comet_debug() << " AddIOps (c2pos_size_value_new): ";
    comet_vdump(c2pos_size_value_new);

    builder.create<memref::StoreOp>(loc, c2pos_size_value_new, c2pos_size_alloc, ValueRange{cst_0_index});
  }

  /// From the W_id_list_size, get the output C and C.rowptr, C.col, and C.val.
  /// ----------------- ///
  /// %55 = "it.ComputeLHS"(%53) {allFormats = [[]], allPerms = [[]]} : (tensor<1xindex>) -> tensor<*xf64>
  /// %56 = "it.Compute"(%54, %55) {MaskType = "none", comp_worksp_opt = true, semiring = "noop_times"} : (tensor<*xindex>, tensor<*xf64>) -> i64
  /// %70 = "it.ComputeRHS"(%50, %51, %52, %53) {allFormats = [["D"]], allPerms = [[1]]} : (tensor<?xf64>, tensor<?xindex>, tensor<?xindex>, tensor<1xindex>) -> tensor<*xf64>
  /// %93 = "it.Compute"(%70, %92) {MaskType = "none", comp_worksp_opt = true, semiring = "noop_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
  /// %92 = "it.ComputeLHS"(%91) {allFormats = [["D", "CU"]], allPerms = [[0, 1]]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
  /// %91 = ta.sptensor_construct(%73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %11, %12) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
  /// %77 = bufferization.to_tensor %alloc_156 : memref<?xindex>
  /// %alloc_156 = memref.alloc(%71) : memref<?xindex>
  void getOutputMtxCRowptrAndDims(indexTree::IndexTreeComputeOp &cur_op,
                                  Value &W_id_list_size,
                                  SymbolicInfo &symbolicInfo /* output */)
  {
    Value mtxC = nullptr;
    for (Operation *u_rhs : W_id_list_size.getUsers())
    {
      if (indexTree::IndexTreeComputeRHSOp rhs_op = dyn_cast<indexTree::IndexTreeComputeRHSOp>(u_rhs))
      {
        /// rhs_op is %70
        for (Operation *u_cmpt : u_rhs->getUsers())
        {
          if (indexTree::IndexTreeComputeOp cmpt_op = dyn_cast<indexTree::IndexTreeComputeOp>(u_cmpt))
          {
            /// cmpt_op is %93
            /// then %93's Operand[1] is %92
            /// %92's Operand[0] is %91 which is the sparse tensor
            Value lhs_op = cmpt_op.getOperand(1);         /// lhs_op is %92
            mtxC = lhs_op.getDefiningOp()->getOperand(0); /// mtxC is %91
            break;
          }
        }
      }
    }

    assert(mtxC && "Error: cannot find mtxC as the output.");
    /// %77 is mtxC.getDefiningOp()->getOperand(A2POS)
    /// %alloc_156 is C_rowptr
    /// %71 is mtxC_rowptr_size
    Value C_rowptr = mtxC.getDefiningOp()->getOperand(CSR_A2POS).getDefiningOp()->getOperand(0); /// A2POS is rowptr's location
    Value C_rowptr_size = C_rowptr.getDefiningOp()->getOperand(0);
    Value C_num_rows = mtxC.getDefiningOp()->getOperand(CSR_DIM1_SIZE);
    Value C_num_cols = mtxC.getDefiningOp()->getOperand(CSR_DIM2_SIZE);
    symbolicInfo.mtxC = mtxC;
    symbolicInfo.mtxC_rowptr = C_rowptr;
    symbolicInfo.mtxC_rowptr_size = C_rowptr_size;
    symbolicInfo.mtxC_num_rows = C_num_rows;
    symbolicInfo.mtxC_num_cols = C_num_cols;
    {
      comet_vdump(mtxC);
      comet_vdump(C_rowptr);
      comet_vdump(C_rowptr_size);
      comet_vdump(C_num_rows);
      comet_vdump(C_num_cols);
    }
  }

  /// Generate mark before the outer-most symbolic for-loop,
  /// and update mark for every idx at the beginning of the outer-most symbolic for-loop.
  void genSymbolicMarkAndUpdate(OpBuilder &builder,
                                Location &loc,
                                ///                              std::vector<scf::ForOp> &symbolic_nested_forops, /* from innermost to outermost */
                                scf::ForOp &outermost_forLoop, /// the outermost for-loop
                                Value &mark_alloc /* output */,
                                Value &mark_new_val /* output */)
  {
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Set the insertion point before the outer-most symbolic for-loop
    builder.setInsertionPoint(outermost_forLoop);

    /// Generate the variable mark
    ///   %mark = memref.alloc() : memref<1xindex>
    ///   memref.store %c0, %mark[%c0] : memref<1xindex>
    MemRefType memTy_1xindex = MemRefType::get({1}, builder.getIndexType());
    mark_alloc = builder.create<memref::AllocOp>(loc, memTy_1xindex);
    Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
    builder.create<memref::StoreOp>(loc,
                                    const_index_0,
                                    mark_alloc,
                                    ValueRange{const_index_0});
    {
      comet_vdump(mark_alloc);
    }

    /// Generate updating mark += 2
    ///   %c2 = arith.constant 2 : index
    ///   %old_val = memref.load %mark[%c0] : memref<1xindex>
    ///   %new_mark = arith.addi %old_val, %c2 : index
    ///   memref.store %new_mark, %mark[%c0] : memref<1xindex>
    builder.setInsertionPointToStart(outermost_forLoop.getBody());
    Value const_index_2 = builder.create<ConstantIndexOp>(loc, 2);
    Value old_mark_val = builder.create<memref::LoadOp>(loc, mark_alloc, ValueRange{const_index_0});
    mark_new_val = builder.create<AddIOp>(loc, old_mark_val, const_index_2);
    builder.create<memref::StoreOp>(loc,
                                    mark_new_val,
                                    mark_alloc,
                                    ValueRange{const_index_0});
    {
      comet_vdump(outermost_forLoop);
    }

    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// Generate symbolic if statement condition in the CmptOp
  /// -------No masking---------- ///
  ///      if (mark_array[j_idx] != mark) {
  ///        mark_array[j_idx] = mark;  /// C[i_idx, j_idx] has been visited
  ///        W_id_list_size += 1;
  ///      }
  /// -------Push masking---------- ///
  ///      if (mark_array[j_idx] == mark) {
  ///        mark_array[j_idx] = mark + 1;  /// C[i_idx, j_idx] has been visited
  ///        W_id_list_size += 1;
  ///      }
  void genSymbolicIfStatementCondition(OpBuilder &builder,
                                       Location &loc,
                                       scf::ForOp &semiringLoop, /// symbolic_nested_forops[0]
                                       Value &mark_array_alloc,  /// tensors_lhs_Allocs[1][0]
                                       Value &valueAccessIdx,    /// allValueAccessIdx[lhs_loc][0]
                                       Value &mark_new_val,
                                       scf::IfOp &if_statement /* output */,
                                       MaskingInfo &maskingInfo)
  {
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Set the insertion point at the end of the inner-most symbolic for-loop
    builder.setInsertionPoint(semiringLoop.getBody()->getTerminator());

    {
      comet_vdump(semiringLoop);
    }
    /// Generate If statement condition
    Value ele_mark_val = builder.create<memref::LoadOp>(loc, mark_array_alloc, ValueRange{valueAccessIdx});

    if (PUSH_BASED_MASKING == maskingInfo.mask_type)
    {
      Value equal_mask = builder.create<CmpIOp>(loc,
                                                CmpIPredicate::eq,
                                                ele_mark_val,
                                                mark_new_val);
      if_statement = builder.create<scf::IfOp>(loc, equal_mask, false /* No Else Region */);
    }
    else if (NO_MASKING == maskingInfo.mask_type)
    {

      Value not_equal_mark = builder.create<CmpIOp>(loc,
                                                    CmpIPredicate::ne,
                                                    ele_mark_val,
                                                    mark_new_val);
      if_statement = builder.create<scf::IfOp>(loc, not_equal_mark, false /* No Else Region */);
    }
    else
    {
      llvm::errs() << "Error: mask_type " << maskingInfo.mask_type << " is not supported.\n";
    }
    {
      comet_vdump(ele_mark_val);
      comet_vdump(if_statement);
      comet_vdump(semiringLoop);
    }
    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// Generate symbolic if statement then region in the CmptOp
  /// -------No masking---------- ///
  ///      if (mark_array[j_idx] != mark) {
  ///        mark_array[j_idx] = mark;  /// C[i_idx, j_idx] has been visited
  ///        W_id_list_size += 1;
  ///      }
  /// -------Push masking---------- ///
  ///      if (mark_array[j_idx] == mark) {
  ///        mark_array[j_idx] = mark + 1;  /// C[i_idx, j_idx] has been visited
  ///        W_id_list_size += 1;
  ///      }
  void genSymbolicIfStatementThenRegion(OpBuilder &builder,
                                        Location &loc,
                                        scf::IfOp &if_statement,
                                        Value &mark_array_alloc, /// tensors_lhs_Allocs[1][0]
                                        Value &valueAccessIdx,   /// allValueAccessIdx[lhs_loc][0]
                                        Value &W_id_list_size,
                                        Value &mark_new_val,
                                        MaskingInfo &maskingInfo)
  {
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Set the insertion point to the beginning of the if statement then region
    builder.setInsertionPointToStart(&if_statement.getThenRegion().front());

    if (PUSH_BASED_MASKING == maskingInfo.mask_type)
    {
      /// mark_array[j_idx] = mark + 1;
      Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
      Value mark_value_plus_one = builder.create<AddIOp>(loc, mark_new_val, const_index_1);
      builder.create<memref::StoreOp>(loc,
                                      mark_value_plus_one,
                                      mark_array_alloc,
                                      ValueRange{valueAccessIdx});
    }
    else if (NO_MASKING == maskingInfo.mask_type)
    {
      /// mark_array[j_idx] = mark
      builder.create<memref::StoreOp>(loc,
                                      mark_new_val,
                                      mark_array_alloc,
                                      ValueRange{valueAccessIdx});
    }
    /// W_id_list_size += 1;

    Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
    Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
    Value old_val = builder.create<memref::LoadOp>(loc, W_id_list_size, ValueRange{const_index_0});
    Value new_val = builder.create<AddIOp>(loc, old_val, const_index_1);
    builder.create<memref::StoreOp>(loc,
                                    new_val,
                                    W_id_list_size,
                                    ValueRange{const_index_0});

    {
      comet_vdump(if_statement);
    }
    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// Updating output
  ///     C.rowptr[idx] = W_id_list_size;
  void genSymbolicUpdateCRowptr(OpBuilder &builder,
                                Location &loc,
                                scf::ForOp &outermost_forLoop,
                                Value &mtxC_rowptr,
                                Value &valueAccessIdx,
                                Value &W_id_list_size)
  {
    {
      comet_vdump(mtxC_rowptr);
      comet_vdump(valueAccessIdx);
      comet_vdump(W_id_list_size);
    }
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Set the insertion point at the end of the outermost for-loop body
    builder.setInsertionPoint(outermost_forLoop.getBody()->getTerminator());

    Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
    Value rowptr_val = builder.create<memref::LoadOp>(loc, W_id_list_size, ValueRange{const_index_0});
    builder.create<memref::StoreOp>(loc,
                                    rowptr_val,
                                    mtxC_rowptr,
                                    ValueRange{valueAccessIdx});

    {
      comet_vdump(outermost_forLoop);
    }
    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// Generate the reduce of the output C.rowptr after the outermost for-loop
  ///   C.rowptr[M] = 0;
  ///   int C_val_size = 0;
  ///   for (int i_idx = 0; i_idx < M + 1; ++i_idx) {
  ///     int curr = C.rowptr[i_idx];
  ///     C.rowptr[i_idx] = C_val_size;
  ///     C_val_size += curr;
  ///   }
  ///   C.col = new int[C_val_size]
  ///   C.val = new f64[C_val_size]
  void genSymbolicReduceOutputCRowptrCColCVal(OpBuilder &builder,
                                              Location &loc,
                                              scf::ForOp &outermost_forLoop,
                                              SymbolicInfo &symbolicInfo /* output */)
  {
    Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
    Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);

    /// C.rowptr[M] = 0
    Value &mtxC_rowptr = symbolicInfo.mtxC_rowptr;
    Value &num_rows = symbolicInfo.mtxC_num_rows;
    builder.create<memref::StoreOp>(loc,
                                    const_index_0,
                                    mtxC_rowptr,
                                    ValueRange{num_rows});

    /// C_val_size = 0;
    MemRefType memTy_1xindex = MemRefType::get({1}, builder.getIndexType());
    Value C_val_size = builder.create<memref::AllocOp>(loc, memTy_1xindex);
    builder.create<memref::StoreOp>(loc,
                                    const_index_0,
                                    C_val_size,
                                    ValueRange{const_index_0});

    /// for (int i_idx = 0; i_idx < M + 1; ++i_idx) {
    ///   int curr = C.rowptr[i_idx];
    ///   C.rowptr[i_idx] = C_val_size;
    ///   C_val_size += curr;
    /// }
    Value &num_rows_plus_one = symbolicInfo.mtxC_rowptr_size;
    scf::ForOp reduce_forLoop = builder.create<scf::ForOp>(loc,
                                                           const_index_0 /* lowerBound */,
                                                           num_rows_plus_one /* upperBound */,
                                                           const_index_1 /* step */);
    builder.setInsertionPointToStart(reduce_forLoop.getBody());
    Value i_idx = reduce_forLoop.getInductionVar();
    Value curr = builder.create<memref::LoadOp>(loc, mtxC_rowptr, ValueRange{i_idx});
    Value size_val = builder.create<memref::LoadOp>(loc, C_val_size, ValueRange{const_index_0});
    builder.create<memref::StoreOp>(loc,
                                    size_val,
                                    mtxC_rowptr,
                                    ValueRange{i_idx});
    Value new_val = builder.create<AddIOp>(loc, curr, size_val);
    builder.create<memref::StoreOp>(loc,
                                    new_val,
                                    C_val_size,
                                    ValueRange{const_index_0});
    {
      comet_vdump(reduce_forLoop);
    }
    builder.setInsertionPointAfter(reduce_forLoop);
    Value mtxC_val_size = builder.create<memref::LoadOp>(loc, C_val_size, ValueRange{const_index_0});
    symbolicInfo.mtxC_val_size = mtxC_val_size;

    /// Allocate new C.col and new C.val
    MemRefType memTy_alloc_dynamic_index = MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
    MemRefType memTy_alloc_dynamic_f64 = MemRefType::get({ShapedType::kDynamic}, builder.getF64Type());
    Value new_mtxC_col = builder.create<memref::AllocOp>(loc,
                                                         memTy_alloc_dynamic_index,
                                                         ValueRange{mtxC_val_size});
    Value new_mtxC_val = builder.create<memref::AllocOp>(loc,
                                                         memTy_alloc_dynamic_f64,
                                                         ValueRange{mtxC_val_size});
    symbolicInfo.mtxC_col = new_mtxC_col;
    symbolicInfo.mtxC_val = new_mtxC_val;
    {
      comet_vdump(mtxC_val_size);
      comet_vdump(new_mtxC_col);
      comet_vdump(new_mtxC_val);
    }
  }

  /// ----------------- ///
  /// Store new mtxC_val_size to the old mtxC's C_col_size (A2crd_size) and C_val_size (Aval_size).
  /// Just in case for safety.
  /// ----------------- ///
  void storeNewMtxCValeSizeToOldMtxC(OpBuilder &builder,
                                     Location &loc,
                                     SymbolicInfo &symbolicInfo)
  {
    Value &mtxC = symbolicInfo.mtxC;
    Value &mtxC_val_size = symbolicInfo.mtxC_val_size;
    Value const_index_0 = builder.create<ConstantIndexOp>(loc, 0);

    {
      comet_vdump(mtxC);
      comet_vdump(mtxC_val_size);
    }

    /// Find the alloc of C_col_size (Arcrd_size)
    ///     %66 = memref.load %alloc_153[%c0_128] : memref<1xindex>
    Value C_col_size_alloc = mtxC.getDefiningOp()->getOperand(CSR_A2CRD_SIZE).getDefiningOp()->getOperand(0); /// 8
                                                                                                              /// Store the new mtxC_val_size to C_col_size
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    auto store_C_col_size_alloc = builder.create<memref::StoreOp>(loc,
                                                                  mtxC_val_size,
                                                                  C_col_size_alloc,
                                                                  ValueRange{const_index_0});
    comet_vdump(C_col_size_alloc);
    comet_vdump(store_C_col_size_alloc);
#else
    builder.create<memref::StoreOp>(loc,
                                    mtxC_val_size,
                                    C_col_size_alloc,
                                    ValueRange{const_index_0});
#endif

    /// Find the alloc of C_val_size (Aval_size)
    ///     %67 = memref.load %alloc_154[%c0_128] : memref<1xindex>
    Value C_val_size_alloc = mtxC.getDefiningOp()->getOperand(CSR_AVAL_SIZE).getDefiningOp()->getOperand(0); /// 9
                                                                                                             /// Store the new mtxC_val_size to C_val_size
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    auto store_C_val_size_alloc = builder.create<memref::StoreOp>(loc,
                                                                  mtxC_val_size,
                                                                  C_val_size_alloc,
                                                                  ValueRange{const_index_0});
    comet_vdump(C_val_size_alloc);
    comet_vdump(store_C_val_size_alloc);
#else
    builder.create<memref::StoreOp>(loc,
                                    mtxC_val_size,
                                    C_val_size_alloc,
                                    ValueRange{const_index_0});
#endif
  }

  /// Dealloc the old C.val and C.col before the outermost_forLoop.
  /// Replace the old C.val and C.col with new ones.
  void deallocMtxCColCVal(OpBuilder &builder,
                          Location &loc,
                          scf::ForOp &outermost_forLoop,
                          SymbolicInfo &symbolicInfo)
  {
    /// Find old C.col and C.val
    Value &mtxC = symbolicInfo.mtxC;
    Value old_C_col = mtxC.getDefiningOp()->getOperand(CSR_A2CRD).getDefiningOp()->getOperand(0);
    Value old_C_val = mtxC.getDefiningOp()->getOperand(CSR_AVAL).getDefiningOp()->getOperand(0);

    /// Dealloc old C.col and C.val
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Set the insertion point before the symbolic outermost_forloop
    builder.setInsertionPoint(outermost_forLoop);

    builder.create<memref::DeallocOp>(loc, old_C_col);
    builder.create<memref::DeallocOp>(loc, old_C_val);

    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);

    /// -------------- ///
    /// Remove mtxC_col's user who is a memref.store operation
    /// This is very ad-hoc, just to avoid segmentation fault for old very large C.val array and C.col array.
    /// -------------- ///
    removeMemrefStoreUser(old_C_col);
    removeMemrefStoreUser(old_C_col);

    /// Replace old C.col and C.val
    /// Just in case of safety.
    replaceOldValueToNewValue(old_C_col, symbolicInfo.mtxC_col);
    replaceOldValueToNewValue(old_C_val, symbolicInfo.mtxC_val);
  }

  /// Generate a new sparse tensor to replace the old output sparse tensor after the numeric outermost for-loop.
  /// (e.g., ta.print(old_tensor)  ->  ta.print(new_tensor)
  void genReplaceOutputSparseTensorToNewSparseTensor(OpBuilder &builder,
                                                     Location &loc,
                                                     scf::ForOp &numeric_outermost_forLoop,
                                                     SymbolicInfo &symbolicInfo)
  {
    /// Set the insertion point after the outermost_forloop
    builder.setInsertionPointAfter(numeric_outermost_forLoop);

    Value &mtxC = symbolicInfo.mtxC;
    Value &mtxC_col = symbolicInfo.mtxC_col;
    Value &mtxC_val = symbolicInfo.mtxC_val;

    /// Generate the new mtxC_col and new mtxC_val bufferization.to_tensor
    Value mtxC_col_buffer = builder.create<bufferization::ToTensorOp>(loc, mtxC_col);
    Value mtxC_val_buffer = builder.create<bufferization::ToTensorOp>(loc, mtxC_val);

    auto sp_op = cast<tensorAlgebra::SparseTensorConstructOp>(mtxC.getDefiningOp());
    int tensorRanks = sp_op.getTensorRank();

    /// Get the operands and their types for the sparse tensor ta.sptensor_construct() (which is mtxC).
    SmallVector<Value, 20> operands;
    operands.insert(operands.end(),
                    mtxC.getDefiningOp()->getOperands().begin(),
                    mtxC.getDefiningOp()->getOperands().end());
    operands[CSR_A2CRD] = mtxC_col_buffer; /// 3 (A2crd)
    operands[CSR_AVAL] = mtxC_val_buffer;  /// 4 (AVal)
    SmallVector<Type, 20> elementTypes;
    for (Value &opd : operands)
    {
      elementTypes.push_back(opd.getType());
    }
    auto ty = tensorAlgebra::SparseTensorType::get(elementTypes);
    Value sptensor = builder.create<tensorAlgebra::SparseTensorConstructOp>(loc,
                                                                            ty,
                                                                            operands,
                                                                            tensorRanks);
    {
      comet_vdump(mtxC_col_buffer);
      comet_vdump(mtxC_val_buffer);
      comet_vdump(sptensor);
    }

    /// ----------------- ///
    /// Find all users of the old sparse tensor mtxC, and replace those users' corresponding operands
    /// to the new sparse tensor (sptensor). For example,
    /// "ta.print"(%mtxC)  =>  "ta.print"(%sptensor)
    /// ----------------- ///
    replaceOldValueToNewValue(mtxC, sptensor);
  }

  /// Logistics of memory about old mtxC, mtxC.col, and mtxC.val
  /// 1. Dealloc the old C.val and C.col before the outermost_forLoop.
  /// 2. Change mtxC's old value in C_col_size (A2crd_size) and C_val_size (Aval_size) to new mtxC_val_size.
  /// 3. Generate a new sparse tensor to replace the old output sparse tensor after the numeric outermost for-loop.
  void logisticsForMtxCColCVal(OpBuilder &builder,
                               Location &loc,
                               scf::ForOp &symbolic_outermost_forLoop,
                               SymbolicInfo &symbolicInfo,
                               scf::ForOp &numeric_outermost_forLoop)
  {

    /// Dealloc old C.col and C.val
    /// Replace the old C.val and C.col with new ones.
    deallocMtxCColCVal(builder,
                       loc,
                       symbolic_outermost_forLoop,
                       symbolicInfo);

    /// Change mtxC's old value in C_col_size (A2crd_size) and C_val_size (Aval_size) to new mtxC_val_size.
    /// Just in case for safety.
    storeNewMtxCValeSizeToOldMtxC(builder,
                                  loc,
                                  symbolicInfo);

    /// Generate a new sparse tensor to replace the old output sparse tensor after the numeric outermost for-loop.
    /// (e.g., ta.print(old_tensor)  ->  ta.print(new_tensor)
    genReplaceOutputSparseTensorToNewSparseTensor(builder,
                                                  loc,
                                                  numeric_outermost_forLoop,
                                                  symbolicInfo);

    ///  builder.restoreInsertionPoint(last_insertion_point);
  }

  /// Initialize the mark-array according to the mask at the beginning of the symbolic outermost for-loop
  /// ----------------- ///
  ///      %j_loc_start = memref.load %mask_rowptr[%i_idx] : memref<?xindex>
  ///      %j_loc_bound = memref.load %mask_rowptr[%i_idx_plus_one] : memref<?xindex>
  ///      scf.for %j_loc = %j_loc_start to %j_loc_bound step %c1 {
  ///        %val = memref.load %mask_val[%j_loc] : memref<?xf64>
  ///        %70 = arith.cmpf une, %val, %cst : f64
  ///        scf.if %70 {
  ///          %j_idx = memref.load %mask_col[%arg1] : memref<?xindex>
  ///          memref.store %mark, %mark_array[%j_idx] : memref<?xi1>
  ///        }
  ///      }
  void genSymbolicInitMarkArrayByMask(OpBuilder &builder,
                                      Location &loc,
                                      scf::ForOp &symbolic_outermost_forLoop,
                                      Value &outermost_forLoop_valueAccessIdx,
                                      Value &mark_array_alloc,
                                      Value &mark_new_val,
                                      MaskingInfo &maskingInfo)
  {
    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();

    /// Set the Insertion Point at the beginning of the symbolic outermost for-loop but AFTER the mark_new_val
    builder.setInsertionPointAfter(mark_new_val.getDefiningOp());

    /// Generate the for-loop entry
    Value &mask_rowptr = maskingInfo.mask_rowptr;
    Value &mask_col = maskingInfo.mask_col;
    Value &mask_val = maskingInfo.mask_val;
    Value const_index_1 = builder.create<ConstantIndexOp>(loc, 1);
    Value &i_idx = outermost_forLoop_valueAccessIdx;
    Value i_idx_plus_one = builder.create<AddIOp>(loc, i_idx, const_index_1);
    Value j_loc_start = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx});
    Value j_loc_bound = builder.create<memref::LoadOp>(loc, mask_rowptr, ValueRange{i_idx_plus_one});
    auto for_loop = builder.create<scf::ForOp>(loc,
                                               j_loc_start /* lower_bound */,
                                               j_loc_bound /* upper_bound*/,
                                               const_index_1 /* step */);
    {
      comet_vdump(j_loc_start);
      comet_vdump(j_loc_bound);
      comet_vdump(for_loop);
    }

    /// Generate the for-loop body
    builder.setInsertionPointToStart(for_loop.getBody());
    Value const_f64_0 = builder.create<ConstantOp>(loc, builder.getF64Type(), builder.getF64FloatAttr(0));
    Value j_loc = for_loop.getInductionVar();
    Value val = builder.create<memref::LoadOp>(loc, mask_val, ValueRange{j_loc});
    Value not_zero = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, val, const_f64_0);
    auto if_not_zero = builder.create<scf::IfOp>(loc, not_zero, false /*NoElseRegion*/);
    builder.setInsertionPointToStart(&if_not_zero.getThenRegion().front());
    Value j_idx = builder.create<memref::LoadOp>(loc, mask_col, ValueRange{j_loc});
    builder.create<memref::StoreOp>(loc,
                                    mark_new_val,
                                    mark_array_alloc,
                                    ValueRange{j_idx});

    {
      comet_vdump(val);
      comet_vdump(if_not_zero);
      comet_vdump(for_loop);
      comet_vdump(symbolic_outermost_forLoop);
    }

    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// Generate the symbolic phase's kernel to compute the rowptr[i_idx]
  void genSymbolicSemiringLoopBody(OpBuilder &builder,
                                   Location &loc,
                                   int lhs_loc,
                                   std::vector<std::vector<Value>> &tensors_lhs_Allocs,
                                   std::vector<scf::ForOp> &symbolic_nested_forops,
                                   std::vector<Value> &symbolic_nested_AccessIdx,
                                   std::vector<std::vector<Value>> &symbolic_allValueAccessIdx,
                                   SymbolicInfo &symbolicInfo,
                                   std::vector<scf::ForOp> &numeric_nested_forops,
                                   MaskingInfo &maskingInfo)
  {

    scf::ForOp &outermost_forLoop = symbolic_nested_forops.back();
    Value &outermost_forLoop_valueAccessIdx = symbolic_nested_AccessIdx.back();
    scf::ForOp &semiringLoop = symbolic_nested_forops[0];
    Value &mark_array = tensors_lhs_Allocs[1][0];
    Value &W_id_list_size = tensors_lhs_Allocs[3][0];
    Value &semiringLoop_valueAccessIdx = symbolic_allValueAccessIdx[lhs_loc][0];

    /// Generate mark before symbolic outer-most for-loop
    Value mark_alloc;
    Value mark_new_val;
    genSymbolicMarkAndUpdate(builder,
                             loc,
                             outermost_forLoop, /// the outermost for-loop
                             mark_alloc /* output */,
                             mark_new_val /* output */);

    if (PUSH_BASED_MASKING == maskingInfo.mask_type)
    {
      assert(symbolic_nested_forops.size() >= 2 && symbolic_allValueAccessIdx.size() >= 2 &&
             "Error: The symbolic for-loops should be at least 2 level.\n");

      /// Initialize the mark-array according to the mask at the beginning of the symbolic outermost for-loop
      genSymbolicInitMarkArrayByMask(builder,
                                     loc,
                                     outermost_forLoop,
                                     outermost_forLoop_valueAccessIdx,
                                     mark_array,
                                     mark_new_val,
                                     maskingInfo);
    }

    /// Generate if statement condition
    ///      if (mark_array[j_idx] != mark) {
    ///        mark_array[j_idx] = mark;  /// C[i_idx, j_idx] has been visited
    ///        W_id_list_size += 1;
    ///      }
    scf::IfOp if_statement;
    genSymbolicIfStatementCondition(builder,
                                    loc,
                                    semiringLoop,                /// the inner-most for-loop (SemiringLoop)
                                    mark_array,                  /// mark-array
                                    semiringLoop_valueAccessIdx, /// value access index j_idx
                                    mark_new_val,
                                    if_statement /* output */,
                                    maskingInfo);

    /// Generate if statement then region
    ///      if (mark_array[j_idx] != mark) {
    ///        mark_array[j_idx] = mark;  /// C[i_idx, j_idx] has been visited
    ///        W_id_list_size += 1;
    ///      }
    genSymbolicIfStatementThenRegion(builder,
                                     loc,
                                     if_statement,
                                     mark_array,                  /// mark-array
                                     semiringLoop_valueAccessIdx, /// value access index j_idx
                                     W_id_list_size,              /// W_id_list_size
                                     mark_new_val,
                                     maskingInfo);

    /// Updating output
    ///     C.rowptr[idx] = W_id_list_size;
    Value i_idx = outermost_forLoop.getInductionVar();
    genSymbolicUpdateCRowptr(builder,
                             loc,
                             outermost_forLoop,
                             symbolicInfo.mtxC_rowptr, /// mtxC_rowptr
                             i_idx,                    /// value access index i_idx
                             W_id_list_size /* W_id_list_size */);

    /// Store the insertion point
    auto last_insertion_point = builder.saveInsertionPoint();
    /// Set the insertion point after the outermost_forloop
    builder.setInsertionPointAfter(outermost_forLoop);

    /// Generate the reduce of output C.rowptr and new C.col and new C.val
    ///   C.rowptr[M] = 0;
    ///   int C_val_size = 0;
    ///   for (int i_idx = 0; i_idx < M + 1; ++i_idx) {
    ///     int curr = C.rowptr[i_idx];
    ///     C.rowptr[i_idx] = C_val_size;
    ///     C_val_size += curr;
    ///   }
    genSymbolicReduceOutputCRowptrCColCVal(builder,
                                           loc,
                                           outermost_forLoop,
                                           symbolicInfo /* output */);

    /// Logistics of memory about old mtxC, mtxC.col, and mtxC.val
    /// 1. Dealloc the old C.val and C.col before the outermost_forLoop.
    /// 2. Change mtxC's old value in C_col_size (A2crd_size) and C_val_size (Aval_size) to new mtxC_val_size.
    /// 3. Generate a new sparse tensor to replace the old output sparse tensor after the numeric outermost for-loop.
    scf::ForOp &numeric_outermost_forLoop = numeric_nested_forops.back();
    logisticsForMtxCColCVal(builder,
                            loc,
                            outermost_forLoop, /// symbolic_outermost_forLoop
                            symbolicInfo,
                            numeric_outermost_forLoop);

    /// Restore the insertion point
    builder.restoreInsertionPoint(last_insertion_point);
  }

  /// 1. Get the nested loops
  /// ---1.1 the nested loops corresponding indices can be infered from ancestors_wp
  /// 2. get lhs and rhs. if only 1 rhs, then it's a fill op; otherwise, binary op
  /// Note: 1. The auxiliary arrays does not contain the perms/formats information
  ///       2. We only apply the compressed workspace on the output of the tensor, then in this case, the workspace tensors will not be in the same side with the main tensors.
  ///         (main tensors: such as A, B, C, w;  auxiliary tensors: such as w_index_list ...)
  void genCmptOps(indexTree::IndexTreeComputeOp &cur_op,
                  indexTree::IndexTreeOp &rootOp,
                  ///                PatternRewriter &rewriter,
                  OpBuilder &builder,
                  OpsTree *opstree,
                  std::vector<Value> &ancestorsWps,
                  std::vector<Value> &wp_ops,
                  SymbolicInfo &symbolicInfo,
                  NumericInfo &numericInfo)
  {
    comet_debug() << " calling genCmptOps\n";
    Location loc = rootOp.getLoc();
    comet_debug() << " \n";

    comet_debug() << " Current IndexTreeComputeOp:";
    comet_vdump(cur_op);

    const bool comp_worksp_opt(cur_op.getCompWorkspOpt());
    comet_debug() << " comp_worksp_opt (bool: true is compressed): " << comp_worksp_opt << "\n";

    /// Two cases:
    /// 1. for the initial workspace, only 1 auxiliary vector w
    /// 2. for the compressed workspace, there are 4 auxiliaty vectors, w, w_already_set, w_index_list, w_index_list_size

    /// The insertion location should be "the end of the body of parent loop"
    std::vector<OpsTree *> ancestorsOps;
    getAncestorsOps(opstree, ancestorsOps);
    comet_debug() << " ancestorsOps.size(): " << ancestorsOps.size() << "\n";
    for (unsigned int i = 0; i < ancestorsOps.size(); i++)
    {
      comet_debug() << " ancestorsOps[i]->id:" << ancestorsOps[i]->id << "\n";
    }

    /// 1. get the nested loops, from innermost to outermost order
    std::vector<scf::ForOp> nested_forops;
    std::vector<Value> nested_AccessIdx;
    std::vector<int64_t> nested_forops_indices; /// Each nested indexOp's index value (e.g., indices=[0])
    getNumericNestedForOpsAndAccessIdx(ancestorsWps,
                                       ancestorsOps,
                                       nested_forops /* output */,
                                       nested_AccessIdx /* output */,
                                       nested_forops_indices /* output */);

    comet_debug() << " nested_forops_indices.size(): " << nested_forops_indices.size() << "\n";
    assert(
        nested_forops.size() == nested_forops_indices.size() && "nested_forops.size() != nested_forops_indices.size()");

    /// Reset the insertion point: the body of the innermost loop
    assert(nested_forops.size() > 0 && "No loops\n");
    comet_debug() << " ";
    comet_pdump(nested_forops[0].getBody());
    comet_debug() << " ";
    comet_pdump(nested_forops[0].getBody()->getTerminator());
    builder.setInsertionPoint(nested_forops[0].getBody()->getTerminator());
    {
      comet_vdump(nested_forops[0]);
    }

    /// Analyze the leafop, Get the tensors, rhs, lhs, and operator_type
    /// --- only one rhs, it will be a fill op; if two, check op_type (+, +=, *=)
    /// Check the indices contained in each tensor
    /// Generate loadOp, compute ops, StoreOp.
    std::vector<Value> tensors_rhs;
    std::vector<std::vector<Value>> tensors_lhs_Allocs;
    std::vector<std::vector<Value>> tensors_rhs_Allocs;
    std::vector<std::vector<std::string>> allFormats;
    std::vector<std::vector<int>> allPerms;
    std::vector<std::vector<int>> allPerms_rhs;
    std::vector<Value> main_tensors_all; /// main_tensors_all has first RHS tensors then LHS tensors
    std::vector<Value> main_tensors_rhs;
    getNumericTensors(cur_op,
                      tensors_rhs /* output */,
                      tensors_lhs_Allocs /* output */,
                      tensors_rhs_Allocs /* output */,
                      allFormats /* output */,
                      allPerms /* output */,
                      allPerms_rhs /* output */,
                      main_tensors_all /* output */,
                      main_tensors_rhs /* output */);

    /// ----------------- ///
    /// Get main_tensors_all_Allocs
    /// ----------------- ///
    int main_tensor_nums = main_tensors_all.size(); /// output
    comet_debug() << " main_tensor_nums: " << main_tensor_nums << "\n";
    /// Check the loop arg in each tensor
    std::vector<std::vector<Value>> main_tensors_all_Allocs = getAllAllocs(main_tensors_all); /// output
    comet_debug() << " main_tensors_all_Allocs.size(): " << main_tensors_all_Allocs.size() << "\n";

    /// ----------------- ///
    /// Get allValueAccessIdx
    /// ----------------- ///
    /// For every main_tensors_all[i], allAccessIdx[i] is the for-loop's induction variable.
    /// However, allValueAccessIdx[i] is not necessarily the induction variable.
    /// For CSR, for example,
    /// for (j_loc = A.rowptr[idx]; j_loc < A.rowptr[idx + 1]; ++j_loc) { j_idx = A.col[j_loc]; }
    /// j_idx is allValueAccessIdx[i], and j_loc is allAccessIdx[i]
    std::vector<std::vector<Value>> allAccessIdx(main_tensor_nums);
    std::vector<std::vector<Value>> allValueAccessIdx(main_tensor_nums);
    getForLoopsValueAccessIdx(builder,
                              loc,
                              main_tensor_nums,
                              allPerms,
                              allFormats,
                              main_tensors_all,
                              nested_forops,
                              nested_AccessIdx,
                              nested_forops_indices,
                              main_tensors_all_Allocs,
                              allAccessIdx /* output */,
                              allValueAccessIdx /* output */);

    /// Symbolic Phase preparation
    std::vector<scf::ForOp> symbolic_nested_forops;
    std::vector<Value> symbolic_nested_AccessIdx;
    std::vector<int64_t> symbolic_nested_forops_indices;
    std::vector<std::vector<Value>> symbolic_allAccessIdx(main_tensor_nums);
    std::vector<std::vector<Value>> symbolic_allValueAccessIdx(main_tensor_nums);
    if (symbolicInfo.has_symbolic_phase)
    {
      /// Store the insertion point
      auto last_insertion_point = builder.saveInsertionPoint();

      getSymbolicNestedForOpsAndAccessIdx(ancestorsWps,
                                          ancestorsOps,
                                          symbolic_nested_forops /* output */,
                                          symbolic_nested_AccessIdx /* output */,
                                          symbolic_nested_forops_indices /* output */);

      /// Set the insertion point
      builder.setInsertionPoint(symbolic_nested_forops[0].getBody()->getTerminator());

      getForLoopsValueAccessIdx(builder,
                                loc,
                                main_tensor_nums,
                                allPerms,
                                allFormats,
                                main_tensors_all,
                                symbolic_nested_forops,
                                symbolic_nested_AccessIdx,
                                symbolic_nested_forops_indices,
                                main_tensors_all_Allocs,
                                symbolic_allAccessIdx /* output */,
                                symbolic_allValueAccessIdx /* output */);

      /// Restore the insertion point
      builder.restoreInsertionPoint(last_insertion_point);
    }

    int rhs_loc = 0;
    int lhs_loc = main_tensors_rhs.size(); /// lhs_loc is the location of the first LHS tensor in main_tensors_all

    /// New version
    Value lhs = cur_op.getLhs().getDefiningOp()->getOperand(0);
    comet_vdump(lhs);
/// lhs is TensorLoadOp
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
    Value lhs_alloc = (lhs.getDefiningOp())->getOperand(0);
    comet_vdump(lhs_alloc);
#endif
    if (main_tensors_rhs.size() == 1)
    { /// Generate "a = b"
      if (ConstantOp cstop = dyn_cast<ConstantOp>(main_tensors_rhs[0].getDefiningOp()))
      { /// "a = 1.0"
        comet_vdump(cstop);
        if (comp_worksp_opt) /// true attr means compressed workspace
        {
          /// Symbolic Phase
          if (symbolicInfo.has_symbolic_phase)
          {
            /// Store the insertion point
            auto last_insertion_point = builder.saveInsertionPoint();

            /// Set the insertion point
            builder.setInsertionPoint(symbolic_nested_forops[0].getBody()->getTerminator());

            /// Symbolic Phase uses the W_id_list_size in the Index Tree (main_tensors_all_Allocs[lhs_loc].back)
            /// to record the current row size.
            ///     W_id_list_size = 0;
            /// However, Numeric Phase should use C.rowptr[i_idx] to initialize W_id_list_size.
            ///     W_id_list_size = C.rowptr[i_idx];
            genWorkspaceCmptOpInitialAssignment(builder,
                                                loc,
                                                lhs_loc,
                                                cstop,
                                                symbolic_nested_forops,
                                                tensors_lhs_Allocs,
                                                main_tensors_all_Allocs,
                                                false /* use_dynamic_init */,
                                                symbolicInfo);

            /// Prepare C, C.rowptr
            if (symbolicInfo.mtxC_rowptr == nullptr)
            {
              Value &W_id_list_size = lhs;
              {
                comet_vdump(W_id_list_size);
              }
              getOutputMtxCRowptrAndDims(cur_op,
                                         W_id_list_size,
                                         symbolicInfo /* output */);
            }

            /// Restore the insertion point
            builder.restoreInsertionPoint(last_insertion_point);
          } /// End symbolic phase
          if (allFormats[lhs_loc].empty())
          {
            /// The computeOp node is W_id_list_size = 0,
            /// then do W_id_list_size = symbolicInfo.mtxC_rowptr[idx]
            genWorkspaceCmptOpInitialAssignment(builder,
                                                loc,
                                                lhs_loc,
                                                cstop,
                                                nested_forops,
                                                tensors_lhs_Allocs,
                                                main_tensors_all_Allocs,
                                                true /* use_dynamic_init */,
                                                symbolicInfo);
          }
          else
          {
            /// The computeOp node is V[j] = 0,
            /// then do V[j] = 0.0
            genWorkspaceCmptOpInitialAssignment(builder,
                                                loc,
                                                lhs_loc,
                                                cstop,
                                                nested_forops,
                                                tensors_lhs_Allocs,
                                                main_tensors_all_Allocs,
                                                false /* use_dynamic_init */,
                                                symbolicInfo);
          }
        }
        else
        { /// initial workspace
          /// Generate Store 1.0, A[...]  this op
          /// this case: allPerms[0] is empty, allFormats[0] is empty

          genCmptOpGeneralInitialAssignment(builder,
                                            loc,
                                            lhs_loc,
                                            cstop,
                                            nested_forops,
                                            main_tensors_all_Allocs,
                                            allValueAccessIdx);
        }
      }
      else if (main_tensors_rhs[0].getType().isa<mlir::TensorType>())
      { /// Cij = Wj
        /// When Cij is dense type
        if (lhs.getType().isa<mlir::TensorType>())
        {
          /// %1 = load b[...]
          /// store %1, a[...]
          genCmptOpGatherFromDenseToDense(builder,
                                          loc,
                                          rhs_loc,
                                          lhs_loc,
                                          main_tensors_all_Allocs,
                                          allValueAccessIdx);
        }
        /// Cij = Wj
        else if (lhs.getType().isa<tensorAlgebra::SparseTensorType>())
        {

          unsigned int lhs_2crd_size_loc;
          unsigned int lhs_2pos_size_loc;
          Value lhs_nnz;
          Value lhs_nnz_alloc;
          Value lhs_val;
          getLHSBeforeGatherFromWorkspace(builder,
                                          loc,
                                          lhs_loc,
                                          lhs,
                                          main_tensors_all_Allocs,
                                          lhs_2crd_size_loc /* output */,
                                          lhs_2pos_size_loc /* output */,
                                          lhs_nnz /* output */,
                                          lhs_nnz_alloc /* output */,
                                          lhs_val /* output */);

          if (comp_worksp_opt) /// true attr means compressed workspace
          {
            /// Gather results from Workspace to the sparse output
            genWorkspaceCmptOpGatherFromWorkspaceToOutput(builder,
                                                          loc,
                                                          tensors_rhs_Allocs,
                                                          nested_forops,
                                                          nested_AccessIdx,
                                                          symbolicInfo,
                                                          numericInfo);
            ///          }
          }
          else
          {
            /// %1 = load b[...]
            /// if(%1 != 0) {
            ///    Cnnz = load Cop.operand(4d+1)
            ///    store %1, cval[Cnnz]
            ///    store Cnnz+1, Cop.operand(4d+1)
            /// }
            genCmptOpGatherFromDenseToOutput(builder,
                                             loc,
                                             rhs_loc,
                                             lhs_loc,
                                             lhs_2crd_size_loc,
                                             lhs_2pos_size_loc,
                                             lhs,
                                             lhs_nnz,
                                             lhs_nnz_alloc,
                                             lhs_val,
                                             allFormats,
                                             main_tensors_all_Allocs,
                                             allAccessIdx,
                                             allValueAccessIdx,
                                             nested_forops);
          }
        }
      }
      /// Vj = Bij
      else if (main_tensors_rhs[0].getType().isa<tensorAlgebra::SparseTensorType>())
      {
        /// %Bvalue = load %Bval[..]
        /// store %Bvalue, %v[%j]

        /// Symbolic Phase
        if (symbolicInfo.has_symbolic_phase)
        {
          /// Store the insertion point
          auto last_insertion_point = builder.saveInsertionPoint();

          /// Set the insertion point
          builder.setInsertionPoint(symbolic_nested_forops[0].getBody()->getTerminator());

          genWorkspaceCmptOpScatterInputToWorkspace(builder,
                                                    loc,
                                                    main_tensor_nums,
                                                    main_tensors_all_Allocs,
                                                    symbolic_allValueAccessIdx);

          /// Restore the insertion point
          builder.restoreInsertionPoint(last_insertion_point);
        } /// End symbolic phase

        genWorkspaceCmptOpScatterInputToWorkspace(builder,
                                                  loc,
                                                  main_tensor_nums,
                                                  main_tensors_all_Allocs,
                                                  allValueAccessIdx);
      }
    }
    else if (main_tensors_rhs.size() == 2)
    { /// Generate " a = b * c" binary op

      comet_debug() << "No masking codegen...\n";

      auto semiringParts = cur_op.getSemiring().split('_');
      /// check validity of semiring provided by user.
      if (!Semiring_reduceOps.contains(semiringParts.first) || !Semiring_ops.contains(semiringParts.second))
      {
        llvm::errs() << "Not supported semiring operator: "
                     << semiringParts.first << " or " << semiringParts.second << " \n";
        llvm::errs() << "Please report this error to the developers!\n";
        /// we should not proceed forward from this point to avoid faults.
      }

      MaskingInfo maskingInfo;
      maskingInfo.mask_type = NO_MASKING;
      if (symbolicInfo.has_symbolic_phase)
      {
        /// Store the insertion point
        auto last_insertion_point = builder.saveInsertionPoint();

        /// Set the insertion point
        builder.setInsertionPoint(symbolic_nested_forops[0].getBody()->getTerminator());

        genSymbolicSemiringLoopBody(builder,
                                    loc,
                                    lhs_loc,
                                    tensors_lhs_Allocs,
                                    symbolic_nested_forops,
                                    symbolic_nested_AccessIdx,
                                    symbolic_allValueAccessIdx,
                                    symbolicInfo,
                                    nested_forops /* numeric_nested_forops= */,
                                    maskingInfo);

        /// Restore the insertion point
        builder.restoreInsertionPoint(last_insertion_point);
      }

      formSemiringLoopBody(cur_op,
                           comp_worksp_opt,
                           semiringParts.first, semiringParts.second,
                           builder, loc, lhs_loc,
                           main_tensors_all_Allocs,
                           tensors_lhs_Allocs,
                           tensors_rhs_Allocs,
                           allValueAccessIdx,
                           allAccessIdx,
                           nested_forops,
                           nested_AccessIdx,
                           symbolic_nested_forops,
                           allPerms_rhs,
                           symbolicInfo,
                           numericInfo,
                           maskingInfo);
    }
    else if (main_tensors_rhs.size() == 3)
    { /// Generate " a<m> = b * c" binary op with masking

      {
        ///    comet_pdump(rootOp.getOperation()->getParentOfType<ModuleOp>());
        comet_pdump(rootOp->getParentOfType<ModuleOp>());
      }
      auto semiringParts = cur_op.getSemiring().split('_');
      /// check validity of semiring provided by user.
      if (!Semiring_reduceOps.contains(semiringParts.first) || !Semiring_ops.contains(semiringParts.second))
      {
        llvm::errs() << "Not supported semiring operator: "
                     << semiringParts.first << " or " << semiringParts.second << " \n";
        llvm::errs() << "Please report this error to the developers!\n";
        /// we should not proceed forward from this point to avoid faults.
      }

      auto maskingAttr = cur_op.getMaskType();
      std::string maskingAttrStr(maskingAttr.data());
      comet_debug() << "mask attr: " << maskingAttrStr << "\n";

      MASKING_TYPE mask_type;
      if (maskingAttrStr == "push")
        mask_type = MASKING_TYPE::PUSH_BASED_MASKING;
      else if (maskingAttrStr == "pull")
        mask_type = MASKING_TYPE::PULL_BASED_MASKING;
      else if (maskingAttrStr == "auto")
        mask_type = MASKING_TYPE::PUSH_BASED_MASKING;
      else /// none
        mask_type = MASKING_TYPE::NO_MASKING;

      switch (mask_type)
      {
      case NO_MASKING:
      { /// Use no masking; we should not hit this case because it is handled
        /// by the previous if-else branch when main_tensors_rhs.size() == 2
        break;
      }
      case PUSH_BASED_MASKING:
      { /// Use push-based masking
        /// mask_tensor should be the 3rd operand of ComputeRHS (tensors_rhs[2]).
        mlir::Value mask_tensor = tensors_rhs[2];
        {
          comet_debug() << "mask_tensor\n";
          comet_vdump(mask_tensor);
        }

        MaskingInfo maskingInfo;
        maskingInfo.mask_type = PUSH_BASED_MASKING;
        maskingInfo.mask_tensor = mask_tensor;

        /// Get mask_rowptr, mask_col, and mask_val arrays
        getMaskSparseTensorInfo(maskingInfo /* contents updated after call*/);

        if (symbolicInfo.has_symbolic_phase)
        {
          /// Store the insertion point
          auto last_insertion_point = builder.saveInsertionPoint();

          /// Set the insertion point
          builder.setInsertionPoint(symbolic_nested_forops[0].getBody()->getTerminator());

          genSymbolicSemiringLoopBody(builder,
                                      loc,
                                      lhs_loc,
                                      tensors_lhs_Allocs,
                                      symbolic_nested_forops,
                                      symbolic_nested_AccessIdx,
                                      symbolic_allValueAccessIdx,
                                      symbolicInfo,
                                      nested_forops /* numeric_nested_forops= */,
                                      maskingInfo);

          /// Restore the insertion point
          builder.restoreInsertionPoint(last_insertion_point);
        }
        formSemiringLoopBody(cur_op,
                             comp_worksp_opt,
                             semiringParts.first, semiringParts.second,
                             builder, loc, lhs_loc,
                             main_tensors_all_Allocs,
                             tensors_lhs_Allocs,
                             tensors_rhs_Allocs,
                             allValueAccessIdx,
                             allAccessIdx,
                             nested_forops,
                             nested_AccessIdx,
                             symbolic_nested_forops,
                             allPerms_rhs,
                             symbolicInfo,
                             numericInfo,
                             maskingInfo);
        break;
      }
      case PULL_BASED_MASKING: /// Use pull-based masking
        llvm::errs() << "Error: mask type PULL_BASED_MASKING is not supported, yet.\n";
      }
    }
    else
    {
      llvm::errs() << "No support for operation with greater than two operands in workspace transforms!"
                   << "\n";
    }
  }

  /// ----------------- ///
  /// Get the itree roots
  /// ----------------- ///
  void getIndexTreeOps(func::FuncOp &function,
                       std::vector<indexTree::IndexTreeOp> &iTreeRoots /* output */)
  {
    function.walk([&](indexTree::IndexTreeOp op)
                  { iTreeRoots.push_back(op); });
  }

  /// ----------------- ///
  /// Delete every objects in opstree_vec, preventing memory leak.
  /// ----------------- ///
  void cleanOpstreeVec(std::vector<OpsTree *> &opstree_vec)
  {
    for (auto &t : opstree_vec)
    {
      delete t;
    }
  }

  /// ----------------- ///
  /// Check if the Index Tree inputs are all sparse
  /// All inputs are sparse if and only if all computeOp nodes are using workspace transformation.
  /// ----------------- ///
  void checkIfAllSparse(std::vector<mlir::Value> &wp_ops,
                        SymbolicInfo &symbolicInfo /* output */)
  {
    for (Value &op : wp_ops)
    {
      if (indexTree::IndexTreeComputeOp cur_op = dyn_cast<indexTree::IndexTreeComputeOp>(op.getDefiningOp()))
      {
        bool comp_worksp_opt(cur_op.getCompWorkspOpt());
        if (!comp_worksp_opt)
        {
          symbolicInfo.are_inputs_sparse = false;
          return;
        }
      }
    }

    symbolicInfo.are_inputs_sparse = true;
  }

  //===----------------------------------------------------------------------===//
  /// LowerIndexTreeIRToSCF PASS
  //===----------------------------------------------------------------------===//

  /// Lower the ta.tc (tensor contraction operation in TA dialect) into scf dialect.
  struct LowerIndexTreeToSCFPass
      : public PassWrapper<LowerIndexTreeToSCFPass, OperationPass<func::FuncOp>>
  {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIndexTreeToSCFPass)
    void runOnOperation() override;

    void doLoweringIndexTreeToSCF(indexTree::IndexTreeOp &rootOp,
                                  OpBuilder &builder);
  };

} /// end anonymous namespace.

/**
 * @brief :
 * Goal: IndexTreeOp(i.e. a tree structure), convert into OpsTree(also tree structure)
 * Steps: 1.Iterate over IndexTreeOptree
 *        2.pass info to opsGen(), including tensors, current workspacetreeop, parent OpsTree node
 *          -- the parent of "current workspacetreeop" can get from getUser(). Only one user(tree structure)
 *          -- DFS traverse the workspacetreeop. How?
 * */
void LowerIndexTreeToSCFPass::doLoweringIndexTreeToSCF(indexTree::IndexTreeOp &rootOp,
                                                       OpBuilder &builder)
{
  assert(isa<indexTree::IndexTreeOp>(rootOp));
  comet_debug() << "\ndoLoweringIndexTreeToSCF in LowerIndexTreeIRToSCF\n";
  /// auto module = rootOp->getParentOfType<ModuleOp>();
  {
    ///    comet_pdump(rootOp.getOperation()->getParentOfType<ModuleOp>());
    comet_pdump(rootOp->getParentOfType<ModuleOp>());
  }

  ///  comet_pdump(rootOp.getOperation()->getParentOp());
  /// Here, should check the operands, at least one operand should be sparse;
  /// Otherwise, if all dense operands, just return.
  /// rootOp only contains one workspace child, no indices

  std::vector<mlir::Value> wp_ops;
  dfsRootOpTree(rootOp.getChildren(), wp_ops);
  // #ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  comet_debug() << " wp_ops.size(): " << wp_ops.size() << "\n";
  for ([[maybe_unused]] auto n : wp_ops)
  {
    comet_debug() << " ";
    comet_vdump(n);
    /// Declare opsTree
  }
  // #endif

  /// In ops vector, for each op, the parent of each op can get from getUsers()
  /// Since it's a tree structure, only one user ==> which is the parent
  /// We can initialize the OpsTree structure with this relationship.
  /// Search the location of the parent of current op, if rootOp, return ops.size;
  /// Otherwise, return the location index.
  std::vector<unsigned int> parent_idx;
  for (unsigned int i = 0; i < wp_ops.size(); i++)
  {
    mlir::Value wp_op = wp_ops[i];
    mlir::Value wp_parent;

    for (auto n : wp_op.getDefiningOp()->getUsers())
    {
      comet_debug() << " " << i << " ";
      comet_pdump(n);
      wp_parent = n->getResult(0);

      comet_debug() << " parent: " << findIndexInVector_Value(wp_ops, wp_parent) << "\n";
      bool isInTree = false;
      if (findIndexInVector_Value(wp_ops, wp_parent) < wp_ops.size())
      {
        isInTree = true;
      }

      if (isInTree || isRealRoot(wp_op.getDefiningOp()))
        parent_idx.push_back(findIndexInVector_Value(wp_ops, wp_parent));
    }
  }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  comet_debug() << " parent_idx: " << parent_idx.size() << "\n";
  for (auto n : parent_idx)
  {
    comet_debug() << " " << n << " \n";
    /// Declare opsTree
  }
#endif

  std::vector<OpsTree *> opstree_vec;
  for (unsigned int i = 0; i < wp_ops.size(); i++)
  {
    std::vector<scf::ForOp> forOps;
    std::vector<Value> accessIdx;

    OpsTree *parent = nullptr;
    if (i >= 1)
    { /// Not rootop
      parent = opstree_vec[parent_idx[i]];
    }
    comet_debug() << " \n";
    OpsTree *ops = new OpsTree(forOps, accessIdx, parent, i);
    if (parent != nullptr)
    { /// add child to the parent
      parent->addChild(ops);
    }

    opstree_vec.push_back(ops);
  }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  {
    int opstree_i = 0;
    for (auto n : opstree_vec)
    {
      comet_debug() << " " << n->id << "\n";
      comet_debug() << "opstree_vec[" << opstree_i << "] "
                    << "forOps.size():" << n->forOps.size() << " "
                    << "accessIdx.size():" << n->accessIdx.size() << "\n";
      ///              << "cmptOps.size():" << n->cmptOps.size() << "\n";
      if (n->parent != nullptr)
      {
        comet_debug() << "parent: " << n->parent->id << "\n";
      }
      else
      {
        comet_debug() << "parent: null \n";
      }
      ++opstree_i;
    }
  }
#endif

  SymbolicInfo symbolicInfo;
  NumericInfo numericInfo;
  checkIfAllSparse(wp_ops,
                   symbolicInfo /* output */);
  if (symbolicInfo.are_inputs_sparse)
  {
    symbolicInfo.has_symbolic_phase = true;
  }

  for (unsigned int i = 0; i < wp_ops.size(); i++)
  {
    comet_debug() << " i: " << i << "\n";
    comet_vdump(wp_ops[i]);
    if (indexTree::IndexTreeIndicesOp cur_op = dyn_cast<mlir::indexTree::IndexTreeIndicesOp>(wp_ops[i].getDefiningOp()))
    {
      /// Get indices
      ArrayAttr op_indices = cur_op.getIndices();
      comet_debug() << "curOp is IndexTreeIndicesOp\n";
      comet_vdump(cur_op);

      /// cur_op's index attribute, e.g., "indices = [0]"
      std::vector<int> indices;
      for (unsigned int j = 0; j < op_indices.size(); j++)
      {
        /// Get the indices;
        int idx = op_indices[j].cast<IntegerAttr>().getInt();
        indices.push_back(idx);
      }
      comet_debug() << " indices.size(): " << indices.size() << "\n";

      /// Leaves are the computeOp nodes and the children of cur_op (an index node)
      std::vector<Value> leafs;

      /// Find leaves of cur_op in the Index Tree (wp_ops).
      /// A leaf is a computeOp node and cur_op is one its ancestors.
      findLeafs(cur_op, indices, wp_ops, leafs /* output leaves*/);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      comet_debug() << " leafs.size(): " << leafs.size() << "\n";
      for (auto n : leafs)
      {
        comet_debug() << " ";
        comet_vdump(n);
      }
#endif

      /// tensors: the tensors that uses the cur_op (index node) as their iterative index.
      /// ids:     An id is the location (0, 1, 2, ...) of the cur_op (index node) in the tensor's Perms.
      /// formats: The format (e.g., "D", "CU", "CN", etc.) for the id-th dimension of the tensor.
      /// tensors[i] uses the cur_op (index node) as its iterative index (e.g., [0], [1], etc.), and
      /// ids[i] is the location of the iterative index in tensors[i]'s Perms.
      /// For example,
      /// allPerms = [[0, 1]]; the tensor[i]'s Perms is [0, 1]. If cur_op (iterative index) is indices = [1], then
      /// ids[i] = 1, because [1] is at location 1 in [0, 1], i.e., the 1-st dimension of the tensors[i].
      /// formats[i] is "CU" if tensors[i]'s Formats = ["D", "CU"].
      std::vector<Value> tensors;
      std::vector<unsigned int> ids;
      std::vector<std::string> formats;

      comet_vdump(cur_op);

      getFormatsInfo(cur_op,
                     indices,
                     leafs,
                     tensors /* output */,
                     ids /* output */,
                     formats /* output */);

      comet_debug() << " indices.size(): " << indices.size() << " tensors.size(): " << tensors.size() << "\n";
      for ([[maybe_unused]] unsigned int m = 0; m < tensors.size(); m++)
      {
        comet_debug() << " Formats:" << formats[m] << " " << ids[m] << " ";
        comet_vdump(tensors[m]);
      }

      comet_debug() << " call genForOps, i = " << i << "\n";
      genForOps(tensors, ids, formats, rootOp, builder, opstree_vec[i], symbolicInfo);
      {
        comet_pdump(rootOp->getParentOfType<ModuleOp>());
      }
      comet_debug() << " finished call genForOps, i = " << i << "\n";
    }
    else if (indexTree::IndexTreeComputeOp cur_op = dyn_cast<mlir::indexTree::IndexTreeComputeOp>(wp_ops[i].getDefiningOp()))
    {
      /// Generate computation ops.
      std::vector<Value> ancestors_wp; /// workspace tree ancestor
      getAncestorsWp(cur_op, ancestors_wp, wp_ops);
#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
      comet_debug() << " Current Op (IndexTreeComputeOp):";
      comet_vdump(cur_op);
      for (auto n : ancestors_wp)
      {
        comet_debug() << " ";
        comet_vdump(n);
      }
#endif

      comet_debug() << " call genCmptOps, i = " << i << "\n";
      /// ancestors_wp can give all the indices of the nested loops
      genCmptOps(cur_op, rootOp, builder, opstree_vec[i], ancestors_wp,
                 wp_ops, symbolicInfo, numericInfo);
      {
        comet_pdump(rootOp->getParentOfType<ModuleOp>());
      }
      comet_debug() << " finished call genCmptOps, i = " << i << "\n";
    }
  }

  {
    comet_debug() << "End of doLoweringIndexTreeToSCF()\n";
    comet_pdump(rootOp->getParentOfType<ModuleOp>());
  }

  comet_debug() << "Cleaning up IndexTree Operations\n";
  comet_vdump(rootOp);
  std::vector<Operation *> operations_dumpster;
  rootOp.erase();
  for (auto itOp : wp_ops)
  {
    if (indexTree::IndexTreeComputeOp cur_op = dyn_cast<mlir::indexTree::IndexTreeComputeOp>(itOp.getDefiningOp()))
    {
      comet_pdump(itOp.getDefiningOp()->getOperand(0).getDefiningOp()); /// RHS
      comet_pdump(itOp.getDefiningOp()->getOperand(1).getDefiningOp()); /// LHS
      operations_dumpster.push_back(cur_op.getOperand(0).getDefiningOp());
      operations_dumpster.push_back(cur_op.getOperand(1).getDefiningOp());
    }
    comet_pdump(itOp.getDefiningOp());
    itOp.getDefiningOp()->erase();
  }
  for (auto op : operations_dumpster)
  {
    op->erase();
  }

#ifdef DEBUG_MODE_LowerIndexTreeToSCFPass
  {
    int opstree_i = 0;
    for (auto n : opstree_vec)
    {
      comet_debug() << " " << n->id << "\n";
      comet_debug() << "opstree_vec[" << opstree_i << "] "
                    << "forOps.size():" << n->forOps.size() << " "
                    << "accessIdx.size():" << n->accessIdx.size() << "\n";
      ///              << "cmptOps.size():" << n->cmptOps.size() << "\n";
      if (n->parent != nullptr)
      {
        comet_debug() << "parent: " << n->parent->id << "\n";
      }
      else
      {
        comet_debug() << "parent: null \n";
      }
      ++opstree_i;
    }
  }
#endif
  /// ----------------- ///
  /// Free the memory occupied by each element in opstree_vec.
  /// ----------------- ///
  cleanOpstreeVec(opstree_vec);

} /// End doLoweringIndexTreeToSCF()

void LowerIndexTreeToSCFPass::runOnOperation()
{
  comet_debug() << "LowerIndexTreeToSCFPass\n";
  func::FuncOp function = getOperation();
  auto module = function.getOperation()->getParentOfType<ModuleOp>();
  auto *ctx = &getContext();

  /// Declare comet_sort_index()
  declareSortFunc(module,
                  ctx,
                  function.getLoc());

  std::vector<indexTree::IndexTreeOp> iTreeRoots;
  getIndexTreeOps(function, iTreeRoots /* output */);
  for (auto root : iTreeRoots)
  {
    comet_vdump(root);
    OpBuilder builder(root);
    doLoweringIndexTreeToSCF(root, builder);
  }
}

/// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createLowerIndexTreeToSCFPass()
{
  return std::make_unique<LowerIndexTreeToSCFPass>();
}
