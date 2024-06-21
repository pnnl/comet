//===- Utils.h - Utilities to support the Tensor Algebra dialect and Index Tree dialect -===//
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

#ifndef UTILS_H_
#define UTILS_H_

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "ta-utils"

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

/// TODO(gkestor): supports only f64 -  need generalization
std::string VALUETYPE = "f64";

using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::indexTree;

namespace mlir
{
  namespace tensorAlgebra
  {
    /// Convert the given TensorType into the corresponding MemRefType.
    MemRefType convertTensorToMemRef(TensorType type)
    {
      assert(type.hasRank() && "expected only ranked shapes");
      return MemRefType::get(type.getShape(), type.getElementType());
    }

    std::vector<unsigned int> getReverseIdentityPermutation(size_t size)
    {
      std::vector<unsigned int> perm(size);
      for (size_t i = 0; i < size; i++)
      {
        perm[i] = (size - 1) - i;
      }
      return perm;
    }

    std::vector<unsigned int> getIdentityPermutation(size_t size)
    {
      std::vector<unsigned int> perm(size);
      for (size_t i = 0; i < size; i++)
      {
        perm[i] = i;
      }
      return perm;
    }

    bool hasFuncDeclaration(ModuleOp &module, std::string funcName)
    {
      for (auto func : module.getOps<func::FuncOp>())
      {
        StringAttr func_name = func.getSymNameAttr();
        if (funcName == func_name.getValue())
          return true;
      }
      return false;
    }

    /// TODO(gkestor): review the use of this code
    /// Insert an allocation and deallocation for the given MemRefType.
    Value insertAllocAndDealloc(MemRefType memtype, Location loc,
                                PatternRewriter &rewriter)
    {
      /// AllocOp is defined in memref Dialect
      auto alloc = rewriter.create<memref::AllocOp>(loc, memtype, rewriter.getI64IntegerAttr(32));

      /// Make sure to allocate at the beginning of the block.
      auto *parentBlock = alloc.getOperation()->getBlock();

      /// Make sure to deallocate this alloc at the end of the block. This is fine
      /// as functions have no control flow.
      auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
      dealloc.getOperation()->moveBefore(&parentBlock->back());
      return alloc;
    }

    Value insertAllocAndDeallocDynamic(MemRefType memtype, std::vector<Value> &dynamic_sizes, Location loc,
                                       PatternRewriter &rewriter)
    {
      /// AllocOp is defined in memref Dialect
      auto alloc = rewriter.create<memref::AllocOp>(loc, memtype, dynamic_sizes, rewriter.getI64IntegerAttr(32));

      /// Make sure to allocate at the beginning of the block.
      auto *parentBlock = alloc.getOperation()->getBlock();

      /// Make sure to deallocate this alloc at the end of the block. This is fine
      /// as functions have no control flow.
      auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
      dealloc.getOperation()->moveBefore(&parentBlock->back());
      return alloc;
    }

    Value insertAllocAndInitialize(Location loc, MemRefType memtype, ValueRange allocValueRange, PatternRewriter &rewriter)
    {
      /// Memory allocation and initialization
      Value alloc_op = rewriter.create<memref::AllocOp>(loc, memtype, allocValueRange);
      comet_debug() << "Alloc Op for initialization: ";
      comet_vdump(alloc_op);
      auto elementType = memtype.getElementType();

      Value cst_init;
      if (elementType.isF64())
      {
        comet_debug() << "Element type F64\n";
        cst_init = rewriter.create<ConstantOp>(loc, rewriter.getF64FloatAttr(0.0));
      }
      else if (elementType.isF32())
      {
        comet_debug() << "Element type F32\n";
        cst_init = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
      }
      else if (elementType.isIndex())
      {
        comet_debug() << "Element type Index\n";
        cst_init = rewriter.create<ConstantIndexOp>(loc, 0);
      }
      else if (elementType.isInteger(1))
      {
        comet_debug() << "Element type I1 - boolean\n";
        cst_init = rewriter.create<ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(0));
      }
      else
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << "Not supported memory reference type. Supported element Types are F32, F64, Index \n";
      }

      /// TODO(gkestor): add better initialization method based on the dimension, leverage existing operations for initialization
      auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
      auto upperBound = alloc_op.getDefiningOp()->getOperand(0);
      auto step = rewriter.create<ConstantIndexOp>(loc, 1);
      auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      auto insertPt = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(loop.getBody());

      /// Build loop body
      std::vector<Value> indices = {loop.getInductionVar()};
      rewriter.create<memref::StoreOp>(loc, cst_init, alloc_op, ValueRange{indices});

      /// need to restore the insertion point to the previous point
      rewriter.restoreInsertionPoint(insertPt);
      comet_debug() << " insertAllocAndInitialize loop "
                    << "\n";
      comet_vdump(loop);

      return alloc_op;
    }

    /// TODO(gkestor): review this code
    void insertInitialize(Location loc,
                          Value cst_init,
                          Value alloc_op,
                          Value accessIdx,
                          OpBuilder &builder,
                          bool use_dynamic_init,
                          Value dynamic_init)
    {
      [[maybe_unused]] auto lowerBound = builder.create<ConstantIndexOp>(loc, 0);
      MemRefType resultMemTy = alloc_op.getDefiningOp()->getResult(0).getType().cast<MemRefType>();
      std::vector<Value> cur_indices;
      std::vector<int64_t> cur_memref;

      for (int i = 0; i < resultMemTy.getRank(); i++)
      {
        if (resultMemTy.isDynamicDim(i))
          cur_memref.push_back(ShapedType::kDynamic);
        else /// The constant dim size must NOT comes from the sparse matrix
          cur_memref.push_back(resultMemTy.getDimSize(i));
      }

      assert(cur_memref.size() == 1 && " Only handle 1-D vector currently\n");
      if (cur_memref[0] == 1)
      { /// Only 1 element in the array, no need to generate for loop
        if (use_dynamic_init)
        {
          /// For Numeric Phase with workspace transform, generate:
          ///     %rowptr = memref.load %C.rowptr[%idx];
          ///     memref.store %rowptr, %alloc_op[%const_index_0];
          /// idx: accessIdx
          /// C.rowptr: dynamic_init
          comet_vdump(dynamic_init);
          Value rowptr = builder.create<memref::LoadOp>(loc, dynamic_init, ValueRange{accessIdx});
          auto const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
          [[maybe_unused]] auto store_op = builder.create<memref::StoreOp>(loc, rowptr, alloc_op, ValueRange{const_index_0});
          comet_vdump(rowptr);
          comet_vdump(store_op);
        }
        else
        {
          auto const_index_0 = builder.create<ConstantIndexOp>(loc, 0);
#ifdef DEBUG_MODE_UTILS
          auto store_op = builder.create<memref::StoreOp>(loc, cst_init, alloc_op, ValueRange{const_index_0});
          comet_vdump(store_op);
#else
          builder.create<memref::StoreOp>(loc, cst_init, alloc_op, ValueRange{const_index_0});
#endif
        }
      }
      else
      {
        /// alloc_op: a dense vector V
        /// cst_init: 0
        ///     Generate: alloc_op[accessIdx] = cst_init;
        ///     i.e.,   : V[accessIdx] = 0;
#ifdef DEBUG_MODE_UTILS
        auto store_op = builder.create<memref::StoreOp>(loc, cst_init, alloc_op, ValueRange{accessIdx});
        comet_vdump(store_op);
#else
        builder.create<memref::StoreOp>(loc, cst_init, alloc_op, ValueRange{accessIdx});
#endif
      }
    }

    void print_vector_value(std::vector<Value> vec)
    {
/// Special code for Array<bool>
#ifdef DEBUG_MODE_UTILS
      for (auto n : vec)
      {
        comet_vdump(n);
      }
#endif
    }
    /*
     ** Convert the Type* objects' dump() to screen information into string
     */
    std::string dump2str(Value t)
    {
      std::string str;
      llvm::raw_string_ostream rso(str);
      t.print(rso);
      rso.flush();
      return rso.str();
    }

    /*
     * Reference:
     * 1. Why can’t I separate the definition of my templates class from its declaration and put it inside a .cpp file?
     *    https://isocpp.org/wiki/faq/templates#templates-defn-vs-decl
     * 2. How can I avoid linker errors with my template functions?
     *    https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
     */

    bool isDense(std::string s, std::string delim)
    {
      if (s.compare("Dense") == 0)
      {
        return true;
      }

      auto start = 0;
      auto end = s.find(delim);
      while (end != std::string::npos)
      {
        if ((s.substr(start, end - start)).compare("D") != 0)
        {
          return false;
        }

        start = end + delim.length();
        end = s.find(delim, start);
      }
      if ((s.substr(start, end)).compare("D") != 0)
      {
        return false;
      }

      return true;
    }

    /// Determine whether this index's loop in rhs1 should be merged or not
    bool isMergedIndex(std::vector<std::string> format_vec, int cur_idx, int sumIndex)
    {
      if (format_vec[cur_idx] != "singleton")
      {
        for (int i = cur_idx + 1; i <= sumIndex; i++)
        {
          if (format_vec[i] != "singleton")
          {
            return false;
          }
        }
        return true;
      }
      else
      {
        return false;
      }
    }

    std::vector<std::vector<int64_t>> getAllPerms(ArrayAttr indexMaps)
    {
      std::vector<std::vector<int64_t>> allPerms;
      /// Find summation indices
      for (const auto &map : indexMaps)
      {
        auto affineMap = map.cast<AffineMapAttr>().getValue();
        std::vector<int64_t> perm;
        for (size_t i = 0; i < affineMap.getNumResults(); i++)
        {
          auto expr = affineMap.getResult(i);
          perm.push_back(llvm::cast<AffineDimExpr>(expr).getPosition());
        }

        allPerms.push_back(perm);
      }
      return allPerms;
    }

    std::vector<std::vector<int64_t>> getAllPermsWorkspace(ArrayAttr indexMaps)
    {
      comet_debug() << " " << indexMaps.size() << " ";

      std::vector<std::vector<int64_t>> allPerms;
      /// Find summation indices
      for (auto map : indexMaps)
      {
        comet_vdump(map);
        std::vector<int64_t> perm;

        if (auto arrayattr = map.dyn_cast<ArrayAttr>())
        {
          comet_debug() << " ";
          for (auto n : arrayattr)
          {
            comet_debug() << " ";
            if (IntegerAttr i = n.dyn_cast<IntegerAttr>())
            {
              comet_debug() << " " << i.getInt() << "\n";
              perm.push_back(i.getInt());
            }
          }
          comet_debug() << " ";
        }
        else
        {
          llvm::errs() << __LINE__ << " Different type: ";
          /// map.getType().dump();
        }
        allPerms.push_back(perm);
        comet_debug() << " ";
      }
      return allPerms;
    }

    std::vector<unsigned int> getFreeIndices(std::vector<unsigned int> rhs_perm, std::vector<unsigned int> lhs_perm)
    {
      std::vector<unsigned int> rhsFreeIndices;
      std::set<unsigned int> rhsIndices(rhs_perm.begin(), rhs_perm.end());
      std::set<unsigned int> lhsIndices(lhs_perm.begin(), lhs_perm.end());
      std::set_intersection(rhsIndices.begin(), rhsIndices.end(),
                            lhsIndices.begin(), lhsIndices.end(),
                            std::inserter(rhsFreeIndices, rhsFreeIndices.begin()));

      LLVM_DEBUG(llvm::dbgs() << "freeIndices: ");
      print_vector<unsigned int>(rhsFreeIndices);
      return rhsFreeIndices;
    }

    std::vector<unsigned int> getSumIndices(std::vector<unsigned int> rhs_perm, std::vector<unsigned int> rhs_perm_free)
    {
      std::vector<unsigned int> rhsSumIndices;
      std::set<unsigned int> rhsIndices(rhs_perm.begin(), rhs_perm.end());
      std::set<unsigned int> rhsFreeIndices(rhs_perm_free.begin(), rhs_perm_free.end());
      std::set_difference(rhsIndices.begin(), rhsIndices.end(),
                          rhsFreeIndices.begin(), rhsFreeIndices.end(),
                          std::inserter(rhsSumIndices, rhsSumIndices.begin()));

      LLVM_DEBUG(llvm::dbgs() << "sumIndices: ");
      print_vector<unsigned int>(rhsSumIndices);
      return rhsSumIndices;
    }

    /// Assumption: The freeIndices and SumIndices are continuous in the tensor
    std::vector<unsigned int> getIndexIterateOrder(std::vector<unsigned int> rhs1_perm, std::vector<unsigned int> rhs2_perm)
    {
      comet_debug() << "print rhs1, rhs2: \n";
      print_vector<unsigned int>(rhs1_perm);
      print_vector<unsigned int>(rhs2_perm);
      /// sparse * dense
      /// 0 1 2, 3 2 ==> 0, 1, 3, 2
      /// 0 1 2 3, 2 3 4 ==> 0, 1, 4
      /// 0 1 2 3, 4 1 5 3 ==> 0 4 1 2 5 3
      /// C[a, b, c, d] = A[e, a] * B[e, b, c, d];
      std::vector<unsigned int> indexIterateOrder;
      std::vector<unsigned int> rhs1_samePos;
      for (unsigned int i = 0; i < rhs1_perm.size(); i++)
      {
        unsigned int pos = findIndexInVector(rhs2_perm, rhs1_perm[i]);
        if (pos < rhs2_perm.size())
        { /// found
          rhs1_samePos.push_back(i);
          comet_debug() << __LINE__ << " pos: " << i << "\n";
        }
        else
        {
          comet_debug() << __LINE__ << " not in"
                        << "\n";
        }
      }

      std::vector<unsigned int> rhs2_samePos;
      for (unsigned int i = 0; i < rhs2_perm.size(); i++)
      {
        unsigned int pos = findIndexInVector(rhs1_perm, rhs2_perm[i]);
        if (pos < rhs1_perm.size())
        { /// found
          rhs2_samePos.push_back(i);
          comet_debug() << __LINE__ << " pos: " << i << "\n";
        }
        else
        {
          comet_debug() << __LINE__ << " not in"
                        << "\n";
        }
      }

      comet_debug() << __LINE__ << "samePos size: " << rhs1_samePos.size() << ", " << rhs2_samePos.size() << "\n";

      assert(rhs1_samePos.size() == rhs2_samePos.size());

      for (unsigned int samep = 0; samep < rhs1_samePos.size(); samep++)
      {
        /// get rhs1 indices
        if (samep == 0)
        {
          for (unsigned int i = 0; i < rhs1_samePos[samep]; i++)
          {
            indexIterateOrder.push_back(rhs1_perm[i]);
          }
          for (unsigned int i = 0; i < rhs2_samePos[samep]; i++)
          {
            indexIterateOrder.push_back(rhs2_perm[i]);
          }
          indexIterateOrder.push_back(rhs2_perm[rhs2_samePos[samep]]);
        }
        else
        {
          for (unsigned int i = rhs1_samePos[samep - 1] + 1; i < rhs1_samePos[samep]; i++)
          {
            indexIterateOrder.push_back(rhs1_perm[i]);
          }
          for (unsigned int i = rhs2_samePos[samep - 1] + 1; i < rhs2_samePos[samep]; i++)
          {
            indexIterateOrder.push_back(rhs2_perm[i]);
          }
          indexIterateOrder.push_back(rhs2_perm[rhs2_samePos[samep]]);
        }
      }

      for (unsigned int i = rhs1_samePos[rhs1_samePos.size() - 1] + 1; i < rhs1_perm.size(); i++)
      {
        indexIterateOrder.push_back(rhs1_perm[i]);
      }
      for (unsigned int i = rhs2_samePos[rhs1_samePos.size() - 1] + 1; i < rhs2_perm.size(); i++)
      {
        indexIterateOrder.push_back(rhs2_perm[i]);
      }

      /// get rhs2 indices
      print_vector<unsigned int>(indexIterateOrder);

      return indexIterateOrder;
    }

    /// for string delimiter
    std::vector<std::string> stringSplit(std::string s, std::string delimiter)
    {
      /// comet_debug() << "split formats string: " << s << ", deli: "<< delimiter << ".\n";

      std::vector<std::string> res;

      std::string format = "";
      for (unsigned int i = 0; i < s.length(); i++)
      {
        comet_debug() << "s[" << i << "]: " << s[i] << "\n";
        if (s[i] != delimiter[0] && s[i] != delimiter[1])
        {
          /// format.append(s[i]+"");
          format = format + s[i];
          comet_debug() << "format: " << format << "\n";
        }
        else if (s[i] == delimiter[0])
        {
          res.push_back(format);
          format = "";
        }
      }
      comet_debug() << "format: " << format << "\n";
      res.push_back(format);

      comet_debug() << "The final format: ";
      print_vector<std::string>(res);
      return res;
    }

    std::vector<std::vector<std::string>> getAllFormats(ArrayAttr opFormatsArrayAttr, std::vector<std::vector<int64_t>> allPerms)
    {
      std::vector<std::vector<std::string>> allFormats(allPerms.size());
      /// format with each input matrix: ["CSR", "D", "D"] SpMM
      for (unsigned int i = 0; i < opFormatsArrayAttr.size(); i++)
      {
        std::string formats_str(opFormatsArrayAttr[i].cast<mlir::StringAttr>().getValue());
        unsigned int tensorDims = allPerms[i].size();

        comet_debug() << "format_str: " << formats_str << ", tensorDims: " << tensorDims << "\n";

        if (formats_str.compare("CSR") == 0)
        {
          assert(tensorDims == 2 && "formst is CSR, should be a 2D tensor.\n");
          allFormats[i].push_back("D");
          allFormats[i].push_back("CU");
        }
        else if (formats_str.compare("ModeGeneric") == 0)
        {
          /// Currently only support modegeneric on 3 D tensor
          assert(tensorDims == 3 && "formst is ModeGeneric, should be a 3D tensor.\n");
          allFormats[i].push_back("CN");
          allFormats[i].push_back("S");
          allFormats[i].push_back("D");
        }
        else if (formats_str.compare("DCSR") == 0 || formats_str.compare("CSF") == 0)
        {
          assert(tensorDims > 1 && "formst is DCSR or CSF, should be more than 1D.\n");
          for (unsigned int d = 0; d < tensorDims; d++)
          {
            allFormats[i].push_back("CU");
          }
        }
        else if (formats_str.compare("ELL") == 0)
        {
          allFormats[i].push_back("D");
          allFormats[i].push_back("D");
          allFormats[i].push_back("S");
        }
        else if (formats_str.compare("BCSR") == 0)
        {
          allFormats[i].push_back("D");
          allFormats[i].push_back("CN");
          allFormats[i].push_back("D");
          allFormats[i].push_back("D");
        }
        else if (formats_str.compare("CSB") == 0)
        {
          allFormats[i].push_back("D");
          allFormats[i].push_back("D");
          allFormats[i].push_back("CU");
          allFormats[i].push_back("S");
        }
        else if (formats_str.compare("COO") == 0)
        {
          assert(tensorDims > 1 && "formst is COO, should be more than 1D.\n");
          for (unsigned int d = 0; d < tensorDims; d++)
          {
            if (d == 0)
            {
              allFormats[i].push_back("CN");
              comet_debug() << "CN\n";
            }
            else
            {
              allFormats[i].push_back("S");
              comet_debug() << "S\n";
            }
          }
        }
        else if (formats_str.compare("Dense") == 0)
        {
          for (unsigned int d = 0; d < tensorDims; d++)
          {
            allFormats[i].push_back("D");
          }
        }
        else if (formats_str.find("D") != std::string::npos || formats_str.find("CU") != std::string::npos || formats_str.find("CN") != std::string::npos || formats_str.find("S") != std::string::npos)
        {
          allFormats[i] = stringSplit(formats_str, ", ");
        }
        else
        {
          llvm::errs() << "Unsupported formats: " << formats_str << " (tensor dimes: " << tensorDims << ") \n";
        }

        comet_debug() << "allFormats[" << i << "].size(): " << allFormats[i].size() << "\n";
        for (auto n : allFormats[i])
        {
          comet_debug() << "format: " << n << "\n";
          LLVM_DEBUG(llvm::dbgs() << "format: " << n << "\n");
        }
      }
      return allFormats;
    }

    std::vector<std::vector<std::string>> getAllFormatsWorkspace(ArrayAttr opFormatsArrayAttr)
    {
      std::vector<std::vector<std::string>> allFormats(opFormatsArrayAttr.size());

      /// format with each input matrix: ["CSR", "D", "D"] SpMM
      for (unsigned int i = 0; i < opFormatsArrayAttr.size(); i++)
      {
        comet_debug() << " ";
        /// std::string formats_str;
        if (opFormatsArrayAttr[i].dyn_cast<mlir::StringAttr>())
        {
          comet_debug() << " yes ";
        }
        else if (mlir::ArrayAttr formatArrayAttr = opFormatsArrayAttr[i].dyn_cast<mlir::ArrayAttr>())
        {
          comet_debug() << " yes " << formatArrayAttr.size() << " ";
          for (unsigned long j = 0; j < formatArrayAttr.size(); j++)
          {
            if (mlir::StringAttr format = formatArrayAttr[j].dyn_cast<mlir::StringAttr>())
            {
              std::string formats_str(format.getValue());
              comet_debug() << " " << formats_str << " ";
              allFormats[i].push_back(formats_str);
            }
          }
        }
        else
        {
          allFormats[i].push_back("");
        }

        comet_debug() << "allFormats[" << i << "].size(): " << allFormats[i].size() << "\n";
        for (auto n : allFormats[i])
        {
          comet_debug() << "format: " << n << "\n";
          LLVM_DEBUG(llvm::dbgs() << "format: " << n << "\n");
        }
      }
      return allFormats;
    }

    /// Get the format of the tensor
    std::string getTensorFormat(std::vector<std::vector<std::string>> allFormats, unsigned int tensor_id)
    {
      assert(tensor_id < allFormats.size() && "illegal tensor_id\n");
      std::string format_ret = "";
      std::vector<std::string> format = allFormats[tensor_id];

      if (format.size() == 1 && format[0].compare("Dense") == 0)
        format_ret = "Dense";
      else if (format.size() == 2 && (format[0].compare("D") == 0 && format[1].compare("D") == 0))
        format_ret = "Dense";

      else if (format.size() == 1 && format[0].compare("CSR") == 0)
        format_ret = "CSR";
      else if (format.size() == 2 && (format[0].compare("D") == 0 && format[1].compare("CU") == 0))
        format_ret = "CSR";

      else if (format.size() == 1 && format[0].compare("COO") == 0)
        format_ret = "COO";
      else if (format.size() == 2 && (format[0].compare("CN") == 0 && format[1].compare("S") == 0))
        format_ret = "COO";

      else if (format.size() == 1 && format[0].compare("DCSR") == 0)
        format_ret = "DCSR";
      else if (format.size() == 2 && (format[0].compare("CU") == 0 && format[1].compare("CU") == 0))
        format_ret = "DCSR";

      else if (format.size() == 1 && format[0].compare("ELL") == 0)
        format_ret = "ELL";
      /// TODO(gkestor): Individual attributes

      else if (format.size() == 1 && format[0].compare("BCSR") == 0)
        format_ret = "BCSR";
      /// TODO(gkestor): Individual attributes

      else if (format.size() == 1 && format[0].compare("CSB") == 0)
        format_ret = "CSB";
      /// TODO(gkestor): Individual attributes

      else if (format.size() == 3 && (format[0].compare("D") == 0 && format[1].compare("D") == 0 && format[2].compare("D") == 0))
        format_ret = "Dense";
      else if (format.size() == 3 && (format[0].compare("CN") == 0 && format[1].compare("S") == 0 && format[2].compare("S") == 0))
        format_ret = "COO";
      else if (format.size() == 1 && format[0].compare("CSF") == 0)
        format_ret = "CSF";
      else if (format.size() == 3 && (format[0].compare("CU") == 0 && format[1].compare("CU") == 0 && format[2].compare("CU") == 0))
        format_ret = "CSF";
      else if (format.size() == 1 && format[0].compare("ModeGeneric") == 0)
        format_ret = "ModeGeneric";
      else if (format.size() == 3 && (format[0].compare("CN") == 0 && format[1].compare("S") == 0 && format[2].compare("S") == 0))
        format_ret = "ModeGeneric";
      else
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: Unsupported formats\n";
      }

      return format_ret;
    }

    std::string getFormat(std::vector<unsigned int> allLocs, std::vector<std::vector<unsigned int>> allPerms, std::vector<std::vector<std::string>> allFormats)
    {
      std::vector<std::string> formats(allFormats.size());

      for (unsigned int i = 0; i < allFormats.size(); i++)
      {
        if (allLocs[i] < allPerms[i].size())
        {
          formats[i] = allFormats[i][allLocs[i]];
        }
        else
        {
          /// The index is not in this tensor, format is "", i.e. empty
          formats[i] = "";
        }
      }
      comet_debug() << "formats[0], formats[1]: " << formats[0] << ", " << formats[1] << "\n";
      /// Currently only support mix sparse dense tensor contraction
      std::string format;
      /// The index in both tensors are dense
      if (formats[0].compare("D") == 0 && formats[1].compare("D") == 0)
      {
        format = "D";
      }
      /// The index only in one tensor
      else if (formats[1].compare("") == 0)
      {
        assert(formats[0].compare("") != 0 && "index should be in rhs1\n");
        format = formats[0];
      }
      else if (formats[0].compare("") == 0)
      {
        assert(formats[1].compare("") != 0 && "index should be in rhs2\n");
        format = formats[1];
      }
      /// The index in one tensor is dense, another is sparse ==> sparse
      else if (formats[1].compare("D") != 0)
      {
        format = formats[1];
      }
      else if (formats[0].compare("D") != 0)
      {
        format = formats[0];
      }
      else
      {
        llvm::errs() << "Not supported format: " << formats[0] << ", " << formats[1] << "\n";
        format = formats[0] + " " + formats[1];
      }

      return format;
    }

    bool checkIsElementwise(std::vector<std::vector<int>> allPerms)
    {
      bool isElementwise = false;
      if (allPerms.size() > 1 && allPerms[0].size() == allPerms[1].size()) /// to check the case sparse = dense, produced after workspace transformations
      {
        if (std::equal(allPerms[0].begin(), allPerms[0].end(), allPerms[1].begin()))
        {
          isElementwise = true;
        }
      }
      return isElementwise;
    }

    bool checkIsDense(std::vector<std::string> format)
    {
      bool isDense = true;
      comet_debug() << "checkIsDense size: " << format.size() << "\n";
      for (unsigned long i = 0; i < format.size(); i++)
      {
        comet_debug() << "checkIsDense Format: " << format[i] << "\n";
        if (format[i].compare("D"))
        {
          isDense = false;
          break;
        }
      }
      return isDense;
    }

    bool checkIsMixedMode(std::vector<std::vector<std::string>> formats)
    {
      /// TODO(gkestor): review the following code
      comet_debug() << "how many operands format:" << formats.size() << "\n";
      if (formats.size() == 2)
      { /// binary operation
        bool isFirstDense = checkIsDense(formats[0]);
        bool isSecondDense = checkIsDense(formats[1]);

        comet_debug() << "isFirstDense:" << isFirstDense << " isSecondDense: " << isSecondDense << "\n";
        if ((isFirstDense && !isSecondDense) || (!isFirstDense && isSecondDense))
        {
          return true;
        }
        else
          return false;
      }
      if (formats.size() == 1) /// new computeOp produces after workspace transformations. There is only one operand on rhs
        return false;

      return false;
    }

    std::vector<Value> getFormatsValue(std::string formats_str, int rank_size, PatternRewriter &rewriter, Location loc, IndexType indexType)
    {
      Value format_unk = rewriter.create<ConstantOp>(loc, indexType, rewriter.getIndexAttr(-1));
      Value format_dense = rewriter.create<ConstantOp>(loc, indexType, rewriter.getIndexAttr(0));
      Value format_compressed = rewriter.create<ConstantOp>(loc, indexType, rewriter.getIndexAttr(1));
      Value format_compressednonunique = rewriter.create<ConstantOp>(loc, indexType, rewriter.getIndexAttr(2));
      Value format_singleton = rewriter.create<ConstantOp>(loc, indexType, rewriter.getIndexAttr(3));
      /// read_input_sizes_2D_f64 or read_input_sizes_3D_f64
      comet_debug() << "\n";
      std::vector<Value> dim_format;

      if (rank_size == 2)
      { /// 2D
        comet_debug() << " 2D\n";
        /// Value dim0_format, dim1_format;
        if (formats_str.compare(0, 3, "CSR") == 0)
        {
          dim_format.push_back(format_dense);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 4, "DCSR") == 0)
        {
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 3, "COO") == 0)
        { /// COO
          dim_format.push_back(format_compressednonunique);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 3, "ELL") == 0)
        { /// ELL
          dim_format.push_back(format_dense);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 4, "BCSR") == 0)
        { /// BCSR
          dim_format.push_back(format_dense);
          dim_format.push_back(format_compressednonunique);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_dense);
        }
        else if (formats_str.compare(0, 3, "CSB") == 0)
        { /// CSB
          dim_format.push_back(format_dense);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_singleton);
        }
        else if (formats_str.find("D") != std::string::npos || formats_str.find("CU") != std::string::npos || formats_str.find("CN") != std::string::npos || formats_str.find("S") != std::string::npos)
        {
          std::vector<std::string> format_vec = stringSplit(formats_str, ", ");
          for (auto n : format_vec)
          {
            if (n.compare(0, 1, "D") == 0)
            {
              dim_format.push_back(format_dense);
            }
            else if (n.compare(0, 2, "CU") == 0)
            {
              dim_format.push_back(format_compressed);
            }
            else if (n.compare(0, 2, "CN") == 0)
            {
              dim_format.push_back(format_compressednonunique);
            }
            else if (n.compare(0, 1, "S") == 0)
            {
              dim_format.push_back(format_singleton);
            }
          }
        }
        else
        {
          llvm::errs() << "Unsupported formats: " << formats_str << " (tensor dimes: " << rank_size << ") \n";
        }
      }
      else if (rank_size == 3)
      { /// 3D
        comet_debug() << " 3D\n";
        /// Value dim0_format, dim1_format, dim2_format;
        if (formats_str.compare(0, 3, "CSF") == 0)
        {
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 11, "ModeGeneric") == 0)
        {
          dim_format.push_back(format_compressednonunique);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 3, "COO") == 0)
        { /// COO
          dim_format.push_back(format_compressednonunique);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.find("D") != std::string::npos || formats_str.find("CU") != std::string::npos || formats_str.find("CN") != std::string::npos || formats_str.find("S") != std::string::npos)
        {
          std::vector<std::string> format_vec = stringSplit(formats_str, ", ");
          comet_debug() << " format_vec.size(): " << format_vec.size() << " \n";

          for (auto n : format_vec)
          {
            comet_debug() << "Current format attribute: " << n << "---\n";
            if (n.compare(0, 1, "D") == 0)
            {
              dim_format.push_back(format_dense);
            }
            else if (n.compare(0, 2, "CU") == 0)
            {
              dim_format.push_back(format_compressed);
            }
            else if (n.compare(0, 2, "CN") == 0)
            {
              dim_format.push_back(format_compressednonunique);
            }
            else if (n.compare(0, 1, "S") == 0)
            {
              dim_format.push_back(format_singleton);
            }
            else
            {
              llvm::errs() << "Uncorrect format attribute: " << n << "---\n";
            }
            comet_debug() << " dim_format.size(): " << dim_format.size() << " \n";
          }
          comet_debug() << " formats_str: " << formats_str << ", dim_format.size(): " << dim_format.size() << " \n";
        }
      }
      else
      {
        llvm::errs() << "Unsupported formats: " << formats_str << " (tensor dimes: " << rank_size << ") \n";
      }

      comet_debug() << " print dim_format: ";
      for (auto n : dim_format)
      {
        comet_debug() << n << " ";
      }
      comet_debug() << "\n";
      return dim_format;
    }

    std::vector<Value> getFormatsValueInt(std::string formats_str, int rank_size, PatternRewriter &rewriter, Location loc, IntegerType intType)
    {
      Value format_unk = rewriter.create<ConstantOp>(loc, intType, rewriter.getIntegerAttr(intType, -1));
      Value format_dense = rewriter.create<ConstantOp>(loc, intType, rewriter.getIntegerAttr(intType, 0));
      Value format_compressed = rewriter.create<ConstantOp>(loc, intType, rewriter.getIntegerAttr(intType, 1));
      Value format_compressednonunique = rewriter.create<ConstantOp>(loc, intType, rewriter.getIntegerAttr(intType, 2));
      Value format_singleton = rewriter.create<ConstantOp>(loc, intType, rewriter.getIntegerAttr(intType, 3));
      /// read_input_sizes_2D_f64 or read_input_sizes_3D_f64
      comet_debug() << "\n";
      std::vector<Value> dim_format;

      if (rank_size == 2)
      { /// 2D
        comet_debug() << " 2D\n";
        /// Value dim0_format, dim1_format;
        if (formats_str.compare(0, 3, "CSR") == 0)
        {
          dim_format.push_back(format_dense);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 4, "DCSR") == 0)
        {
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 3, "COO") == 0)
        { /// COO
          dim_format.push_back(format_compressednonunique);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 3, "ELL") == 0)
        { /// ELL
          dim_format.push_back(format_dense);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 4, "BCSR") == 0)
        { /// BCSR
          dim_format.push_back(format_dense);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_dense);
        }
        else if (formats_str.compare(0, 3, "CSB") == 0)
        { /// CSB
          dim_format.push_back(format_dense);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_singleton);
        }
        else if (formats_str.find("D") != std::string::npos || formats_str.find("CU") != std::string::npos || formats_str.find("CN") != std::string::npos || formats_str.find("S") != std::string::npos)
        {
          std::vector<std::string> format_vec = stringSplit(formats_str, ", ");
          for (auto n : format_vec)
          {
            if (n.compare(0, 1, "D") == 0)
            {
              dim_format.push_back(format_dense);
            }
            else if (n.compare(0, 2, "CU") == 0)
            {
              dim_format.push_back(format_compressed);
            }
            else if (n.compare(0, 2, "CN") == 0)
            {
              dim_format.push_back(format_compressednonunique);
            }
            else if (n.compare(0, 1, "S") == 0)
            {
              dim_format.push_back(format_singleton);
            }
          }
        }
        else
        {
          llvm::errs() << "Unsupported formats: " << formats_str << " (tensor dimes: " << rank_size << ") \n";
        }
      }
      else if (rank_size == 3)
      { /// 3D
        comet_debug() << " 3D\n";
        /// Value dim0_format, dim1_format, dim2_format;
        if (formats_str.compare(0, 3, "CSF") == 0)
        {
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 11, "ModeGeneric") == 0)
        {
          dim_format.push_back(format_compressednonunique);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.compare(0, 3, "COO") == 0)
        { /// COO
          dim_format.push_back(format_compressednonunique);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
          dim_format.push_back(format_singleton);
          dim_format.push_back(format_unk);
        }
        else if (formats_str.find("D") != std::string::npos || formats_str.find("CU") != std::string::npos || formats_str.find("CN") != std::string::npos || formats_str.find("S") != std::string::npos)
        {
          std::vector<std::string> format_vec = stringSplit(formats_str, ", ");
          comet_debug() << " format_vec.size(): " << format_vec.size() << " \n";
          /// print_vector<std::string>(format_vec);
          for (auto n : format_vec)
          {
            comet_debug() << "Current format attribute: " << n << "---\n";
            if (n.compare(0, 1, "D") == 0)
            {
              dim_format.push_back(format_dense);
            }
            else if (n.compare(0, 2, "CU") == 0)
            {
              dim_format.push_back(format_compressed);
            }
            else if (n.compare(0, 2, "CN") == 0)
            {
              dim_format.push_back(format_compressednonunique);
            }
            else if (n.compare(0, 1, "S") == 0)
            {
              dim_format.push_back(format_singleton);
            }
            else
            {
              llvm::errs() << "Uncorrect format attribute: " << n << "---\n";
            }
            comet_debug() << " dim_format.size(): " << dim_format.size() << " \n";
          }
          comet_debug() << " formats_str: " << formats_str << ", dim_format.size(): " << dim_format.size() << " \n";
        }
      }
      else
      {
        llvm::errs() << "Unsupported formats: " << formats_str << " (tensor dimes: " << rank_size << ") \n";
      }

      comet_debug() << " print dim_format: ";
      for (auto n : dim_format)
      {
        comet_debug() << n << " ";
      }
      comet_debug() << "\n";
      return dim_format;
    }

    unsigned int findIndexInVector_Value(std::vector<Value> vec, Value e)
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

    bool isRealRoot(Operation *op)
    {
      bool isrealroot = false;

      if (op->getResult(0).hasOneUse() && isa<IndexTreeOp>(*(op->getResult(0).getUsers().begin())))
      {
        isrealroot = true;
      }
      comet_debug() << " isrealroot = " << isrealroot << " ";
      comet_pdump(op);
      return isrealroot;
    }

    /// dfs traverse the tcRootOp,
    /// parent node can get from getUser() function, only one user since tree structure
    void dfsRootOpTree(Value tcRootOp, std::vector<Value> &ret)
    {
      if (isa<indexTree::IndexTreeIndicesOp>(tcRootOp.getDefiningOp()))
      {
        IndexTreeIndicesOp workspaceop = dyn_cast<indexTree::IndexTreeIndicesOp>(tcRootOp.getDefiningOp());

        comet_debug() << " dfsRootOpTree\n";
        comet_vdump(workspaceop);

        unsigned int sz = workspaceop.getChildren().size();

        comet_debug() << " " << sz << " ";
        ret.push_back(workspaceop);
        comet_debug() << " ";
        comet_vdump(workspaceop);

        for (unsigned int i = 0; i < sz; i++)
        {
          Value t = workspaceop.getChildren()[i];
          dfsRootOpTree(t, ret);
        }
      }
      else if (isa<indexTree::IndexTreeComputeOp>(tcRootOp.getDefiningOp()))
      {
        indexTree::IndexTreeComputeOp leafop = dyn_cast<indexTree::IndexTreeComputeOp>(tcRootOp.getDefiningOp());
        /// comet_debug() <<  " dfsRootOpTree\n";
        comet_vdump(leafop);
        ret.push_back(leafop);
      }
    }

    void getAncestorsWp(Value op, std::vector<Value> &ret /* output ancestors*/, std::vector<Value> &dfsOps)
    {
      comet_debug() << " ";
      comet_vdump(op);
      bool hasParent = true;
      while ((isa<indexTree::IndexTreeIndicesOp>(op.getDefiningOp()) || isa<indexTree::IndexTreeComputeOp>(op.getDefiningOp())) && hasParent)
      {
        comet_debug() << " ";
        comet_vdump(op);
        /// Should only one user in the index tree
        for (auto n : op.getDefiningOp()->getUsers())
        {
          /// comet_debug() <<  " "; comet_pdump(n); comet_vdump(n->getResult(0));

          bool isInTree = false;
          if (findIndexInVector_Value(dfsOps, n->getResult(0)) < dfsOps.size())
          {
            isInTree = true;
          }

          comet_debug() << " isInTree: " << isInTree << " \n";
          if (isInTree)
          {
            ret.push_back(n->getResult(0));
            op = n->getResult(0);
            comet_debug() << " ";
            comet_pdump(n);
            hasParent = true;
          }
          else
          {
            comet_debug() << " the user op does not belong to the index tree\n";
            hasParent = false;
          }
        }
      }
    }

    std::vector<int> getUnionOf2Dvector(std::vector<std::vector<int>> perms_int)
    {
      comet_debug() << " perms_int.size(): " << perms_int.size() << "\n";
      std::vector<int> perms;
      if (perms_int.size() == 0)
        return perms;

      if (perms_int.size() == 1)
      {
        perms = perms_int[0];
        return perms;
      }

      /// perms_int >= 2
      perms = perms_int[0];
      comet_debug() << " perms.size(): " << perms.size() << "\n";
      for (unsigned int i = 1; i < perms_int.size(); i++)
      {
        for (unsigned int j = 0; j < perms_int[i].size(); j++)
        {
          if (std::find(perms.begin(), perms.end(), perms_int[i][j]) == perms.end())
          { /// Not in
            perms.push_back(perms_int[i][j]);
          }
        }
      }
      comet_debug() << " perms.size(): " << perms.size() << ", elements: ";
      for (auto n : perms)
      {
        comet_debug() << n << " ";
      }
      comet_debug() << "\n";

      return perms;
    }

    std::vector<std::vector<bool>> createInputOutputMapping(ArrayAttr perms, bool value)
    {
      std::vector<std::vector<bool>> mapping;
      for (unsigned int m = 0; m < perms.size(); m++)
      {
        ArrayAttr aa = perms[m].dyn_cast<mlir::ArrayAttr>();
        std::vector<bool> p;
        for (unsigned int n = 0; n < aa.size(); n++)
        {
          p.push_back(value);
          comet_debug() << " createInputOutputMapping m:" << m << " n:" << n << " value:" << value << "\n";
        }
        mapping.push_back(p);
      }
      return mapping;
    }

    std::vector<std::vector<int>> convertArrayAttrIntTo2DVector(ArrayAttr perms)
    {
      std::vector<std::vector<int>> perms_int;
      for (unsigned int m = 0; m < perms.size(); m++)
      {
        ArrayAttr aa = perms[m].dyn_cast<mlir::ArrayAttr>();
        std::vector<int> p;
        for (unsigned int n = 0; n < aa.size(); n++)
        {
          p.push_back(aa[n].cast<mlir::IntegerAttr>().getInt());
          comet_debug() << " convertArrayAttrIntTo2DVector:" << aa[n].cast<mlir::IntegerAttr>().getInt() << "\n";
        }
        perms_int.push_back(p);
      }
      return perms_int;
    }

    ArrayAttr convert2DVectorToArrayAttrInt(std::vector<std::vector<int>> t1_perms_int, OpBuilder &builder)
    {
      SmallVector<mlir::Attribute, 8> t1_perms;
      for (unsigned int m = 0; m < t1_perms_int.size(); m++)
      {
        SmallVector<int64_t, 8> p;
        for (unsigned int n = 0; n < t1_perms_int[m].size(); n++)
        {
          p.push_back(t1_perms_int[m][n]);
        }
        mlir::Attribute ppp1 = builder.getI64ArrayAttr(p);
        t1_perms.push_back(ppp1);
      }
      return builder.getArrayAttr(t1_perms);
    }

    ArrayAttr convert2DVectorToArrayAttrStr(std::vector<std::vector<std::string>> formats_str, OpBuilder &builder)
    {
      SmallVector<mlir::Attribute, 8> formats;
      for (unsigned int m = 0; m < formats_str.size(); m++)
      {
        SmallVector<mlir::StringRef, 8> format;
        for (unsigned int n = 0; n < formats_str[m].size(); n++)
        {
          format.push_back(formats_str[m][n]);
        }
        mlir::Attribute final_format = builder.getStrArrayAttr(format);
        formats.push_back(final_format);
      }
      return builder.getArrayAttr(formats);
    }

    std::vector<std::vector<std::string>> convertArrayAttrStrTo2DVector(ArrayAttr formats)
    {
      std::vector<std::vector<std::string>> formats_str;
      for (unsigned int m = 0; m < formats.size(); m++)
      {
        ArrayAttr aa = formats[m].dyn_cast<mlir::ArrayAttr>();
        std::vector<std::string> p;
        for (unsigned int n = 0; n < aa.size(); n++)
        {
          std::string format_str(aa[n].cast<mlir::StringAttr>().getValue());
          p.push_back(format_str);
          comet_debug() << " convertArrayAttrStrTo2DVector:" << format_str << "\n";
        }
        formats_str.push_back(p);
      }
      return formats_str;
    }
    /// Get the perms and formats of the itCompute op
    void getFormatsPermsOfComputeOp(Value computeOp,
                                    std::vector<std::vector<std::string>> &opFormats,
                                    std::vector<std::vector<int>> &opPerms,
                                    std::vector<std::vector<bool>> &inputOutputMapping)
    {
      indexTree::IndexTreeComputeRHSOp itComputeOp_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(computeOp.getDefiningOp()->getOperand(0).getDefiningOp());
      ArrayAttr opFormatsArrayAttr_rhs = itComputeOp_rhs.getAllFormats();
      ArrayAttr opPermsArrayAttr_rhs = itComputeOp_rhs.getAllPerms();
      indexTree::IndexTreeComputeLHSOp itComputeOp_lhs = dyn_cast<indexTree::IndexTreeComputeLHSOp>(computeOp.getDefiningOp()->getOperand(1).getDefiningOp());
      ArrayAttr opFormatsArrayAttr_lhs = itComputeOp_lhs.getAllFormats();
      ArrayAttr opPermsArrayAttr_lhs = itComputeOp_lhs.getAllPerms();
      assert(opFormatsArrayAttr_rhs.size() == opPermsArrayAttr_rhs.size() && "not equal RHS formats size with perms size\n");
      assert(opFormatsArrayAttr_lhs.size() == opPermsArrayAttr_lhs.size() && "not equal LHS formats size with perms size\n");

      /// Get output format, vector of vector
      /// Convert ArrayAttr into
      comet_debug() << "Start printing opFormats_rhs\n";
      std::vector<std::vector<std::string>> opFormats_rhs = convertArrayAttrStrTo2DVector(opFormatsArrayAttr_rhs);
      comet_debug() << "End printing opFormats_rhs\n";
      std::vector<std::vector<int>> opPerms_rhs = convertArrayAttrIntTo2DVector(opPermsArrayAttr_rhs);
      std::vector<std::vector<bool>> inputMapping = createInputOutputMapping(opPermsArrayAttr_rhs, true);

      comet_debug() << "Start printing opFormats_lhs\n";
      std::vector<std::vector<std::string>> opFormats_lhs = convertArrayAttrStrTo2DVector(opFormatsArrayAttr_lhs);
      comet_debug() << "End printing opFormats_lhs\n";
      std::vector<std::vector<int>> opPerms_lhs = convertArrayAttrIntTo2DVector(opPermsArrayAttr_lhs);
      std::vector<std::vector<bool>> outputMapping = createInputOutputMapping(opPermsArrayAttr_lhs, false);

      opFormats = opFormats_rhs;
      opFormats.insert(opFormats.end(), opFormats_lhs.begin(), opFormats_lhs.end());
      opPerms = opPerms_rhs;
      opPerms.insert(opPerms.end(), opPerms_lhs.begin(), opPerms_lhs.end());
      inputOutputMapping = inputMapping;
      inputOutputMapping.insert(inputOutputMapping.end(), outputMapping.begin(), outputMapping.end());
    }

    /// Get the formats of the itCompute op
    void getFormatsOfComputeOp(Value computeOp, std::vector<std::vector<std::string>> &opFormats)
    {
      indexTree::IndexTreeComputeRHSOp itComputeOp_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(computeOp.getDefiningOp()->getOperand(0).getDefiningOp());
      ArrayAttr opFormatsArrayAttr_rhs = itComputeOp_rhs.getAllFormats();
      indexTree::IndexTreeComputeLHSOp itComputeOp_lhs = dyn_cast<indexTree::IndexTreeComputeLHSOp>(computeOp.getDefiningOp()->getOperand(1).getDefiningOp());
      ArrayAttr opFormatsArrayAttr_lhs = itComputeOp_lhs.getAllFormats();

      /// Get output format, vector of vector
      /// Convert ArrayAttr into
      std::vector<std::vector<std::string>> opFormats_rhs = convertArrayAttrStrTo2DVector(opFormatsArrayAttr_rhs);
      std::vector<std::vector<std::string>> opFormats_lhs = convertArrayAttrStrTo2DVector(opFormatsArrayAttr_lhs);

      opFormats = opFormats_rhs;
      opFormats.insert(opFormats.end(), opFormats_lhs.begin(), opFormats_lhs.end());
    }

    /// Get the rhs formats of the itCompute op
    void getRHSFormatsOfComputeOp(Value computeOp, std::vector<std::vector<std::string>> &opFormats)
    {
      indexTree::IndexTreeComputeRHSOp itComputeOp_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(computeOp.getDefiningOp()->getOperand(0).getDefiningOp());
      ArrayAttr opFormatsArrayAttr_rhs = itComputeOp_rhs.getAllFormats();

      /// Get output format, vector of vector
      /// Convert ArrayAttr into
      std::vector<std::vector<std::string>> opFormats_rhs = convertArrayAttrStrTo2DVector(opFormatsArrayAttr_rhs);

      opFormats = opFormats_rhs;
    }

    /// Get the LHS formats of the itCompute op
    void getLHSFormatsOfComputeOp(Value computeOp, std::vector<std::vector<std::string>> &opFormats)
    {
      indexTree::IndexTreeComputeLHSOp itComputeOp_lhs = dyn_cast<indexTree::IndexTreeComputeLHSOp>(computeOp.getDefiningOp()->getOperand(1).getDefiningOp());
      ArrayAttr opFormatsArrayAttr_lhs = itComputeOp_lhs.getAllFormats();
      std::vector<std::vector<std::string>> opFormats_lhs = convertArrayAttrStrTo2DVector(opFormatsArrayAttr_lhs);
      opFormats = opFormats_lhs;
    }

    /// Get the input tensors of the itCompute op
    void getInputTensorsOfComputeOp(Value computeOp, std::vector<Value> &inputTensors /* output */)
    {
      /// indexTree::IndexTreeComputeOp itComputeOp = dyn_cast<indexTree::IndexTreeComputeOp>(computeOp.getDefiningOp());
      indexTree::IndexTreeComputeRHSOp itComputeOp_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(computeOp.getDefiningOp()->getOperand(0).getDefiningOp());
      comet_debug() << " ";
      comet_vdump(itComputeOp_rhs);
      for (unsigned int i = 0; i < itComputeOp_rhs.getOperation()->getNumOperands(); i++)
      {
        comet_debug() << " ";
        comet_vdump(itComputeOp_rhs.getOperation()->getOperand(i));
        inputTensors.push_back(itComputeOp_rhs.getOperation()->getOperand(i));
      }
    }

    /// Get the output tensors of the itCompute op
    void getOutputTensorsOfComputeOp(Value computeOp, std::vector<Value> &outputTensors /* output */)
    {
      indexTree::IndexTreeComputeLHSOp itComputeOp_lhs = dyn_cast<indexTree::IndexTreeComputeLHSOp>(computeOp.getDefiningOp()->getOperand(1).getDefiningOp());
      for (unsigned int i = 0; i < itComputeOp_lhs.getOperation()->getNumOperands(); i++)
      {
        outputTensors.push_back(itComputeOp_lhs.getOperation()->getOperand(i));
      }
    }

    /// Get indices in current WorkspaceOp cur_op
    void getFormatsInfo(Value cur_op,
                        std::vector<int> &indices,
                        std::vector<Value> &leafs,
                        std::vector<Value> &tensors /* output */,
                        std::vector<unsigned int> &ids /* output */,
                        std::vector<std::string> &formats /* output */)
    {
      /// For each indices, find in each leaf, which tensor, the corresponding format
      /// If in all tensors, the formats of the index are D, then D
      ///                    If only one Sparse, then sparse
      comet_debug() << " getFormatsInfo:Start Current op\n";
      comet_vdump(cur_op);
      comet_debug() << " getFormatsInfo:indices.size(): " << indices.size() << "\n";
      for (unsigned long i = 0; i < indices.size(); i++)
      {
        comet_debug() << " getFormatsInfo:indices[" << i << "]: " << indices[i] << "\n";
        /// Info for each index
        std::string format;
        Value tensor;
        unsigned int id;
        bool isSet = false;

        std::vector<std::string> formats_leafs;
        std::vector<Value> tensors_leafs;
        std::vector<unsigned int> ids_leafs;

        for (unsigned long j = 0; j < leafs.size(); j++)
        {
          /// Info for each index in leaf[j]
          comet_debug() << " getFormatsInfo:LeafOp: ";
          comet_vdump(leafs[j]);
          std::string format_in_leaf;
          Value tensor_in_leaf;
          unsigned int id_in_leaf;
          bool isSetInLeaf = false;

          /// get All perms and formats info
          if (indexTree::IndexTreeComputeOp leafop = dyn_cast<mlir::indexTree::IndexTreeComputeOp>(leafs[j].getDefiningOp()))
          {
            comet_debug() << " getFormatsInfo:leafs[" << j << "] is computeOp\n";
            std::vector<std::vector<std::string>> allFormats;
            std::vector<std::vector<int>> allPerms;
            std::vector<std::vector<bool>> inputOutputMapping;
            OpBuilder builder(leafop);
            getFormatsPermsOfComputeOp(leafop, allFormats, allPerms, inputOutputMapping);

            comet_debug() << " getFormatsInfo:Allformats allFormats.size(): " << allFormats.size() << "\n";
            for (auto m : allFormats)
            {
              comet_debug() << " ";
              for (auto n : m)
              {
                comet_debug() << n << " ";
              }
              comet_debug() << "\n";
            }

            std::vector<Value> leafop_inputTensors;
            getInputTensorsOfComputeOp(leafop, leafop_inputTensors /* output */);
            comet_debug() << " getFormatsInfo:leafop_inputTensors.size(): " << leafop_inputTensors.size() << "\n";

            std::vector<Value> leafop_outputTensors;
            getOutputTensorsOfComputeOp(leafop, leafop_outputTensors /* output */);
            comet_debug() << " getFormatsInfo:leafop_outputTensors.size(): " << leafop_outputTensors.size() << "\n";

            std::vector<Value> leafop_tensors = leafop_inputTensors;
            leafop_tensors.insert(leafop_tensors.end(), leafop_outputTensors.begin(), leafop_outputTensors.end());
#ifdef DEBUG_MODE_UTILS
            comet_debug() << " getFormatsInfo:leafop_tensors.size(): " << leafop_tensors.size() << "\n";
            for (auto n : leafop_tensors)
            {
              comet_debug() << " ";
              comet_vdump(n);
            }
#endif
            /// Check if this index is in this leaf's perms

            std::vector<std::string> formats_local;
            std::vector<Value> tensors_local;
            std::vector<unsigned int> ids_local;
            std::vector<bool> rhs_vs_lhs;
            /// This leafOp contain multiple tensors.
            comet_debug() << " getFormatsInfo:allPerms.size()" << allPerms.size() << "\n";
            for (unsigned long k = 0; k < allPerms.size(); k++)
            {
              comet_debug() << " getFormatsInfo:allPerms[" << k << "].size(): " << allPerms[k].size() << ", print allPerms[" << k << "]: ";
              print_vector<int>(allPerms[k]);
              comet_debug() << " getFormatsInfo:indices[" << i << "]: " << indices[i] << "\n";
              unsigned int idx = findIndexInVector(allPerms[k], indices[i]);
              comet_debug() << " getFormatsInfo:idx: " << idx << ", allPerms[" << k << "].size(): " << allPerms[k].size() << "\n";
              if (idx < allPerms[k].size())
              { /// In tensor k
                comet_debug() << " getFormatsInfo:AddingLocalFormat[" << k << "][" << idx << "]: " << allFormats[k][idx] << " ";
                comet_vdump(leafop_tensors[k]);
                formats_local.push_back(allFormats[k][idx]);
                tensors_local.push_back(leafop_tensors[k]);
                ids_local.push_back(idx);
                rhs_vs_lhs.push_back(inputOutputMapping[k][idx]);
              }
            }

            comet_debug() << " getFormatsInfo:formats_local.size(): " << formats_local.size() << " \n";
            for (unsigned long k = 0; k < formats_local.size(); k++)
            {
              comet_debug() << " getFormatsInfo:formats_local[k]:" << formats_local[k] << " " << ids_local[k] << " ";
              comet_vdump(tensors_local[k]);
            }

            /// analyze _local arrays, to get final formats, tensors, idx
            if (formats_local.size() > 0)
            {
              isSetInLeaf = true;
              format_in_leaf = formats_local[0];
              tensor_in_leaf = tensors_local[0];
              id_in_leaf = ids_local[0];

              for (unsigned long k = 1; k < formats_local.size(); k++)
              {
                if (format_in_leaf.compare(0, 1, "D") == 0 && formats_local[k].compare(0, 1, "D") != 0 && rhs_vs_lhs[k])
                /// if the next format in the local format is not dense and not output
                /// rhs_vs_lhs determines if the format comes from input (lhs) or output (rhs)
                /// C[i,j] =  A[i,k] * B[k, j] -> i is in both input A and output C
                /// -> j is in both input B and output C
                /// -> k is in both inputs A and B
                /// index format information stores in formats_local
                {
                  format_in_leaf = formats_local[k];
                  tensor_in_leaf = tensors_local[k];
                  id_in_leaf = ids_local[k];
                  break; /// Get the first sparse case
                }
              }
            }

          } /// if(indexTree::IndexTreeComputeOp leafop

          if (isSetInLeaf)
          {
            comet_debug() << " getFormatsInfo:isSetInLeaf: " << isSetInLeaf << ", format_in_leaf: " << format_in_leaf << ", id_in_leaf: " << id_in_leaf << ", tensor: ";
            comet_vdump(tensor_in_leaf);
            formats_leafs.push_back(format_in_leaf);
            tensors_leafs.push_back(tensor_in_leaf);
            ids_leafs.push_back(id_in_leaf);
          }

        } /// for(auto j = 0; j < leafs.size(); j++){

        comet_debug() << " getFormatsInfo:formats_leafs.size(): " << formats_leafs.size() << "\n";
        for ([[maybe_unused]] unsigned long k = 0; k < formats_leafs.size(); k++)
        {
          comet_debug() << " getFormatsInfo:formats_leafs[k]:" << formats_leafs[k] << "\n";
        }

        /// analyze the _leafs info to get the current index format, tensor, id information
        for (unsigned long j = 0; j < formats_leafs.size(); j++)
        {
          if (j == 0)
          {
            format = formats_leafs[j];
            tensor = tensors_leafs[j];
            id = ids_leafs[j];
            isSet = true;
          }
          else
          {
            if (formats_leafs[j].compare(0, 1, "D") != 0)
            { /// not D
              format = formats_leafs[j];
              tensor = tensors_leafs[j];
              id = ids_leafs[j];
              isSet = true;
              break; /// Get the first sparse case
            }
          }
        }

        if (isSet)
        {
          comet_debug() << " getFormatsInfo:EndFormat: " << format << ", id: " << id << ", tensor: ";
          comet_vdump(tensor);

          formats.push_back(format);
          tensors.push_back(tensor);
          ids.push_back(id);
        }

      } /// for(auto i = 0; i < indices.size(); i++){
    }

    /// Find leaves of tcRootOp in the Index Tree (dfsOp).
    /// A leaf is a computeOp node and tcRootOp is one of its ancestors.
    /// Method 0:
    /// Search for the tensor which contains index i from workspace tree ops: ta.tc_root
    /// Return the tensor name and the index in the tensor
    /// step: find the ancestor of each leaf, check the workspaceOp is in whose ancestorWP
    void findLeafs(Value tcRootOp,
                   std::vector<int> &indices,
                   std::vector<Value> &dfsOps,
                   std::vector<Value> &ret /* output leaves */)
    {
      std::vector<std::vector<Value>> allAncestors(dfsOps.size());
      for (unsigned int i = 0; i < dfsOps.size(); i++)
      {
        if (IndexTreeComputeOp cur_op = dyn_cast<IndexTreeComputeOp>(dfsOps[i].getDefiningOp()))
        {
          getAncestorsWp(dfsOps[i], allAncestors[i] /* output ancestors */, dfsOps);
          comet_debug() << " print allAncestors[" << i << "]: ";
          print_vector_value(allAncestors[i]);
        }
      }

      /// Each wp op in which tensors
      if (IndexTreeIndicesOp cur_op = dyn_cast<IndexTreeIndicesOp>(tcRootOp.getDefiningOp()))
      {
        comet_debug() << " ";
        comet_vdump(tcRootOp);
        for (unsigned int j = 0; j < dfsOps.size(); j++)
        {
          auto idx = findIndexInVector_Value(allAncestors[j], tcRootOp);
          if (idx < allAncestors[j].size())
          {
            if (indexTree::IndexTreeComputeOp cur_op = dyn_cast<IndexTreeComputeOp>(dfsOps[j].getDefiningOp()))
            {
              ret.push_back(dfsOps[j]);
              comet_debug() << " ";
              comet_vdump(dfsOps[j]);
            }
          }
        }
      }
    }

    /// new version for new children ops
    /// Only one user, because of the tree structure
    void replaceOperands(Operation *itComputeOp, std::vector<Value> newComputeOps)
    {
      Operation *parentIndicesOp = *(itComputeOp->getResult(0).getUsers().begin());
      comet_debug() << " ";
      comet_pdump(parentIndicesOp);
      std::vector<Value> parentOldChildren;
      for (unsigned int m = 0; m < parentIndicesOp->getNumOperands(); m++)
      {
        comet_debug() << " ";
        comet_vdump(parentIndicesOp->getOperand(m));
        parentOldChildren.push_back(parentIndicesOp->getOperand(m));
      }
      comet_debug() << " ";
      comet_vdump(itComputeOp->getResult(0));
      unsigned int whichOperand = findIndexInVector_Value(parentOldChildren, itComputeOp->getResult(0));
      comet_debug() << " which operand: " << whichOperand << "\n";
      std::vector<Value> newChildren;
      if (whichOperand < parentIndicesOp->getNumOperands())
      {
        for (unsigned int m = 0; m < whichOperand; m++)
        {
          newChildren.push_back(parentIndicesOp->getOperand(m));
        }
        for (auto n : newComputeOps)
        {
          newChildren.push_back(n);
        }
        for (unsigned int m = whichOperand + 1; m < parentIndicesOp->getNumOperands(); m++)
        {
          newChildren.push_back(parentIndicesOp->getOperand(m));
        }
      }
      parentIndicesOp->setOperands(newChildren);
    }

    /// Get the output tensors of the itCompute op
    void getTensorsOfComputeOp(Value computeOp, std::vector<Value> &tensors)
    {
      indexTree::IndexTreeComputeRHSOp itComputeOp_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(computeOp.getDefiningOp()->getOperand(0).getDefiningOp());
      comet_debug() << " ";
      comet_vdump(itComputeOp_rhs);
      for (unsigned int i = 0; i < itComputeOp_rhs.getOperation()->getNumOperands(); i++)
      {
        comet_debug() << " ";
        comet_vdump(itComputeOp_rhs.getOperation()->getOperand(i));
        tensors.push_back(itComputeOp_rhs.getOperation()->getOperand(i));
      }
      indexTree::IndexTreeComputeLHSOp itComputeOp_lhs = dyn_cast<indexTree::IndexTreeComputeLHSOp>(computeOp.getDefiningOp()->getOperand(1).getDefiningOp());
      for (unsigned int i = 0; i < itComputeOp_lhs.getOperation()->getNumOperands(); i++)
      {
        tensors.push_back(itComputeOp_lhs.getOperation()->getOperand(i));
      }
    }

    /// Get the perms and formats of the itCompute op
    void getRHSPermsOfComputeOp(Value computeOp, std::vector<std::vector<int>> &opPerms)
    {
      indexTree::IndexTreeComputeRHSOp itComputeOp_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(computeOp.getDefiningOp()->getOperand(0).getDefiningOp());
      ArrayAttr opPermsArrayAttr_rhs = itComputeOp_rhs.getAllPerms();
      /// Get output format, vector of vector
      /// Convert ArrayAttr into
      std::vector<std::vector<int>> opPerms_rhs = convertArrayAttrIntTo2DVector(opPermsArrayAttr_rhs);
      opPerms = opPerms_rhs;
    }

    /// Get the perms and formats of the itCompute op
    void getLHSPermsOfComputeOp(Value computeOp, std::vector<std::vector<int>> &opPerms)
    {
      indexTree::IndexTreeComputeLHSOp itComputeOp_lhs = dyn_cast<indexTree::IndexTreeComputeLHSOp>(computeOp.getDefiningOp()->getOperand(1).getDefiningOp());
      ArrayAttr opPermsArrayAttr_lhs = itComputeOp_lhs.getAllPerms();

      /// Get output format, vector of vector
      /// Convert ArrayAttr into
      std::vector<std::vector<int>> opPerms_lhs = convertArrayAttrIntTo2DVector(opPermsArrayAttr_lhs);

      opPerms = opPerms_lhs;
    }

    /// Get the perms and formats of the itCompute op
    void getPermsOfComputeOp(Value computeOp, std::vector<std::vector<int>> &opPerms)
    {
      indexTree::IndexTreeComputeRHSOp itComputeOp_rhs = dyn_cast<indexTree::IndexTreeComputeRHSOp>(computeOp.getDefiningOp()->getOperand(0).getDefiningOp());
      ArrayAttr opPermsArrayAttr_rhs = itComputeOp_rhs.getAllPerms();
      indexTree::IndexTreeComputeLHSOp itComputeOp_lhs = dyn_cast<indexTree::IndexTreeComputeLHSOp>(computeOp.getDefiningOp()->getOperand(1).getDefiningOp());
      ArrayAttr opPermsArrayAttr_lhs = itComputeOp_lhs.getAllPerms();

      /// Get output format, vector of vector
      /// Convert ArrayAttr into
      std::vector<std::vector<int>> opPerms_rhs = convertArrayAttrIntTo2DVector(opPermsArrayAttr_rhs);
      std::vector<std::vector<int>> opPerms_lhs = convertArrayAttrIntTo2DVector(opPermsArrayAttr_lhs);

      opPerms = opPerms_rhs;
      opPerms.insert(opPerms.end(), opPerms_lhs.begin(), opPerms_lhs.end());
    }

    double loopCostHeuristic(const std::vector<unsigned> &loopOrder, size_t dim_,
                             std::vector<unsigned> &sourceOrder, std::vector<unsigned> &destOrder)
    {
      double loopCost = 0.0;
      /// for(unsigned i=1;i < dim_ ; ++i){ /// column major: first one has no penalty
      for (unsigned i = 0; i < dim_ - 1; ++i)
      { /// row major: last one has no penalty
        /// const int idx = loopOrder[dim_-1-i];
        /// const int posB = findPos(idx, perm_);
        int idx;      /// position in sourceOrder
        int posB = 0; /// position in destOrder
        for (unsigned ii = 0; ii < sourceOrder.size(); ii++)
        {
          if (sourceOrder[ii] == loopOrder[i])
          {
            idx = ii;
            break;
          }
        }
        for (unsigned ii = 0; ii < destOrder.size(); ii++)
        {
          if (destOrder[ii] == loopOrder[i])
          {
            posB = ii;
            break;
          }
        }
        /*column major: */
        /// int importanceA = (1<<(dim_ - idx)); /// stride-1 has the most importance .
        /// int importanceB = (1<<(dim_ - posB));/// subsequent indices are half as important
        /// int penalty = 10 * (1<<(i-1)); /// smaller i has smaller penalty
        /*row major: */
        int importanceA = (1 << idx);               /// stride-1 has the most importance. Larger pos, more important
        int importanceB = (1 << posB);              /// subsequent indices are half as important
        int penalty = 10 * (1 << (dim_ - (i + 2))); /// smaller i has larger penalty
        double bias = 1.01;                         /// destOrder is more important
        loopCost += (importanceA + importanceB * bias) * penalty;
      }

      return loopCost;
    }

    void getLoopOrders(std::vector<std::vector<unsigned>> &loopOrders, size_t dim_,
                       std::vector<unsigned> &sourceOrder, std::vector<unsigned> &destOrder)
    {
      loopOrders.clear();
      std::vector<unsigned> loopOrder;
      for (unsigned i = 0; i < dim_; i++)
        loopOrder.push_back(i);

      /// create all loopOrders
      do
      {
        if (loopOrder[dim_ - 1] != destOrder[dim_ - 1] && loopOrder[dim_ - 1] != sourceOrder[dim_ - 1])
          /// ATTENTION: we skip all loop-orders where the stride-1 index is not the inner-most loop iff perm[0] == 0 (both for perf & correctness)
          continue;
        loopOrders.push_back(loopOrder);
      } while (std::next_permutation(loopOrder.begin(), loopOrder.end()));

      /// Sort the loopOrders based on loopCostHeuristic results
      for (unsigned k = 0; k < loopOrders.size(); k++)
      {
        for (unsigned i = 0; i < loopOrders.size() - 1; i++)
        {
          if (loopCostHeuristic(loopOrders[i], dim_, sourceOrder, destOrder) > loopCostHeuristic(loopOrders[i + 1], dim_, sourceOrder, destOrder))
          {
            std::swap(loopOrders[i], loopOrders[i + 1]);
          }
        }
      }
    }

    bool hasSameOrder(const std::vector<unsigned> &initial, const std::vector<unsigned> &final)
    {
      assert(initial.size() == final.size());
      assert(initial.size() > 2);
      if (initial == final)
        return true;

      if (initial[0] == final[1] && initial[1] == final[0])
      {
        for (size_t i = 2; i < initial.size(); i++)
        {
          if (initial[i] != final[i])
            return false;
        }
        return true;
      }
      else
        return false;
    }

    std::vector<Operation *>
    getSumLabels(const std::vector<Operation *> &rhs1Labels,
                 const std::vector<Operation *> &rhs2Labels,
                 const std::vector<Operation *> &lhsLabels)
    {
      std::set<Operation *> inLabels(rhs1Labels.begin(), rhs1Labels.end());
      inLabels.insert(rhs2Labels.begin(), rhs2Labels.end());
      std::set<Operation *> outLabels(lhsLabels.begin(), lhsLabels.end());
      std::vector<Operation *> result;

      std::set_difference(inLabels.begin(), inLabels.end(), outLabels.begin(),
                          outLabels.end(), std::back_inserter(result));

      return result;
    }

    void createTensorContraction(Location loc, Value rhs1Tensor,
                                 ArrayRef<Value> rhs1Labels,
                                 Value rhs2Tensor,
                                 ArrayRef<Value> rhs2Labels, Value lhsTensor,
                                 ArrayRef<Value> lhsLabels,
                                 ConversionPatternRewriter &rewriter,
                                 double beta)
    {

      std::map<Operation *, mlir::AffineExpr> allLabels;
      double alpha = 1.0;
      auto rhs1AlphaAttr = rhs1Tensor.getDefiningOp()->getAttr("__alpha__");
      auto rhs2AlphaAttr = rhs2Tensor.getDefiningOp()->getAttr("__alpha__");

      alpha *= rhs1AlphaAttr.cast<FloatAttr>().getValueAsDouble();
      alpha *= rhs2AlphaAttr.cast<FloatAttr>().getValueAsDouble();

      unsigned idx = 0;
      for (auto lbl : rhs1Labels)
      {
        if (allLabels.find(lbl.getDefiningOp()) == allLabels.end())
        {
          allLabels[lbl.getDefiningOp()] =
              getAffineDimExpr(idx++, rewriter.getContext());
        }
      }

      for (auto lbl : rhs2Labels)
      {
        if (allLabels.find(lbl.getDefiningOp()) == allLabels.end())
        {
          allLabels[lbl.getDefiningOp()] =
              getAffineDimExpr(idx++, rewriter.getContext());
        }
      }

      for (auto lbl : lhsLabels)
      {
        if (allLabels.find(lbl.getDefiningOp()) == allLabels.end())
        {
          allLabels[lbl.getDefiningOp()] =
              getAffineDimExpr(idx++, rewriter.getContext());
        }
      }

      std::vector<mlir::AffineExpr> rhs1Exprs;
      std::vector<mlir::AffineExpr> rhs2Exprs;
      std::vector<mlir::AffineExpr> lhsExprs;

      for (const auto &lbl : rhs1Labels)
      {
        rhs1Exprs.push_back(allLabels[lbl.getDefiningOp()]);
      }

      for (const auto &lbl : rhs2Labels)
      {
        rhs2Exprs.push_back(allLabels[lbl.getDefiningOp()]);
      }

      for (const auto &lbl : lhsLabels)
      {
        lhsExprs.push_back(allLabels[lbl.getDefiningOp()]);
      }

      auto context = rewriter.getContext();
      SmallVector<mlir::AffineMap, 8> affineMaps{
          mlir::AffineMap::get(idx, 0, rhs1Exprs, context),
          mlir::AffineMap::get(idx, 0, rhs2Exprs, context),
          mlir::AffineMap::get(idx, 0, lhsExprs, context)};

      auto affineMapArrayAttr = rewriter.getAffineMapArrayAttr(affineMaps);
      comet_debug() << "\n";
      SmallVector<mlir::StringRef, 8> formats;
      std::vector<mlir::Operation *> defops{rhs1Tensor.getDefiningOp(), rhs2Tensor.getDefiningOp(), lhsTensor.getDefiningOp()};
      for (auto defop : defops)
      {
        comet_debug() << " ";
        comet_pdump(defop);
        if (isa<DenseTensorDeclOp>(defop))
        {
          comet_debug() << " is TensorDeclOp\n";

          /// infer the format
          auto lhs_format = dyn_cast<DenseTensorDeclOp>(defop).getFormat();
          comet_debug() << " lhs_format: " << lhs_format << "\n";
          formats.push_back(lhs_format);
        }
        else if (isa<SparseTensorDeclOp>(defop))
        {
          comet_debug() << " is TensorDeclOp\n";

          /// infer the format
          auto lhs_format = dyn_cast<SparseTensorDeclOp>(defop).getFormat();
          comet_debug() << " lhs_format: " << lhs_format << "\n";
          formats.push_back(lhs_format);
        }
        else
        {
          comet_debug() << " not TensorDeclOp\n";
        }
      }

      auto formatAttr = rewriter.getStrArrayAttr(formats);
      comet_debug() << " formatAttr: " << formatAttr << "\n";

      auto SemiringAttr = rewriter.getStringAttr("none");
      auto MaskingAttr = rewriter.getStringAttr("none");
      auto tc = rewriter.create<tensorAlgebra::TensorMultOp>(loc, lhsTensor.getType(),
                                                             rhs1Tensor, rhs2Tensor,
                                                             lhsLabels, affineMapArrayAttr,
                                                             formatAttr, SemiringAttr, MaskingAttr,
                                                             nullptr); /// TODO: masking is an optional operand
      tc.getOperation()->setAttr("__alpha__", rewriter.getF64FloatAttr(alpha));
      tc.getOperation()->setAttr("__beta__", rewriter.getF64FloatAttr(beta));
      comet_debug() << " ";
      comet_vdump(tc);
    }

    std::vector<unsigned> constructPermutationMapAttr(const std::vector<Operation *> &rhs_labels,
                                                      const std::vector<Operation *> &lhs_labels)
    {
      std::vector<unsigned> perm_vec;
      for (size_t i = 0; i < rhs_labels.size(); i++)
      {
        auto it = std::find(lhs_labels.begin(), lhs_labels.end(), rhs_labels[i]);
        perm_vec.push_back(it - lhs_labels.begin());
      }
      return perm_vec;
    }

    std::vector<Value> createInductionVar(std::vector<scf::ForOp> forloops,
                                          std::vector<unsigned int> indexIterateOrder,
                                          std::vector<unsigned int> inputPerm)
    {
      std::vector<Value> inputInductionVars;
      unsigned int perm_loc;

      print_vector<unsigned int>(inputPerm);
      for (unsigned int i = 0; i < inputPerm.size(); i++)
      {
        perm_loc = findIndexInVector(indexIterateOrder, inputPerm[i]);
        comet_debug() << "indexIterateOrder:" << indexIterateOrder[i] << " perm_loc:" << perm_loc << "\n";
        inputInductionVars.push_back(forloops[perm_loc].getInductionVar());
      }
      return inputInductionVars;
    }

    std::vector<Value> createInductionVarAffine(std::vector<AffineForOp> affineloops,
                                                std::vector<int64_t> indexIterateOrder,
                                                SmallVector<ReassociationIndices> inputPerm)
    {
      std::vector<Value> inputInductionVars;
      unsigned int perm_loc;

      for (unsigned int i = 0; i < inputPerm[0].size(); i++)
      {
        perm_loc = findIndexInVector(indexIterateOrder, inputPerm[0][i]);
        comet_debug() << "indexIterateOrder:" << indexIterateOrder[i] << " perm_loc:" << perm_loc << "\n";
        inputInductionVars.push_back(affineloops[perm_loc].getInductionVar());
      }
      return inputInductionVars;
    }

    SmallVector<ReassociationIndices> getReassociationIndices(ArrayRef<AffineMap> maps)
    {
      SmallVector<ReassociationIndices> reassociation;
      for (AffineMap map : maps)
      {
        ReassociationIndices indices;
        for (unsigned i = 0, e = map.getNumResults(); i < e; i++)
        {
          unsigned pos = llvm::cast<AffineDimExpr>(map.getResult(i)).getPosition();
          indices.push_back(pos);
        }
        reassociation.push_back(indices);
      }
      return reassociation;
    }

  } //// namespace tensorAlgebra
} //// namespace mlir

#endif //// UTILS_H_
