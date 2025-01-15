//===- Utils.h - Utilities to support the Tensor Algebra dialect --------*- C++ -*-===//
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

#ifndef TENSORALGEBRA_UTILS_H_
#define TENSORALGEBRA_UTILS_H_

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

#include "mlir/Transforms/DialectConversion.h"

#include <set>
#include <unordered_map>
#include <typeinfo>

/// TODO(gkestor): supports only f64 -  need generalization

using namespace mlir::linalg;

namespace mlir
{
  namespace tensorAlgebra
  {
    enum TargetDevice { CPU, GPU};

    using IndexSizeMap = std::unordered_map<unsigned, int64_t>;
    using IndexVector = std::vector<unsigned>;
    using TensorShape = ArrayRef<int64_t>;

    MemRefType convertTensorToMemRef(TensorType type);
    Value insertAllocAndDealloc(MemRefType memtype, Location loc, PatternRewriter &rewriter);
    Value insertAllocAndDeallocDynamic(MemRefType memtype, std::vector<Value> &dynamic_sizes, Location loc,
                                       PatternRewriter &rewriter);
    Value insertAllocAndInitialize(Location loc, MemRefType memtype, ValueRange allocValueRange, PatternRewriter &rewriter);
    void insertInitialize(Location loc,
                          Value cst_init,
                          Value alloc_op,
                          Value accessIdx,
                          OpBuilder &builder,
                          bool use_dynamic_init,
                          Value dynamic_init);
    bool hasFuncDeclaration(ModuleOp &module, std::string funcName);

    /*
     * We should put template function definition in the header rather than in the cpp file.
     * Reference:
     * 1. Why can’t I separate the definition of my templates class from its declaration and put it inside a .cpp file?
     *    https://isocpp.org/wiki/faq/templates#templates-defn-vs-decl
     * 2. How can I avoid linker errors with my template functions?
     *    https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
     */
    template <class T>
    unsigned int findIndexInVector(std::vector<T> const &vec, T e)
    {
      // Check if element e exists in vector
      auto it = std::find(vec.begin(), vec.end(), e);

      // It accepts a range and an element to search in the given range. If element is found then
      // it returns an iterator to the first element in the given range that’s equal to given element,
      // else it returns an end of the list.
      unsigned int ret = 0;
      if (it != vec.end())
      {
        ret = std::distance(vec.begin(), it);
      }
      else
      {
        ret = vec.size();
      }
      return ret;
    }

    template <typename T>
    void print_vector(std::vector<T> vec)
    {
      for ([[maybe_unused]] auto n : vec)
      {
        // comet_debug() << n << " ";
      }
      // comet_debug() << "\n";
    }

    void print_vector_value(std::vector<Value> vec);

    std::string dump2str(Value t);
    std::vector<llvm::StringRef> stringSplit(llvm::StringRef s, llvm::StringRef delimiter);
    std::vector<unsigned> getReverseIdentityPermutation(size_t size);
    std::vector<unsigned> getIdentityPermutation(size_t size);

    std::vector<std::vector<int64_t>> getAllPerms(ArrayAttr indexMaps);
    std::vector<std::vector<int64_t>> getAllPermsWorkspace(ArrayAttr indexMaps);

    // Tensor algebra dialect
    std::string getTensorFormat(std::vector<std::vector<std::string>> allFormats,
                                unsigned tensor_id);
    std::string getFormat(std::vector<unsigned> allLocs,
                          std::vector<std::vector<unsigned>> allPerms,
                          std::vector<std::vector<std::string>> allFormats);
    std::vector<unsigned> getFreeIndices(std::vector<unsigned> rhs_perm, std::vector<unsigned> lhs_perm);
    std::vector<unsigned> getSumIndices(std::vector<unsigned> rhs_perm, std::vector<unsigned> rhs_perm_free);
    std::vector<unsigned> getIndexIterateOrder(std::vector<unsigned> rhs1_perm, std::vector<unsigned> rhs2_perm);
    std::vector<std::vector<llvm::StringRef>> getAllFormats(ArrayAttr opFormatsArrayAttr, std::vector<std::vector<int64_t>> allPerms);
    bool checkIsElementwise(std::vector<std::vector<int>> allPerms);
    bool checkIsMixedMode(std::vector<std::vector<std::string>> formats);
    bool checkIsDense(std::vector<std::string> format);

    bool isDense(std::string s, std::string delim);
    bool isMergedIndex(std::vector<std::string> format_vec, int cur_idx, int sumIndex);

    std::vector<Value> getFormatsValue(llvm::StringRef formats_str, int rank_size,
                                       PatternRewriter &rewriter, Location loc, IndexType indexType);
    std::vector<Value> getFormatsValueInt(llvm::StringRef formats_str, int rank_size,
                                          PatternRewriter &rewriter, Location loc, IntegerType intType);
    std::vector<TensorFormatEnum> getFormats(llvm::StringRef formats_str, int rank_size, MLIRContext* ctx);

    double loopCostHeuristic(const std::vector<unsigned> &loopOrder, size_t dim_,
                             std::vector<unsigned> &sourceOrder, std::vector<unsigned> &destOrder);

    void getLoopOrders(std::vector<std::vector<unsigned>> &loopOrders, size_t dim_,
                       std::vector<unsigned> &sourceOrder, std::vector<unsigned> &destOrder);

    int64_t labelSize(Operation *op);

    bool hasSameOrder(const std::vector<unsigned> &initial,
                      const std::vector<unsigned> &final);

    std::vector<Operation *>
    getSumLabels(const std::vector<Operation *> &rhs1Labels,
                 const std::vector<Operation *> &rhs2Labels,
                 const std::vector<Operation *> &lhsLabels);

    void createTensorContraction(Location loc, Value rhs1Tensor,
                                 ArrayRef<Value> rhs1Labels,
                                 Value rhs2Tensor,
                                 ArrayRef<Value> rhs2Labels, Value lhsTensor,
                                 ArrayRef<Value> lhsLabels,
                                 ConversionPatternRewriter &rewriter,
                                 double beta = 0.0);

    std::vector<unsigned> constructPermutationMapAttr(const std::vector<Operation *> &rhs_labels,
                                                      const std::vector<Operation *> &lhs_labels);

    std::vector<Value> createInductionVar(std::vector<scf::ForOp> forloops,
                                          std::vector<unsigned int> indexIterateOrder,
                                          std::vector<unsigned int> inputPerm);

    std::vector<Value> createInductionVarAffine(std::vector<affine::AffineForOp> affineloops,
                                                std::vector<int64_t> indexIterateOrder,
                                                SmallVector<ReassociationIndices> inputPerm);

    SmallVector<ReassociationIndices> getReassociationIndices(ArrayRef<AffineMap> maps);

    // For workspace transformations and Index Tree dialect
    std::vector<int> getUnionOf2Dvector(std::vector<std::vector<int>> perms_int);
    unsigned int findIndexInVector_Value(std::vector<Value> vec, Value e);

    /// dfs traverse the tcRootOp,
    /// parent node can get from getUser() function, only one user since tree structure
    void dfsRootOpTree(Value tcRootOp, std::vector<Value> &ret);
    void getAncestorsWp(Value op, std::vector<Value> &ret /* output ancestors*/, std::vector<Value> &dfsOps);

    /// Method 0:
    /// Search for the tensor which contains index i from workspace tree ops: ta.tc_root
    /// Return the tensor name and the index in the tensor
    /// step: find the ancestor of each leaf, check the workspaceOp is in whose ancestorWP
    void findLeafs(Value tcRootOp, std::vector<int> &indices, std::vector<Value> &dfsOps, std::vector<Value> &ret /* output leaves */);
    bool isRealRoot(Operation *op);
    std::vector<std::vector<int>> convertArrayAttrIntTo2DVector(ArrayAttr perms);
    ArrayAttr convert2DVectorToArrayAttrInt(std::vector<std::vector<int>> t1_perms_int, OpBuilder &builder);
    ArrayAttr convert2DVectorToArrayAttrStr(std::vector<std::vector<std::string>> t1_formats_str, OpBuilder &builder);
    std::vector<std::vector<std::string>> convertArrayAttrStrTo2DVector(ArrayAttr formats);
    std::vector<std::vector<bool>> createInputOutputMapping(ArrayAttr perms, bool value);

    void getInputTensorsOfComputeOp(Value computeOp, std::vector<Value> &inputTensors);
    void getOutputTensorsOfComputeOp(Value computeOp, std::vector<Value> &outputTensors);
    void getTensorsOfComputeOp(Value computeOp, std::vector<Value> &tensors);

    void getPermsOfComputeOp(Value computeOp, std::vector<std::vector<int>> &opPerms);
    void getRHSPermsOfComputeOp(Value computeOp, std::vector<std::vector<int>> &opPerms);
    void getLHSPermsOfComputeOp(Value computeOp, std::vector<std::vector<int>> &opPerms);

    void getFormatsOfComputeOp(Value computeOp, std::vector<std::vector<std::string>> &opFormats);
    void getRHSFormatsOfComputeOp(Value computeOp, std::vector<std::vector<std::string>> &opFormats);
    void getLHSFormatsOfComputeOp(Value computeOp, std::vector<std::vector<std::string>> &opFormats);

    void getFormatsPermsOfComputeOp(Value computeOp,
                                    std::vector<std::vector<std::string>> &opFormats,
                                    std::vector<std::vector<int>> &opPerms,
                                    std::vector<std::vector<bool>> &inputOutputMapping);

    void getFormatsInfo(Value cur_op,
                        std::vector<int> &indices,
                        std::vector<Value> &leafs,
                        std::vector<Value> &tensors,
                        std::vector<unsigned int> &ids,
                        std::vector<std::string> &formats);

    void replaceOperands(Operation *itComputeOp, std::vector<Value> newComputeOps);

    // For TTGT transformations
    struct ContractionPlan
    {
      ContractionPlan(IndexVector a_perm, TensorShape a_shape, IndexVector b_perm,
                      TensorShape b_shape, IndexVector c_perm, TensorShape c_shape)
          : a_perm_{a_perm}, b_perm_{b_perm}, c_perm_{c_perm}
      {

        inA_ =
            std::find(a_perm_.begin(), a_perm_.end(), c_perm_[0]) == a_perm_.end()
                ? true
                : false;

        /// get M-N-K indices for gemm
        std::tie(m_indices_, n_indices_, k_indices_) =
            getIndices(a_perm_, b_perm_, c_perm_);

        /// compute size map for each index
        for (size_t i = 0; i < a_perm_.size(); i++)
        {
          size_map_.insert({a_perm_[i], a_shape[i]});
        }
        for (size_t i = 0; i < b_perm_.size(); i++)
        {
          size_map_.insert({b_perm_[i], b_shape[i]});
        }
        for (size_t i = 0; i < c_perm_.size(); i++)
        {
          size_map_.insert({c_perm_[i], c_shape[i]});
        }

        /// compute sizes for M-N-K sizes
        m_size_ = 1;
        for (const auto &idx : m_indices_)
        {
          m_size_ *= size_map_[idx];
        }

        n_size_ = 1;
        for (const auto &idx : n_indices_)
        {
          n_size_ *= size_map_[idx];
        }

        k_size_ = 1;
        for (const auto &idx : k_indices_)
        {
          k_size_ *= size_map_[idx];
        }
      }

      std::tuple<IndexVector, IndexVector, IndexVector>
      getIndices(IndexVector A_perm, IndexVector B_perm, IndexVector C_perm) const
      {
        IndexVector mIndices, nIndices, kIndices;

        std::set<unsigned> A_perm_set(A_perm.begin(), A_perm.end()),
            B_perm_set(B_perm.begin(), B_perm.end()),
            C_perm_set(C_perm.begin(), C_perm.end());

        std::set<unsigned> *A_perm_ptr{&A_perm_set}, *B_perm_ptr{&B_perm_set};

        std::set<unsigned> A_int_C, B_int_C, A_un_B;

        std::set_intersection(A_perm_ptr->begin(), A_perm_ptr->end(),
                              C_perm_set.begin(), C_perm_set.end(),
                              std::inserter(A_int_C, A_int_C.begin()));
        std::set_difference(A_int_C.begin(), A_int_C.end(), B_perm_ptr->begin(),
                            B_perm_ptr->end(), std::back_inserter(mIndices));

        std::set_intersection(B_perm_ptr->begin(), B_perm_ptr->end(),
                              C_perm_set.begin(), C_perm_set.end(),
                              std::inserter(B_int_C, B_int_C.begin()));
        std::set_difference(B_int_C.begin(), B_int_C.end(), A_perm_ptr->begin(),
                            A_perm_ptr->end(), std::back_inserter(nIndices));

        std::set_union(A_perm_ptr->begin(), A_perm_ptr->end(), B_perm_ptr->begin(),
                       B_perm_ptr->end(), std::inserter(A_un_B, A_un_B.begin()));
        std::set_difference(A_un_B.begin(), A_un_B.end(), C_perm_set.begin(),
                            C_perm_set.end(), std::back_inserter(kIndices));

        return std::make_tuple(mIndices, nIndices, kIndices);
      }

      double getTransposeTime(uint64_t mem_size, const IndexVector &perm) const
      {
        double result;
        if (perm[0] != 0)
        {
          // TODO(gkestor): needs to be adjusted according to our transpose method
          result = mem_size / 0.71;
        }
        else
        {
          result = mem_size;
        }
        return result;
      }

      double flopCount() const
      {
        double overall_size = m_size_ * n_size_ * k_size_;
        int op_factor = n_indices_.size() == 0 ? 1 : 2;

        return overall_size * op_factor;
      }

      std::string contractionString(const IndexVector &a_idx,
                                    const IndexVector &b_idx,
                                    const IndexVector &c_idx) const
      {
        std::string result = "contr_C";

        for (const auto &idx : c_idx)
        {
          result += "_" + std::to_string(idx);
        }
        result += "_A";

        for (const auto &idx : a_idx)
        {
          result += "_" + std::to_string(idx);
        }
        result += "_B";

        for (const auto &idx : b_idx)
        {
          result += "_" + std::to_string(idx);
        }

        return result;
      }

      IndexVector getPermutation(const IndexVector &in_idx,
                                 const IndexVector &out_idx) const
      {
        IndexVector result;
        for (const auto &idx : out_idx)
        {
          auto it = std::find(in_idx.begin(), in_idx.end(), idx);
          assert(it != in_idx.end() && "Wrong permutation");
          result.push_back(std::distance(in_idx.begin(), it));
        }
        return result;
      }

      double getTotalTime()
      {
        IndexVector a_perm, b_perm, c_perm;
        double minTime;
        std::tie(a_perm, b_perm, c_perm, minTime) = computeBestPermutations();
        double result = flopCount() + minTime;

        return result;
      }

      std::tuple<IndexVector, IndexVector, IndexVector> computePermutations(bool isbestperm, int whichpermutation)
      {
        IndexVector a_perm, b_perm, c_perm;
        double minTime;

        if (isbestperm)
        {
          std::tie(a_perm, b_perm, c_perm, minTime) = computeBestPermutations();
        }
        else
        {
          /// chose the permutation identified with which permutation_ order
          std::tie(a_perm, b_perm, c_perm) = findPermutationsAtN(whichpermutation);
        }
        return std::make_tuple(a_perm, b_perm, c_perm);
      }

      std::tuple<IndexVector, IndexVector, IndexVector> findPermutationsAtN(int whichperm)
      {
        int curper = 1;
        IndexVector m_idx{m_indices_}, n_idx{n_indices_}, k_idx{k_indices_};
        std::sort(m_idx.begin(), m_idx.end());
        std::sort(n_idx.begin(), n_idx.end());
        std::sort(k_idx.begin(), k_idx.end());

        IndexVector a_candidate, b_candidate, c_candidate;
        for (size_t i = 0; i < 2; i++)
        {
          bool doSwap = (i != 0);
          do
          {
            do
            {
              do
              {
                if (curper == whichperm)
                {
                  IndexVector a_idx, b_idx, c_idx;

                  if (i == 1)
                  {
                    a_idx.insert(a_idx.end(), k_idx.begin(), k_idx.end());
                    a_idx.insert(a_idx.end(), m_idx.begin(), m_idx.end());
                    b_idx.insert(b_idx.end(), n_idx.begin(), n_idx.end());
                    b_idx.insert(b_idx.end(), k_idx.begin(), k_idx.end());
                    c_idx.insert(c_idx.end(), n_idx.begin(), n_idx.end());
                    c_idx.insert(c_idx.end(), m_idx.begin(), m_idx.end());
                  }
                  else
                  {
                    a_idx.insert(a_idx.end(), m_idx.begin(), m_idx.end());
                    a_idx.insert(a_idx.end(), k_idx.begin(), k_idx.end());
                    b_idx.insert(b_idx.end(), k_idx.begin(), k_idx.end());
                    b_idx.insert(b_idx.end(), n_idx.begin(), n_idx.end());
                    c_idx.insert(c_idx.end(), m_idx.begin(), m_idx.end());
                    c_idx.insert(c_idx.end(), n_idx.begin(), n_idx.end());
                  }

                  a_candidate = a_idx;
                  b_candidate = b_idx;
                  c_candidate = c_idx;
                  swapAB_ = doSwap;
                }
                curper++;

              } while (std::next_permutation(k_idx.begin(), k_idx.end()));
            } while (std::next_permutation(n_idx.begin(), n_idx.end()));
          } while (std::next_permutation(m_idx.begin(), m_idx.end()));
        }

        assert(whichperm <= curper && "Cannot find the selected permutation");
        IndexVector best_a_perm, best_b_perm, best_c_perm;
        bestPermStr_ = contractionString(a_candidate, b_candidate, c_candidate);
        best_a_perm = getPermutation(a_perm_, a_candidate);
        best_b_perm = getPermutation(b_perm_, b_candidate);
        best_c_perm = getPermutation(c_perm_, c_candidate);

        return std::make_tuple(best_a_perm, best_b_perm, best_c_perm);
      }

      std::tuple<IndexVector, IndexVector, IndexVector, double> computeBestPermutations()
      {
        IndexVector m_idx{m_indices_}, n_idx{n_indices_}, k_idx{k_indices_};
        std::sort(m_idx.begin(), m_idx.end());
        std::sort(n_idx.begin(), n_idx.end());
        std::sort(k_idx.begin(), k_idx.end());

        IndexVector a_candidate, b_candidate, c_candidate;

        double minTranspose = std::numeric_limits<double>::max();

        do
        {
          do
          {
            do
            {
              for (size_t i = 0; i < 2; i++)
              {
                IndexVector a_idx, b_idx, c_idx;
                uint64_t a_size, b_size, c_size;
                double transposeTime = 0.0;

                if (i == 1)
                {
                  a_idx.insert(a_idx.end(), k_idx.begin(), k_idx.end());
                  a_idx.insert(a_idx.end(), m_idx.begin(), m_idx.end());
                  b_idx.insert(b_idx.end(), n_idx.begin(), n_idx.end());
                  b_idx.insert(b_idx.end(), k_idx.begin(), k_idx.end());
                  c_idx.insert(c_idx.end(), n_idx.begin(), n_idx.end());
                  c_idx.insert(c_idx.end(), m_idx.begin(), m_idx.end());
                }
                else
                {
                  a_idx.insert(a_idx.end(), m_idx.begin(), m_idx.end());
                  a_idx.insert(a_idx.end(), k_idx.begin(), k_idx.end());
                  b_idx.insert(b_idx.end(), k_idx.begin(), k_idx.end());
                  b_idx.insert(b_idx.end(), n_idx.begin(), n_idx.end());
                  c_idx.insert(c_idx.end(), m_idx.begin(), m_idx.end());
                  c_idx.insert(c_idx.end(), n_idx.begin(), n_idx.end());
                }

                a_size = k_size_ * m_size_;
                b_size = n_size_ * k_size_;
                c_size = m_size_ * n_size_;

                if (a_perm_ != a_idx)
                {
                  transposeTime +=
                      getTransposeTime(a_size, getPermutation(a_perm_, a_idx));
                }

                if (b_perm_ != b_idx)
                {
                  transposeTime +=
                      /// b_size is the memory size of B, the current heuristics is only based on memsize
                      getTransposeTime(b_size, getPermutation(b_perm_, b_idx));
                }

                if (c_perm_ != c_idx)
                {
                  transposeTime +=
                      getTransposeTime(c_size, getPermutation(c_perm_, c_idx));
                }

                if (transposeTime < minTranspose)
                {
                  a_candidate = a_idx;
                  b_candidate = b_idx;
                  c_candidate = c_idx;
                  minTranspose = transposeTime;
                  swapAB_ = (i == 1) ? true : false;
                }
              }

            } while (std::next_permutation(k_idx.begin(), k_idx.end()));

          } while (std::next_permutation(n_idx.begin(), n_idx.end()));

        } while (std::next_permutation(m_idx.begin(), m_idx.end()));

        IndexVector best_a_perm, best_b_perm, best_c_perm;

        bestPermStr_ = contractionString(a_candidate, b_candidate, c_candidate);
        best_a_perm = getPermutation(a_perm_, a_candidate);
        best_b_perm = getPermutation(b_perm_, b_candidate);
        best_c_perm = getPermutation(c_perm_, c_candidate);

        return std::make_tuple(best_a_perm, best_b_perm, best_c_perm, minTranspose);
      }

      IndexVector a_perm_;
      IndexVector b_perm_;
      IndexVector c_perm_;

      IndexVector m_indices_;
      IndexVector n_indices_;
      IndexVector k_indices_;

      int64_t m_size_;
      int64_t n_size_;
      int64_t k_size_;

      IndexSizeMap size_map_;

      bool swapAB_;
      bool inA_;

      std::string bestPermStr_;
    }; /// struct ContractionPlan

  } /// namespace tensorAlgebra
} /// namespace mlir

#endif /// TENSORALGEBRA_UTILS_H_
