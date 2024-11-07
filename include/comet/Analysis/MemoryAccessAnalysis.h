//===- MemoryAccessPatternAnalysis.h  ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memory access pattern analysis
//
//===----------------------------------------------------------------------===//

#ifndef COMET_MEMORY_ACCESS_PATTERN_ANALYSIS_H
#define COMET_MEMORY_ACCESS_PATTERN_ANALYSIS_H

#include <string>

namespace mlir
{
  class Pass;
  class RewritePatternSet;
  namespace comet
  {

#define GEN_PASS_DECL_MEMORYANALYSIS
#include "comet/Analysis/Passes.h.inc"

    /// pass performs memory access frequency analysis
    std::unique_ptr<Pass> createMemoryAccessFrequencyAnalysisPass();

    /// pass performs memory access pattern analysis.
    std::unique_ptr<Pass> createMemoryAccessPatternAnalysisPass();


    /// For example, time complexity O(L*N*M + I*J), `L*N*M` is a term, so is `I*J`. And `L`, `N`, etc. are factors.
    /// A factor could be
    /// 1. const, e.g., O(1)
    /// 2. var, e.g., O(%alloc[%c4])
    /// 3. var - var, e.g., O(%alloc_9[%14]-%alloc_9[%arg0])
    /// 4. var - const, e.g., O(%alloc[%c4] - 4) (this is very rare)
    /// -------------------- ///
    /// class TimeFactor
    /// -------------------- ///
    class TimeFactor
    {
    public:
      enum Kind
      {
        Variable = 0,
        Constant
      };

      TimeFactor() = default;
      TimeFactor(std::string name) : name_(name) {}
      TimeFactor(std::string name, Kind kind) : name_(name), kind_(kind) {}
      TimeFactor(std::string name, Kind kind, float priority) : name_(name), kind_(kind), priority_(priority) {}

      const std::string &getName() const
      {
        return name_;
      }

      void setName(const std::string &name)
      {
        name_ = name;
      }

      Kind getKind() const
      {
        return kind_;
      }

      void setKind(Kind kind)
      {
        kind_ = kind;
      }

      float getPriority() const
      {
        return priority_;
      }

      void setPriority(float priority)
      {
        priority_ = priority;
      }

      bool operator==(const TimeFactor &rhs) const
      {
        return name_ == rhs.name_ &&
               kind_ == rhs.kind_ &&
               priority_ == rhs.priority_;
      }

      bool operator!=(const TimeFactor &rhs) const
      {
        return !(rhs == *this);
      }

      bool operator<(const TimeFactor &rhs) const;

      bool operator>(const TimeFactor &rhs) const
      {
        return rhs < *this;
      }

      bool operator<=(const TimeFactor &rhs) const
      {
        return !(rhs < *this);
      }

      bool operator>=(const TimeFactor &rhs) const
      {
        return !(*this < rhs);
      }

      bool isConstant() const
      {
        return kind_ == Constant;
      }

      bool isVariable() const
      {
        return kind_ == Variable;
      }

      std::string toString() const
      {
        return name_;
      }

      void dump() const;

    private:
      std::string name_;
      Kind kind_ = Constant;
      float priority_ = 0.0;  /// Used for sorting. The larger priority is considered bigger the TimeFactor.

    };  /// End class TimeFactor

    /// -------------------- ///
    /// class TimeTerm
    /// -------------------- ///
    class TimeTerm
    {
    public:
      const std::vector<TimeFactor> &getFactors() const
      {
        return factors_;
      }

      const comet::TimeFactor getFactor(uint64_t index) const;

      void setFactors(const std::vector<TimeFactor> &factors)
      {
        factors_ = factors;
      }

      void setFactor(uint64_t index, const TimeFactor &factor);

      bool operator==(const TimeTerm &rhs) const
      {
        return factors_ == rhs.factors_ &&
               hasConstant_ == rhs.hasConstant_;
      }

      bool operator!=(const TimeTerm &rhs) const
      {
        return !(rhs == *this);
      }

      bool operator<(const TimeTerm &rhs) const;

      bool operator>(const TimeTerm &rhs) const
      {
        return rhs < *this;
      }

      bool operator<=(const TimeTerm &rhs) const
      {
        return !(rhs < *this);
      }

      bool operator>=(const TimeTerm &rhs) const
      {
        return !(*this < rhs);
      }

      /// Add factor to the term.
      void addFactor(const TimeFactor &factor);

      /// Check if this term only contains one single constant
      bool isSingleConstant() const;

      std::string toString() const;

      void dump() const;

    private:
      std::vector<TimeFactor> factors_;
      bool hasConstant_ = false;
    };  /// End class TimeTerm

    /// -------------------- ///
    /// class TimeComplexity
    /// -------------------- ///
    class TimeComplexity
    {
    public:
      bool operator==(const TimeComplexity &rhs) const
      {
        return terms_ == rhs.terms_;
      }

      bool operator!=(const TimeComplexity &rhs) const
      {
        return !(rhs == *this);
      }

      bool operator<(const TimeComplexity &rhs) const;

      bool operator>(const TimeComplexity &rhs) const
      {
        return rhs < *this;
      }

      bool operator<=(const TimeComplexity &rhs) const
      {
        return !(rhs < *this);
      }

      bool operator>=(const TimeComplexity &rhs) const
      {
        return !(*this < rhs);
      }

//      void addTerm(const std::string &term);

      void addTerm(const TimeTerm &term);


      std::string toString() const;

      void dump() const;

    private:
      std::vector<TimeTerm> terms_;
      uint64_t locMaxTerm_ = (uint64_t) -1;  /// pointer to the largest term
    };  /// End class TimeComplexity
  } // namespace comet
} // namespace mlir

#endif // COMET_MEMORY_ACCESS_PATTERN_ANALYSIS_H
