#include "comet/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <unordered_map>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::affine;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
//#undef COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
// *********** For debug purpose *********//

namespace
{
  struct MemoryAccessFreqeuncyAnalysisPass
      : PassWrapper<MemoryAccessFreqeuncyAnalysisPass, OperationPass<func::FuncOp>>
  {

    // A map to store the frequency of accesses for each memory location.
//    llvm::DenseMap<Value, int> accessFrequencyMap;

    llvm::DenseMap<Value, std::string> object_name_map_;
    llvm::DenseMap<Value, comet::TimeComplexity> read_count_map_;
    llvm::DenseMap<Value, comet::TimeComplexity> write_count_map_;

    // StringRef getArgument() const final
    // {
    //   return "memory-access-pattern-analysis";
    // }
    // StringRef getDescription() const final
    // {
    //   return "Test alias analysis extending.";
    // }

//    void incrementAccessFrequency(Value memref, int64_t count);
//    int64_t calculateConditionFactor(Operation *op, int64_t iterations);
//    int64_t estimateAccessMultiplierFromAffineMap(AffineMap affineMap);
//    void estimateAccessFrequencyWithAffineMap(Value memref, Operation *op, AffineMap affineMap, int64_t iterations);
//    void estimateAccessFrequency(Value memref, Operation *op, int64_t iterations);

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryAccessFreqeuncyAnalysisPass)
    void runOnOperation() override;
  };

  struct MemoryAccessPatternAnalysisPass
      : PassWrapper<MemoryAccessPatternAnalysisPass, OperationPass<func::FuncOp>>
  {

    // bool isSequentialAffineExpr(AffineExpr expr);
    // bool isRandomAccess(AffineMap affineMap);

    bool isSimpleAffineExpr(AffineExpr expr);
    bool isSequentialAccess(AffineMap affineMap);

    void analyzeAccessPattern(Value memref, AffineMap affineMap, OperandRange indices, Operation *op);

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryAccessPatternAnalysisPass)
    void runOnOperation() override;
  };

} // end anonymous namespace

///===----------------------------------------------------------------------===//
/// Memory Access Frequency Analysis
///===----------------------------------------------------------------------===//
namespace mlir
{
  namespace comet
  {
    /// -------------------- ///
    /// class TimeFactor
    /// -------------------- ///
    bool TimeFactor::operator<(const TimeFactor &rhs) const
    {
      /// Compare priority at first
      if (priority_ != rhs.priority_)
      {
        return priority_ < rhs.priority_;
      }
      if (kind_ == Constant && rhs.kind_ == Constant)
      {
        /// If both constant, compare their values
        return std::stoull(name_) < std::stoull(rhs.name_);
      }
      /// Assume Constant should be less than Variable, so that O(1) < O(n).
      if (kind_ == Constant && rhs.kind_ == Variable)
      {
        return true;
      }
      else if (kind_ == Variable && rhs.kind_ == Constant)
      {
        return false;
      }
      /// If both Variable, and equal priority, then don't know
      return false;
    }

    void TimeFactor::dump() const
    {
      llvm::errs() << toString() << "\n";
    }

    bool TimeFactor::isConstantFloat() const
    {

      if (this->isConstant() && name_.find('.') != std::string::npos)
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    void TimeFactor::addConstant(const mlir::comet::TimeFactor &factor)
    {
      assert(kind_ == Constant && "Error: this kind is not constant.");
      assert(factor.getKind() == Constant && "Error: the factor is not constant.");

      if (this->isConstantFloat() || factor.isConstantFloat())
      {
        double old_value = std::stod(name_);
        double new_value = old_value + std::stod(factor.getName());
        name_ = std::to_string(new_value);
      }
      else
      {
        auto old_value = std::stoull(name_);
        auto new_value = old_value + std::stoull(factor.getName());
        name_ = std::to_string(new_value);
      }
    }

    void TimeFactor::multiplyConstant(const mlir::comet::TimeFactor &factor)
    {
      assert(kind_ == Constant && "Error: this kind is not constant.");
      assert(factor.getKind() == Constant && "Error: the factor is not constant.");

      if (this->isConstantFloat() || factor.isConstantFloat())
      {
        double old_value = std::stod(name_);
        double new_value = old_value * std::stod(factor.getName());
        name_ = std::to_string(new_value);
      }
      else
      {
        auto old_value = std::stoull(name_);
        auto new_value = old_value * std::stoull(factor.getName());
        name_ = std::to_string(new_value);
      }
    }
    /// End class TimeFactor

    /// -------------------- ///
    /// class TimeTerm
    /// -------------------- ///

    const TimeFactor &TimeTerm::getFactor(uint64_t index) const
    {
      assert(index < factors_.size() && "Error: index >= factors_.size()");
      return factors_[index];
    }

    void TimeTerm::setFactor(uint64_t index, const TimeFactor &factor)
    {
      assert(index < factors_.size() && "Error: index >= factors_.size()");
      factors_[index] = factor;

      /// Update hasConstant_
      if (hasConstant_)
      {
        if (0 == index && !factor.isConstant())
        {
          hasConstant_ = false;
        }
      }
      else if (factor.isConstant())
      {
        hasConstant_ = true;
        if (index != 0)
        {
          std::swap(factors_.front(), factors_[index]);
        }
      }
    }

    /// Add factor to the term. The Constant factor, if existed, should always be the first factor in the term.
    void TimeTerm::addFactor(const TimeFactor &factor)
    {
      if (factor.isConstant())
      {
        /// If factor is a Constant, then try to update the existing constant factor.
        if (hasConstant_)
        {
          /// Update the existing constant factor
          TimeFactor &c = factors_.front();
          c.multiplyConstant(factor);
        }
        else
        {
          /// This factor is the first constant factor, and put it on the front.
          factors_.insert(factors_.begin(), factor);
          hasConstant_ = true;
        }
      }
      else
      {
        /// The factor is not constant.
        factors_.push_back(factor);
      }
    }

    bool TimeTerm::operator<(const TimeTerm &rhs) const
    {
      if (factors_.size() != rhs.factors_.size())
      {
        return factors_.size() < rhs.factors_.size();
      }

      if (factors_.size() == 1 && rhs.factors_.size() == 1)
      {
        return factors_.front() < rhs.factors_.front();
      }

      return factors_ < rhs.factors_;
    }

    /// If has only one single constant factor, return true, otherwise false
    bool TimeTerm::isSingleConstant() const
    {
      if (factors_.size() == 1 && factors_.front().isConstant())
      {
        return true;
      }
      return false;
    }

    std::string TimeTerm::toString() const
    {
      std::string text;
      for (auto iter = factors_.begin(); iter != factors_.end(); ++iter)
      {
        if (factors_.begin() == iter)
        {
          text += iter->toString();
        }
        else
        {
          text += "*" + iter->toString();
        }
      }

      return text;
    }

    void TimeTerm::dump() const
    {
      llvm::errs() << toString() << "\n";
    }
    /// End class TimeTerm

    /// -------------------- ///
    /// class TimeComplexity
    /// -------------------- ///
    bool TimeComplexity::operator<(const TimeComplexity &rhs) const
    {
      if ((uint64_t) -1 != locMaxTerm_ && (uint64_t) -1 != rhs.locMaxTerm_)
      /// If lhs and rhs are not empty
      {
        return terms_[locMaxTerm_] < rhs.terms_[rhs.locMaxTerm_];
      }
      else if ((uint64_t) -1 == rhs.locMaxTerm_)
      /// If rhs is empty, then no matter lhs is empty or not, return false.
      {
        return false;
      }
      else
      /// If rhs is not empty and lhs is empty, return true.
      {
        return true;
      }
    }

    /// Add term to the time complexity. The constant term, if existed, should always be the first term.
    void TimeComplexity::addTerm(const TimeTerm &term)
    {
      if (term.isSingleConstant())
      {
        if (hasConstant_)
        {
          /// Update the existing constant term
          TimeFactor factor = terms_.front().getFactor(0);
          factor.addConstant(term.getFactor(0));
          terms_.front().setFactor(0, factor);
        }
        else
        {
          /// This term is the first constant term, and put it on the front.
          hasConstant_ = true;
          terms_.insert(terms_.begin(), term);
        }
      }
      else
      {
        terms_.push_back(term);
      }
      /// locMaxTerm_ pointer to the largest term.
      if (terms_.size() == 1)
      {
        locMaxTerm_ = 0;
      }
      else
      {
        if (term > terms_[locMaxTerm_])
        {
          locMaxTerm_ = terms_.size() - 1;
        }
      }

//      //////////////////
//      /// If current time complexity only contains a single constant and the new term is also a single constant,
//      /// then update the constant factor.
//      if (terms_.size() == 1 && terms_.front().isSingleConstant() && term.isSingleConstant())
//      {
//        TimeFactor new_factor = terms_.front().getFactor(0);
//        auto old_value = std::stoull(new_factor.getName());
//        auto new_value = old_value + std::stoull(term.getFactors().front().getName());
//        new_factor.setName(std::to_string(new_value));
//        terms_.front().setFactor(0, new_factor);
//        return;
//      }
//      terms_.push_back(term);
//      /// locMaxTerm_ pointer to the largest term.
//      if (terms_.size() == 1)
//      {
//        locMaxTerm_ = 0;
//      }
//      else
//      {
//        if (term > terms_[locMaxTerm_])
//        {
//          locMaxTerm_ = terms_.size() - 1;
//        }
//      }
    }

    std::string TimeComplexity::toString() const
    {
      std::string text = "O(";
      for (auto iter = terms_.begin(); iter != terms_.end(); ++iter)
      {
        if (iter == terms_.begin())
        {
          text += iter->toString();
        }
        else
        {
          text += " + " + iter->toString();
        }
      }
      text += ")";

      return text;
    }

    void TimeComplexity::dump() const
    {
      llvm::errs() << toString() << "\n";
    }

    /// End class TimeComplexity
  }
}

namespace
{
  /// Check if an op is within any for-loops
  /// TODO: how about while-loop?
  bool isOpInForLoop(Operation *op)
  {
    if(op->getParentOfType<scf::ForOp>() ||
       op->getParentOfType<scf::ParallelOp>() ||
       op->getParentOfType<omp::WsLoopOp>())
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  /// For example, for `%alloc_9 = memref.alloc(%4) : memref<?xindex>`, return `%alloc_9`.
  std::string getValueName(mlir::Value value)
  {
    std::string buffer;
    llvm::raw_string_ostream rso(buffer);
    value.print(rso);
    comet_debug() << "buffer: " << buffer << "\n";
    auto i_bound = buffer.find(' ');
    if (i_bound != std::string::npos)
    {
      std::string name = buffer.substr(0, i_bound);
      comet_debug() << "name: " << name << "\n";
      return name;
    }
    else
    {
      llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: Value is not in supported format.\n";
      return "";
    }
  }

  /// Get a for-loop's iterator name.
  /// For example, for `scf.parallel (%arg0) = (%c0) to (%12) step (%c1) {}`, return `%arg0`.
  std::string getForLoopIteratorName(mlir::Operation *op)
  {
    std::string buffer;
    llvm::raw_string_ostream rso(buffer);
    op->print(rso);
    comet_debug() << "buffer: " << buffer << "\n";
    if (mlir::dyn_cast<omp::WsLoopOp>(op))
    /// `omp.wsloop for  (%arg0) : index = (%c0) to (%22) step (%c1)`
    {
      auto i_start = buffer.find('%');
      if (i_start == std::string::npos)
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: Op is not in supported format.\n";
        return "";
      }
      auto i_bound = buffer.find(')', /*pos=*/i_start + 1);
      if (i_bound == std::string::npos)
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: Op is not in supported format.\n";
        return "";
      }
      std::string arg = buffer.substr(i_start, i_bound - i_start);
      comet_debug() << "arg: " << arg << "\n";
      return arg;
    }
    else
    /// `scf.for %arg1 = %26 to %27 step %c1 {}`
    /// `scf.parallel (%arg0) = (%c0) to (%12) step (%c1) {}`
    {
      auto i_start = buffer.find('%');
      if (i_start == std::string::npos)
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: Op is not in supported format.\n";
        return "";
      }
      auto i_bound = buffer.find(' ', /*pos=*/i_start + 1);
      if (i_bound == std::string::npos)
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: Op is not in supported format.\n";
        return "";
      }
      if (buffer[i_bound - 1] == ')')
      {
        --i_bound;
      }
      std::string arg = buffer.substr(i_start, i_bound - i_start);
      comet_debug() << "arg: " << arg << "\n";
      return arg;
    }
  }


  /// Convert a boundary (lowerBound or upperBound) to a string
  std::string getBoundaryStr(Value &value, bool &is_number /*output*/)
  {
    comet_vdump(value);
    is_number = false;
    if (mlir::isa<BlockArgument>(value))
    /// Boundary is a outer-level loop's iterator variable
    {
      /// For example,
      /// scf.for %arg0 = %c0_3 to %c2000_4 step %c1_5 {
      ///   %c0_6 = arith.constant 0 : index
      ///   %c1_7 = arith.constant 1 : index
      ///   scf.for %arg1 = %c0_6 to %arg0 step %c1_7 { ... }}
      /// Here %arg0 is a boundary of the inner for-loop, and also the iterator variable of the outer for-loop.
      /// Then dyn_cast<...>(value.getDefiningOp()) would crash.
      auto parentOp = value.getParentBlock()->getParentOp();
      std::string boundary_str = getForLoopIteratorName(parentOp);
      comet_debug() << boundary_str << "\n";
      return boundary_str;
    }
    else if (auto cio = mlir::dyn_cast<arith::ConstantIndexOp>(value.getDefiningOp()))
    /// Boundary is a constant
    {
      comet_vdump(cio);
      /// Get ConstantIndexOp's constant value.
      mlir::Attribute valueAttr = cio->getAttr("value");
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(valueAttr))
      {
        auto intVal = intAttr.getInt();
        comet_debug() << "intVal: " << intVal << "\n";
        is_number = true;
        return std::to_string(intVal);
      }
      else
      {
        llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: expected an integer attribute for ConstantIndexOp.\n";
        return "";
      }
    }
    else if (auto mlo = mlir::dyn_cast<memref::LoadOp>(value.getDefiningOp()))
    /// Boundary is a memref.load from a memref.alloc.
    {
      /// For example,
      /// %16 = memref.load %alloc_9[%14] : memref<?xindex>
      /// scf.for %arg1 = %15 to %16 step %c1 {
      /// Here mlo is %16
      comet_vdump(mlo);
      Value memrefOperand = mlo.getMemRef();
      comet_vdump(memrefOperand);
      /// Get the memref part
      std::string memref_str = getValueName(memrefOperand);
      /// Get the index part. May contains multiple indices
      std::vector<std::string> indices_str_list;
      for (auto index : mlo.getIndices())
      {
        /// If index is a for-loop iterator variable, for example
        /// scf.parallel (%arg0) = (%c0) to (%12) step (%c1) {
        ///   %15 = memref.load %alloc_9[%arg0] : memref<?xindex>
        ///   scf.for %arg1 = %15 to %16 step %c1 {
        /// Here boundary %15's index %arg0 is a for-loop iterator variable.
        if (mlir::isa<BlockArgument>(index))
        {
          /// Notes: surprisingly, `index.getDefiningOp()` would crash here.
          comet_vdump(index);
          comet_pdump(index.getParentBlock()->getParentOp());
          auto parentOp = index.getParentBlock()->getParentOp();
          std::string ind_str = getForLoopIteratorName(parentOp);
          indices_str_list.push_back(ind_str);
        }
        else
        /// if index is a ConstantIndexOp or register values. For example,
        /// Index (%c0) is ConstantIndexOp:
        /// %c0 = arith.constant 0 : index
        /// %12 = memref.load %alloc_1[%c0] : memref<?xindex>
        /// Index (%14) is register values:
        /// %14 = arith.addi %arg0, %c1 : index
        /// %16 = memref.load %alloc_9[%14] : memref<?xindex>
        {
          std::string value_name = getValueName(index);
          indices_str_list.push_back(value_name);
        }
      }

      /// Format the boundary string as for example `%alloc_19[%17, %arg2]`
      std::string boundary_str = memref_str + "[";
      for (auto iter = indices_str_list.begin(); iter != indices_str_list.end(); ++iter)
      {
        if (iter == indices_str_list.begin())
        {
          boundary_str += *iter;
        }
        else
        {
          boundary_str += ", " + *iter;
        }
      }
      boundary_str += "]";
      return boundary_str;
    }
    else
    {
//      llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: Value `";
//      value.print(llvm::errs());
//      llvm::errs() << "` type is not supported yet.\n";
      std::string boundary_str = getValueName(value);
      comet_debug() << "boundary_str: " << boundary_str << "\n";
      return boundary_str;
    }
  }

  /// Get a for-loop's range as (upperBound - lowerBound)
  comet::TimeFactor getForLoopRangeAsFactor(mlir::Operation *forLoop)
  {
    auto getLowerBoundHelper =
      [&](mlir::Operation *op) -> mlir::Value
      {
        if (auto forOp = mlir::dyn_cast<scf::ForOp>(op))
        {
          return forOp.getLowerBound();
        }
        else if (auto parallelOp = mlir::dyn_cast<scf::ParallelOp>(op))
        {
          return parallelOp.getLowerBound()[0];
        }
        else if (auto wsLoopOp = mlir::dyn_cast<omp::WsLoopOp>(op))
        {
          return wsLoopOp.getLowerBound()[0];
        }
        else
        {
          llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: The for-loop type is not supported.\n";
          return nullptr;
        }
      };
    auto getUpperBoundHelper =
      [&](mlir::Operation *op) -> mlir::Value
      {
        if (auto forOp = mlir::dyn_cast<scf::ForOp>(op))
        {
          return forOp.getUpperBound();
        }
        else if (auto parallelOp = mlir::dyn_cast<scf::ParallelOp>(op))
        {
          return parallelOp.getUpperBound()[0];
        }
        else if (auto wsLoopOp = mlir::dyn_cast<omp::WsLoopOp>(op))
        {
          return wsLoopOp.getUpperBound()[0];
        }
        else
        {
          llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: The for-loop type is not supported.\n";
          return nullptr;
        }
      };

    Value lowerBound = getLowerBoundHelper(forLoop);
    bool lower_is_number = false;
    std::string lowerBoundStr = getBoundaryStr(lowerBound, lower_is_number/*output*/);
    Value upperBound = getUpperBoundHelper(forLoop);
    bool upper_is_number = false;
    std::string upperBoundStr = getBoundaryStr(upperBound, upper_is_number/*output*/);
//    std::string term;
    comet::TimeFactor factor;
    if (lower_is_number && upper_is_number)
    {
      auto lower_int = std::strtoll(lowerBoundStr.c_str(), nullptr, 0);
      auto upper_int = std::strtoll(upperBoundStr.c_str(), nullptr, 0);
      auto range = upper_int - lower_int;
      /// TODO: what if
      /// 1. the step is negative then the upper_int < lower_int.
      /// 2. range is 0.
      /// 3. upper bound or lower bound are not integer.
      /// 4. step is not 1.
      factor.setName(std::to_string(range));
    }
    else if (lower_is_number && lowerBoundStr == "0")
    {
      factor.setName(upperBoundStr);
      factor.setKind(comet::TimeFactor::Variable);
    }
    else
    {
      factor.setName("(" + upperBoundStr + "-" + lowerBoundStr + ")");
      factor.setKind(comet::TimeFactor::Variable);
    }

    return factor;
  }

  /// Find the inner most for-loop operation
  void findInnerMostForLoop(mlir::Operation *op,
                           scf::ForOp &forOp /*output*/,
                           scf::ParallelOp &parallelOp /*output*/,
                           omp::WsLoopOp &wsLoopOp /*output*/)
  {
    mlir::DominanceInfo domInfo(op);
    mlir::Operation *curr = op;
    forOp = mlir::dyn_cast<scf::ForOp>(curr->getParentOp());
    parallelOp = mlir::dyn_cast<scf::ParallelOp>(curr->getParentOp());
    wsLoopOp = mlir::dyn_cast<omp::WsLoopOp>(curr->getParentOp());
    curr = curr->getParentOp();

    while (curr && (!forOp && !parallelOp && !wsLoopOp))
    {
      forOp = mlir::dyn_cast<scf::ForOp>(curr->getParentOp());
      parallelOp = mlir::dyn_cast<scf::ParallelOp>(curr->getParentOp());
      wsLoopOp = mlir::dyn_cast<omp::WsLoopOp>(curr->getParentOp());
      curr = curr->getParentOp();
    }

    comet_debug() << "forOp: " << (void *) forOp << "\n";
    comet_debug() << "parallelOp: " << (void *) parallelOp << "\n";
    comet_debug() << "wsLoopOp: " << (void *) wsLoopOp << "\n";
  }


  /// If op is within for-loops, get its time complexity
  comet::TimeTerm getForLoopRangesAsTerm(mlir::Operation *op)
  {
//    std::vector<std::string> factors;  /// Each level has one factor
    comet::TimeTerm term;
    scf::ForOp forOp;
    scf::ParallelOp parallelOp;
    omp::WsLoopOp wsLoopOp;
    findInnerMostForLoop(op, forOp /*output*/, parallelOp /*output*/, wsLoopOp /*output*/);
//    scf::ForOp forOp = op->getParentOfType<scf::ForOp>();
//    scf::ParallelOp parallelOp = op->getParentOfType<scf::ParallelOp>();
//    omp::WsLoopOp wsLoopOp = op->getParentOfType<omp::WsLoopOp>();
    auto getUpperLevelForLoop =
      [&](mlir::Operation *currOp) -> bool
      {
        if ((forOp = currOp->getParentOfType<scf::ForOp>()))
        {
          parallelOp = nullptr;
          wsLoopOp = nullptr;
        }
        else if ((parallelOp = currOp->getParentOfType<scf::ParallelOp>()))
        {
          forOp = nullptr;
          wsLoopOp = nullptr;
        }
        else if ((wsLoopOp = currOp->getParentOfType<omp::WsLoopOp>()))
        {
          forOp = nullptr;
          parallelOp = nullptr;
        }
        else
        /// No more upper-level for-loop
        {
          return false;
        }
        return true;
      };
    while (forOp || parallelOp || wsLoopOp)
    {
      if (forOp)
      {
        comet_vdump(forOp);
        comet::TimeFactor factor = getForLoopRangeAsFactor(forOp);
        term.addFactor(factor);
        if (!getUpperLevelForLoop(forOp))
        {
          break;
        }
      }
      else if (parallelOp)
      {  /// parallel Op
        comet_vdump(parallelOp);
        comet::TimeFactor factor = getForLoopRangeAsFactor(parallelOp);
        term.addFactor(factor);
        if (!getUpperLevelForLoop(parallelOp))
        {
          break;
        }
      }
      else if (wsLoopOp)
      {
        comet_vdump(wsLoopOp);
        comet::TimeFactor factor = getForLoopRangeAsFactor(wsLoopOp);
        term.addFactor(factor);
        if (!getUpperLevelForLoop(wsLoopOp))
        {
          break;
        }
      }
    }

    return term;
  }


  bool isReadOp(mlir::Operation *op)
  {
    if (mlir::isa<memref::LoadOp>(op))
    {
      return true;
    }

    return false;
  }


  bool isWriteOp(mlir::Operation *op)
  {
    if (mlir::isa<memref::StoreOp>(op) ||
        mlir::isa<linalg::FillOp>(op))
    {
      return true;
    }

    return false;
  }


  bool isIrrelevantOp(mlir::Operation *op)
  {
    if (mlir::isa<memref::DeallocOp>(op) ||
        mlir::isa<memref::CastOp>(op))
    {
      return true;
    }

    return false;
  }


  /// Analyze a memory object in terms of read and write frequency that is represented as time complexity
  void analyzeMemrefAllocOp(memref::AllocOp &memAllocOp,
                              comet::TimeComplexity &read_TC /*output*/,
                              comet::TimeComplexity &write_TC /*output*/)
  {
    comet_vdump(memAllocOp);
    comet_debug() << "\n";
    for (mlir::Operation *user : memAllocOp->getUsers())
    {
      comet_pdump(user);
//      std::string term = "1";
      comet::TimeTerm term;
      if (isOpInForLoop(user))
      {
        comet_debug() << "in a loop\n";
        term = getForLoopRangesAsTerm(user);
        comet_vdump(term);
      }
      else
      {
        term.addFactor(comet::TimeFactor("1"));
      }

      if (isReadOp(user))
      {
        read_TC.addTerm(term);
      }
      else if (isWriteOp(user))
      {
        write_TC.addTerm(term);
      }
      else
      {
        if (!isIrrelevantOp(user))
        {
          llvm::errs() << __FILE__ << ":" << __LINE__ << " Error: Operation `";
          user->print(llvm::errs());
          llvm::errs() << "` is not read either write.\n";
          comet_pdump(user);
        }
      }
    }

    comet_vdump(read_TC);
    comet_vdump(write_TC);
  }
} /// End anonymous namespace
//void MemoryAccessFreqeuncyAnalysisPass::incrementAccessFrequency(Value memref, int64_t count)
//{
//  comet_debug() << "before adding incrementAccessFrequency:" << accessFrequencyMap[memref] << "\n";
//  accessFrequencyMap[memref] += count;
//  comet_debug() << "input incrementAccessFrequency:" << count << "\n";
//  comet_debug() << "after adding incrementAccessFrequency:" << accessFrequencyMap[memref] << "\n";
//}

//int64_t MemoryAccessFreqeuncyAnalysisPass::calculateConditionFactor(Operation *op, int64_t iterations)
//{
//  int64_t conditionFactor = 1;
//
//  if (auto ifOp = op->getParentOfType<AffineIfOp>())
//  {
//    conditionFactor = iterations / 2;
//  }
//  else
//  {
//    conditionFactor = iterations;
//  }
//
//  return conditionFactor;
//}

//// Estimate the effect of the affine map on access frequency.
//int64_t MemoryAccessFreqeuncyAnalysisPass::estimateAccessMultiplierFromAffineMap(AffineMap affineMap)
//{
//  // TODO(gkestor): need better heuristics
//  // Simple heuristic: This is a heuristic that assumes each result of the affine map corresponds to a distinct access.
//  // The actual effect might be more complex, especially for multi-dimensional accesses, but this provides a basic approach.
//  return affineMap.getNumResults();
//}

//// Estimate access frequency considering affine maps.
//void MemoryAccessFreqeuncyAnalysisPass::estimateAccessFrequencyWithAffineMap(Value memref, Operation *op, AffineMap affineMap, int64_t iterations)
//{
//  int64_t conditionFactor = calculateConditionFactor(op, iterations);
//
//  // Consider the complexity of the affine map.
//  int64_t accessMultiplier = 1;
//  accessMultiplier = estimateAccessMultiplierFromAffineMap(affineMap);
//  comet_vdump(memref);
//  comet_debug() << "incrementAccessFrequency before calling:" << conditionFactor * accessMultiplier << " \n";
//  incrementAccessFrequency(memref, conditionFactor * accessMultiplier);
//}

//// Estimate the access frequency for the given memory location considering conditions.
//void MemoryAccessFreqeuncyAnalysisPass::estimateAccessFrequency(Value memref, Operation *op, int64_t iterations)
//{
//  int64_t conditionFactor = calculateConditionFactor(op, iterations);
//  incrementAccessFrequency(memref, conditionFactor);
//}

void MemoryAccessFreqeuncyAnalysisPass::runOnOperation()
{
//  accessFrequencyMap.clear();
  /// map: {op -> vector<string>}
  /// example: {memref::AllocOp -> {"10", "%alloc_9[%14]"}}
  read_count_map_.clear();
  write_count_map_.clear();

  // Get the current function.
  func::FuncOp function = getOperation();
  comet_vdump(function->getParentOfType<ModuleOp>());
//  comet_vdump(function);
  function.walk(
    [&](mlir::Operation *op)
    {
      comet_pdump(op);
      /// If the op is memref.alloc
      if (auto memAllocOp = mlir::dyn_cast<memref::AllocOp>(op))
      {
        comet_vdump(memAllocOp);
        comet::TimeComplexity read_TC;
        comet::TimeComplexity write_TC;
        /// Analyze the memory object
        analyzeMemrefAllocOp(memAllocOp, read_TC /*output*/, write_TC /*output*/);
        read_count_map_[memAllocOp] = read_TC;
        write_count_map_[memAllocOp] = write_TC;
        /// Get its name
//        std::string object_name = getValueName(mlir::cast<mlir::Value>(memAllocOp));
        std::string object_name = getValueName(memAllocOp);
        object_name_map_[memAllocOp] = object_name;
      }
    }
  );

  /// dump maps
  if (read_count_map_.empty()) {
    return;
  }
  llvm::errs() << "####====---------------------------------====####\n";
  std::vector<std::pair<comet::TimeComplexity, std::string>> read_tc;
  std::vector<std::pair<comet::TimeComplexity, std::string>> write_tc;
  for (auto &entry : read_count_map_)
  {
    llvm::errs() << "\n";
    llvm::errs() << "memory object: `";
    entry.first.print(llvm::errs());
    llvm::errs() << "`\n";
    llvm::errs() << "object name: ";
    llvm::errs() << object_name_map_[entry.first] << "\n";
    llvm::errs() << "read time-complexity: " << entry.second.toString() << "\n";
    llvm::errs() << "write time-complexity: " << write_count_map_[entry.first].toString() << "\n";
    read_tc.push_back(std::make_pair(entry.second, object_name_map_[entry.first]));
    write_tc.push_back(std::make_pair(write_count_map_[entry.first], object_name_map_[entry.first]));
  }
  std::sort(read_tc.rbegin(), read_tc.rend());
  std::sort(write_tc.rbegin(), write_tc.rend());
  llvm::errs() << "\nRead Time Complexity:\n";
  for (auto &tc : read_tc)
  {
    llvm::errs() << tc.first.toString() << "\t|\t" << tc.second << "\n";
  }
  llvm::errs() << "\nWrite Time Complexity:\n";
  for (auto &tc : write_tc)
  {
    llvm::errs() << tc.first.toString() << "\t|\t" << tc.second << "\n";
  }

//  for (auto &entry : write_count_map_)
//  {
//    entry.first.print(llvm::errs());
//    llvm::errs() << " write time-complexity: " << entry.second.toString() << "\n";
//  }
  llvm::errs() << "####====---------------------------------====####\n";

}


/// backup
//void MemoryAccessFreqeuncyAnalysisPass::runOnOperation()
//{
//  accessFrequencyMap.clear();
//
//  // Get the current function.
//  func::FuncOp function = getOperation();
//  function.dump();
//
//  // Traverse each operation in the function.
//  function.walk([&](Operation *op)
//                {
//    if (auto affineForOp = dyn_cast<AffineForOp>(op))
//    {
//      // Calculate number of iterations considering the step value
//      int64_t lowerBound = affineForOp.getConstantLowerBound();
//      int64_t upperBound = affineForOp.getConstantUpperBound();
//      auto step = affineForOp.getStepAsInt();
//      int64_t iterations = (upperBound - lowerBound + step - 1) / step;
//
//        affineForOp.getBody()->walk([&](Operation *nestedOp) {
//        // AffineLoadOp and AffineStoreOp
//        if (auto loadOp = dyn_cast<AffineLoadOp>(nestedOp))
//        {
//          comet_debug() << "Inside loop body\n";
//          comet_vdump(loadOp);
//          estimateAccessFrequencyWithAffineMap(loadOp.getMemRef(), nestedOp, loadOp.getAffineMap(), iterations);
//        }
//        else if (auto storeOp = dyn_cast<AffineStoreOp>(nestedOp))
//        {
//          estimateAccessFrequencyWithAffineMap(storeOp.getMemRef(), nestedOp, storeOp.getAffineMap(), iterations);
//          // LoadOp and StoreOp
//        }
//        else if (auto loadOp = dyn_cast<memref::LoadOp>(nestedOp))
//        {
//          estimateAccessFrequency(loadOp.getMemRef(), nestedOp, iterations);
//        }
//        else if (auto storeOp = dyn_cast<memref::StoreOp>(nestedOp))
//        {
//          estimateAccessFrequency(storeOp.getMemRef(), nestedOp, iterations);
//        }
//        });
//    }
//    else
//    {
//      if (auto loadOp = dyn_cast<AffineLoadOp>(op))
//      {
//        comet_debug() << "Outside loop body\n";
//        comet_vdump(loadOp);
//        incrementAccessFrequency(loadOp.getMemRef(), 1);
//      }
//      else if (auto storeOp = dyn_cast<AffineStoreOp>(op))
//      {
//        incrementAccessFrequency(storeOp.getMemRef(), 1);
//      }
//      else if (auto loadOp = dyn_cast<memref::LoadOp>(op))
//      {
//        incrementAccessFrequency(loadOp.getMemRef(), 1);
//      }
//      else if (auto storeOp = dyn_cast<memref::StoreOp>(op))
//      {
//        incrementAccessFrequency(storeOp.getMemRef(), 1);
//      }
//    } });
//
//  // Print the access frequency of each memory location.
//  for (auto &entry : accessFrequencyMap)
//  {
//    llvm::outs() << "Memory location ";
//    entry.first.print(llvm::outs());
//    llvm::outs() << " was accessed " << entry.second << " times.\n";
//  }
//}

///===----------------------------------------------------------------------===//
/// Memory Access Pattern Analysis (sequential or random)
///===----------------------------------------------------------------------===//

// Recursive helper function to check if an affine expression is simple.
bool MemoryAccessPatternAnalysisPass::isSimpleAffineExpr(AffineExpr expr)
{
  comet_vdump(expr);
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
  {
    comet_debug() << "Direct access to a loop index can be sequential";
    return true; // Direct loop index access is simple.
  }

  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
  {
    return true; // Constant expressions are simple.
  }

  // Handle simple addition or multiplication.
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr))
  {
    switch (binExpr.getKind())
    {
    case AffineExprKind::Add:
      // Both sides must be simple.
      return isSimpleAffineExpr(binExpr.getLHS()) && isSimpleAffineExpr(binExpr.getRHS());
    case AffineExprKind::Mul:
      // Allow multiplication only if one side is a constant.
      if (isa<AffineConstantExpr>(binExpr.getLHS()) || isa<AffineConstantExpr>(binExpr.getRHS()))
      {
        return isSimpleAffineExpr(binExpr.getLHS()) || isSimpleAffineExpr(binExpr.getRHS());
      }
      return false; // Non-constant multiplication is considered complex.
    case AffineExprKind::Mod:
    case AffineExprKind::CeilDiv:
    case AffineExprKind::FloorDiv:
      // Modulo and division are always considered complex.
      return false;
    default:
      break;
    }
  }

  // Any other case (mod, div, etc.) is considered complex and thus random.
  return false;
}

// Check if an access pattern is sequential.
bool MemoryAccessPatternAnalysisPass::isSequentialAccess(AffineMap affineMap)
{
  if (affineMap.getNumResults() != 1)
  {
    return false; // Multiple results generally indicate complexity.
  }

  auto expr = affineMap.getResult(0);
  return isSimpleAffineExpr(expr);
}

void MemoryAccessPatternAnalysisPass::analyzeAccessPattern(Value memref, AffineMap affineMap, OperandRange indices, Operation *op)
{
  if (affineMap)
  {
    comet_vdump(affineMap);
    if (isSequentialAccess(affineMap))
    {
      llvm::outs() << "Sequential access detected for memory location: ";
    }
    else
    {
      llvm::outs() << "Random access detected for memory location: ";
    }
  }
  else
  {
    llvm::outs() << "Access pattern analysis could not be determined for: ";
  }
  memref.print(llvm::outs());
  llvm::outs() << "\n";
}

void MemoryAccessPatternAnalysisPass::runOnOperation()
{
  // Get the current function.
  func::FuncOp function = getOperation();

  function.walk([&](Operation *op)
                {
      if (auto affineForOp = dyn_cast<AffineForOp>(op)) {
        affineForOp.getBody()->walk([&](Operation *nestedOp) {
          //AffineLoad and AffineStore operation
          if (auto loadOp = dyn_cast<AffineLoadOp>(nestedOp)) {
            // TODO(gkestor): it assumes that affinemap is embedded in affine.load.
            // It doesn't support the following case. Affine map will return (d0) ->  (do)
            // %index = affine.apply affine_map<(d0) -> ((d0 * 5) mod 37)>(%i)
            // %v = affine.load %A[%index] : memref<100xi32>
            analyzeAccessPattern(loadOp.getMemRef(), loadOp.getAffineMap(), loadOp.getMapOperands(), nestedOp);
          } else if (auto storeOp = dyn_cast<AffineStoreOp>(nestedOp)) {
            analyzeAccessPattern(storeOp.getMemRef(), storeOp.getAffineMap(), storeOp.getMapOperands(), nestedOp);
          } 
          
          //LoadOp and StoreOp
          else if (auto loadOp = dyn_cast<memref::LoadOp>(nestedOp)) {
            // Handle standard load ops without affine map.
            analyzeAccessPattern(loadOp.getMemRef(), {}, loadOp.getIndices(), nestedOp);
          } else if (auto storeOp = dyn_cast<memref::StoreOp>(nestedOp)) {
            // Handle standard store ops without affine map.
            analyzeAccessPattern(storeOp.getMemRef(), {}, storeOp.getIndices(), nestedOp);
          }
        });
      } });
}

///===----------------------------------------------------------------------===//
///===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::comet::createMemoryAccessFrequencyAnalysisPass()
{
  return std::make_unique<MemoryAccessFreqeuncyAnalysisPass>();
}

std::unique_ptr<Pass> mlir::comet::createMemoryAccessPatternAnalysisPass()
{
  return std::make_unique<MemoryAccessPatternAnalysisPass>();
}