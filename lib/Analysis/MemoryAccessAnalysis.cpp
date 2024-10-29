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
#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
//#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

namespace
{
  /// For example, time complexity O(L*N*M + I*J), `L*N*M` is a term, so is `I*J`. And `L`, `N`, etc. are factors.
  /// A factor could be
  /// 1. const, e.g., O(1)
  /// 2. var, e.g., O(%alloc[%c4])
  /// 3. var - var, e.g., O(%alloc_9[%14]-%alloc_9[%arg0])
  /// 4. var - const, e.g., O(%alloc[%c4] - 4) (this is very rare)
  class TimeTerm
  {
    std::vector<std::string> factors_;
  };
  class TimeComplexity
  {
    std::vector<std::string> list_;

  public:
    void addTerm(const std::string &term)
    {
      list_.push_back(term);
    }

    std::string toString()
    {
      std::string text = "O(";
      for (auto iter = list_.begin(); iter != list_.end(); ++iter)
      {
        if (iter == list_.begin())
        {
          text += *iter;
        }
        else
        {
          text += " + " + *iter;
        }
      }
      text += ")";

      return text;
    }

    void dump()
    {
      llvm::errs() << toString() << "\n";
    }
  };


  struct MemoryAccessFreqeuncyAnalysisPass
      : PassWrapper<MemoryAccessFreqeuncyAnalysisPass, OperationPass<func::FuncOp>>
  {

    // A map to store the frequency of accesses for each memory location.
    llvm::DenseMap<Value, int> accessFrequencyMap;

    llvm::DenseMap<Value, std::string> object_name_map_;
    llvm::DenseMap<Value, TimeComplexity> read_count_map_;
    llvm::DenseMap<Value, TimeComplexity> write_count_map_;

    // StringRef getArgument() const final
    // {
    //   return "memory-access-pattern-analysis";
    // }
    // StringRef getDescription() const final
    // {
    //   return "Test alias analysis extending.";
    // }

    void incrementAccessFrequency(Value memref, int64_t count);
    int64_t calculateConditionFactor(Operation *op, int64_t iterations);
    int64_t estimateAccessMultiplierFromAffineMap(AffineMap affineMap);
    void estimateAccessFrequencyWithAffineMap(Value memref, Operation *op, AffineMap affineMap, int64_t iterations);
    void estimateAccessFrequency(Value memref, Operation *op, int64_t iterations);

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
    is_number = false;
    if (auto cio = mlir::dyn_cast<arith::ConstantIndexOp>(value.getDefiningOp()))
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
  std::string getForLoopRange(mlir::Operation *forLoop)
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
    std::string term;
    if (lower_is_number && upper_is_number)
    {
      auto lower_int = std::strtoll(lowerBoundStr.c_str(), nullptr, 0);
      auto upper_int = std::strtoll(upperBoundStr.c_str(), nullptr, 0);
      auto range = upper_int - lower_int;
      /// TODO: what if
      /// 1. the step is negative then the upper_int < lower_int
      /// 2. range is 0
      /// 3. upper bound or lower bound are not integer.
      term = std::to_string(range);
    }
    else if (lower_is_number && lowerBoundStr == "0")
    {
      term = upperBoundStr;
    }
    else
    {
      term = "(" + upperBoundStr + "-" + lowerBoundStr + ")";
    }

    return term;
  }

  /// If op is within for-loops, get its time complexity
  std::string getForLoopFactors(mlir::Operation *op)
  {
    std::vector<std::string> factors;  /// Each level has one factor
    scf::ForOp forOp = op->getParentOfType<scf::ForOp>();
    scf::ParallelOp parallelOp = op->getParentOfType<scf::ParallelOp>();
    omp::WsLoopOp wsLoopOp = op->getParentOfType<omp::WsLoopOp>();
//    scf::ForOp forOp = mlir::dyn_cast<scf::ForOp>(op->getParentOp());
//    scf::ParallelOp parallelOp = mlir::dyn_cast<scf::ParallelOp>(op->getParentOp());
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
        std::string term = getForLoopRange(forOp);
        factors.push_back(term);
        if (!getUpperLevelForLoop(forOp))
        {
          break;
        }
      }
      else if (parallelOp)
      {  /// parallel Op
        comet_vdump(parallelOp);
        std::string term = getForLoopRange(parallelOp);
        factors.push_back(term);
        if (!getUpperLevelForLoop(parallelOp))
        {
          break;
        }
      }
      else if (wsLoopOp)
      {
        comet_vdump(wsLoopOp);
        std::string term = getForLoopRange(wsLoopOp);
        factors.push_back(term);
        if (!getUpperLevelForLoop(wsLoopOp))
        {
          break;
        }
      }
    }

    /// Combine strings to a final string
    std::string factors_str = "";
    for (auto iter = factors.rbegin(); iter != factors.rend(); ++iter)
    {
      if (iter == factors.rbegin())
      {
        factors_str += *iter;
      }
      else
      {
        factors_str += "*" + *iter;
      }
    }

//    comet_debug() << "factors_str: " << factors_str << "\n";
    return factors_str;
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
  void analyzeOneMemoryObject(memref::AllocOp &memAllocOp,
                              TimeComplexity &read_TC /*output*/,
                              TimeComplexity &write_TC /*output*/)
  {
    comet_vdump(memAllocOp);
    for (mlir::Operation *user : memAllocOp->getUsers())
    {
      comet_pdump(user);
      std::string term = "1";
      if (isOpInForLoop(user))
      {
        comet_debug() << "in a loop\n";
        term = getForLoopFactors(user);
        comet_debug() << "term: " << term << "\n";
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
void MemoryAccessFreqeuncyAnalysisPass::incrementAccessFrequency(Value memref, int64_t count)
{
  comet_debug() << "before adding incrementAccessFrequency:" << accessFrequencyMap[memref] << "\n";
  accessFrequencyMap[memref] += count;
  comet_debug() << "input incrementAccessFrequency:" << count << "\n";
  comet_debug() << "after adding incrementAccessFrequency:" << accessFrequencyMap[memref] << "\n";
}

int64_t MemoryAccessFreqeuncyAnalysisPass::calculateConditionFactor(Operation *op, int64_t iterations)
{
  int64_t conditionFactor = 1;

  if (auto ifOp = op->getParentOfType<AffineIfOp>())
  {
    conditionFactor = iterations / 2;
  }
  else
  {
    conditionFactor = iterations;
  }

  return conditionFactor;
}

// Estimate the effect of the affine map on access frequency.
int64_t MemoryAccessFreqeuncyAnalysisPass::estimateAccessMultiplierFromAffineMap(AffineMap affineMap)
{
  // TODO(gkestor): need better heuristics
  // Simple heuristic: This is a heuristic that assumes each result of the affine map corresponds to a distinct access.
  // The actual effect might be more complex, especially for multi-dimensional accesses, but this provides a basic approach.
  return affineMap.getNumResults();
}

// Estimate access frequency considering affine maps.
void MemoryAccessFreqeuncyAnalysisPass::estimateAccessFrequencyWithAffineMap(Value memref, Operation *op, AffineMap affineMap, int64_t iterations)
{
  int64_t conditionFactor = calculateConditionFactor(op, iterations);

  // Consider the complexity of the affine map.
  int64_t accessMultiplier = 1;
  accessMultiplier = estimateAccessMultiplierFromAffineMap(affineMap);
  comet_vdump(memref);
  comet_debug() << "incrementAccessFrequency before calling:" << conditionFactor * accessMultiplier << " \n";
  incrementAccessFrequency(memref, conditionFactor * accessMultiplier);
}

// Estimate the access frequency for the given memory location considering conditions.
void MemoryAccessFreqeuncyAnalysisPass::estimateAccessFrequency(Value memref, Operation *op, int64_t iterations)
{
  int64_t conditionFactor = calculateConditionFactor(op, iterations);
  incrementAccessFrequency(memref, conditionFactor);
}

void MemoryAccessFreqeuncyAnalysisPass::runOnOperation()
{
//  accessFrequencyMap.clear();
  /// map: {op -> vector<string>}
  /// example: {memref::AllocOp -> {"10", "%alloc_9[%14]"}}
  read_count_map_.clear();
  write_count_map_.clear();

  // Get the current function.
  func::FuncOp function = getOperation();
//  comet_vdump(function);
  function.walk(
    [&](mlir::Operation *op)
    {
      comet_pdump(op);
      /// If the op is memref.alloc
      if (auto memAllocOp = mlir::dyn_cast<memref::AllocOp>(op))
      {
        comet_vdump(memAllocOp);
        TimeComplexity read_TC;
        TimeComplexity write_TC;
        /// Analyze the memory object
        analyzeOneMemoryObject(memAllocOp, read_TC /*output*/, write_TC /*output*/);
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
  llvm::errs() << "####====---------------------------------====####\n";
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