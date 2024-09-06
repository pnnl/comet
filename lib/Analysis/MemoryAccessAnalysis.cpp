#include "comet/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::affine;

// *********** For debug purpose *********//
//#define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

namespace
{
  struct MemoryAccessFreqeuncyAnalysisPass
      : PassWrapper<MemoryAccessFreqeuncyAnalysisPass, OperationPass<func::FuncOp>>
  {

    // A map to store the frequency of accesses for each memory location.
    llvm::DenseMap<Value, int> accessFrequencyMap;

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

void MemoryAccessFreqeuncyAnalysisPass::incrementAccessFrequency(Value memref, int64_t count)
{
  accessFrequencyMap[memref] += count;
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
  accessFrequencyMap.clear();

  // Get the current function.
  func::FuncOp function = getOperation();

  // Traverse each operation in the function.
  function.walk([&](Operation *op)
                {
      if (auto affineForOp = dyn_cast<AffineForOp>(op)) {
        // Calculate number of iterations considering the step value
        int64_t lowerBound = affineForOp.getConstantLowerBound();
        int64_t upperBound = affineForOp.getConstantUpperBound();
        auto step = affineForOp.getStepAsInt();
        int64_t iterations = (upperBound - lowerBound + step - 1) / step;

        affineForOp.getBody()->walk([&](Operation *nestedOp) {
          //AffineLoadOp and AffineStoreOp
          if (auto loadOp = dyn_cast<AffineLoadOp>(nestedOp)) {
            estimateAccessFrequencyWithAffineMap(loadOp.getMemRef(), nestedOp, loadOp.getAffineMap(), iterations);
          } else if (auto storeOp = dyn_cast<AffineStoreOp>(nestedOp)) {
            estimateAccessFrequencyWithAffineMap(storeOp.getMemRef(), nestedOp, storeOp.getAffineMap(), iterations);
          
          //LoadOp and StoreOp
          } else if (auto loadOp = dyn_cast<memref::LoadOp>(nestedOp)) {
            estimateAccessFrequency(loadOp.getMemRef(), nestedOp, iterations);
          } else if (auto storeOp = dyn_cast<memref::StoreOp>(nestedOp)) {
            estimateAccessFrequency(storeOp.getMemRef(), nestedOp, iterations);
          }
        });
      } else {
        if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
          incrementAccessFrequency(loadOp.getMemRef(), 1);
        } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
          incrementAccessFrequency(storeOp.getMemRef(), 1);
        } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
          incrementAccessFrequency(loadOp.getMemRef(), 1);
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
          incrementAccessFrequency(storeOp.getMemRef(), 1);
        }
      } });

  // Print the access frequency of each memory location.
  for (auto &entry : accessFrequencyMap)
  {
    llvm::outs() << "Memory location ";
    entry.first.print(llvm::outs());
    llvm::outs() << " was accessed " << entry.second << " times.\n";
  }
}

///===----------------------------------------------------------------------===//
/// Memory Access Pattern Analysis (sequential or random)
///===----------------------------------------------------------------------===//

// Simple heuristic to determine if the affine map represents sequential access.
// bool MemoryAccessPatternAnalysisPass::isSequentialAccess(AffineMap affineMap)
// {
//   if (affineMap.getNumResults() != 1)
//   {
//     return false; // Not a single-dimensional access.
//   }

//   auto resultExpr = affineMap.getResult(0);
//   if (auto dimExpr = llvm::dyn_cast<AffineDimExpr>(resultExpr))
//   {
//     // Simple case: access is directly based on the loop index.
//     return true;
//   }
//   else if (auto binExpr = llvm::dyn_cast<AffineBinaryOpExpr>(resultExpr))
//   {
//     // Handle simple affine expressions like `2 * i + 1`.
//     if (binExpr.getKind() == AffineExprKind::Add || binExpr.getKind() == AffineExprKind::Mul)
//     {
//       return llvm::isa<AffineDimExpr>(binExpr.getLHS()) || llvm::isa<AffineDimExpr>(binExpr.getRHS());
//     }
//   }

//   return false;
// }

// Simple heuristic to determine if indices represent sequential access.
// Heuristics for Sequential Access: If the index is an affine function of the loop iterator
// with a stride of 1 or a constant stride, it's sequential. If the index depends on some complex non-affine
// expression or varies non-uniformly, it's considered random.
// bool MemoryAccessPatternAnalysisPass::isSequentialAccess(OperandRange indices)
// {
//   if (indices.size() != 1)
//   {
//     return false; // Not a single-dimensional access.
//   }

//   if (auto defOp = indices[0].getDefiningOp())
//   {
//     if (isa<AffineApplyOp>(defOp))
//     {
//       // Further analysis can be added here to analyze complex affine expressions.
//       return true; // Assume affine apply represents sequential access.
//     }
//   }

//   return false;
// }

// Determine if the affine expression represents a sequential access pattern.
// bool MemoryAccessPatternAnalysisPass::isSequentialAffineExpr(AffineExpr expr) {
//   if (auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr)) {
//     comet_debug() << "Direct mapping from loop index to memory index\n";
//     return true; // Direct mapping from loop index to memory index.
//   }

//   if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
//     comet_debug() << "Constant expressions are sequential\n";
//     return true; // Constant expressions are sequential.
//   }

//   if (auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
//     // Check for add, mul with constants and loop indices.
//     if (binExpr.getKind() == AffineExprKind::Add) {
//       return isSequentialAffineExpr(binExpr.getLHS()) && isSequentialAffineExpr(binExpr.getRHS());
//     }
//     if (binExpr.getKind() == AffineExprKind::Mul) {
//       if (auto lhsConst = binExpr.getLHS().dyn_cast<AffineConstantExpr>()) {
//         return lhsConst.getValue() == 1 && isSequentialAffineExpr(binExpr.getRHS());
//       }
//       if (auto rhsConst = binExpr.getRHS().dyn_cast<AffineConstantExpr>()) {
//         return rhsConst.getValue() == 1 && isSequentialAffineExpr(binExpr.getLHS());
//       }
//     }
//   }

//   return false; // If none of the conditions match, consider it random.
// }

// Recursive helper function to check if an affine expression is simple.
bool MemoryAccessPatternAnalysisPass::isSimpleAffineExpr(AffineExpr expr)
{
  comet_vdump(expr);
  if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
    comet_debug() << "Direct access to a loop index can be sequential";
      return true; // Direct loop index access is simple.
    }

    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
      return true; // Constant expressions are simple.
    }

    // Handle simple addition or multiplication.
    if (auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
      switch (binExpr.getKind()) {
        case AffineExprKind::Add:
          // Both sides must be simple.
          return isSimpleAffineExpr(binExpr.getLHS()) && isSimpleAffineExpr(binExpr.getRHS());
        case AffineExprKind::Mul:
          // Allow multiplication only if one side is a constant.
          if (binExpr.getLHS().isa<AffineConstantExpr>() || binExpr.getRHS().isa<AffineConstantExpr>()) {
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