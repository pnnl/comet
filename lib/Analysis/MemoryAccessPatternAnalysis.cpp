#include "comet/Analysis/MemoryAccessPatternAnalysis.h"
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
#define COMET_DEBUG_MODE
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
} // end anonymous namespace

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
  //TODO(gkestor): need better heuristics
  // Simple heuristic: assume each dimension's impact on unique accesses.
  // This can be refined to analyze the actual structure of the affine map.
  return affineMap.getNumResults();
}

// Estimate access frequency considering affine maps.
void MemoryAccessFreqeuncyAnalysisPass::estimateAccessFrequencyWithAffineMap(Value memref, Operation *op, AffineMap affineMap, int64_t iterations)
{
  int64_t conditionFactor = calculateConditionFactor(op,iterations);

  // Consider the complexity of the affine map.
  int64_t accessMultiplier = 1;
  accessMultiplier = estimateAccessMultiplierFromAffineMap(affineMap);
  incrementAccessFrequency(memref, conditionFactor * accessMultiplier);
}

// Estimate the access frequency for the given memory location considering conditions.
void MemoryAccessFreqeuncyAnalysisPass::estimateAccessFrequency(Value memref, Operation *op, int64_t iterations)
{
  int64_t conditionFactor = calculateConditionFactor(op,iterations);
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



std::unique_ptr<Pass> mlir::comet::createMemoryAccessFrequencyAnalysisPass()
{
  return std::make_unique<MemoryAccessFreqeuncyAnalysisPass>();
}
