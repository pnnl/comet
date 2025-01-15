//===- LinalgTransforms.cpp - Linalg transformation patterns -----===//
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
// This file implements logic for Linalg transformations after lowering TA to Linalg operations
//
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// suppress all warnings coming from inclusion of blis.h in source tree
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wcast-qual"
#include "blis.h"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wcast-qual"
#include "blis.h"
#endif

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::arith;
using namespace mlir::tensorAlgebra;
using namespace mlir::affine;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

//===----------------------------------------------------------------------===//
// LinAlg Matmul Tiling
//===----------------------------------------------------------------------===//
namespace
{
  // Marker used as attribute name in generated Linalg rewriting transformations.
  const StringLiteral kLinalgTransformMarker = "__with_tiling__";
  std::string linalgMatmulUkrFname = "linalg_matmul_viewsxs_viewsxs_viewsxs";

  /// Helper class to control application of linalg transformation patterns.
  /// Control comes in 2 forms:
  ///   1. attribute matching and setting behavior using the attribute named
  ///      `kLinalgTransformMarker`. This can be used to build a state machine
  ///      using attributes and incrementally applying patterns to advance states.
  ///   2. filter function, which is a simple lambda on the Operation* that
  ///      returns a LogicalResult.
  struct LinalgTransformationFilter
  {
    using FilterFunction = std::function<LogicalResult(Operation *)>;

    explicit LinalgTransformationFilter(
        ArrayRef<StringAttr> matchDisjunction = {},
        std::optional<StringAttr> replacement = std::nullopt);

    explicit LinalgTransformationFilter(
        const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction = {},
        std::optional<StringAttr> replacement = std::nullopt);

    LinalgTransformationFilter(LinalgTransformationFilter &&) = default;
    LinalgTransformationFilter(const LinalgTransformationFilter &) = default;
    LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
    void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
                                           Operation *op) const;

    LinalgTransformationFilter &addFilter(const FilterFunction &f)
    {
      if (f)
        filters.push_back(f);
      return *this;
    }

    template <typename... OpTypes>
    LinalgTransformationFilter &addOpFilter()
    {
      return addFilter(
          [](Operation *op)
          { return success(isa<OpTypes...>(op)); });
    }

    LinalgTransformationFilter &addOpNameFilter(StringRef opName)
    {
      return addFilter([opName](Operation *op)
                       { return success(op->getName().getStringRef() == opName); });
    }

    LinalgTransformationFilter &setMatchByDefault()
    {
      matchByDefault = true;
      return *this;
    }

  private:
    SmallVector<FilterFunction> filters;
    SmallVector<StringAttr> matchDisjunction;
    std::optional<StringAttr> replacement;
    /// When set to true, if the attribute is not set, it will be treated as
    /// a match. Default is false.
    bool matchByDefault;
  };

  LinalgTransformationFilter::LinalgTransformationFilter(
      ArrayRef<StringAttr> matchDisjunction,
      std::optional<StringAttr> replacement)
      : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
        replacement(replacement), matchByDefault(false) {}

  LogicalResult
  LinalgTransformationFilter::checkAndNotify(PatternRewriter &rewriter,
                                             Operation *op) const
  {
    if (llvm::any_of(filters,
                     [&](const FilterFunction &f)
                     { return failed(f(op)); }))
      return failure();

    auto attr = op->template getAttrOfType<StringAttr>(kLinalgTransformMarker);

    if (!attr)
    {
      /// 1. Has no filter case and matchDisjunction is empty.
      if (matchDisjunction.empty() || matchByDefault)
        return success();

      /// 2. Has no filter but was expecting a filter.
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag)
                                         {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag); });
    }

    /// 4. Match explicit filter.
    for (auto filter : matchDisjunction)
      if (attr.getValue() == filter)
        return success();

    /// 5. Fail to match.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag)
                                       {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag); });
  }

  void LinalgTransformationFilter::replaceLinalgTransformationFilter(
      PatternRewriter &rewriter, Operation *op) const
  {
    if (replacement.has_value())
      op->setAttr(kLinalgTransformMarker, *replacement);
    else
      op->removeAttr(rewriter.getStringAttr(kLinalgTransformMarker));
  }

  /// Pattern that tiles linalg operations using the `TilingInterface`
  /// with `scf.for` ops for iterating over the tiles) while
  /// using a `filter` to avoid recursive application.
  struct LinalgTilingLoops
      : public OpInterfaceRewritePattern<TilingInterface>
  {
    LinalgTilingLoops(
        MLIRContext *context, scf::SCFTilingOptions options,
        LinalgTransformationFilter filter = LinalgTransformationFilter(),
        PatternBenefit benefit = 1)
        : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
          options(std::move(options)), filter(std::move(filter)) {}

    /// Construct a generic pattern applied to `opName`.
    LinalgTilingLoops(
        StringRef opName, MLIRContext *context, scf::SCFTilingOptions options,
        LinalgTransformationFilter filter = LinalgTransformationFilter(),
        PatternBenefit benefit = 1)
        : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
          options(std::move(options)), filter(std::move(filter)) {}

    LogicalResult matchAndRewrite(TilingInterface op,
                                  PatternRewriter &rewriter) const override
    {
      if (failed(filter.checkAndNotify(rewriter, op)))
        return failure();

      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCFForOp(rewriter, op, options);
      if (failed(tilingResult))
        return rewriter.notifyMatchFailure(op, "failed to tile operation");

      if (op->getNumResults())
      {
        rewriter.replaceOp(op, tilingResult->replacements);
      }
      else
      {
        rewriter.eraseOp(op);
      }

      for (auto *tiledOp : tilingResult->tiledOps)
        filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
      return success();
    }

  private:
    scf::SCFTilingOptions options;
    LinalgTransformationFilter filter;
  };

} /// namespace

void get_level3_blocksizes(int *mc, int *kc, int *nc, int *mr, int *nr, int size_dt)
{
  /// Query a native context.
  const cntx_t *cntx = bli_gks_query_nat_cntx();

  *mc = (int)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx);
  *kc = (int)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx);
  *nc = (int)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx);
  *mr = (int)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
  *nr = (int)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);

  /// printf("mc= %d, kc= %d, nc=%d, mr=%d, nr=%d\n", *mc, *kc, *nc, *mr, *nr);
  return;
}

static void addPatternForTiling(MLIRContext *context,
                                RewritePatternSet &patterns,
                                StringRef filterName,
                                StringRef updatedFilterName,
                                ArrayRef<int64_t> tileSizes,
                                ArrayRef<int64_t> interchange = {})
{
  scf::SCFTilingOptions tilingOptions;
  SmallVector<OpFoldResult> tileSizesOfr =
      getAsIndexOpFoldResult(context, tileSizes);
  tilingOptions.setTileSizes(tileSizesOfr).setInterchange(interchange);
  LinalgTransformationFilter filter(StringAttr::get(context, filterName),
                                    StringAttr::get(context, updatedFilterName));
  patterns.add<LinalgTilingLoops>(context, tilingOptions, filter);
}

namespace
{
  class LinAlgMatmulTilingPass : public PassWrapper<LinAlgMatmulTilingPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinAlgMatmulTilingPass)
    void runOnOperation() override
    {
      func::FuncOp func = getOperation();
      MLIRContext *ctx = func.getContext();
      RewritePatternSet tilingPatterns(ctx);

      int mc, kc, nc, mr, nr = 0;
      get_level3_blocksizes(&mc, &kc, &nc, &mr, &nr, sizeof(double));

      addPatternForTiling(ctx, tilingPatterns, "__with_tiling__", "__L2__with_tiling__", {mc, nc, kc}, {1, 2, 0});
      addPatternForTiling(ctx, tilingPatterns, "__L2__with_tiling__", "__micro_kernel__", {mr, nr, kc}, {1, 0, 2});

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(tilingPatterns))))
        return signalPassFailure();
    }
  };
} /// end anonymous namespace

/////////////////////////////////////////////
//////////LinAlg Matmul microkernel//////////
/////////////////////////////////////////////
namespace
{
  class LinalgMatMulOpToLibraryCallPattern : public OpRewritePattern<MatmulOp>
  {
  public:
    using OpRewritePattern<MatmulOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(MatmulOp op,
                                  PatternRewriter &rewriter) const override;
  };
}

static MemRefType makeStridedLayoutDynamic(MemRefType type)
{
  return MemRefType::Builder(type).setLayout(StridedLayoutAttr::get(
      type.getContext(), ShapedType::kDynamic,
      SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)));
}

/// Helper function to extract the operand types that are passed to the
/// generated CallOp. MemRefTypes have their layout canonicalized since the
/// information is not used in signature generation.
/// Note that static size information is not modified.
static SmallVector<Type, 4> extractOperandTypes(Operation *op)
{
  SmallVector<Type, 4> result;
  result.reserve(op->getNumOperands());
  for (auto type : op->getOperandTypes())
  {
    /// The underlying descriptor type (e.g. LLVM) does not have layout
    /// information. Canonicalizing the type at the level of std when going into
    /// a library call avoids needing to introduce DialectCastOp.
    if (auto memrefType = type.dyn_cast<MemRefType>())
      result.push_back(makeStridedLayoutDynamic(memrefType));
    else
      result.push_back(type);
  }

  return result;
}

static SmallVector<Value, 4>
createTypeCanonicalizedMemRefOperands(OpBuilder &b, Location loc,
                                      ValueRange operands)
{
  SmallVector<Value, 4> res;
  res.reserve(operands.size());
  for (auto op : operands)
  {
    auto memrefType = op.getType().dyn_cast<MemRefType>();
    if (!memrefType)
    {
      res.push_back(op);
      continue;
    }
    Value cast =
        b.create<memref::CastOp>(loc, makeStridedLayoutDynamic(memrefType), op);
    res.push_back(cast);
  }
  return res;
}

/// Get a SymbolRefAttr containing the library function name for the LinalgOp.
/// If the library function does not exist, insert a declaration.
static FailureOr<FlatSymbolRefAttr> getLibraryCallSymbolRef(Operation *op, PatternRewriter &rewriter)
{
  auto fnName = linalgMatmulUkrFname;
  if (fnName.empty())
    return rewriter.notifyMatchFailure(op, "No library call defined for: ");

  /// fnName is a dynamic std::string, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr()))
    return fnNameAttr;

  SmallVector<Type, 4> inputTypes(extractOperandTypes(op));

  /// Add inputTypes for mr and nr
  inputTypes.push_back(IntegerType::get(rewriter.getContext(), 32));
  inputTypes.push_back(IntegerType::get(rewriter.getContext(), 32));

  if (op->getNumResults() != 0)
  {
    return rewriter.notifyMatchFailure(
        op,
        "Library call for linalg operation can be generated only for ops that "
        "have void return types");
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, {});

  OpBuilder::InsertionGuard guard(rewriter);
  /// Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(
      op->getLoc(), fnNameAttr.getValue(), libFnType);
  /// Insert a function attribute that will trigger the emission of the
  /// corresponding `_mlir_ciface_xxx` interface so that external libraries see
  /// a normalized ABI. This interface is added during std to llvm conversion.
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(op->getContext()));
  funcOp.setPrivate();
  return fnNameAttr;
}

LogicalResult LinalgMatMulOpToLibraryCallPattern::matchAndRewrite(
    MatmulOp op, PatternRewriter &rewriter) const
{
  if (!isa<MatmulOp>(op))
    return failure();

  auto libraryCallName = getLibraryCallSymbolRef(op, rewriter);
  if (failed(libraryCallName))
    return failure();

  int mc, kc, nc, mr, nr = 0;
  get_level3_blocksizes(&mc, &kc, &nc, &mr, &nr, sizeof(double));

  IntegerType i32Type = IntegerType::get(rewriter.getContext(), 32);
  Value mrValue = rewriter.create<ConstantOp>(op->getLoc(), i32Type, rewriter.getIntegerAttr(i32Type, mr));
  Value nrValue = rewriter.create<ConstantOp>(op->getLoc(), i32Type, rewriter.getIntegerAttr(i32Type, nr));

  comet_vdump(mrValue);
  comet_vdump(nrValue);

  std::vector<Value> operands;
  operands.insert(operands.end(), op->getOperands().begin(), op->getOperands().end());
  operands.push_back(mrValue);
  operands.push_back(nrValue);

  rewriter.replaceOpWithNewOp<func::CallOp>(
      op, libraryCallName->getValue(), TypeRange(),
      createTypeCanonicalizedMemRefOperands(rewriter, op->getLoc(), operands));
  return success();
}

namespace
{
  class LinAlgMatmulMicroKernelPass : public PassWrapper<LinAlgMatmulMicroKernelPass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinAlgMatmulMicroKernelPass)
    void runOnOperation() override
    {
      func::FuncOp func = getOperation();
      MLIRContext *ctx = func.getContext();

      RewritePatternSet patterns(&getContext());

      /// Replace the inner linalg.matmul with the blis microkernel
      patterns.insert<LinalgMatMulOpToLibraryCallPattern>(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }
  };
} /// end anonymous namespace

//===----------------------------------------------------------------------===//
/// LinAlg.transpose Optimization
//===----------------------------------------------------------------------===//

struct OptDenseTranspose : public ConversionPattern
{
  OptDenseTranspose(MLIRContext *ctx, uint64_t tile_size, bool seperate_tiles)
      : ConversionPattern(linalg::TransposeOp::getOperationName(), 1, ctx),
        tile_size(tile_size), seperate_tiles(seperate_tiles) {}

  LogicalResult
  matchAndRewrite(Operation *input_op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final
  {
    comet_debug() << " OptDenseTranspose : public ConversionPattern\n";
    auto op = dyn_cast<linalg::TransposeOp>(input_op);
    comet_debug() << " Lowering dense transpose\n";
    assert(isa<linalg::TransposeOp>(op) &&
           "this operation is not a linalg.transpose");

    auto ctx = rewriter.getContext();
    Builder builder(ctx);
    Location loc = op.getLoc();

    //auto module = op->getParentOfType<ModuleOp>();
    comet_vdump(op);

    auto tensorize_input = op->getOperand(0).getDefiningOp();
    auto inputMemref = tensorize_input->getOperand(0);
    auto inputType = inputMemref.getType();

    auto tensorize_output = op->getOperand(1).getDefiningOp();
    auto outputMemref = tensorize_output->getOperand(0);

    comet_debug() << " Input Type:\n";
    comet_vdump(inputType);
    auto inputRank = inputType.cast<mlir::MemRefType>().getRank();

    std::vector<AffineForOp> loops;
    std::vector<int64_t> indexIterateOrder;
    for (int64_t rank = 0; rank < inputRank; rank++)
    {
      indexIterateOrder.push_back(rank);
      int64_t upperBound = inputType.cast<mlir::MemRefType>().getDimSize(rank);
      if (upperBound == ShapedType::kDynamic)
      {
        assert(false && "TODO: This dimension is a dynamic size");
      }
      /// create for loops
      auto loop = rewriter.create<AffineForOp>(loc, 0, upperBound, 1);
      loops.push_back(loop);
      comet_vdump(loop);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    AffineMap inputIndexingMap = builder.getMultiDimIdentityMap(inputRank);
    auto inputIndices = getReassociationIndices(inputIndexingMap);
    auto inputIVs = createInductionVarAffine(loops, indexIterateOrder, inputIndices);

    AffineMap outputIndexingMap = AffineMap::getPermutationMap(llvm::to_vector_of<unsigned>(op.getPermutation()), ctx);
    SmallVector<ReassociationIndices> outputIndices =
        getReassociationIndices(outputIndexingMap);
    auto outputIVs = createInductionVarAffine(loops, indexIterateOrder, outputIndices);

    /// Build loop body
    auto load_rhs = rewriter.create<memref::LoadOp>(loc, inputMemref, inputIVs);
    comet_vdump(load_rhs);
#ifdef COMET_DEBUG_MODE
    comet_vdump(load_rhs);
    auto store_lhs = rewriter.create<memref::StoreOp>(loc, load_rhs, outputMemref, outputIVs);
    comet_vdump(store_lhs);
#else
    rewriter.create<memref::StoreOp>(loc, load_rhs, outputMemref, outputIVs);
#endif

    /// TransposeOp index permutation
    ArrayRef<AffineExpr> invresults = inputIndexingMap.getResults();
    std::vector<unsigned> sourceOrder;
    for (auto a : invresults)
    {
      if (a.getKind() == AffineExprKind::DimId)
      {
        AffineDimExpr *b = (AffineDimExpr *)&a; /// down_casting
        sourceOrder.push_back(b->getPosition());
        comet_debug() << "Source order: " << b->getPosition() << "\n";
      }
    }

    /// AffineMap outvmap = op.getPermutation().getValue();
    ArrayRef<AffineExpr> outvresults = outputIndexingMap.getResults();
    /// From outer to inner, the destOrder[size -1] is the most important,
    std::vector<unsigned> destOrder;
    for (auto a : outvresults)
    {
      if (a.getKind() == AffineExprKind::DimId)
      {
        AffineDimExpr *b = (AffineDimExpr *)&a; /// down_casting
        destOrder.push_back(b->getPosition());
        comet_debug() << "destination order: " << b->getPosition() << "\n";
      }
    }

    if (loops.size() > 0)
    {
      /* Suppose Given best order: a0, a3, a1, a2
      ** Then first step: a0, a1, a3, a2 (exchange loop index 1 and 2)
      ** Then second step: a0, a1, a2, a3 (exchange loop order index 2 and 3)
      */
      std::vector<unsigned> optimalOrder = destOrder;
      /// Call an getLoopOrder algorithm to get the best order
      std::vector<std::vector<unsigned>> loopOrders;
      getLoopOrders(loopOrders, destOrder.size(), sourceOrder, destOrder);
      optimalOrder = loopOrders[0];

      std::vector<unsigned> currentOrder;
      for (unsigned i = 0; i < destOrder.size(); i++)
      {
        currentOrder.push_back(i);
      }

      for (unsigned i = 0; i < optimalOrder.size(); i++)
      {
        comet_debug() << "currentOrder[i]: " << currentOrder[i] << " optimalOrder[i]: " << optimalOrder[i] << "\n";
        /// This loop index is the correct loop index, no loop interchange
        if (optimalOrder[i] == currentOrder[i])
        {
          continue;
        }
        else
        { /// Get the location of the right loop index
          for (unsigned j = i + 1; j < currentOrder.size(); j++)
          {
            if (optimalOrder[i] == currentOrder[j])
            { /// loop j and i exchange
              unsigned k = j;
              /// k = (i,j]. k is unsigned, should be >= 0. use k-1, so k>=1
              while (k > 0 && k > i)
              {
                mlir::affine::interchangeLoops(loops[currentOrder[k - 1]], loops[currentOrder[k]]);
                std::swap(currentOrder[k - 1], currentOrder[k]);
                k--;
              }
              break;
            }
          }
        }
      }

      std::vector<AffineForOp> newLoops;
      for (unsigned i = 0; i < currentOrder.size(); i++)
      {
        newLoops.push_back(loops[currentOrder[i]]);
      }
      loops.clear();

      /// Possible to assign different tile size based on the dimension
      if (tile_size > 1)
      {
        std::vector<unsigned> tileSizes;
        for (unsigned i = 0; i < currentOrder.size(); i++)
        {
          tileSizes.push_back(tile_size);
        }
        SmallVector<AffineForOp, 6> tiledNest;
        if (failed(mlir::affine::tilePerfectlyNested(newLoops, tileSizes, &tiledNest)))
          return failure();

        comet_vdump(tiledNest[0]);

        /// Separate full and partial tiles.
        if (seperate_tiles)
        {
          auto intraTileLoops =
              MutableArrayRef<AffineForOp>(tiledNest).drop_front(newLoops.size());
          if (failed(separateFullTiles(intraTileLoops)))
            return failure();
        }
      } /// end if (tilesize > 1)

    } /// end loops.size() < 0

    rewriter.replaceAllUsesWith(op->getResult(0), outputMemref);
    rewriter.eraseOp(op);

    //module.dump();
    return success();
  }

private:
  uint64_t tile_size;
  bool seperate_tiles;
}; /// Lower Dense Transpose to loops after optimizations

namespace
{
  class OptDenseTransposePass : public PassWrapper<OptDenseTransposePass, OperationPass<func::FuncOp>>
  {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptDenseTransposePass)
    OptDenseTransposePass(uint64_t tile_size, bool seperate_tiles) : tile_size(tile_size), seperate_tiles(seperate_tiles){};
    void runOnOperation() override
    {
      comet_debug() << "OptDenseTransposePass : public PassWrapper<OptDenseTransposePass, FunctionPass>\n";
      func::FuncOp func = getOperation();
      ConversionTarget target(getContext());
      target.addLegalDialect<ArithDialect, AffineDialect, memref::MemRefDialect>();
      RewritePatternSet patterns(&getContext());
      patterns.insert<OptDenseTranspose>(&getContext(), tile_size, seperate_tiles);

      if (failed(applyPartialConversion(func, target, std::move(patterns))))
      {
        llvm::errs() << "Failed to Lower dense transpose operation\n";
        signalPassFailure();
      }
      comet_debug() << "OptDenseTransposePass done\n";
    }

  private:
    uint64_t tile_size;
    bool seperate_tiles;
  };
} /// end anonymous namespace

/// Create a pass to optimize LinAlg Matmul Op with tiling
std::unique_ptr<mlir::Pass> mlir::comet::createLinAlgMatmulTilingPass()
{
  return std::make_unique<LinAlgMatmulTilingPass>();
}

/// Create a pass to call a blis micro kernel for the inner linalg.matmul after tiling
std::unique_ptr<mlir::Pass> mlir::comet::createLinAlgMatmulMicroKernelPass()
{
  return std::make_unique<LinAlgMatmulMicroKernelPass>();
}

/// Create a pass to optimize LinAlg transposeOp - follow in HPTT paper
/// HPTT: A High-Performance Tensor Transposition C++ Library
/// https://arxiv.org/abs/1704.04374
std::unique_ptr<mlir::Pass> mlir::comet::createOptDenseTransposePass(uint64_t tile_size,
                                                                     bool seperate_tiles)
{
  comet_debug() << "LinAlgTransforms createOptDenseTransposePass\n";
  return std::make_unique<OptDenseTransposePass>(tile_size, seperate_tiles);
}
