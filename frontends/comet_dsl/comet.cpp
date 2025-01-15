//===- comet.cpp - The COMET Compiler ----===//
//
/// Copyright 2022 Battelle Memorial Institute
///
/// Redistribution and use in source and binary forms, with or without modification,
/// are permitted provided that the following conditions are met:
///
/// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
/// and the following disclaimer.
///
/// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
/// and the following disclaimer in the documentation and/or other materials provided with the distribution.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
/// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
/// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
/// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
/// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
/// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///
/// =============================================================================
///
/// This file implements the entry point for the COMET compiler.
///
//===----------------------------------------------------------------------===//

#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/TensorAlgebra/Passes.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/IndexTree/Passes.h"

#include "comet/Conversion/Passes.h"

#include "MLIRGen.h"
#include "Parser.h"

#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"


#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
// #include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"

#include "mlir/IR/Verifier.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#ifdef ENABLE_GPU_TARGET
#include "comet/Conversion/ParallelLoopsToGpu/ParallelLoopsToGpu.h"
#include "comet/Conversion/GpuToTriton/GpuToTritonPass.h"
#include "comet/Conversion/TritonToCuda/TritonToCudaPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#endif

#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/Passes.h"
// #ifdef ENABLE_GPU_TARGET
// #include "comet/TritonConfig.h"
// #endif
#include <cstdio>
#include <cstdlib>
#include <string>

int exec(const char* cmd, std::string& result) {
    char buffer[128];
    result = "";
    FILE* pipe = popen(cmd, "r");
    while (fgets(buffer, sizeof buffer, pipe) != NULL) {
        result += buffer;
    }
    return pclose(pipe);
}

mlir::OwningOpRef<mlir::ModuleOp> createModuleFromString(mlir::MLIRContext &context, const std::string &moduleStr) {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(moduleStr), llvm::SMLoc());
    
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module)
        llvm::errs() << "Error: failed to parse module from string.\n";
    
    return module;
}

using namespace mlir::tensorAlgebra;
using namespace mlir::indexTree;

#define DEBUG_TYPE "comet_dsl"
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input tensorAlgebra file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace
{
  enum InputType
  {
    TensorAlgebra,
    MLIR
  };
}

static cl::opt<enum InputType> inputType(
    "x", cl::init(TensorAlgebra), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(TensorAlgebra, "ta", "load the input file as a TensorAlgebra source.")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

/// List of TA compiler parameters

/// =============================================================================
/// Emit AST or Tensor Algebra dialect
/// =============================================================================
static cl::opt<bool> emitAST("emit-ast", cl::desc("Output the AST dump"));
static cl::opt<bool> emitTA("emit-ta", cl::desc("output the Tensor Algebra dialect dump"));
static cl::opt<bool> emitIT("emit-it", cl::desc("output the Index Tree dialect dump"));
static cl::opt<bool> emitLoops("emit-loops", cl::desc("output the SCF dialect dump"));
#ifdef ENABLE_GPU_TARGET
static cl::opt<bool> emitTriton("emit-triton", cl::desc("output the Triton dialect dump"));
#endif
static cl::opt<bool> emitLLVM("emit-llvm", cl::desc("output the LLVM dialect dump"));

/// =============================================================================
/// Godegen Target
/// =============================================================================

static cl::opt<TargetDevice> CodegenTarget("target", cl::init(CPU), cl::desc("Code generation target"), 
    cl::values(
      clEnumVal(CPU, "Codegen target is CPU")
      #ifdef ENABLE_GPU_TARGET
      , 
      clEnumVal(GPU, "Codegen target is GPU")
      #endif
    )
  );

#ifdef ENABLE_GPU_TARGET
static cl::opt<int> GPUBlockSizeX("gpu-block-x-size", cl::init(32), cl::desc("GPU Block size in X direction"));
static cl::opt<int> GPUBlockSizeY("gpu-block-y-size", cl::init(8), cl::desc("GPU Block size in Y direction"));
static cl::opt<int> GPUBlockSizeR("gpu-block-r-size", cl::init(32), cl::desc("GPU Block size in R direction"));
static cl::opt<int> GPUComputeCapability("gpu-compute-capability", cl::init(CUDA_COMPUTE_CAPABILITY), cl::desc("GPU compute capability"));
static cl::opt<int> GPUNumWarps("gpu-num-warps", cl::init(4), cl::desc("GPU number of warps"));
static cl::opt<int> GPUThreadsPerWarp("gpu-threads-per-warp", cl::init(32), cl::desc("GPU threads per warp"));
static cl::opt<int> GPUNumCTAs("gpu-num-ctas", cl::init(1), cl::desc("GPU num CTAs"));
static cl::opt<int> GPUNumStages("gpu-num-stages", cl::init(1), cl::desc("GPU num stages"));
#endif

/// =============================================================================
/// Optimization at the TA dialect (High-level optimizations) for tensor contraction operations
/// =============================================================================
static cl::opt<bool> OptMultiOpFactorization("opt-multiop-factorize",
                                             cl::desc("Multi operations factorization optimization"));

static cl::opt<bool> IsSelectBestPermTTGT("opt-bestperm-ttgt",
                                          cl::desc("Select the best index permutation for TTGT, otherwise the first appropriate permutation"));

static cl::opt<int> selectedPermNum("perm-num", cl::init(1),
                                    cl::ZeroOrMore, cl::desc("Select the permutation number to choose"));

/// =============================================================================
/// Operation based optimizations
/// =============================================================================
static cl::opt<bool> OptMatmulTiling("opt-matmul-tiling",
                                     cl::desc("Optimize LinAlg matmul operation with tiling"));

static cl::opt<bool> OptCallToMatMulMicroKernel("opt-matmul-mkernel",
                                                cl::desc("Replace the inner linalg.matmul that introduced after tiling with the blis micro kernel"));

static cl::opt<bool> OptDenseTransposeOp("opt-dense-transpose",
                                         cl::desc("Optimize transpose operation: optimal loop ordering and tiling"));

/// =============================================================================
/// Sparse kernel optimizations
/// =============================================================================
static cl::opt<bool> OptWorkspace("opt-comp-workspace", cl::init(false),
                                  cl::desc("Optimize sparse output code generation while reducing iteration space for nonzero elements"));

/// The details of the fusion algorithm can be found in the following paper.
/// ReACT: Redundancy-Aware Code Generation for Tensor Expressions.
/// Tong Zhou, Ruiqin Tian, Rizwan A Ashraf, Roberto Gioiosa, Gokcen Kestor, Vivek Sarkar.
/// 2022 31st International Conference on Parallel Architectures and Compilation Techniques (PACT). October 2022.
///  =============================================================================
///  Partial Fusion (ReACT Fusion) on Index Tree dialect
///  =============================================================================
static cl::opt<bool> OptKernelFusion("opt-fusion", cl::init(false),
                                     cl::desc("Output IT dialect after redundancy-aware fusion"));

/// =============================================================================
/// TTGT reformulation for tensor contraction operations
/// =============================================================================
static cl::opt<bool> IsLoweringTCtoTTGT("convert-tc-to-ttgt",
                                        cl::desc("Output IR after lowering dense tensor contractions operations through a TTGT approach"));

/// =============================================================================
/// Lowering TA operations to IT dialect
/// =============================================================================
static cl::opt<bool> IsLoweringtoIndexTree("convert-ta-to-it", /// Lower sparse/dense mult (semiring) and elemwise (monoid) ops to index-tree dialect
                                           cl::desc("Output IT dialect after processing dense sparse/dense mult and elemwise ops"));

/// =============================================================================
/// Lowering IT operations to loops
/// =============================================================================
static cl::opt<bool> IsLoweringtoSCF("convert-to-loops",
                                     cl::desc("Output SCF dialect after lowering all operations"));

#ifdef ENABLE_GPU_TARGET

/// =============================================================================
/// Lowering loops to Triton
/// =============================================================================
static cl::opt<bool> IsLoweringtoTriton("convert-to-triton",
                                     cl::desc("Output Triton dialect after lowering all operations"));
#endif

/// =============================================================================
/// Lowering to LLVM
/// =============================================================================
static cl::opt<bool> isLoweringToLLVM("convert-to-llvm",
                                      cl::desc("Output LLVM IR"));

/// =============================================================================
/// Utility functions
/// =============================================================================
static cl::opt<bool> IsPrintFlops("print-flops", cl::init(false),
                                  cl::desc("Print the flops per tensor contraction"));

/// Returns a Tensor Algebra AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<tensorAlgebra::ModuleAST> parseInputFile(llvm::StringRef filename)
{
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError())
  {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  tensorAlgebra::LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  tensorAlgebra::Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module)
{
  /// Handle '.ta' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir"))
  {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  /// Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError())
  {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  /// Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module)
  {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module)
{
#ifdef ENABLE_GPU_TARGET
  bool emitTriton_ = emitTriton && CodegenTarget == TargetDevice::GPU;
#else
  bool emitTriton_ = false;
#endif

  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  /// Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  /// Lower tensorAlgebra:FuncOp to func::FuncOp
  pm.addPass(mlir::comet::createFuncOpLoweringPass());

  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();

  /// Check to see if we are dumping to TA dialect.
  if (emitTA)
  {
    if (mlir::failed(pm.run(*module)))
      return 4;
    return 0;
  }

  ///  =============================================================================
  ///  High-level optimization at the TA dialect
  ///  Such as finding the optimal ordering of dense tensor contractions, or reformulating tensor contractions
  ///  operations via TTGT
  ///  =============================================================================
  if (OptMultiOpFactorization)
  {
    /// createFindOptimalTCFactorizationPass should be before lowering of input/output tensor declarations
    /// because this pass finds the optimal ordering of dense tensor multiplication
    /// operations before lowering them specific tc operations
    optPM.addPass(mlir::comet::createFindOptimalTCFactorizationPass());
  }

  ///  =============================================================================
  ///  Check if there are missing tensor declaration operations introduced by compound expressions.
  ///  If so, add a new tensor declaration to represent intermediate tensors
  ///  =============================================================================
  optPM.addPass(mlir::comet::createTensorAlgebraCheckImplicitTensorDeclPass());
  ///  =============================================================================
  /// Check to see if we are dumping to TA dialect.
  if (emitTA)
  {
    if (mlir::failed(pm.run(*module)))
      return 4;
    return 0;
  }

  /// ===================================================================================
  /// Lowering of TC (tensor contraction) operation to Index Tree dialect
  /// Also performs optimization at the Index Tree dialect
  /// ===================================================================================
  if (IsLoweringtoIndexTree || emitIT || emitLoops || emitTriton_ || emitLLVM)
  {
    /// Generate the index tree IR
    optPM.addPass(mlir::comet::createLowerTensorAlgebraToIndexTreePass(CodegenTarget));

    // Create new pass manager to optimize the index tree dialect
    optPM.addPass(mlir::comet::createIndexTreeDomainInferencePass());

    // if (OptKernelFusion)
    // {
    //   /// Apply partial fusion on index tree dialect for some compound expressions.
    //   optPM.addPass(mlir::comet::createIndexTreeKernelFusionPass());
    // }
  }

  /// =============================================================================

  /// =============================================================================
  /// Early lowering for the following passes to lower IndexTree and TA dialect to loops
  /// Early lowering pases based on the sparsity property of inputs/outputs
  /// =============================================================================

  /// Sparse input tensor declararion should be lowered before dense input tensor declaration
  /// sparse input tensor declaration lowering, also generate sparse_output_tensor declaration if needed
  /// input and output sparse tensor declaration lowering are distant and need different information
  optPM.addPass(mlir::comet::createSparseTensorDeclLoweringPass());
  optPM.addPass(mlir::comet::createDenseTensorDeclLoweringPass());
  optPM.addPass(mlir::comet::createSparseTempOutputTensorDeclLoweringPass());
  optPM.addPass(mlir::comet::createSparseOutputTensorDeclLoweringPass());
  optPM.addPass(mlir::comet::createTensorFillLoweringPass());
  optPM.addPass(mlir::comet::createDimOpLoweringPass());

  /// =============================================================================

  /// TTGT reformulation for dense tensor contraction operations
  if (IsLoweringTCtoTTGT)
  {
    /// Sparse input and dense input/output tensor declarations needed be lowered before for TTGT pass
    optPM.addPass(mlir::comet::createLoweringTTGTPass(IsSelectBestPermTTGT, selectedPermNum, IsPrintFlops));
  }

  // /// =============================================================================
  // /// Operation based optimizations
  // /// =============================================================================
  if (OptMatmulTiling)
  {
    optPM.addPass(mlir::comet::createLinAlgMatmulTilingPass());
  }

  if (OptCallToMatMulMicroKernel)
  {
    optPM.addPass(mlir::comet::createLinAlgMatmulMicroKernelPass());
  }

  /// =============================================================================
  /// Lowering all the operations to loops
  /// =============================================================================
  if (IsLoweringtoSCF || emitLoops || emitLLVM)
  { 
    /// =============================================================================
    /// Lowering of other operations such as transpose, sum, etc. to SCF dialect
    /// =============================================================================
    /// If it is a transpose of dense tensor, the rewrites rules replaces ta.transpose with linalg.copy.
    /// If it is a transpose of sparse tensor, it lowers the code to make a runtime call to specific sorting algorithm
    optPM.addPass(mlir::comet::createLowerTensorAlgebraToSCFPass());

    /// Concretize the domains of all the index variables
    optPM.addPass(mlir::comet::createIndexTreeDomainConcretizationPass());

    if (OptWorkspace) {
      /// Optimized workspace transformations, reduce iteration space for nonzero elements
      optPM.addPass(mlir::comet::createIndexTreeWorkspaceTransformationsPass());
    }

    optPM.addPass(mlir::comet::createIndexTreeSymbolicComputePass());

    /// Dump index tree dialect.
    if (emitIT)
    {
      if (mlir::failed(pm.run(*module)))
        return 4;
      return 0;
    }

    /// Finally lowering index tree to SCF dialect
    optPM.addPass(mlir::comet::createLowerIndexTreeToSCFPass());
    optPM.addPass(mlir::comet::createConvertSymbolicDomainsPass());
    optPM.addPass(mlir::comet::createSparseTensorConversionPass());
    optPM.addPass(mlir::comet::createIndexTreeInliningPass());
    optPM.addPass(mlir::createCanonicalizerPass());

    if (OptDenseTransposeOp) /// Optimize Dense Transpose operation
    {
      /// If it is a dense transpose ops, the rewrites rules replaces ta.transpose with linalg.transpose, then
      /// Create a pass to optimize LinAlg Copy Op - follow in HPTT paper
      /// HPTT: A High-Performance Tensor Transposition C++ Library
      /// https://arxiv.org/abs/1704.04374
      optPM.addPass(mlir::comet::createOptDenseTransposePass());
    }

    /// Dump scf dialect.
    if (emitLoops)
    {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      if (mlir::failed(pm.run(*module)))
        return 4;
      return 0;
    }
    ///  =============================================================================
  }



  // /// =============================================================================
  // /// Late lowering passes
  // /// =============================================================================
  // pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::comet::createTABufferizeFunc());
  pm.addPass(mlir::createCanonicalizerPass());

  mlir::bufferization::OneShotBufferizationOptions opts;
  opts.allowUnknownOps = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));

  mlir::OpPassManager &late_lowering_pm = pm.nest<mlir::func::FuncOp>();
  late_lowering_pm.addPass(mlir::comet::createSTCRemoveDeadOpsPass());
  late_lowering_pm.addPass(mlir::comet::createLateLoweringPass());
  
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

#ifdef ENABLE_GPU_TARGET
  if (CodegenTarget == TargetDevice::GPU && (emitTriton_ || emitLLVM || IsLoweringtoTriton))
  {
    pm.addNestedPass<mlir::func::FuncOp>(mlir::comet::createConvertParallelLoopsToGpuPass(GPUBlockSizeX, GPUBlockSizeY, GPUBlockSizeR));
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::createParallelLoopToGpuPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::comet::createConvertGpuKernelToTritonPass());

    if (emitTriton_)
    {
      if (mlir::failed(pm.run(*module)))
        return 4;
      return 0;
    }

  }
#endif
  pm.addPass(mlir::createCanonicalizerPass());

  /// =============================================================================

  if (isLoweringToLLVM || emitLLVM)
  {
#ifdef ENABLE_GPU_TARGET
    if (CodegenTarget == GPU)
    {
      pm.addPass(mlir::comet::createLowerTritonDeviceToCudaPass(GPUNumWarps, GPUThreadsPerWarp, GPUNumCTAs, GPUNumStages, GPUComputeCapability));
      pm.addPass(mlir::comet::createLowerGpuHostToCudaPass());
    }
#endif

    optPM.addPass(mlir::createCanonicalizerPass());
    /// Blanket-convert any remaining high-level vector ops to loops if any remain.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToSCFPass());
    /// Blanket-convert any remaining linalg ops to loops if any remain.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    /// Blanket-convert any remaining affine ops if any remain.
    pm.addPass(mlir::createLowerAffinePass());
    /// Convert SCF to CF (always needed).
    pm.addPass(mlir::createConvertSCFToCFPass());
    /// Sprinkle some cleanups.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    /// Convert vector to LLVM (always needed).
    pm.addPass(mlir::createConvertVectorToLLVMPass()); // TODO: add more options on a per-need basis.
    //// Convert Math to LLVM (always needed).
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToLLVMPass());
    /// Expand complicated MemRef operations before lowering them.
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    /// The expansion may create affine expressions. Get rid of them.
    pm.addPass(mlir::createLowerAffinePass());
    /// Convert MemRef to LLVM (always needed).
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    /// Convert Func to LLVM (always needed).
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    /// Convert Index to LLVM (always needed).
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    /// Convert remaining unrealized_casts (always needed).
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(*module)))
      return 4;
    return 0;
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int dumpAST()
{
  if (inputType == InputType::MLIR)
  {
    llvm::errs() << "Can't dump a TensorAlgebra AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv)
{

  mlir::MLIRContext context;
  mlir::registerAllDialects(context);

  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "Tensor Algebra compiler\n");

  if (emitAST)
    return dumpAST();

  /// If we aren't dumping the AST, then we are compiling with/to MLIR.
  /// Register our Dialect with MLIR.
#ifdef ENABLE_GPU_TARGET
  context.loadDialect<mlir::triton::TritonDialect>();
  registerLLVMDialectTranslation(context);
  registerLLVMDialectTranslation(context);
  registerBuiltinDialectTranslation(context);
  registerNVVMDialectTranslation(context);
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
#endif
  context.loadDialect<mlir::tensorAlgebra::TADialect>();
  context.loadDialect<mlir::indexTree::IndexTreeDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<mlir::bufferization::BufferizationDialect>();
  context.loadDialect<mlir::index::IndexDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;

  if (int error = loadAndProcessMLIR(context, module))
    return error;

  /// If we aren't exporting to non-mlir, then we are done.
  module->dump();
  return 0;
}