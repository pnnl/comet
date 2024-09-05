
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Analysis)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Analysis)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Analysis)
add_public_tablegen_target(COMETAnalysisPassIncGen)
