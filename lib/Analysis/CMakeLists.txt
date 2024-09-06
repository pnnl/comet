add_llvm_library(COMETAnalysis
  MemoryAliasAnalysis.cpp
  MemoryAccessAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Analysis/
  )


add_dependencies(
  COMETAnalysis
  COMETAnalysisPassIncGen
  MLIRAnalysis
  MLIRSupport
  )

target_link_libraries(COMETAnalysis MLIRIR)

