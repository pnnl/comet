add_mlir_conversion_library(COMETIndexTreeToSCF
  IndexTreeToSCF.cpp
  AbstractLoopOp.cpp
  SymbolicDomainConversion.cpp
  IndexTreeConversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${COMET_MAIN_INCLUDE_DIR}/comet/Conversion/IndexTreeToSCF

  DEPENDS
  COMETConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIndexDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  )
