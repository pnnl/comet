set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(comet-opt
  comet.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
)

include_directories(include/)
if(ENABLE_GPU_TARGET)
  include_directories("${TRITON_PATH}")
  include_directories("${TRITON_PATH}/include")
  include_directories("${TRITON_BUILD_PATH}/include")
  get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
endif()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

llvm_update_compile_flags(comet-opt)

set(LIBS
MLIRAnalysis
MLIRIR
MLIRParser
MLIRPass
MLIRTransforms
COMETUtils
COMETTensorAlgebraDialect
COMETIndexTreeDialect
COMETIndexTreeToSCF
)

if(ENABLE_GPU_TARGET)
  set(LIBS
    ${LIBS}
    COMETParallelLoopsToGpu
    COMETGpuToTriton
    COMETTritonToCuda
    ${triton_libs}
  )
endif()

target_link_libraries(comet-opt 
    PRIVATE MLIRIR 
    ${LIBS} 
    ${dialect_libs} 
    ${conversion_libs}
    )