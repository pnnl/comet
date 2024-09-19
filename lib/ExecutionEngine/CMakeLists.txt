if(ENABLE_GPU_TARGET)
find_package(CUDAToolkit REQUIRED)
endif()

set(SOURCES
  blis_interface.cpp
  StatUtils.cpp
  SparseUtils.cpp
  TransposeUtils.cpp
  
)

if(ENABLE_GPU_TARGET)
  set(SOURCES
    ${SOURCES}
    GpuUtils.cpp
  )
endif()

add_llvm_library(comet_runner_utils
  PARTIAL_SOURCES_INTENDED
  SHARED
  ${SOURCES}
)

target_compile_definitions(comet_runner_utils PRIVATE comet_runner_utils_EXPORTS comet_blis_interface_EXPORTS)
set(LIBS
  ${BLAS_LIBRARIES} 
  LLVMSupport
)

if(ENABLE_GPU_TARGET)
set(LIBS
  ${LIBS}
  CUDA::cuda_driver
  )
endif()
target_link_libraries(comet_runner_utils ${LIBS})