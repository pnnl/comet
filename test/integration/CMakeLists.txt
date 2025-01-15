set(COMET_INTEGRATION_TEST_DEPENDS
  FileCheck count not
  comet-opt
  mlir-opt
  mlir-cpu-runner
  )



set(COMET_INTEGRATION_TIMEOUT 60) # Set a 60s timeout on individual tests.

# Passed to lit.site.cfg.py.in to set up the path where to find the libraries.
set(MLIR_UTILITY_LIBRARY_DIR  ${LLVM_BUILD_LIBRARY_DIR})
message(STATUS "Using MLIR_UTILITY_LIBRARIES: ${MLIR_UTILITY_LIBRARY_DIR}")

set(COMET_UTILITY_LIBRARY_DIR  ${LLVM_LIBRARY_OUTPUT_INTDIR})
message(STATUS "Using COMET_UTILITY_LIBRARIES: ${COMET_UTILITY_LIBRARY_DIR}")

set(COMET_INTEGRATION_TEST_DATA_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/data/)
message(STATUS "Using COMET_INTEGRATION_TEST_DATA_DIR: ${COMET_INTEGRATION_TEST_DATA_DIR}")

if(ENABLE_GPU_TARGET)
set(COMET_ENABLE_GPU  True)
message(STATUS "Using COMET_ENABLE_GPU: ${COMET_ENABLE_GPU}")
else()
set(COMET_ENABLE_GPU  False)
message(STATUS "Using COMET_ENABLE_GPU: ${COMET_ENABLE_GPU}")
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  INTEGRATION_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-comet-integration "Running the COMET integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${COMET_INTEGRATION_TEST_DEPENDS}
  )
set_target_properties(check-comet-integration PROPERTIES FOLDER "IntegrationTests")

add_lit_testsuites(COMET_INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${COMET_INTEGRATION_TEST_DEPS}
)