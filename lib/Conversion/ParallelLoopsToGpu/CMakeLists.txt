add_llvm_library(COMETParallelLoopsToGpu
    ParallelLoopsToGpu.cpp

    DEPENDS
    ParallelLoopsConversionPassIncGen

    # LINK_LIBS
    # MLIRPASS
) 