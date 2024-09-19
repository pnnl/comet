add_llvm_library(COMETTritonToCuda
    # GpuToTritonConversion.cpp
    TritonToCudaPass.cpp

    DEPENDS
    TritonDeviceToCudaConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    NVGPUIR
)
