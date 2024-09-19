add_llvm_library(COMETGpuToTriton
    GpuToTritonConversion.cpp
    GpuToTritonPass.cpp

    DEPENDS
    GpuToTritonConversionPassIncGen

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
