#include <cstdint>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <algorithm>

#define CU_CHECK(call) \
do { \
    CUresult res = call; \
    if (res != CUDA_SUCCESS) { \
        fprintf(stderr, "CU Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d\n", res); \
        exit(1); \
    } \
} while (0)

CUcontext cuContext = NULL;
CUmodule cuModule = NULL;
char* moduleImg = NULL;

void initCudaCtx()
{
    if(!cuContext)
    {
        CU_CHECK(cuInit(0));
        CU_CHECK(cuCtxCreate(&cuContext, 0, 0));
    }

    if(!cuModule && moduleImg)
    {
        // printf("PTX: \n %s\n", moduleImg);
        CU_CHECK(cuModuleLoadData(&cuModule, moduleImg));       
    }
}


template<typename T>
int64_t cudaMalloc(int64_t size) {
    initCudaCtx();
    CUdeviceptr device_ptr;
    CU_CHECK(cuMemAlloc(&device_ptr,   size * sizeof(T)));

    // return (int64_t)malloc(size * sizeof(T));
    return device_ptr;
}

extern "C" __attribute__((visibility("default"))) void cudaSetModuleImage(char* ptx)
{
    // printf("Called cudaSetModuleImage");

    if (moduleImg == NULL) 
    {
        // printf("Setting PTX");
        moduleImg = ptx;
    }
}

extern "C" __attribute__((visibility("default"))) int64_t cudaMallocF64(int64_t size)  {
    // printf("Allocating memory of size: %ld\n", size);
    return cudaMalloc<double>(size);
}

extern "C" __attribute__((visibility("default"))) int64_t cudaMallocF32(int64_t size)  {
    // printf("Allocating memory of size: %ld\n", size);
    return cudaMalloc<float>(size);
}

extern "C" __attribute__((visibility("default"))) int64_t cudaMallocI64(int64_t size)  {
    // printf("Allocating memory of size: %ld\n", size);
    return cudaMalloc<int64_t>(size);
}

extern "C" __attribute__((visibility("default"))) int64_t cudaMallocI32(int64_t size)  {
    // printf("Allocating memory of size: %ld\n", size);
    return cudaMalloc<int32_t>(size);
}

extern "C" __attribute__((visibility("default"))) void cudaFree(int64_t ptr) {
    initCudaCtx();

    // printf("Freeing memory\n");
    CU_CHECK(cuMemFree(ptr));
    // free((void*)ptr);
}

template<typename T>
void cudaMemcpy(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction) {
    initCudaCtx();

    // printf("Memcpy memory of size: %ld\n", size);

    if(direction == 0) // Host to Device
    {
        CU_CHECK(cuMemcpyHtoD(device, aligned_ptr, size * sizeof(T)));
        // memcpy((void*)device, aligned_ptr, size * sizeof(T));
    }
    else  // Device to Host
    {
        CU_CHECK(cuMemcpyDtoH(aligned_ptr, device, size * sizeof(T)));
        // memcpy(aligned_ptr, (void*)device, size * sizeof(T));
    }
}


extern "C" __attribute__((visibility("default"))) void cudaMemcpyI64(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction) {
    cudaMemcpy<int64_t>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

extern "C" __attribute__((visibility("default"))) void cudaMemcpyF64(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction) {
    cudaMemcpy<double>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

extern "C" __attribute__((visibility("default"))) void cudaMemcpyI32(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction) {
    cudaMemcpy<int32_t>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

extern "C" __attribute__((visibility("default"))) void cudaMemcpyF32(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction) {
    cudaMemcpy<float>(device, ptr, aligned_ptr, offset, size, stride, direction);
}

struct UnrankedMemRef {
    
    void* ptr;
    void* aligned_ptr;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

const int64_t MAX_NUM_BLOCKS_X = 2147483647;
const int64_t MAX_NUM_BLOCKS_Y = 65535;
const int64_t MAX_NUM_BLOCKS_Z = 65535;

extern "C" __attribute__((visibility("default"))) void cudaLaunchKernel(int64_t realblocksX, int64_t realblocksY, int64_t realblocksZ, int64_t tritonBlockX, int64_t tritonBlockY, int64_t tritonBlockZ, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, char* kernel, int64_t kernel_name_size, int64_t sharedMem, int64_t numWraps, int64_t threadsPerWarp) 
{
    initCudaCtx();
    CUfunction cuFunction = NULL;
    unsigned blocksPerGridX = std::min(realblocksX, MAX_NUM_BLOCKS_X);
    unsigned blocksPerGridY = std::min(realblocksY, MAX_NUM_BLOCKS_Y);
    unsigned blocksPerGridZ = std::min(realblocksZ, MAX_NUM_BLOCKS_Y);
    char* name_with_suffix = (char*)malloc(kernel_name_size+1);
    memcpy(name_with_suffix, kernel, kernel_name_size);
    name_with_suffix[kernel_name_size] = '\0';
    CU_CHECK(cuModuleGetFunction(&cuFunction, cuModule, name_with_suffix));
    // printf("Kernel name: %s\n", name_with_suffix);
    void** cast_args = (void**)aligned_ptr;
    // for(int i = 0; i < size; i++)
    // {
    //     printf("%p\n", cast_args[i]);
    // }

    free(name_with_suffix);

    CU_CHECK(cuLaunchKernel(cuFunction, blocksPerGridX, blocksPerGridY, blocksPerGridZ, numWraps* threadsPerWarp, 1, 1, sharedMem, 0, cast_args, 0));
}