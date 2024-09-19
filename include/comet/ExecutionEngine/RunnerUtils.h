//===- RunnerUtils.h - Utils for MLIR execution -----------------===//
//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to debug structured MLIR
// types at runtime. Entities in this file may not be compatible with targets
// without a C++ runtime.
//
//===----------------------------------------------------------------------===//

#ifndef COMET_EXECUTIONENGINE_RUNNERUTILS_H_
#define COMET_EXECUTIONENGINE_RUNNERUTILS_H_

#ifdef _WIN32
#ifndef COMET_RUNNERUTILS_EXPORT
#ifdef comet_runner_utils_EXPORTS
#define COMET_RUNNERUTILS_EXPORT __declspec(dllexport)
#else
#define COMET_RUNNERUTILS_EXPORT __declspec(dllimport)
#endif // comet_runner_utils_EXPORTS
#endif // COMET_RUNNERUTILS_EXPORT
#else
#define COMET_RUNNERUTILS_EXPORT
#endif // _WIN32

#include <assert.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include <cmath>
#include <cstring>

#include <iterator>
#include <string>
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <list>
#include <stdio.h>
#include <string.h>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "comet/ExecutionEngine/blis_interface.h"

using namespace std;

enum SparseFormatAttribute
{
    Dense,
    Compressed_unique,
    Compressed_nonunique,
    singleton
};

/**************************************/
/// Currently exposed C API.
/**************************************/

/// Read matrices and tensors
extern "C" COMET_RUNNERUTILS_EXPORT void read_input_sizes_2D_f32(int32_t fileID,
                                                                 int32_t A1format, int32_t A1_tile_format,
                                                                 int32_t A2format, int32_t A2_tile_format,
                                                                 int A1pos_rank, void *A1pos_ptr, int32_t readMode);

extern "C" COMET_RUNNERUTILS_EXPORT void read_input_2D_f32(int32_t fileID,
                                                           int32_t A1format, int32_t A1_tile_format,
                                                           int32_t A2format, int32_t A2_tile_format,
                                                           int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                                           int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                                           int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                                           int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                                           int Aval_rank, void *Aval_ptr, int32_t readMode);

extern "C" COMET_RUNNERUTILS_EXPORT void read_input_sizes_2D_f64(int32_t fileID,
                                                                 int32_t A1format, int32_t A1_tile_format,
                                                                 int32_t A2format, int32_t A2_tile_format,
                                                                 int A1pos_rank, void *A1pos_ptr, int32_t readMode);

extern "C" COMET_RUNNERUTILS_EXPORT void read_input_2D_f64(int32_t fileID,
                                                           int32_t A1format, int32_t A1_tile_format,
                                                           int32_t A2format, int32_t A2_tile_format,
                                                           int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                                           int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                                           int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                                           int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                                           int Aval_rank, void *Aval_ptr, int32_t readMode);

extern "C" COMET_RUNNERUTILS_EXPORT void read_input_sizes_3D_f32(int32_t fileID,
                                                                 int32_t A1format, int32_t A1_tile_format,
                                                                 int32_t A2format, int32_t A2_tile_format,
                                                                 int32_t A3format, int32_t A3_tile_format,
                                                                 int A1pos_rank, void *A1pos_ptr, int32_t readMode);

extern "C" COMET_RUNNERUTILS_EXPORT void read_input_3D_f32(int32_t fileID,
                                                           int32_t A1format, int32_t A1_tile_format,
                                                           int32_t A2format, int32_t A2_tile_format,
                                                           int32_t A3format, int32_t A3_tile_format,
                                                           int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                                           int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                                           int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                                           int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                                           int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                                           int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                                                           int Aval_rank, void *Aval_ptr, int32_t readMode);

extern "C" COMET_RUNNERUTILS_EXPORT void read_input_sizes_3D_f64(int32_t fileID,
                                                                 int32_t A1format, int32_t A1_tile_format,
                                                                 int32_t A2format, int32_t A2_tile_format,
                                                                 int32_t A3format, int32_t A3_tile_format,
                                                                 int A1pos_rank, void *A1pos_ptr, int32_t readMode);

extern "C" COMET_RUNNERUTILS_EXPORT void read_input_3D_f64(int32_t fileID,
                                                           int32_t A1format, int32_t A1_tile_format,
                                                           int32_t A2format, int32_t A2_tile_format,
                                                           int32_t A3format, int32_t A3_tile_format,
                                                           int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                                           int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                                           int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                                           int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                                           int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                                           int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                                                           int Aval_rank, void *Aval_ptr, int32_t readMode);

// Transpose operations
extern "C" COMET_RUNNERUTILS_EXPORT void transpose_2D_f32(int32_t A1format, int32_t A1tile_format, int32_t A2format, int32_t A2tile_format,
                                                          int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                                          int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                                          int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                                          int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                                          int Aval_rank, void *Aval_ptr,
                                                          int32_t B1format, int32_t B1tile_format, int32_t B2format, int32_t B2tile_format,
                                                          int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                                                          int B1tile_pos_rank, void *B1tile_pos_ptr, int B1tile_crd_rank, void *B1tile_crd_ptr,
                                                          int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                                                          int B2tile_pos_rank, void *B2tile_pos_ptr, int B2tile_crd_rank, void *B2tile_crd_ptr,
                                                          int Bval_rank, void *Bval_ptr,
                                                          int sizes_rank, void *sizes_ptr);

extern "C" COMET_RUNNERUTILS_EXPORT void transpose_3D_f32(int32_t input_permutation, int32_t output_permutation,
                                                          int32_t A1format, int32_t A1tile_format,
                                                          int32_t A2format, int32_t A2tile_format,
                                                          int32_t A3format, int32_t A3tile_format,
                                                          int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                                          int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                                          int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                                          int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                                          int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                                          int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                                                          int Aval_rank, void *Aval_ptr,
                                                          int32_t B1format, int32_t B1tile_format,
                                                          int32_t B2format, int32_t B2tile_format,
                                                          int32_t B3format, int32_t B3tile_format,
                                                          int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                                                          int B1tile_pos_rank, void *B1tile_pos_ptr, int B1tile_crd_rank, void *B1tile_crd_ptr,
                                                          int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                                                          int B2tile_pos_rank, void *B2tile_pos_ptr, int B2tile_crd_rank, void *B2tile_crd_ptr,
                                                          int B3pos_rank, void *B3pos_ptr, int B3crd_rank, void *B3crd_ptr,
                                                          int B3tile_pos_rank, void *B3tile_pos_ptr, int B3tile_crd_rank, void *B3tile_crd_ptr,
                                                          int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr);

extern "C" COMET_RUNNERUTILS_EXPORT void transpose_2D_f64(int32_t A1format, int32_t A1tile_format, int32_t A2format, int32_t A2tile_format,
                                                          int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                                          int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                                          int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                                          int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                                          int Aval_rank, void *Aval_ptr,
                                                          int32_t B1format, int32_t B1tile_format, int32_t B2format, int32_t B2tile_format,
                                                          int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                                                          int B1tile_pos_rank, void *B1tile_pos_ptr, int B1tile_crd_rank, void *B1tile_crd_ptr,
                                                          int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                                                          int B2tile_pos_rank, void *B2tile_pos_ptr, int B2tile_crd_rank, void *B2tile_crd_ptr,
                                                          int Bval_rank, void *Bval_ptr,
                                                          int sizes_rank, void *sizes_ptr);

extern "C" COMET_RUNNERUTILS_EXPORT void transpose_3D_f64(int32_t input_permutation, int32_t output_permutation,
                                                          int32_t A1format, int32_t A1tile_format,
                                                          int32_t A2format, int32_t A2tile_format,
                                                          int32_t A3format, int32_t A3tile_format,
                                                          int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                                          int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                                          int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                                          int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                                          int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                                          int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                                                          int Aval_rank, void *Aval_ptr,
                                                          int32_t B1format, int32_t B1tile_format,
                                                          int32_t B2format, int32_t B2tile_format,
                                                          int32_t B3format, int32_t B3tile_format,
                                                          int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                                                          int B1tile_pos_rank, void *B1tile_pos_ptr, int B1tile_crd_rank, void *B1tile_crd_ptr,
                                                          int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                                                          int B2tile_pos_rank, void *B2tile_pos_ptr, int B2tile_crd_rank, void *B2tile_crd_ptr,
                                                          int B3pos_rank, void *B3pos_ptr, int B3crd_rank, void *B3crd_ptr,
                                                          int B3tile_pos_rank, void *B3tile_pos_ptr, int B3tile_crd_rank, void *B3tile_crd_ptr,
                                                          int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr);

///===----------------------------------------------------------------------===///
/// Small runtime support library for timing execution, printing elapse time, printing GFLOPS
//===----------------------------------------------------------------------===///
extern "C" COMET_RUNNERUTILS_EXPORT double getTime();
extern "C" COMET_RUNNERUTILS_EXPORT void printElapsedTime(double stime, double etime);
extern "C" COMET_RUNNERUTILS_EXPORT void print_flops(double flops);

///===----------------------------------------------------------------------===///
/// Small runtime support library for printing output scalar and tensors
///===----------------------------------------------------------------------===///

template <typename T>
void printData(std::ostream &os, T *base, int64_t dim,
               int64_t rank, int64_t offset,
               const int64_t *sizes, const int64_t *strides)
{
    if (dim == 0)
    {
        os << base[offset] << ",";
        return;
    }
    for (unsigned i = 0; i < sizes[0]; i++)
    {
        printData(os, base, dim - 1, rank, offset + i * strides[0], sizes + 1,
                  strides + 1);
    }
}

template <typename T>
void cometPrint(const DynamicMemRefType<T> &M)
{
    if (M.sizes[0] <= 0)
    {
        return;
    }
    std::cout << "data = " << std::endl;
    printData(std::cout, M.data, M.rank, M.rank, M.offset,
              M.sizes, M.strides);
    std::cout << "\n";
}

template <typename T>
void cometPrintMemRef(UnrankedMemRefType<T> &M)
{
    cometPrint(DynamicMemRefType<T>(M));
}

///===----------------------------------------------------------------------===///
/// Small runtime support library for memset for tensors
//===----------------------------------------------------------------------===///

template <typename T>
void RTmemset(const DynamicMemRefType<T> &M)
{
    unsigned num_elements = 1;
    for (unsigned i = 0; i < M.rank; i++)
    {
        num_elements = num_elements * M.sizes[0];
    }
    memset(M.data, 0, sizeof(T) * num_elements);
}

template <typename T>
void cometMemset(UnrankedMemRefType<T> &M)
{
    RTmemset(DynamicMemRefType<T>(M));
}

///===----------------------------------------------------------------------===///
/// Small runtime support library for std::sort indices
///===----------------------------------------------------------------------===///

template <typename T>
void RTSortIndex(const DynamicMemRefType<T> &M, int64_t index_first, int64_t index_last)
{
    std::sort(M.data + index_first, M.data + index_last);
}

template <typename T>
void cometSortIndex(UnrankedMemRefType<T> &M, int64_t index_first, int64_t index_last)
{
    RTSortIndex(DynamicMemRefType<T>(M), index_first, index_last);
}

#ifdef ENABLE_GPU_TARGET
extern "C" COMET_RUNNERUTILS_EXPORT void cudaSetModuleImage(char* ptx);
extern "C" COMET_RUNNERUTILS_EXPORT int64_t cudaMallocF64(int64_t size);
extern "C" COMET_RUNNERUTILS_EXPORT int64_t cudaMallocF32(int64_t size);
extern "C" COMET_RUNNERUTILS_EXPORT int64_t cudaMallocI64(int64_t size);
extern "C" COMET_RUNNERUTILS_EXPORT int64_t cudaMallocI32(int64_t size);
extern "C" COMET_RUNNERUTILS_EXPORT void cudaFree(int64_t ptr);
extern "C" COMET_RUNNERUTILS_EXPORT void cudaMemcpyI64(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction);
extern "C" COMET_RUNNERUTILS_EXPORT void cudaMemcpyF64(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction);
extern "C" COMET_RUNNERUTILS_EXPORT void cudaMemcpyI32(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction);
extern "C" COMET_RUNNERUTILS_EXPORT void cudaMemcpyF32(int64_t device, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, int64_t direction);
extern "C" COMET_RUNNERUTILS_EXPORT void cudaLaunchKernel(int64_t realblocksX, int64_t realblocksY, int64_t realblocksZ, int64_t tritonBlockX, int64_t tritonBlockY, int64_t tritonBlockZ, void* ptr, void* aligned_ptr, int64_t offset, int64_t size, int64_t stride, char* kernel, int64_t kernel_name_size, int64_t sharedMem, int64_t numWraps, int64_t threadsPerWarp);
#endif


#endif /// COMET_EXECUTIONENGINE_RUNNERUTILS_H_
