//===- StatUtils.cpp - Utils to collect some statistics -----------------===//
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
//===----------------------------------------------------------------------===//

#include "comet/ExecutionEngine/RunnerUtils.h"

#include <assert.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <cstdio>
#include <limits>
#include <iomanip>
#include <stdio.h>

#include <random>

//===----------------------------------------------------------------------===//
/// Small runtime support library for print some statistics.
//===----------------------------------------------------------------------===//
/// Returns the number of seconds since Epoch 1970-01-01 00:00:00 +0000 (UTC).
extern "C" double getTime()
{
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, NULL);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  fprintf(stderr, "Timing utility not implemented on Windows\n");
  return 0.0;
#endif // _WIN32
}

extern "C" void printElapsedTime(double stime, double etime)
{
  fprintf(stdout, "ELAPSED_TIME = %lf\n", etime - stime);
}

extern "C" void print_f64(double val)
{
  fprintf(stdout, "VAL = %lf\n", val);
}

extern "C" void print_f32(float val)
{
  fprintf(stdout, "VAL = %f\n", val);
}

extern "C" void print_i32(int32_t t)
{
  std::cout << t << std::endl;
}

extern "C" void print_range(int32_t t, int32_t e)
{
  std::cout << "range: (" << t << ", " << e << ")" << std::endl;
}

extern "C" void print_space()
{
  std::cout << " " << std::endl;
}

//===----------------------------------------------------------------------===//
/// Small runtime support library for printing output scalar and tensors
//===----------------------------------------------------------------------===//
extern "C" void _mlir_ciface_comet_print_memref_f64(UnrankedMemRefType<double> *M)
{
  cometPrintMemRef(*M);
}

extern "C" void _mlir_ciface_comet_print_memref_f32(UnrankedMemRefType<float> *M)
{
  cometPrintMemRef(*M);
}

extern "C" void _mlir_ciface_comet_print_memref_i64(UnrankedMemRefType<int64_t> *M)
{
  cometPrintMemRef(*M);
}

extern "C" void _mlir_ciface_comet_print_memref_i32(UnrankedMemRefType<int32_t> *M)
{
  cometPrintMemRef(*M);
}

extern "C" void comet_print_memref_f64(int64_t rank, void *ptr)
{
  UnrankedMemRefType<double> descriptor = {rank, ptr};
  _mlir_ciface_comet_print_memref_f64(&descriptor);
}

extern "C" void comet_print_memref_f32(int64_t rank, void *ptr)
{
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  _mlir_ciface_comet_print_memref_f32(&descriptor);
}

extern "C" void comet_print_memref_i64(int64_t rank, void *ptr)
{
  UnrankedMemRefType<int64_t> descriptor = {rank, ptr};
  _mlir_ciface_comet_print_memref_i64(&descriptor);
}

extern "C" void comet_print_memref_i32(int64_t rank, void *ptr)
{
  UnrankedMemRefType<int32_t> descriptor = {rank, ptr};
  _mlir_ciface_comet_print_memref_i32(&descriptor);
}

extern "C" void comet_print_memref_index(int64_t rank, void *ptr)
{
  UnrankedMemRefType<int64_t> descriptor = {rank, ptr};
  _mlir_ciface_comet_print_memref_i64(&descriptor);
}

//===----------------------------------------------------------------------===//
/// Small runtime support library for memset
//===----------------------------------------------------------------------===//
extern "C" void _mlir_ciface_comet_memset_f64(UnrankedMemRefType<double> *M)
{
  cometMemset(*M);
}

extern "C" void _mlir_ciface_comet_memset_i64(UnrankedMemRefType<int64_t> *M)
{
  cometMemset(*M);
}

extern "C" void _mlir_ciface_comet_memset_i1(UnrankedMemRefType<bool> *M)
{
  cometMemset(*M);
}

extern "C" void comet_memset_f64(int64_t rank, void *ptr)
{
  UnrankedMemRefType<double> descriptor = {rank, ptr};
  _mlir_ciface_comet_memset_f64(&descriptor);
}

extern "C" void comet_memset_i64(int64_t rank, void *ptr)
{
  UnrankedMemRefType<int64_t> descriptor = {rank, ptr};
  _mlir_ciface_comet_memset_i64(&descriptor);
}

extern "C" void comet_memset_i1(int64_t rank, void *ptr)
{
  UnrankedMemRefType<bool> descriptor = {rank, ptr};
  _mlir_ciface_comet_memset_i1(&descriptor);
}
