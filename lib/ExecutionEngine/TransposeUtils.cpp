
//===- TransposeUtils.cpp - Utility functions for Tensor transpose -----------------===//
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
// This file includes runtime functions for sparse tensor
// transpose operations.
//
//===----------------------------------------------------------------------===//

#include "comet/ExecutionEngine/RunnerUtils.h"

#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>

/// Parallel sorting algorithms
#include <algorithm>

using namespace std;

enum SortingOption
{
  NO_SORT = 1, /// default: re-traverse instead of sorting
  SEQ_QSORT = 2,
  PAR_QSORT = 3,
  RADIX_BUCKET = 4,
  COUNT_RADIX = 5,
  COUNT_QUICK = 6
};

void getSortType(int &selected_sort_type)
{
  selected_sort_type = -1;
  if (getenv("SORT_TYPE"))
  {
    char *sort_type = getenv("SORT_TYPE");
    if (strcmp(sort_type, "NO_SORT") == 0)
      selected_sort_type = NO_SORT;
    else if (strcmp(sort_type, "SEQ_QSORT") == 0)
      selected_sort_type = SEQ_QSORT;
    else if (strcmp(sort_type, "PAR_QSORT") == 0)
      selected_sort_type = PAR_QSORT;
    else if (strcmp(sort_type, "RADIX_BUCKET") == 0)
      selected_sort_type = RADIX_BUCKET;
    else if (strcmp(sort_type, "COUNT_RADIX") == 0)
      selected_sort_type = COUNT_RADIX;
    else if (strcmp(sort_type, "COUNT_QUICK") == 0)
      selected_sort_type = COUNT_QUICK;
    else
      assert(selected_sort_type != -1 && "\n\nError: SORT_TYPE environmental variable for sparse transpose is not recognized!\n"
                                         "\tValid Options are: NO_SORT, SEQ_QSORT, PAR_QSORT, RADIX_BUCKET, COUNT_RADIX, COUNT_QUICK.\n\n\n");
  }
  else
  {
    selected_sort_type = NO_SORT; /// default
  }
}

struct coo_t
{
  vector<int32_t> coords;
  double val;
};

struct bucket
{
  int left;
  int right;
};

//===----------------------------------------------------------------------===//
/// Different sorting algorithms
//===----------------------------------------------------------------------===//

bool sort_compare_coords(struct coo_t p, struct coo_t q)
{
  /// Get the values at given addresses
  struct coo_t l = p;
  struct coo_t r = q;

  for (unsigned long i = 0; i < l.coords.size(); ++i)
  {
    if (l.coords[i] < r.coords[i])
    {
      return true;
    }
    else if (l.coords[i] > r.coords[i])
    {
      return false;
    }
  }
  return false;
}

int qsort_compare_coords(const void *p, const void *q)
{
  /// Get the values at given addresses
  const struct coo_t l = *(const struct coo_t *)p;
  const struct coo_t r = *(const struct coo_t *)q;

  for (unsigned long i = 0; i < l.coords.size(); ++i)
  {
    if (l.coords[i] < r.coords[i])
    {
      return -1;
    }
    else if (l.coords[i] > r.coords[i])
    {
      return 1;
    }
  }
  return 0;
}

bool sort_compare_coord_2(struct coo_t p, struct coo_t q)
{
  /// sort with k dimension as the key in (i, j, k)
  return (p.coords[2] < q.coords[2]);
}

bool sort_compare_coord_1(struct coo_t p, struct coo_t q)
{
  /// sort with j dimension as the key in (i, j, k)
  return (p.coords[1] < q.coords[1]);
}

bool sort_compare_coord_0(struct coo_t p, struct coo_t q)
{
  /// sort with i dimension as the key in (i, j, k)
  return (p.coords[0] < q.coords[0]);
}

/// A function to swap two elements
void swap(struct coo_t *a, struct coo_t *b)
{
  struct coo_t t = *a;
  *a = *b;
  *b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
array, and places all smaller (smaller than pivot) to
left of pivot and all greater elements to right of pivot */
int partition(vector<struct coo_t> &ary, int low, int high, int mod)
{
  /// pivot
  struct coo_t pivot = ary[high];
  /// Index of smaller element and indicates the right position of pivot found so far
  int i = (low - 1);

  for (int j = low; j <= high - 1; j++)
  {
    /// If current element is smaller than the pivot
    if (ary[j].coords[mod] < pivot.coords[mod])
    {
      i++; /// increment index of smaller element
      swap(&ary[i], &ary[j]);
    }
  }
  swap(&ary[i + 1], &ary[high]);
  return (i + 1);
}

/* The QuickSort function
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index,
mod --> the coords mode to sort on (0, 1, 2, 3, ..)
*/
void quick_sort(vector<struct coo_t> &ary, int low, int high, int mod)
{
  if (low < high)
  {
    /* pi is partitioning index, ary[p] is now at right place */
    int pi = partition(ary, low, high, mod);

    /// Separately sort elements before
    /// partition and after partition
    quick_sort(ary, low, pi - 1, mod);
    quick_sort(ary, pi + 1, high, mod);
  }
}

void count_sort(vector<struct coo_t> &ary, int n, int mode)
{
  /// count sort for mode m
  vector<coo_t> sorted_ary(n);
  int maxx = 0, i;
  for (i = 0; i < n; i++)
  {
    if (maxx < ary[i].coords[mode])
      maxx = ary[i].coords[mode];
  }
  /// Create a count array to store count of individual
  /// characters and initialize count array as 0
  vector<int> count(maxx + 1);
  fill(count.begin(), count.end(), 0);

  /// Store count of each number
  for (i = 0; i < n; ++i)
    ++count[ary[i].coords[mode]];

  /// Change count[i] so that count[i] now contains actual
  /// position of this character in output array
  for (i = 1; i <= maxx; ++i)
  {
    count[i] += count[i - 1];
  }

  /// Build the output character array
  for (i = n - 1; i >= 0; --i)
  {
    sorted_ary[count[ary[i].coords[mode]] - 1] = ary[i];
    --count[ary[i].coords[mode]];
  }

  /// Copy the sorted array to arr
  for (i = 0; i < n; ++i)
    ary[i] = sorted_ary[i];
}

void radix_bucket(vector<struct coo_t> &ary, int n)
{
  int bucket_cnt[10];
  int i, j, k, r, NOP = 0 /*#digits*/, divisor = 1, maxx = 0, pass1, pass2;
  struct coo_t **bucket = new struct coo_t *[10];

  for (i = 0; i < 10; i++)
  {
    bucket[i] = new struct coo_t[n];
  }

  for (pass1 = 1; pass1 >= 0; pass1--)
  {
    /// find the max val
    maxx = 0;
    for (i = 0; i < n; i++)
    {
      if (maxx < ary[i].coords[pass1])
        maxx = ary[i].coords[pass1];
    }

    /// digists
    NOP = 0;
    while (maxx != 0)
    {
      maxx = maxx / 10;
      ++NOP;
    }

    divisor = 1;
    for (pass2 = 0; pass2 < NOP; pass2++)
    {
      for (i = 0; i < 10; i++)
      {
        bucket_cnt[i] = 0;
      }
      for (i = 0; i < n; i++)
      {
        r = (ary[i].coords[pass1] / divisor) % 10;
        bucket[r][bucket_cnt[r]] = ary[i];
        bucket_cnt[r] += 1;
      }
      i = 0;
      for (k = 0; k < 10; k++)
      {
        for (j = 0; j < bucket_cnt[k]; j++)
        {
          ary[i] = bucket[k][j];
          i++;
        }
      }
      divisor *= 10;
    }
  }

  /// deallocate array
  for (i = 0; i < 10; i++)
  {
    delete[] bucket[i];
  }
  delete[] bucket;
}

/// generate buckets for next sorting based on mode in question
void generate_buckets(vector<coo_t> &coo_ts, int mode, vector<bucket> &buckets)
{
  int count = 1;
  vector<int> counts;
  counts.push_back(0);
  for (unsigned long i = 1; i < coo_ts.size(); i++)
  {
    if (coo_ts[i].coords[mode] == coo_ts[i - 1].coords[mode])
    {
      count++;
    }
    else
    {
      counts.push_back(count);
      count++;
    }
  }
  counts.push_back(count);

  /// merge previous bucket into new counts
  for (unsigned long i = 0; i < buckets.size(); i++)
  {
    counts.push_back(buckets[i].right);
  }
  sort(counts.begin(), counts.end());
  buckets.clear();
  buckets.shrink_to_fit();

  for (unsigned int i = 1; i < counts.size(); i++)
  {
    if (counts[i - 1] < counts[i])
    {
      /// if (bucket)
      struct bucket abucket
      {
        counts[i - 1], counts[i]
      };
      buckets.push_back(abucket);
    }
  }
}

/// This hybrid sort function that combines count sort and quick sort,
/// which use count sort for the 1st mode, and then quick sort on the
/// remaining modes within buckets
void count_quick(vector<struct coo_t> &ary, int n, int num_dims)
{
  vector<bucket> buckets; /// buckets generated by first sort
  count_sort(ary, n, 0);
  generate_buckets(ary, 0, buckets);
  for (unsigned long i = 0; i < buckets.size(); i++)
  {
    sort(&ary[buckets[i].left], &ary[buckets[i].right], sort_compare_coord_1);
  }
  if (num_dims > 2)
  {
    generate_buckets(ary, 1, buckets); /// buckets generated by second sort
    for (unsigned long i = 0; i < buckets.size(); i++)
    {
      sort(&ary[buckets[i].left], &ary[buckets[i].right], sort_compare_coord_2);
    }
  }
}

void count_radix(vector<struct coo_t> &ary, int n, int num_dims)
{

  if (num_dims == 2)
  {
    /// it only reqires to sort the first dimension
    count_sort(ary, n, 0);
  }

  if (num_dims == 3)
  {
    /// for best performance, we want something to decide how many dimensions to sort
    /// all dimensions or partial dimensions?
    int sort_dims = 3;
    switch (sort_dims)
    {
    case 0:
      /// neglect sorting
      break;
    case 1:
      count_sort(ary, n, 0);
      break;
    case 2:
      count_sort(ary, n, 1);
      count_sort(ary, n, 0);
      break;
    case 3:
      count_sort(ary, n, 2);
      count_sort(ary, n, 1);
      count_sort(ary, n, 0);
      break;
    default:
      llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: wrong dimensions for 3D tensors\n";
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
/**
 * @brief Differet sorting algorithm for sparse transpse taking COO coordinates as input
 *
 * @param sort_type specify the sort algorithm to use
 * @param coo_ts the COO coordinates: for 2D, for example, (i, j, val)
 * @param sz number of coordinates within coo_ts
 * @param num_dims number of dimensions: 2D or 3D
 * @param output_permutation the output permutation to transpose,
 *      such as i, j, k (input perm) to j, k, i (output perm)
 */
void transpose_sort(int sort_type, vector<coo_t> &coo_ts, int sz,
                    int num_dims, int output_permutation)
{
  switch (sort_type)
  {
  case SEQ_QSORT:
    std::qsort((void *)&coo_ts[0], sz, sizeof(struct coo_t), qsort_compare_coords);
    break;
  case PAR_QSORT:
    //__gnu_parallel::sort(coo_ts.begin(), coo_ts.end(), sort_compare_coords);
    sort(coo_ts.begin(), coo_ts.end(), sort_compare_coords);
    break;
  case RADIX_BUCKET:
    radix_bucket(coo_ts, sz);
    break;
  case COUNT_RADIX:
    count_radix(coo_ts, sz, num_dims);
    break;
  case COUNT_QUICK:
    count_quick(coo_ts, sz, num_dims);
    break;
  case NO_SORT:
    break;
  }
}

/**
 * @brief transpose a sparse matrix
 *
 * @tparam T type of matrix values
 * @param A1format sparse storage format of A1 dimension, which could be CN, CU, S, or D
 * @param A2format sparse storage format of A2 dimension, which could be CN, CU, S, or D
 * @param A1pos_rank ?
 * @param A1pos_ptr A1 pos array
 * @param A1crd_rank ?
 * @param A1crd_ptr A1 crd array
 * @param A2pos_rank ?
 * @param A2pos_ptr A2 pos array
 * @param A2crd_rank ?
 * @param A2crd_ptr A2 crd array
 * @param Aval_rank ?
 * @param Aval_ptr A val array
 * @param B1format sparse storage format of B1 dimension, which could be CN, CU, S, or D
 * @param B2format sparse storage format of B2 dimension, which could be CN, CU, S, or D
 * @param B1pos_rank ?
 * @param B1pos_ptr B1 pos array
 * @param B1crd_rank ?
 * @param B1crd_ptr B1 crd array
 * @param B2pos_rank ?
 * @param B2pos_ptr B2 pos array
 * @param B2crd_rank ?
 * @param B2crd_ptr B2 crd array
 * @param Bval_rank ?
 * @param Bval_ptr B val array
 * @param sizes_rank ?
 * @param sizes_ptr array of length of those arrays
 */
template <typename T, typename IndicesType>
void transpose_2D(int32_t A1format, int32_t A1tile_format, int32_t A2format, int32_t A2tile_format,
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
                  int sizes_rank, void *sizes_ptr)
{
  /// Get sort type
  int selected_sort_type = 0;
  getSortType(selected_sort_type);

  /// int i = 0;
  int num_dims = 2;
  std::string Aspformat;
  std::string Bspformat;

  auto *desc_A1pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A1pos_ptr);
  auto *desc_A1crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A1crd_ptr);
  auto *desc_A1tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A1tile_pos_ptr);
  auto *desc_A1tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A1tile_crd_ptr);
  auto *desc_A2pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A2pos_ptr);
  auto *desc_A2crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A2crd_ptr);
  [[maybe_unused]] auto *desc_A2tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A2tile_pos_ptr);
  [[maybe_unused]] auto *desc_A2tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A2tile_crd_ptr);
  auto *desc_Aval = static_cast<StridedMemRefType<T, 1> *>(Aval_ptr);

  auto *desc_B1pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B1pos_ptr);
  auto *desc_B1crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B1crd_ptr);
  auto *desc_B2pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B2pos_ptr);
  auto *desc_B2crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B2crd_ptr);
  auto *desc_Bval = static_cast<StridedMemRefType<T, 1> *>(Bval_ptr);

  auto *dim_sizes = static_cast<StridedMemRefType<IndicesType, 1> *>(sizes_ptr);

  auto *desc_B1tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B1tile_pos_ptr);
  auto *desc_B1tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B1tile_crd_ptr);
  auto *desc_B2tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B2tile_pos_ptr);
  auto *desc_B2tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B2tile_crd_ptr);
  desc_B1tile_pos->data[0] = -1;
  desc_B1tile_crd->data[0] = -1;
  desc_B2tile_pos->data[0] = -1;
  desc_B2tile_crd->data[0] = -1;

  desc_B1tile_pos->sizes[0] = -1;
  desc_B1tile_crd->sizes[0] = -1;
  desc_B2tile_pos->sizes[0] = -1;
  desc_B2tile_crd->sizes[0] = -1;

  /*
  std::cout << "desc_sizes detail: \n"
              << "desc_sizes->data[0]: " << desc_sizes->data[0] << "\n"
              << "desc_sizes->data[1]: " << desc_sizes->data[1] << "\n"
              << "desc_sizes->data[2]: " << desc_sizes->data[2] << "\n"
              << "desc_sizes->data[3]: " << desc_sizes->data[3] << "\n"
              << "desc_sizes->data[4]: " << desc_sizes->data[4] << "\n"
              << "desc_sizes->data[5]: " << desc_sizes->data[5] << "\n"
              << "desc_sizes->data[6]: " << desc_sizes->data[6] << "\n"
              << "desc_sizes->data[7]: " << desc_sizes->data[7] << "\n"
              << "desc_sizes->data[8]: " << desc_sizes->data[8] << "\n"
              << "desc_sizes->data[9]: " << desc_sizes->data[9] << "\n"
              << "desc_sizes->data[10]: " << desc_sizes->data[10] << "\n";
  */

  int rowSize = dim_sizes->data[0];
  int colSize = dim_sizes->data[1];

  if (A1format == Dense && A1tile_format == Dense && A2format == singleton)
  {
    Aspformat.assign("ELL");
  }
  else if ((A1format == Compressed_nonunique && A2format == singleton) || (A1format == singleton && A2format == Compressed_nonunique))
  {
    Aspformat.assign("COO");
  }
  else if ((A1format == Dense && A2format == Compressed_unique) || (A1format == Compressed_unique && A2format == Dense))
  {
    Aspformat.assign("CSR");
  }
  else
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: At this time, only ELL, COO, CSR formats are supported for input for transpose (A)\n";
  }

  if (B1format == Dense && B1tile_format == Dense && B2format == singleton)
  {
    Bspformat.assign("ELL");
  }
  else if ((B1format == Compressed_nonunique && B2format == singleton) || (B1format == singleton && B2format == Compressed_nonunique))
  {
    Bspformat.assign("COO");
  }
  else if ((B1format == Dense && B2format == Compressed_unique) || (B1format == Compressed_unique && B2format == Dense))
  {
    Bspformat.assign("CSR");
  }
  else
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: At this time, only ELL, COO, and CSR formats are supported for output for transpose (B)\n";
  }

  if (Aspformat.compare("ELL") == 0 && Bspformat.compare("ELL") == 0)
  {
    /// Copy A1 and A1_tile
    desc_B1crd->data[0] = desc_A1crd->data[0];
    desc_B1crd->sizes[0] = desc_A1crd->sizes[0];

    desc_B1pos->data[0] = desc_A1pos->data[0];
    desc_B1pos->sizes[0] = desc_A1pos->sizes[0];

    desc_B1tile_crd->data[0] = desc_A1tile_crd->data[0];
    desc_B1tile_crd->sizes[0] = desc_A1tile_crd->sizes[0];

    desc_B1tile_pos->data[0] = desc_A1tile_pos->data[0];
    desc_B1tile_pos->sizes[0] = desc_A1tile_pos->sizes[0];

    desc_B2pos->data[0] = desc_A2pos->data[0];
    desc_B2pos->sizes[0] = desc_A2pos->sizes[0];

    /// Rearrange the values
    int val_sz = desc_Aval->sizes[0];
    int rows = desc_A1pos->data[0];
    // int cols = desc_A1tile_pos->data[0];
    int index = 0;

    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < val_sz; j++)
      {
        int col = desc_A2crd->data[j];
        if (col == i)
        {
          desc_B2crd->data[j] = desc_A2crd->data[j];
          desc_Bval->data[index] = desc_Aval->data[j];
          ++index;
        }
      }
    }
  }

  if (Aspformat.compare("COO") == 0 && Bspformat.compare("COO") == 0)
  {
    int sz = desc_Aval->sizes[0];
    /// vector of coordinates
    vector<coo_t> coo_ts(sz);

    int m = 0;
    if (selected_sort_type == NO_SORT)
    { /// coordinates are not sorted

      for (int i = 0; i < colSize + 1; ++i)
      {
        for (int j = 0; j < rowSize + 1; ++j)
        {
          for (int k = 0; k < sz; ++k)
          {
            if (desc_A1crd->data[k] == j && desc_A2crd->data[k] == i)
            {
              coo_ts[m].coords.push_back(desc_A2crd->data[k]);
              coo_ts[m].coords.push_back(desc_A1crd->data[k]);
              coo_ts[m].val = desc_Aval->data[k];
              ++m;
            }
          }
        }
      }
    }
    else
    {

      /// dimension, so we need to use indexing map for transpose
      //===----------------------------------------------------------------------===//
      /// marshalling data for each sorting algorithms
      //===----------------------------------------------------------------------===//
      for (int i = 0; i < sz; ++i)
      {
        coo_ts[i].coords.push_back(desc_A2crd->data[i]);
        coo_ts[i].coords.push_back(desc_A1crd->data[i]);
        coo_ts[i].val = desc_Aval->data[i];
      }

      //===----------------------------------------------------------------------===//
      /// Different sorting algorithm
      //===----------------------------------------------------------------------===//
      transpose_sort(selected_sort_type, coo_ts, sz, num_dims, 0);
    }

    //===----------------------------------------------------------------------===//
    /// push transposed coords to output tensors
    //===----------------------------------------------------------------------===//
    for (int i = 0; i < sz; ++i)
    {
      desc_B1crd->data[i] = coo_ts[i].coords[0];
      desc_B2crd->data[i] = coo_ts[i].coords[1];
      desc_Bval->data[i] = coo_ts[i].val;
    }

    /// B2 pos should have two values: data[0]: 0 and data[1]: sz
    desc_B1pos->sizes[0] = 2;
    desc_B1pos->data[1] = sz;

    /// to be consistent, desc_B2pos is set to -1
    desc_B2pos->sizes[0] = desc_A2pos->sizes[0];
    desc_B2pos->data[0] = desc_A2pos->data[0];

    /// switch row and col size
    dim_sizes->data[0] = colSize;
    dim_sizes->data[1] = rowSize;
  }

  if (Aspformat.compare("CSR") == 0 && Bspformat.compare("CSR") == 0)
  {
    if (selected_sort_type == NO_SORT) /// coordinates are not sorted
    {
      /// 1) not by sorting: only works for CSR/matrices
      /// Atomic-based Transposition: retraverse the matrix from the transposed direction
      /// B's row size == input's col size
      int BRowSize = dim_sizes->data[0];
      int BColSize = dim_sizes->data[1];

      /// B's col pos size == B's #rows + 1
      desc_B2pos->sizes[0] = BRowSize + 1;
      int count = 0;
      desc_B2pos->data[0] = count;
      int i, j, k;
      for (k = 0; k < BRowSize; k++)
      {
        for (i = 0; i < BColSize; i++)
        {
          for (j = desc_A2pos->data[i]; j < desc_A2pos->data[i + 1]; j++)
          {
            if (desc_A2crd->data[j] == k)
            {
              desc_B2crd->data[count] = i;
              desc_Bval->data[count] = desc_Aval->data[j];
              count++;
            }
          }
        }
        desc_B2pos->data[k + 1] = count;
      }

      /// push sorted data back to B1
      desc_B1pos->data[0] = BRowSize;
      desc_B1crd->data[0] = -1;

      /// switch row and col size
      dim_sizes->data[0] = colSize;
      dim_sizes->data[1] = rowSize;
    }
    else
    {
      /// 2) by sorting: work for both CSR/matrices and CSF/tensors
      /// CSR -> COO -> Transpose -> Sort -> COO -> CSR
      //===----------------------------------------------------------------------===//
      /// marshalling data
      //===----------------------------------------------------------------------===//
      /// vector of coordinates
      int i, j;
      int BNnz = desc_Aval->sizes[0];
      vector<coo_t> coo_ts(BNnz);
      int pos = 0;
      int counter = 0;
      for (i = 0; i < BNnz; ++i)
      {
        coo_ts[i].coords.push_back(desc_A2crd->data[i]);
        coo_ts[i].val = desc_Aval->data[i];
        counter++;
        for (j = 0; j < desc_A2pos->sizes[0] - 1; j++)
        {
          if (counter > desc_A2pos->data[j] && counter <= desc_A2pos->data[j + 1])
          {
            pos = j;
          }
        }
        coo_ts[i].coords.push_back(pos);
      }
      /// sort the first dim
      //===----------------------------------------------------------------------===//
      /// Different sorting algorithm
      //===----------------------------------------------------------------------===//
      //
      transpose_sort(selected_sort_type, coo_ts, BNnz, num_dims, 0);

      /// push sorted data back to B
      int BRowSize = colSize;
      desc_B1pos->data[0] = BRowSize;
      desc_B1crd->data[0] = -1;

      desc_B2pos->sizes[0] = BRowSize + 1; /// resize
      /// push pos to B
      counter = 1;
      j = 1;
      for (i = 1; i < BNnz; i++)
      {
        if (coo_ts[i - 1].coords[0] != coo_ts[i].coords[0])
        {
          desc_B2pos->data[j] = counter;
          j++;
        }
        /// for cases having a gap larger than 1, e.g., 0 0 1 1 (gap > 1) 3 4 4
        if (coo_ts[i].coords[0] - coo_ts[i - 1].coords[0] > 1)
        {
          int gap = coo_ts[i].coords[0] - coo_ts[i - 1].coords[0] - 1;
          while (gap > 0)
          {
            desc_B2pos->data[j] = counter;
            j++;
            gap--;
          }
        }
        counter++;
      }
      desc_B2pos->data[j] = counter;
      /// push crd to B
      for (i = 0; i < BNnz; i++)
      {
        desc_B2crd->data[i] = coo_ts[i].coords[1];
        desc_Bval->data[i] = coo_ts[i].val;
      }

      /// switch row and col size
      dim_sizes->data[0] = colSize;
      dim_sizes->data[1] = rowSize;
    }
  }

  if (Aspformat.compare("CSR") == 0 && Bspformat.compare("COO") == 0)
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: 'CSR->COO' is not supported for sparse tensor transpose.\n";
  }

  if (Aspformat.compare("COO") == 0 && Bspformat.compare("CSR") == 0)
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: 'COO->CSR' is not supported for sparse tensor transpose.\n";
  }
}

template <typename T, typename  IndicesType>
void transpose_3D(int32_t input_permutation, int32_t output_permutation,
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
                  int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  /// Get sort type
  int selected_sort_type = 0;
  getSortType(selected_sort_type);

  int num_dims = 3;

  std::string Aspformat;
  std::string Bspformat;

  if ((A1format == Compressed_nonunique && A2format == singleton && A3format == singleton) ||
      (A1format == singleton && A2format == Compressed_nonunique && A3format == singleton) ||
      (A1format == singleton && A2format == singleton && A3format == Compressed_nonunique))
  {
    Aspformat.assign("COO");
  }
  else if (A1format == Compressed_unique &&
           A2format == Compressed_unique &&
           A3format == Compressed_unique)
  {
    Aspformat.assign("CSF");
  }
  else
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: At this time, only COO and CSF formats are supported for input tensor.\n";
  }
  if ((B1format == Compressed_nonunique && B2format == singleton && B3format == singleton) ||
      (B1format == singleton && B2format == Compressed_nonunique && B3format == singleton) ||
      (B1format == singleton && B2format == singleton && B3format == Compressed_nonunique))
  {
    Bspformat.assign("COO");
  }
  else if (B1format == Compressed_unique &&
           B2format == Compressed_unique &&
           B3format == Compressed_unique)
  {
    Bspformat.assign("CSF");
  }
  else
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: At this time, only COO and CSF formats are supported for output tensor.\n";
  }

  /// auto *desc_A1pos = static_cast<StridedMemRefType<int64_t, 1> *>(A1pos_ptr);
  auto *desc_A1crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A1crd_ptr);
  auto *desc_A2pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A2pos_ptr);
  auto *desc_A2crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A2crd_ptr);
  auto *desc_A3pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A3pos_ptr);
  auto *desc_A3crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A3crd_ptr);
  auto *desc_Aval = static_cast<StridedMemRefType<T, 1> *>(Aval_ptr);

  auto *desc_B1pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B1pos_ptr);
  auto *desc_B1crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B1crd_ptr);
  auto *desc_B2pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B2pos_ptr);
  auto *desc_B2crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B2crd_ptr);
  auto *desc_B3pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B3pos_ptr);
  auto *desc_B3crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B3crd_ptr);
  auto *desc_Bval = static_cast<StridedMemRefType<T, 1> *>(Bval_ptr);

  auto *desc_sizes = static_cast<StridedMemRefType<IndicesType, 1> *>(sizes_ptr);

  auto *desc_B1tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B1tile_pos_ptr);
  auto *desc_B1tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B1tile_crd_ptr);
  auto *desc_B2tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B2tile_pos_ptr);
  auto *desc_B2tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B2tile_crd_ptr);
  auto *desc_B3tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(B3tile_pos_ptr);
  auto *desc_B3tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(B3tile_crd_ptr);
  desc_B1tile_pos->data[0] = -1;
  desc_B1tile_crd->data[0] = -1;
  desc_B2tile_pos->data[0] = -1;
  desc_B2tile_crd->data[0] = -1;
  desc_B3tile_pos->data[0] = -1;
  desc_B3tile_crd->data[0] = -1;

  desc_B1tile_pos->sizes[0] = -1;
  desc_B1tile_crd->sizes[0] = -1;
  desc_B2tile_pos->sizes[0] = -1;
  desc_B2tile_crd->sizes[0] = -1;
  desc_B3tile_pos->sizes[0] = -1;
  desc_B3tile_crd->sizes[0] = -1;

  int mode_sz0 = desc_sizes->data[13];
  int mode_sz1 = desc_sizes->data[14];
  int mode_sz2 = desc_sizes->data[15];

  int dim_sizes[3] = {mode_sz0, mode_sz1, mode_sz2};
  int trans_dim_sizes[3] = {mode_sz0, mode_sz1, mode_sz2}; /// sizes of dimensions after transposition of dimensions

  if (Aspformat.compare("COO") == 0 && Bspformat.compare("COO") == 0)
  {
    int sz = desc_Aval->sizes[0];

    /// vector of coordinates
    vector<coo_t> coo_ts(sz);

    //===----------------------------------------------------------------------===//
    /// marshalling data for permutation for transpose
    //===----------------------------------------------------------------------===//
    /// There are 5 input cases: 012, 021, 102, 120, 201, 210
    /// There are 5 output cases: 012, 021, 102, 120, 201, 210
    int idigists[3], odigists[3], i = 0;
    for (int j = num_dims - 1; j >= 0; j--)
    {
      int tmp = pow(10, j);
      idigists[i] = (input_permutation / tmp) % 10;
      odigists[i] = (output_permutation / tmp) % 10;
      ++i;
    }

    /// the order depends on both input and output permutations
    /// for example, if the input is 201 and output is 102, the order
    /// should be (k, j, i)
    for (int i = 0; i < num_dims; ++i)
    {
      for (int j = 0; j < num_dims; ++j)
      {
        if (odigists[i] == idigists[j])
        {
          trans_dim_sizes[i] = dim_sizes[j];
          for (int k = 0; k < sz; ++k)
          {
            switch (j)
            {
            case 0:
              coo_ts[k].coords.push_back(desc_A1crd->data[k]);
              break;

            case 1:
              coo_ts[k].coords.push_back(desc_A2crd->data[k]);
              break;

            case 2:
              coo_ts[k].coords.push_back(desc_A3crd->data[k]);
              break;

            default:;
              llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: incorrect output permutation\n";
            }
            coo_ts[k].val = desc_Aval->data[k];
          }
        }
      }
    }

    if (selected_sort_type == NO_SORT)
    {
      vector<coo_t> perm_coo_ts(sz);

      /// re-traverse from the target dimension
      int m = 0;
      for (int i = 0; i < trans_dim_sizes[0] + 1; ++i)
      {
        for (int j = 0; j < trans_dim_sizes[1] + 1; ++j)
        {
          for (int k = 0; k < trans_dim_sizes[2] + 1; ++k)
          {
            for (int l = 0; l < sz; ++l)
            {
              if (i == coo_ts[l].coords[0] && j == coo_ts[l].coords[1] && k == coo_ts[l].coords[2])
              {
                perm_coo_ts[m].coords.push_back(coo_ts[l].coords[0]);
                perm_coo_ts[m].coords.push_back(coo_ts[l].coords[1]);
                perm_coo_ts[m].coords.push_back(coo_ts[l].coords[2]);
                perm_coo_ts[m].val = coo_ts[l].val;
                ++m;
              }
            }
          }
        }
      }

      //===----------------------------------------------------------------------===//
      /// push transposed coords to output tensors
      //===----------------------------------------------------------------------===//
      for (i = 0; i < sz; ++i)
      {
        desc_B1crd->data[i] = perm_coo_ts[i].coords[0];
        desc_B2crd->data[i] = perm_coo_ts[i].coords[1];
        desc_B3crd->data[i] = perm_coo_ts[i].coords[2];
        desc_Bval->data[i] = perm_coo_ts[i].val;
      }
    }
    else
    {
      //===----------------------------------------------------------------------===//
      /// Different sorting algorithm
      //===----------------------------------------------------------------------===//
      transpose_sort(selected_sort_type, coo_ts, sz, num_dims, output_permutation);

      //===----------------------------------------------------------------------===//
      /// push transposed coords to output tensors
      //===----------------------------------------------------------------------===//
      for (i = 0; i < sz; ++i)
      {
        desc_B1crd->data[i] = coo_ts[i].coords[0];
        desc_B2crd->data[i] = coo_ts[i].coords[1];
        desc_B3crd->data[i] = coo_ts[i].coords[2];
        desc_Bval->data[i] = coo_ts[i].val;
      }
    }

    /// B2 pos should have two values: data[0]: 0 and data[1]: sz
    desc_B1pos->sizes[0] = 2;
    desc_B1pos->data[1] = sz;
  }

  if (Aspformat.compare("CSF") == 0 && Bspformat.compare("CSF") == 0)
  {
    int sz = desc_Aval->sizes[0];

    /// vector of coordinates
    vector<coo_t> coo_ts(sz);

    /// initialization
    for (int i = 0; i < sz; ++i)
    {
      coo_ts[i].coords.push_back(desc_A3crd->data[i]);
      coo_ts[i].coords.push_back(desc_A3crd->data[i]);
      coo_ts[i].coords.push_back(desc_A3crd->data[i]);
      coo_ts[i].val = desc_Aval->data[i];
    }

    /// fix coords for A1 and A2
    int i = 0, j = 0, k = 0;
    for (i = 0; i < desc_A2crd->sizes[0]; ++i)
    {
      for (j = desc_A3pos->data[i]; j < desc_A3pos->data[i + 1]; ++j)
      {
        coo_ts[k].coords[1] = desc_A2crd->data[i]; /// for A2
        k++;
      }
    }
    k = 0;
    for (i = 0; i < desc_A2pos->sizes[0] - 1; ++i)
    {
      for (j = desc_A3pos->data[desc_A2pos->data[i]]; j < desc_A3pos->data[desc_A2pos->data[i + 1]]; ++j)
      {
        coo_ts[k].coords[0] = desc_A1crd->data[i]; /// for A1
        k++;
      }
    }

    vector<coo_t> perm_coo_ts(sz);
    //===----------------------------------------------------------------------===//
    /// marshalling data for permutation for transpose
    //===----------------------------------------------------------------------===//
    /// There are 5 input cases: 012, 021, 102, 120, 201, 210
    /// There are 5 output cases: 012, 021, 102, 120, 201, 210
    int idigists[3], odigists[3];
    i = 0;
    for (int j = num_dims - 1; j >= 0; j--)
    {
      int tmp = pow(10, j);
      idigists[i] = (input_permutation / tmp) % 10;
      odigists[i] = (output_permutation / tmp) % 10;
      ++i;
    }

    /// the order depends on both input and output permutations
    /// for example, if the input is 201 and output is 102, the order
    /// should be (k, j, i)
    for (int i = 0; i < num_dims; ++i)
    {
      for (int j = 0; j < num_dims; ++j)
      {
        if (odigists[i] == idigists[j])
        {
          trans_dim_sizes[i] = dim_sizes[j];
          for (int k = 0; k < sz; ++k)
          {
            switch (j)
            {
            case 0:
              perm_coo_ts[k].coords.push_back(coo_ts[k].coords[0]);
              break;

            case 1:
              perm_coo_ts[k].coords.push_back(coo_ts[k].coords[1]);
              break;

            case 2:
              perm_coo_ts[k].coords.push_back(coo_ts[k].coords[2]);
              break;

            default:;
              llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: incorrect output permutation\n";
            }
            perm_coo_ts[k].val = coo_ts[k].val;
          }
        }
      }
    }

    if (selected_sort_type == NO_SORT)
    {

      /// re-traverse from the target dimension
      int m = 0, l = 0;
      for (i = 0; i < trans_dim_sizes[0] + 1; ++i)
      {
        for (j = 0; j < trans_dim_sizes[1] + 1; ++j)
        {
          for (k = 0; k < trans_dim_sizes[2] + 1; ++k)
          {
            for (l = 0; l < sz; ++l)
            {
              if (i == perm_coo_ts[l].coords[0] && j == perm_coo_ts[l].coords[1] && k == perm_coo_ts[l].coords[2])
              {
                coo_ts[m].coords[0] = perm_coo_ts[l].coords[0];
                coo_ts[m].coords[1] = perm_coo_ts[l].coords[1];
                coo_ts[m].coords[2] = perm_coo_ts[l].coords[2];
                coo_ts[m].val = perm_coo_ts[l].val;
                ++m;
              }
            }
          }
        }
      }

      //===----------------------------------------------------------------------===//
      /// Convert COO back to CSF
      //===----------------------------------------------------------------------===//
      /// calculate B1crd, B2crd, B3crd
      int counter = 0;
      for (i = 0; i < sz - 1; i++)
      {
        desc_B1crd->data[counter] = coo_ts[i].coords[0];
        if (coo_ts[i].coords[0] != coo_ts[i + 1].coords[0])
        {
          counter++;
        }
      }
      desc_B1crd->data[counter] = coo_ts[i].coords[0];
      desc_B1crd->sizes[0] = counter + 1;

      counter = 0;
      for (i = 0; i < sz - 1; i++)
      {
        desc_B2crd->data[counter] = coo_ts[i].coords[1];
        if (coo_ts[i].coords[1] != coo_ts[i + 1].coords[1])
        {
          counter++;
        }
      }
      desc_B2crd->data[counter] = coo_ts[i].coords[1];
      desc_B2crd->sizes[0] = counter + 1;

      for (i = 0; i < sz; i++)
      {
        desc_B3crd->data[i] = coo_ts[i].coords[2];
        desc_Bval->data[i] = coo_ts[i].val;
      }
      desc_B3crd->sizes[0] = sz;
      desc_Bval->sizes[0] = sz;

      /// calculate B1pos, B2pos, B3pos
      desc_B1pos->data[0] = 0;
      desc_B1pos->data[1] = desc_B1crd->sizes[0];
      desc_B1pos->sizes[0] = 2;

      desc_B2pos->data[0] = 0;
      counter = 1;
      j = 1;
      for (i = 0; i < sz - 1; ++i)
      {
        if (coo_ts[i].coords[0] != coo_ts[i + 1].coords[0])
        {
          desc_B2pos->data[j] = counter;
          j++;
        }
        if (coo_ts[i].coords[1] != coo_ts[i + 1].coords[1] ||
            coo_ts[i].coords[0] != coo_ts[i + 1].coords[0])
        {
          counter++;
        }
      }
      desc_B2pos->data[j] = counter;
      desc_B2pos->sizes[0] = j + 1;

      desc_B3pos->data[0] = 0;
      counter = 1;
      j = 1;
      for (i = 0; i < sz - 1; ++i)
      {
        if (coo_ts[i].coords[1] != coo_ts[i + 1].coords[1] ||
            coo_ts[i].coords[0] != coo_ts[i + 1].coords[0])
        {
          desc_B3pos->data[j] = counter;
          j++;
        }
        if (coo_ts[i].coords[2] != coo_ts[i + 1].coords[2] ||
            coo_ts[i].coords[1] != coo_ts[i + 1].coords[1] ||
            coo_ts[i].coords[0] != coo_ts[i + 1].coords[0])
        {
          counter++;
        }
      }
      desc_B3pos->data[j] = counter;
      desc_B3pos->sizes[0] = j + 1;
    }
    else
    { /// end of NO_SORT
      //===----------------------------------------------------------------------===//
      /// Different sorting algorithm
      //===----------------------------------------------------------------------===//
      transpose_sort(selected_sort_type, perm_coo_ts, sz, num_dims, output_permutation);

      //===----------------------------------------------------------------------===//
      /// Convert COO back to CSF
      //===----------------------------------------------------------------------===//
      /// calculate B1crd, B2crd, B3crd
      int counter = 0;
      for (i = 0; i < sz - 1; i++)
      {
        desc_B1crd->data[counter] = perm_coo_ts[i].coords[0];
        if (perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
        {
          counter++;
        }
      }
      desc_B1crd->data[counter] = perm_coo_ts[i].coords[0];
      desc_B1crd->sizes[0] = counter + 1;

      counter = 0;
      for (i = 0; i < sz - 1; i++)
      {
        desc_B2crd->data[counter] = perm_coo_ts[i].coords[1];
        if (perm_coo_ts[i].coords[1] != perm_coo_ts[i + 1].coords[1])
        {
          counter++;
        }
      }
      desc_B2crd->data[counter] = perm_coo_ts[i].coords[1];
      desc_B2crd->sizes[0] = counter + 1;

      for (i = 0; i < sz; i++)
      {
        desc_B3crd->data[i] = perm_coo_ts[i].coords[2];
        desc_Bval->data[i] = perm_coo_ts[i].val;
      }
      desc_B3crd->sizes[0] = sz;
      desc_Bval->sizes[0] = sz;

      /// calculate B1pos, B2pos, B3pos
      desc_B1pos->data[0] = 0;
      desc_B1pos->data[1] = desc_B1crd->sizes[0];
      desc_B1pos->sizes[0] = 2;

      desc_B2pos->data[0] = 0;
      counter = 1;
      j = 1;
      for (i = 0; i < sz - 1; ++i)
      {
        if (perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
        {
          desc_B2pos->data[j] = counter;
          j++;
        }
        if (perm_coo_ts[i].coords[1] != perm_coo_ts[i + 1].coords[1] ||
            perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
        {
          counter++;
        }
      }
      desc_B2pos->data[j] = counter;
      desc_B2pos->sizes[0] = j + 1;

      desc_B3pos->data[0] = 0;
      counter = 1;
      j = 1;
      for (i = 0; i < sz - 1; ++i)
      {
        if (perm_coo_ts[i].coords[1] != perm_coo_ts[i + 1].coords[1] ||
            perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
        {
          desc_B3pos->data[j] = counter;
          j++;
        }
        if (perm_coo_ts[i].coords[2] != perm_coo_ts[i + 1].coords[2] ||
            perm_coo_ts[i].coords[1] != perm_coo_ts[i + 1].coords[1] ||
            perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
        {
          counter++;
        }
      }
      desc_B3pos->data[j] = counter;
      desc_B3pos->sizes[0] = j + 1;
    } /// end of sorting
  }

  if (Aspformat.compare("CSF") == 0 && Bspformat.compare("COO") == 0)
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: 'CSF->COO' is not supported for sparse tensor transpose.\n";
  }

  if (Aspformat.compare("COO") == 0 && Bspformat.compare("CSF") == 0)
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: 'COO->CSF' is not supported for sparse tensor transpose.\n";
  }
}

/// 2D tensors
extern "C" void transpose_2D_f32_i32(int32_t A1format, int32_t A1tile_format, int32_t A2format, int32_t A2tile_format,
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
                                 int sizes_rank, void *sizes_ptr)
{
  transpose_2D<float, int32_t>(A1format, A1tile_format, A2format, A2tile_format,
                      A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                      A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                      A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                      A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                      Aval_rank, Aval_ptr,
                      B1format, B1tile_format, B2format, B2tile_format,
                      B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                      B1tile_pos_rank, B1tile_pos_ptr, B1tile_crd_rank, B1tile_crd_ptr,
                      B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                      B2tile_pos_rank, B2tile_pos_ptr, B2tile_crd_rank, B2tile_crd_ptr,
                      Bval_rank, Bval_ptr,
                      sizes_rank, sizes_ptr);
}

/// 2D tensors
extern "C" void transpose_2D_f32_i64(int32_t A1format, int32_t A1tile_format, int32_t A2format, int32_t A2tile_format,
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
                                 int sizes_rank, void *sizes_ptr)
{
  transpose_2D<float, int64_t>(A1format, A1tile_format, A2format, A2tile_format,
                      A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                      A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                      A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                      A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                      Aval_rank, Aval_ptr,
                      B1format, B1tile_format, B2format, B2tile_format,
                      B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                      B1tile_pos_rank, B1tile_pos_ptr, B1tile_crd_rank, B1tile_crd_ptr,
                      B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                      B2tile_pos_rank, B2tile_pos_ptr, B2tile_crd_rank, B2tile_crd_ptr,
                      Bval_rank, Bval_ptr,
                      sizes_rank, sizes_ptr);
}

extern "C" void transpose_2D_f64_i32(int32_t A1format, int32_t A1tile_format, int32_t A2format, int32_t A2tile_format,
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
                                 int sizes_rank, void *sizes_ptr)
{
  transpose_2D<double, int32_t>(A1format, A1tile_format, A2format, A2tile_format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                       Aval_rank, Aval_ptr,
                       B1format, B1tile_format, B2format, B2tile_format,
                       B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                       B1tile_pos_rank, B1tile_pos_ptr, B1tile_crd_rank, B1tile_crd_ptr,
                       B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                       B2tile_pos_rank, B2tile_pos_ptr, B2tile_crd_rank, B2tile_crd_ptr,
                       Bval_rank, Bval_ptr,
                       sizes_rank, sizes_ptr);
}

extern "C" void transpose_2D_f64_i64(int32_t A1format, int32_t A1tile_format, int32_t A2format, int32_t A2tile_format,
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
                                 int sizes_rank, void *sizes_ptr)
{
  transpose_2D<double, int64_t>(A1format, A1tile_format, A2format, A2tile_format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                       Aval_rank, Aval_ptr,
                       B1format, B1tile_format, B2format, B2tile_format,
                       B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                       B1tile_pos_rank, B1tile_pos_ptr, B1tile_crd_rank, B1tile_crd_ptr,
                       B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                       B2tile_pos_rank, B2tile_pos_ptr, B2tile_crd_rank, B2tile_crd_ptr,
                       Bval_rank, Bval_ptr,
                       sizes_rank, sizes_ptr);
}

/// 3D tensors
extern "C" void transpose_3D_f32_i32(int32_t input_permutation, int32_t output_permutation,
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
                                 int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  transpose_3D<float, int32_t>(input_permutation, output_permutation,
                      A1format, A1tile_format,
                      A2format, A2tile_format,
                      A3format, A3tile_format,
                      A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                      A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                      A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                      A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                      A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                      A3tile_pos_rank, A3tile_pos_ptr, A3tile_crd_rank, A3tile_crd_ptr,
                      Aval_rank, Aval_ptr,
                      B1format, B1tile_format,
                      B2format, B2tile_format,
                      B3format, B3tile_format,
                      B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                      B1tile_pos_rank, B1tile_pos_ptr, B1tile_crd_rank, B1tile_crd_ptr,
                      B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                      B2tile_pos_rank, B2tile_pos_ptr, B2tile_crd_rank, B2tile_crd_ptr,
                      B3pos_rank, B3pos_ptr, B3crd_rank, B3crd_ptr,
                      B3tile_pos_rank, B3tile_pos_ptr, B3tile_crd_rank, B3tile_crd_ptr,
                      Bval_rank, Bval_ptr,
                      sizes_rank, sizes_ptr);
}

extern "C" void transpose_3D_f64_i64(int32_t input_permutation, int32_t output_permutation,
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
                                 int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  transpose_3D<double, int64_t>(input_permutation, output_permutation,
                       A1format, A1tile_format,
                       A2format, A2tile_format,
                       A3format, A3tile_format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                       A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                       A3tile_pos_rank, A3tile_pos_ptr, A3tile_crd_rank, A3tile_crd_ptr,
                       Aval_rank, Aval_ptr,
                       B1format, B1tile_format,
                       B2format, B2tile_format,
                       B3format, B3tile_format,
                       B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                       B1tile_pos_rank, B1tile_pos_ptr, B1tile_crd_rank, B1tile_crd_ptr,
                       B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                       B2tile_pos_rank, B2tile_pos_ptr, B2tile_crd_rank, B2tile_crd_ptr,
                       B3pos_rank, B3pos_ptr, B3crd_rank, B3crd_ptr,
                       B3tile_pos_rank, B3tile_pos_ptr, B3tile_crd_rank, B3tile_crd_ptr,
                       Bval_rank, Bval_ptr,
                       sizes_rank, sizes_ptr);
}

/// 3D tensors
extern "C" void transpose_3D_f32_i64(int32_t input_permutation, int32_t output_permutation,
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
                                 int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  transpose_3D<float, int64_t>(input_permutation, output_permutation,
                      A1format, A1tile_format,
                      A2format, A2tile_format,
                      A3format, A3tile_format,
                      A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                      A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                      A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                      A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                      A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                      A3tile_pos_rank, A3tile_pos_ptr, A3tile_crd_rank, A3tile_crd_ptr,
                      Aval_rank, Aval_ptr,
                      B1format, B1tile_format,
                      B2format, B2tile_format,
                      B3format, B3tile_format,
                      B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                      B1tile_pos_rank, B1tile_pos_ptr, B1tile_crd_rank, B1tile_crd_ptr,
                      B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                      B2tile_pos_rank, B2tile_pos_ptr, B2tile_crd_rank, B2tile_crd_ptr,
                      B3pos_rank, B3pos_ptr, B3crd_rank, B3crd_ptr,
                      B3tile_pos_rank, B3tile_pos_ptr, B3tile_crd_rank, B3tile_crd_ptr,
                      Bval_rank, Bval_ptr,
                      sizes_rank, sizes_ptr);
}

extern "C" void transpose_3D_f64_i32(int32_t input_permutation, int32_t output_permutation,
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
                                 int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  transpose_3D<double, int32_t>(input_permutation, output_permutation,
                       A1format, A1tile_format,
                       A2format, A2tile_format,
                       A3format, A3tile_format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                       A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                       A3tile_pos_rank, A3tile_pos_ptr, A3tile_crd_rank, A3tile_crd_ptr,
                       Aval_rank, Aval_ptr,
                       B1format, B1tile_format,
                       B2format, B2tile_format,
                       B3format, B3tile_format,
                       B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                       B1tile_pos_rank, B1tile_pos_ptr, B1tile_crd_rank, B1tile_crd_ptr,
                       B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                       B2tile_pos_rank, B2tile_pos_ptr, B2tile_crd_rank, B2tile_crd_ptr,
                       B3pos_rank, B3pos_ptr, B3crd_rank, B3crd_ptr,
                       B3tile_pos_rank, B3tile_pos_ptr, B3tile_crd_rank, B3tile_crd_ptr,
                       Bval_rank, Bval_ptr,
                       sizes_rank, sizes_ptr);
}
