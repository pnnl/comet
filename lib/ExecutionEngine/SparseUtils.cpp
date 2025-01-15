//===- SparseUtils.cpp - Sparse utils for MLIR execution -----------------===//
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

#include "llvm/Support/raw_ostream.h"
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <cstdio>
#include <limits>
#include <iomanip>

#include <random>
#include <map>

enum MatrixReadOption
{
  DEFAULT = 1, /// standard matrix read
  LOWER_TRI_STRICT = 2,
  LOWER_TRI = 3,
  UPPER_TRI_STRICT = 4,
  UPPER_TRI = 5
};

/// helper func: inquire matrix read type
int getMatrixReadOption(int32_t readMode)
{
  int selected_matrix_read = DEFAULT;

  if (readMode == LOWER_TRI_STRICT)
    selected_matrix_read = LOWER_TRI_STRICT;
  else if (readMode == LOWER_TRI)
    selected_matrix_read = LOWER_TRI;
  else if (readMode == UPPER_TRI_STRICT)
    selected_matrix_read = UPPER_TRI_STRICT;
  else if (readMode == UPPER_TRI)
    selected_matrix_read = UPPER_TRI;

  return selected_matrix_read;
}

//===----------------------------------------------------------------------===//
/// Small runtime support library for sparse matrices/tensors.
//===----------------------------------------------------------------------===//

/// COO edge tuple
template <typename T>
struct CooTuple
{
  uint64_t row;
  uint64_t col;
  T val;

  CooTuple() {}
  CooTuple(uint64_t row, uint64_t col) : row(row), col(col) {}
  CooTuple(uint64_t row, uint64_t col, T val) : row(row), col(col), val(val) {}
};

//===----------------------------------------------------------------------===//
/// COO matrix type.  A COO matrix is just a vector of edge tuples.  Tuples are sorted
/// first by row, then by column.
//===----------------------------------------------------------------------===//
template <typename T>
struct CooMatrix
{
  ///---------------------------------------------------------------------
  /// Data members
  ///---------------------------------------------------------------------

  /// Fields
  uint64_t num_rows;
  uint64_t num_cols;
  uint64_t num_nonzeros;
  uint64_t num_nonzeros_lowerTri; /// triangular matrix read stats
  uint64_t num_nonzeros_upperTri;
  uint64_t num_nonzeros_lowerTri_strict;
  uint64_t num_nonzeros_upperTri_strict;
  CooTuple<T> *coo_tuples;

  ///---------------------------------------------------------------------
  /// Methods
  ///---------------------------------------------------------------------

  /// Constructor
  CooMatrix() : num_rows(0), num_cols(0), num_nonzeros(0),
                num_nonzeros_lowerTri(0), num_nonzeros_upperTri(0),
                num_nonzeros_lowerTri_strict(0), num_nonzeros_upperTri_strict(0),
                coo_tuples(NULL) {}

  //===----------------------------------------------------------------------===//
  // Clear
  //===----------------------------------------------------------------------===//
  void Clear()
  {
    if (coo_tuples)
      delete[] coo_tuples;
    coo_tuples = NULL;
  }

  /// Destructor
  ~CooMatrix()
  {
    /// do nothing. coo_tuples is now cleared from else-where.
    /// Clear();
  }

  /// Display matrix to stdout
  void Display()
  {
    cout << "COO Matrix (" << num_rows << " rows, " << num_cols << " columns, " << num_nonzeros << " non-zeros):\n";
    cout << "Ordinal, Row, Column, Value\n";
    for (uint64_t i = 0; i < num_nonzeros; i++)
    {
      cout << '\t' << i << ',' << coo_tuples[i].row << ',' << coo_tuples[i].col << ',' << coo_tuples[i].val << "\n";
    }
  }

  //===----------------------------------------------------------------------===//
  /// Builds a MARKET COO sparse from the given file.
  //===----------------------------------------------------------------------===//
  void InitMarket(
      const string &market_filename,
      T default_value = 1.0,
      bool verbose = false)
  {
    if (verbose)
    {
      printf("Reading... ");
      fflush(stdout);
    }

    if (coo_tuples)
    {
      fprintf(stderr, "ERROR: Matrix already constructed (abrupt exit)!\n");
      // updated code should avoid coming to this path
      exit(1);
    }

    std::ifstream ifs;
    ifs.open(market_filename.c_str(), std::ifstream::in);
    if (!ifs.good())
    {
      fprintf(stderr, "Error opening file\n");
      exit(1);
    }

    bool array = false;
    bool symmetric = false;
    bool skew = false;
    uint64_t current_nz = 0;
    int nparsed = -1;
    char line[1024];

    if (verbose)
    {
      printf("Parsing... ");
      fflush(stdout);
    }

    while (true)
    {
      ifs.getline(line, 1024);
      if (!ifs.good())
      {
        /// Done
        break;
      }

      if (line[0] == '%')
      {
        /// Comment
        if (line[1] == '%')
        {
          /// Banner
          symmetric = (strstr(line, "symmetric") != NULL);
          skew = (strstr(line, "skew") != NULL);
          array = (strstr(line, "array") != NULL);

          if (verbose)
          {
            printf("(symmetric: %d, skew: %d, array: %d) ", symmetric, skew, array);
            fflush(stdout);
          }
        }
      }
      else if (nparsed == -1)
      {
        /// Problem description
        nparsed = sscanf(line, "%" PRIu64 " %" PRIu64 " %" PRIu64, &num_rows, &num_cols, &num_nonzeros);
        if ((!array) && (nparsed == 3))
        {
          if (symmetric)
            num_nonzeros *= 2;

          /// Allocate coo matrix
          coo_tuples = new CooTuple<T>[num_nonzeros];
          current_nz = 0;
        }
        else if (array && (nparsed == 2))
        {
          /// Allocate coo matrix
          num_nonzeros = num_rows * num_cols;
          coo_tuples = new CooTuple<T>[num_nonzeros];
          current_nz = 0;
        }
        else
        {
          fprintf(stderr, "Error parsing MARKET matrix: invalid problem description: %s\n", line);
          exit(1);
        }
      }
      else
      {
        /// Edge
        if (current_nz >= num_nonzeros)
        {
          fprintf(stderr, "Error parsing MARKET matrix: encountered more than %" PRIu64 " num_nonzeros\n", num_nonzeros);
          exit(1);
        }

        uint64_t row, col;
        T val;
        double tempVal = 0.0;

        if (array)
        {
          if (sscanf(line, "%lf", &tempVal) != 1) /// using tempVal instead of templated T val to avoid warning
          {
            fprintf(stderr, "Error parsing MARKET matrix: badly formed current_nz: '%s' at edge %" PRIu64 "\n", line, current_nz);
            exit(1);
          }
          val = (T)tempVal;
          col = (current_nz / num_rows);
          row = (current_nz - (num_rows * col));
          coo_tuples[current_nz] = CooTuple<T>(row, col, val); /// Convert indices to zero-based

          if (row > col)
          {
            num_nonzeros_lowerTri_strict++;
            num_nonzeros_lowerTri++;
          }
          else if (row < col)
          {
            num_nonzeros_upperTri_strict++;
            num_nonzeros_upperTri++;
          }
          else /// equal or diagonals
          {
            num_nonzeros_lowerTri++;
            num_nonzeros_upperTri++;
          }
        }
        else
        {
          /// Parse nonzero (note: using strtol and strtod is 2x faster than sscanf or istream parsing)
          char *l = line;
          char *t = NULL;

          /// parse row
          row = strtol(l, &t, 0);
          if (t == l)
          {
            fprintf(stderr, "Error parsing MARKET matrix: badly formed row at edge %" PRIu64 "\n", current_nz);
            exit(1);
          }
          l = t;

          /// parse col
          col = strtol(l, &t, 0);
          if (t == l)
          {
            fprintf(stderr, "Error parsing MARKET matrix: badly formed col at edge %" PRIu64 "\n", current_nz);
            exit(1);
          }
          l = t;

          /// parse val
          val = strtod(l, &t);
          if (t == l)
          {
            val = default_value;
          }

          coo_tuples[current_nz] = CooTuple<T>(row - 1, col - 1, val); /// Convert indices to zero-based

          if (row - 1 > col - 1)
          {
            num_nonzeros_lowerTri_strict++;
            num_nonzeros_lowerTri++;
          }
          else if (row - 1 < col - 1)
          {
            num_nonzeros_upperTri_strict++;
            num_nonzeros_upperTri++;
          }
          else /// equal or diagonals
          {
            num_nonzeros_lowerTri++;
            num_nonzeros_upperTri++;
          }
        }

        current_nz++;

        if (symmetric && (row != col))
        {
          coo_tuples[current_nz].row = coo_tuples[current_nz - 1].col;
          coo_tuples[current_nz].col = coo_tuples[current_nz - 1].row;
          coo_tuples[current_nz].val = coo_tuples[current_nz - 1].val * (skew ? -1 : 1);
          current_nz++;
        }
      }
    }

    /// Adjust nonzero count (nonzeros along the diagonal aren't reversed)
    num_nonzeros = current_nz;

    if (symmetric)
    {
      /// we only do one half if matrix is symmetric, so update num according to lower Triangle read.
      num_nonzeros_upperTri = num_nonzeros_lowerTri;
      num_nonzeros_upperTri_strict = num_nonzeros_lowerTri_strict;
    }

    if (verbose)
    {
      printf("done. ");
      fflush(stdout);
    }

    ifs.close();
  }
};

/// Sort by rows, then columns
struct CooComparatorRow
{
  template <typename CooTuple>
  bool operator()(const CooTuple &a, const CooTuple &b) const
  {
    return ((a.row < b.row) || ((a.row == b.row) && (a.col < b.col)));
  }
};

/// Sort by cols, then rows
struct CooComparatorCol
{
  template <typename CooTuple>
  bool operator()(const CooTuple &a, const CooTuple &b) const
  {
    return ((a.col < b.col) || ((a.col == b.col) && (a.row < b.row)));
  }
};

//===----------------------------------------------------------------------===//
/// CSR matrix type
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// CSR sparse format matrix
//===----------------------------------------------------------------------===//
template <typename T>
struct CsrMatrix
{

  uint64_t num_rows;
  uint64_t num_cols;
  uint64_t num_nonzeros;
  uint64_t num_nonzeros_lowerTri; /// triangular matrix read stats from COO
  uint64_t num_nonzeros_upperTri;
  uint64_t num_nonzeros_lowerTri_strict;
  uint64_t num_nonzeros_upperTri_strict;
  uint64_t *row_offsets;
  uint64_t *column_indices;
  T *values;

  /**
   * Initializer
   */
  void Init(
      CooMatrix<T> *coo_matrix,
      int readMode,
      bool verbose = false)
  {
    num_rows = coo_matrix->num_rows;
    num_cols = coo_matrix->num_cols;
    num_nonzeros = coo_matrix->num_nonzeros;
    num_nonzeros_lowerTri = coo_matrix->num_nonzeros_lowerTri;
    num_nonzeros_upperTri = coo_matrix->num_nonzeros_upperTri;
    num_nonzeros_lowerTri_strict = coo_matrix->num_nonzeros_lowerTri_strict;
    num_nonzeros_upperTri_strict = coo_matrix->num_nonzeros_upperTri_strict;

    /// Sort by rows, then columns
    if (verbose)
      printf("Ordering...");
    fflush(stdout);
    std::stable_sort(coo_matrix->coo_tuples, coo_matrix->coo_tuples + num_nonzeros, CooComparatorRow());
    if (verbose)
      printf("done.");
    fflush(stdout);

    row_offsets = new uint64_t[num_rows + 1];

    if (readMode == LOWER_TRI_STRICT)
    {
      column_indices = new uint64_t[num_nonzeros_lowerTri_strict];
      values = new T[num_nonzeros_lowerTri_strict];
    }
    else if (readMode == LOWER_TRI)
    {
      column_indices = new uint64_t[num_nonzeros_lowerTri];
      values = new T[num_nonzeros_lowerTri];
    }
    else if (readMode == UPPER_TRI_STRICT)
    {
      column_indices = new uint64_t[num_nonzeros_upperTri_strict];
      values = new T[num_nonzeros_upperTri_strict];
    }
    else if (readMode == UPPER_TRI)
    {
      column_indices = new uint64_t[num_nonzeros_upperTri];
      values = new T[num_nonzeros_upperTri];
    }
    else /// DEFAULT
    {
      column_indices = new uint64_t[num_nonzeros];
      values = new T[num_nonzeros];
    }

    uint64_t prev_row = -1;
    uint64_t curr_idx = 0;
    for (uint64_t current_nz = 0; current_nz < num_nonzeros; current_nz++)
    {
      uint64_t current_row = coo_matrix->coo_tuples[current_nz].row;
      uint64_t current_col = coo_matrix->coo_tuples[current_nz].col;

      if (((readMode == LOWER_TRI_STRICT) || (readMode == LOWER_TRI)) &&
          (current_row > current_col))
      {
        /// Fill in rows up to and including the current row
        /// printf("\t\tLower> current_nz: %lld, curr_idx: %lld\n", current_nz, curr_idx);
        for (uint64_t row = prev_row + 1; row <= current_row; row++)
        {
          row_offsets[row] = curr_idx;
        }
        prev_row = current_row;

        column_indices[curr_idx] = coo_matrix->coo_tuples[current_nz].col;
        values[curr_idx] = coo_matrix->coo_tuples[current_nz].val;
        curr_idx++;
      }
      else if (((readMode == UPPER_TRI_STRICT) || (readMode == UPPER_TRI)) &&
               (current_row < current_col))
      {
        for (uint64_t row = prev_row + 1; row <= current_row; row++)
        {
          row_offsets[row] = curr_idx;
        }
        prev_row = current_row;

        column_indices[curr_idx] = coo_matrix->coo_tuples[current_nz].col;
        values[curr_idx] = coo_matrix->coo_tuples[current_nz].val;
        curr_idx++;
      }
      else if (((readMode == UPPER_TRI) || (readMode == LOWER_TRI)) &&
               (current_row == current_col)) /// diagonals
      {
        for (uint64_t row = prev_row + 1; row <= current_row; row++)
        {
          row_offsets[row] = curr_idx;
        }
        prev_row = current_row;

        column_indices[curr_idx] = coo_matrix->coo_tuples[current_nz].col;
        values[curr_idx] = coo_matrix->coo_tuples[current_nz].val;
        curr_idx++;
      }
      else if (readMode == DEFAULT) /// standard matrix read
      {
        /// Fill in rows up to and including the current row
        /// printf("\t\tLower> current_nz: %lld, curr_idx: %lld\n", current_nz, curr_idx);
        for (uint64_t row = prev_row + 1; row <= current_row; row++)
        {
          row_offsets[row] = curr_idx;
        }
        prev_row = current_row;

        column_indices[curr_idx] = coo_matrix->coo_tuples[current_nz].col;
        values[curr_idx] = coo_matrix->coo_tuples[current_nz].val;
        curr_idx++;
      }

    } /// end-loop

    /// Fill out any trailing edgeless vertices (and the end-of-list element)
    for (uint64_t row = prev_row + 1; row <= num_rows; row++)
    {
      if (readMode == LOWER_TRI_STRICT)
        row_offsets[row] = num_nonzeros_lowerTri_strict;
      else if (readMode == UPPER_TRI_STRICT)
        row_offsets[row] = num_nonzeros_upperTri_strict;
      else if (readMode == UPPER_TRI)
        row_offsets[row] = num_nonzeros_upperTri;
      else if (readMode == LOWER_TRI)
        row_offsets[row] = num_nonzeros_lowerTri;
      else /// DEFAULT
        row_offsets[row] = num_nonzeros;
    }
  }

  //===----------------------------------------------------------------------===//
  /// Clear
  //===----------------------------------------------------------------------===//
  void Clear()
  {
    if (row_offsets)
      delete[] row_offsets;
    if (column_indices)
      delete[] column_indices;
    if (values)
      delete[] values;

    row_offsets = NULL;
    column_indices = NULL;
    values = NULL;
  }

  //===----------------------------------------------------------------------===//
  /// Constructor
  //===----------------------------------------------------------------------===//
  CsrMatrix(
      CooMatrix<T> *coo_matrix,
      int readMode,
      bool verbose = false)
  {
    Init(coo_matrix, readMode, verbose);
  }

  //===----------------------------------------------------------------------===//
  /// Destructor
  //===----------------------------------------------------------------------===//
  ~CsrMatrix()
  {
    Clear();
  }
};

//===----------------------------------------------------------------------===//
/// CSC sparse format matrix
//===----------------------------------------------------------------------===//
template <typename T>
struct CscMatrix
{

  uint64_t num_rows;
  uint64_t num_cols;
  uint64_t num_nonzeros;
  uint64_t *col_offsets;
  uint64_t *row_indices;
  T *values;

  //===----------------------------------------------------------------------===//
  /// Initializer
  //===----------------------------------------------------------------------===//
  void Init(
      CooMatrix<T> *coo_matrix,
      bool verbose = false)
  {
    num_rows = coo_matrix->num_rows;
    num_cols = coo_matrix->num_cols;
    num_nonzeros = coo_matrix->num_nonzeros;

    /// Sort by rows, then columns
    if (verbose)
      printf("Ordering...");
    fflush(stdout);
    std::stable_sort(coo_matrix->coo_tuples, coo_matrix->coo_tuples + num_nonzeros, CooComparatorCol());
    if (verbose)
      printf("done.");
    fflush(stdout);

    col_offsets = new uint64_t[num_cols + 1];
    row_indices = new uint64_t[num_nonzeros];
    values = new T[num_nonzeros];

    uint64_t prev_col = -1;
    for (uint64_t current_nz = 0; current_nz < num_nonzeros; current_nz++)
    {
      uint64_t current_col = coo_matrix->coo_tuples[current_nz].col;

      /// Fill in cols up to and including the current col
      for (uint64_t col = prev_col + 1; col <= current_col; col++)
      {
        col_offsets[col] = current_nz;
      }
      prev_col = current_col;

      row_indices[current_nz] = coo_matrix->coo_tuples[current_nz].row;
      values[current_nz] = coo_matrix->coo_tuples[current_nz].val;
    }

    /// Fill out any trailing edgeless vertices (and the end-of-list element)
    for (uint64_t col = prev_col + 1; col <= num_cols; col++)
    {
      col_offsets[col] = num_nonzeros;
    }
  }

  /// Clear
  void Clear()
  {
    if (col_offsets)
      delete[] col_offsets;
    if (row_indices)
      delete[] row_indices;
    if (values)
      delete[] values;

    col_offsets = NULL;
    row_indices = NULL;
    values = NULL;
  }

  /// Constructor
  CscMatrix(
      CooMatrix<T> *coo_matrix,
      bool verbose = false)
  {
    Init(coo_matrix, verbose);
  }

  /// Destructor
  ~CscMatrix()
  {
    Clear();
  }
};

//===----------------------------------------------------------------------===//
/// DCSR matrix type
//===----------------------------------------------------------------------===//

/// DCSR sparse format matrix
template <typename T>
struct DcsrMatrix
{
  uint64_t num_rows;
  uint64_t num_cols;
  uint64_t num_nonzeros;
  uint64_t A1pos_size = 0;
  uint64_t A1crd_size = 0;
  uint64_t A2pos_size = 0;
  uint64_t A2crd_size = 0;
  uint64_t Aval_size = 0;
  uint64_t *A1pos;
  uint64_t *A1crd;
  uint64_t *A2pos;
  uint64_t *A2crd;
  T *Aval;

  /// Initializer
  void Init(
      CooMatrix<T> *coo_matrix,
      bool verbose = false)
  {
    num_rows = coo_matrix->num_rows;
    num_cols = coo_matrix->num_cols;
    num_nonzeros = coo_matrix->num_nonzeros;

    /// Sort by rows, then columns
    if (verbose)
      printf("Ordering...");
    fflush(stdout);
    std::stable_sort(coo_matrix->coo_tuples, coo_matrix->coo_tuples + num_nonzeros, CooComparatorRow());
    if (verbose)
      printf("done.");
    fflush(stdout);

    A1pos = new uint64_t[2];
    A1crd = new uint64_t[num_rows];
    A2pos = new uint64_t[num_rows + 1];
    A2crd = new uint64_t[num_nonzeros];
    Aval = new T[num_nonzeros];

    uint64_t prev_row = -1;
    for (uint64_t current_nz = 0; current_nz < num_nonzeros; current_nz++)
    {
      uint64_t current_row = coo_matrix->coo_tuples[current_nz].row;

      /// Fill in rows up to and including the current row
      if (current_row == prev_row)
      {
        /// A1crd[A1_crd] = current_row;
        A2pos[A2pos_size] = current_nz;
      }
      else
      {
        A2pos[A2pos_size++] = current_nz;
        A1crd[A1crd_size++] = current_row;

        prev_row = current_row;
      }

      A2crd[A2crd_size++] = coo_matrix->coo_tuples[current_nz].col;
      Aval[Aval_size++] = coo_matrix->coo_tuples[current_nz].val;
    }

    A2pos[A2pos_size++] = A2crd_size;
    A1pos[A1pos_size++] = 0;
    A1pos[A1pos_size++] = A1crd_size;
  }

  /// Clear
  void Clear()
  {
    if (A1pos)
      delete[] A1pos;
    if (A1crd)
      delete[] A1crd;
    if (A2pos)
      delete[] A2pos;
    if (A2crd)
      delete[] A2crd;
    if (Aval)
      delete[] Aval;

    A1pos = NULL;
    A1crd = NULL;
    A2pos = NULL;
    A2crd = NULL;
    Aval = NULL;
  }

  /// Constructor
  DcsrMatrix(
      CooMatrix<T> *coo_matrix,
      bool verbose = false)
  {
    Init(coo_matrix, verbose);
  }

  /// Destructor
  ~DcsrMatrix()
  {
    Clear();
  }
};

///===----------------------------------------------------------------------===//
/// ELLPACK matrix type
///===----------------------------------------------------------------------===//

template <typename T>
struct EllpackMatrix
{
  uint64_t num_rows;
  uint64_t num_cols;
  uint64_t num_nonzeros;
  uint64_t *col_crd;
  T *Aval;

  /// Initializer
  void Init(CooMatrix<T> *coo_matrix, bool verbose = false)
  {
    num_rows = coo_matrix->num_rows;
    num_cols = 0;
    num_nonzeros = coo_matrix->num_nonzeros;

    /// Sort by rows, then columns
    if (verbose)
      printf("Ordering...");
    fflush(stdout);
    std::stable_sort(coo_matrix->coo_tuples, coo_matrix->coo_tuples + num_nonzeros, CooComparatorRow());
    if (verbose)
      printf("done.");
    fflush(stdout);

    /// Calculate the column count
    uint64_t max = 0;
    uint64_t buffer = 0;
    // int current = -1;
    uint64_t current = num_nonzeros > 0 ? coo_matrix->coo_tuples[0].row : 0;
    for (uint64_t i = 0; i < num_nonzeros; i++)
    {
      if (coo_matrix->coo_tuples[i].row == current)
      {
        ++buffer;
      }
      else
      {
        if (buffer > max)
          max = buffer;
        buffer = 1;
        current = coo_matrix->coo_tuples[i].row;
      }
    }
    if (buffer > max)
      max = buffer;
    num_cols = max;

    /// Create the column coordinate list
    /// TODO: This terrible, but it works
    col_crd = new uint64_t[num_rows * num_cols];
    Aval = new T[num_rows * num_cols];
    uint64_t index = 0;

    /// Build the column coordinates/value array
    for (uint64_t i = 0; i < num_rows; i++)
    {
      /// In this loop, get all non-zero column coordinates and track
      /// how many we have found
      uint64_t found_cols = 0;
      for (uint64_t j = 0; j < num_nonzeros; j++)
      {
        if (coo_matrix->coo_tuples[j].row == i)
        {
          ++found_cols;
        }
      }

      /// If the number of columns we have found in the row is less than
      /// the block, we need to add some zeros to create the block
      if (found_cols == num_cols)
      {
        for (uint64_t j = 0; j < num_nonzeros; j++)
        {
          if (coo_matrix->coo_tuples[j].row == i)
          {
            col_crd[index] = coo_matrix->coo_tuples[j].col;
            Aval[index] = coo_matrix->coo_tuples[j].val;
            ++index;
          }
        }
        continue;
      }

      /// If there were no found columns, we'll just add zeros
      if (found_cols == 0)
      {
        for (uint64_t j = 0; j < num_cols; j++)
        {
          col_crd[index] = j;
          Aval[index] = 0;
          ++index;
        }
        continue;
      }

      /// If we have an odd number, we need to add preceeding elements before
      /// the actual non-zero indicies
      for (uint64_t j = 0; j < num_nonzeros; j++)
      {
        if (coo_matrix->coo_tuples[j].row == i)
        {
          // TODO: This IS NOT portable
          for (uint64_t k = 0; k < coo_matrix->coo_tuples[j].col && found_cols < num_cols; k++)
          {
            col_crd[index] = k;
            Aval[index] = 0;
            ++index;
            ++found_cols;
          }
          col_crd[index] = coo_matrix->coo_tuples[j].col;
          Aval[index] = coo_matrix->coo_tuples[j].val;
          ++index;
        }
      }
    }
  }

  /// Clear matrix
  void Clear()
  {
    delete[] col_crd;
    delete[] Aval;
  }

  /// The constructor- calls the initializer
  EllpackMatrix(CooMatrix<T> *coo_matrix, bool verbose = false)
  {
    Init(coo_matrix, verbose);
  }

  /// Destructor
  ~EllpackMatrix()
  {
    Clear();
  }
};

//===----------------------------------------------------------------------===//
/// COO tensor 3D type.  A COO tensor is just a vector of edge tuples.  Tuples are sorted
/// first by first dim, then by second dim and so on.
//===----------------------------------------------------------------------===//
template <typename T>
struct Coo3DTensor
{
  //---------------------------------------------------------------------
  // Type definitions and constants
  //---------------------------------------------------------------------

  /// COO edge tuple
  struct Coo3DTuple
  {
    uint64_t index_i;
    uint64_t index_j;
    uint64_t index_k;
    T val;

    Coo3DTuple() {}
    Coo3DTuple(uint64_t index_i, uint64_t index_j, uint64_t index_k) : index_i(index_i), index_j(index_j), index_k(index_k) {}
    Coo3DTuple(uint64_t index_i, uint64_t index_j, uint64_t index_k, T val) : index_i(index_i), index_j(index_j), index_k(index_k), val(val) {}
  };

  //---------------------------------------------------------------------
  /// Data members
  //---------------------------------------------------------------------

  /// Fields
  uint64_t num_index_i;
  uint64_t num_index_j;
  uint64_t num_index_k;
  uint64_t num_nonzeros;
  Coo3DTuple *coo_3dtuples;

  //---------------------------------------------------------------------
  /// Methods
  //---------------------------------------------------------------------

  /// Constructor
  Coo3DTensor() : num_index_i(0), num_index_j(0), num_index_k(0), num_nonzeros(0), coo_3dtuples(NULL) {}

  /// Clear
  void Clear()
  {
    if (coo_3dtuples)
      delete[] coo_3dtuples;
    coo_3dtuples = NULL;
  }

  /// Destructor
  ~Coo3DTensor()
  {
    /// do nothing. coo_3dtuples is now cleared from else-where.
    // Clear();
  }

  // Display matrix to stdout
  void Display()
  {
    cout << "COO Tensor 3D (" << num_index_i << " index_i, " << num_index_j << " index_j, " << num_index_k << " index_k, " << num_nonzeros << " non-zeros):\n";
    cout << "Ordinal, index_i, index_j, index_k, Value\n";
    for (uint64_t i = 0; i < num_nonzeros; i++)
    {
      cout << '\t' << i << ',' << coo_3dtuples[i].index_i << ',' << coo_3dtuples[i].index_j << ',' << coo_3dtuples[i].index_k << ',' << coo_3dtuples[i].val << "\n";
    }
  }

  /// Builds a Frostt COO sparse from the given file.
  void InitFrostt(
      const string &filename,
      T default_value = 1.0,
      bool verbose = false)
  {
    if (verbose)
    {
      printf("Reading... ");
      fflush(stdout);
    }

    if (coo_3dtuples)
    {
      fprintf(stderr, "\tERROR: Tensor already constructed (abrupt exit)!\n");
      /// updated code should avoid coming to this path
      exit(1);
    }

    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);
    if (!ifs.good())
    {
      fprintf(stderr, "Error opening file\n");
      exit(1);
    }

    uint64_t current_nz = 0;
    int nparsed = -1;
    char line[1024];

    if (verbose)
    {
      printf("Parsing... ");
      fflush(stdout);
    }

    while (true)
    {
      ifs.getline(line, 1024);
      if (!ifs.good())
      {
        /// Done
        break;
      }

      if (nparsed == -1)
      {
        /// Problem description
        nparsed = sscanf(line, "%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64, &num_index_i, &num_index_j, &num_index_k, &num_nonzeros);

        if (nparsed == 4)
        {
          /// Allocate coo matrix
          coo_3dtuples = new Coo3DTuple[num_nonzeros];
          current_nz = 0;
        }
        else
        {
          llvm::errs() << __FILE__ << " " << __LINE__ << "ERROR: parsing FROSTT tensor: invalid problem description: " << line << "\n";
        }
      }
      else
      {
        /// Edge
        if (current_nz >= num_nonzeros)
        {
          llvm::errs() << __FILE__ << " " << __LINE__ << "ERROR: parsing FROSTT tensor: encountered more than " << num_nonzeros << "\n";
        }

        uint64_t idx_i, idx_j, idx_k;
        T val;

        /// Parse nonzero (note: using strtol and strtod is 2x faster than sscanf or istream parsing)
        char *l = line;
        char *t = NULL;

        /// parse idx_i
        idx_i = strtol(l, &t, 0);
        if (t == l)
        {
          llvm::errs() << __FILE__ << " " << __LINE__ << "ERROR: parsing FROSTT tensor: badly formed row at edge " << current_nz << "\n";
        }
        l = t;

        /// parse idx_j
        idx_j = strtol(l, &t, 0);
        if (t == l)
        {
          llvm::errs() << __FILE__ << " " << __LINE__ << "ERROR: parsing FROSTT tensor: badly formed col at edge " << current_nz << "\n";
        }
        l = t;

        /// parse idx_k
        idx_k = strtol(l, &t, 0);
        if (t == l)
        {
          llvm::errs() << __FILE__ << " " << __LINE__ << "ERROR: parsing FROSTT tensor: badly formed col at edge " << current_nz << "\n";
        }
        l = t;

        /// parse val
        val = strtod(l, &t);
        if (t == l)
        {
          val = default_value;
        }

        // coo_3dtuples[current_nz] = Coo3DTuple(idx_i - 1, idx_j - 1, idx_k - 1, val);    // Convert indices to zero-based --> This is incorrect!!
        coo_3dtuples[current_nz] = Coo3DTuple(idx_i, idx_j, idx_k, val);

        current_nz++;
      }
    }

    /// Adjust nonzero count (nonzeros along the diagonal aren't reversed)
    num_nonzeros = current_nz;

    if (verbose)
    {
      printf("done. ");
      fflush(stdout);
    }

    ifs.close();
  }
};

/// Sort by rows, then columns
struct Coo3DTensorComparator
{
  template <typename Coo3DTuple>
  bool operator()(const Coo3DTuple &a, const Coo3DTuple &b) const
  {
    return ((a.index_i < b.index_i) || ((a.index_i == b.index_i) && (a.index_j < b.index_j)) || ((a.index_i == b.index_i) && (a.index_j == b.index_j) && (a.index_k < b.index_k)));
  }
};

//===----------------------------------------------------------------------===//
///  CSF tensor type
//===----------------------------------------------------------------------===//

/// CSF sparse format tensor

template <typename T>
struct Csf3DTensor
{
  uint64_t num_index_i;
  uint64_t num_index_j;
  uint64_t num_index_k;
  uint64_t num_nonzeros;
  uint64_t A1pos_size = 0;
  uint64_t A1crd_size = 0;
  uint64_t A2pos_size = 0;
  uint64_t A2crd_size = 0;
  uint64_t A3pos_size = 0;
  uint64_t A3crd_size = 0;
  uint64_t Aval_size = 0;
  uint64_t *A1pos;
  uint64_t *A1crd;
  uint64_t *A2pos;
  uint64_t *A2crd;
  uint64_t *A3pos;
  uint64_t *A3crd;
  T *Aval;

  /// Initializer
  void Init(
      Coo3DTensor<T> *coo_3dtensor,
      bool verbose = false)
  {
    num_index_i = coo_3dtensor->num_index_i;
    num_index_j = coo_3dtensor->num_index_j;
    num_index_k = coo_3dtensor->num_index_k;
    num_nonzeros = coo_3dtensor->num_nonzeros;

    /// Sort by rows, then columns
    if (verbose)
      printf("Ordering...");
    fflush(stdout);
    std::stable_sort(coo_3dtensor->coo_3dtuples, coo_3dtensor->coo_3dtuples + num_nonzeros, Coo3DTensorComparator());
    if (verbose)
      printf("done.");
    fflush(stdout);

    A1pos = new uint64_t[2];
    A1crd = new uint64_t[num_nonzeros];
    A2pos = new uint64_t[num_nonzeros + 1];
    A2crd = new uint64_t[num_nonzeros];
    A3pos = new uint64_t[num_nonzeros + 1];
    A3crd = new uint64_t[num_nonzeros];
    Aval = new T[num_nonzeros];

    uint64_t prev_index_i = -1;
    uint64_t prev_index_j = -1;
    for (uint64_t current_nz = 0; current_nz < num_nonzeros; current_nz++)
    {
      uint64_t current_index_i = coo_3dtensor->coo_3dtuples[current_nz].index_i;
      uint64_t current_index_j = coo_3dtensor->coo_3dtuples[current_nz].index_j;

      if (current_index_j != prev_index_j)
      {
        A3pos[A3pos_size++] = current_nz;
        A2crd[A2crd_size++] = current_index_j;

        if (current_index_i != prev_index_i)
        {
          A2pos[A2pos_size++] = A2crd_size - 1;
          A1crd[A1crd_size++] = current_index_i;
          prev_index_i = current_index_i;
        }
        else
        { /// current_index_i == prev_index_i
          /// NOTHING
          ;
        }
        prev_index_j = current_index_j;
      }
      else
      { /// current_index_j == prev_index_j
        if (current_index_i != prev_index_i)
        {
          A3pos[A3pos_size++] = current_nz;
          A2crd[A2crd_size++] = current_index_j;
          A2pos[A2pos_size++] = A2crd_size - 1;
          A1crd[A1crd_size++] = current_index_i;
          prev_index_i = current_index_i;
        }
        else
        { /// current_index_i == prev_index_i
          /// Nothing
          ;
        }
      }

      A3crd[A3crd_size++] = coo_3dtensor->coo_3dtuples[current_nz].index_k;
      Aval[Aval_size++] = coo_3dtensor->coo_3dtuples[current_nz].val;
    }

    A3pos[A3pos_size++] = A3crd_size;
    A2pos[A2pos_size++] = A2crd_size;
    A1pos[A1pos_size++] = 0;
    A1pos[A1pos_size++] = A1crd_size;
  }

  /// Clear
  void Clear()
  {
    if (A1pos)
      delete[] A1pos;
    if (A1crd)
      delete[] A1crd;
    if (A2pos)
      delete[] A2pos;
    if (A2crd)
      delete[] A2crd;
    if (A3pos)
      delete[] A3pos;
    if (A3crd)
      delete[] A3crd;
    if (Aval)
      delete[] Aval;

    A1pos = NULL;
    A1crd = NULL;
    A2pos = NULL;
    A2crd = NULL;
    A3pos = NULL;
    A3crd = NULL;
    Aval = NULL;
  }

  /// Constructor
  Csf3DTensor(
      Coo3DTensor<T> *coo_3dtensor,
      bool verbose = false)
  {
    Init(coo_3dtensor, verbose);
  }

  /// Destructor
  ~Csf3DTensor()
  {
    Clear();
  }
};

//===----------------------------------------------------------------------===//
///  Mode-Generic matrix type
//===----------------------------------------------------------------------===//

/// MG sparse format tensor
template <typename T>
struct Mg3DTensor
{
  uint64_t num_index_i;
  uint64_t num_index_j;
  uint64_t num_index_k;
  uint64_t num_nonzeros;
  uint64_t A1pos_size = 0;
  uint64_t A1crd_size = 0;
  uint64_t A2pos_size = 0;
  uint64_t A2crd_size = 0;
  uint64_t A3pos_size = 0;
  uint64_t A3crd_size = 0;
  uint64_t Aval_size = 0;
  uint64_t *A1pos;
  uint64_t *A1crd;
  uint64_t *A2pos;
  uint64_t *A2crd;
  uint64_t *A3pos;
  uint64_t *A3crd;
  T *Aval;

  /// Initializer
  void Init(
      Coo3DTensor<T> *coo_3dtensor,
      bool verbose = false)
  {
    num_index_i = coo_3dtensor->num_index_i;
    num_index_j = coo_3dtensor->num_index_j;
    num_index_k = coo_3dtensor->num_index_k;
    num_nonzeros = coo_3dtensor->num_nonzeros;

    /// Sort by rows, then columns
    if (verbose)
      printf("Ordering...");
    fflush(stdout);
    std::stable_sort(coo_3dtensor->coo_3dtuples, coo_3dtensor->coo_3dtuples + num_nonzeros, Coo3DTensorComparator());
    if (verbose)
      printf("done.");
    fflush(stdout);

    /// Evaluate size of A1crd, A2crd
    uint64_t alloc_size_A1crd = 0; /// same with alloc_size_A2crd
    uint64_t alloc_prev_index_i = -1;
    uint64_t alloc_prev_index_j = -1;
    for (uint64_t current_nz = 0; current_nz < num_nonzeros; current_nz++)
    {
      uint64_t current_index_i = coo_3dtensor->coo_3dtuples[current_nz].index_i;
      uint64_t current_index_j = coo_3dtensor->coo_3dtuples[current_nz].index_j;

      if (current_index_i != alloc_prev_index_i || current_index_j != alloc_prev_index_j)
      {
        alloc_size_A1crd++;
        alloc_prev_index_i = current_index_i;
        alloc_prev_index_j = current_index_j;
      }
    }
    A1pos = new uint64_t[2];
    A1crd = new uint64_t[alloc_size_A1crd];
    A2pos = new uint64_t[1];
    A2crd = new uint64_t[alloc_size_A1crd];
    A3pos = new uint64_t[1];
    A3crd = new uint64_t[1];
    Aval = new T[alloc_size_A1crd * num_index_k];

    A3pos[A3pos_size++] = num_index_k;

    uint64_t prev_index_i = -1;
    uint64_t prev_index_j = -1;
    for (uint64_t current_nz = 0; current_nz < num_nonzeros; current_nz++)
    {
      uint64_t current_index_i = coo_3dtensor->coo_3dtuples[current_nz].index_i;
      uint64_t current_index_j = coo_3dtensor->coo_3dtuples[current_nz].index_j;
      uint64_t current_index_k = coo_3dtensor->coo_3dtuples[current_nz].index_k;
      uint64_t current_val = coo_3dtensor->coo_3dtuples[current_nz].val;

      /// Fill in rows up to and including the current row
      if (current_index_i != prev_index_i || current_index_j != prev_index_j)
      {
        A1crd[A1crd_size++] = current_index_i;
        A2crd[A2crd_size++] = current_index_j;

        /// Fill previous (i,j)
        if (Aval_size % num_index_k != 0)
        {
          for (uint64_t i = Aval_size % num_index_k; i < num_index_k; i++)
            Aval[Aval_size++] = 0;
        }
        /// Fill current (i,j)
        for (uint64_t i = Aval_size % num_index_k; i < current_index_k; i++)
          Aval[Aval_size++] = 0;
        Aval[Aval_size++] = current_val;

        prev_index_i = current_index_i;
        prev_index_j = current_index_j;
      }
      else
      {
        for (uint64_t i = Aval_size % num_index_k; i < current_index_k; i++)
          Aval[Aval_size++] = 0;
        Aval[Aval_size++] = current_val;
      }
    }

    A1pos[A1pos_size++] = 0;
    A1pos[A1pos_size++] = A1crd_size;
  }

  /// Clear
  void Clear()
  {
    if (A1pos)
      delete[] A1pos;
    if (A1crd)
      delete[] A1crd;
    if (A2pos)
      delete[] A2pos;
    if (A2crd)
      delete[] A2crd;
    if (A3pos)
      delete[] A3pos;
    if (A3crd)
      delete[] A3crd;
    if (Aval)
      delete[] Aval;

    A1pos = NULL;
    A1crd = NULL;
    A2pos = NULL;
    A2crd = NULL;
    A3pos = NULL;
    A3crd = NULL;
    Aval = NULL;
  }

  /// Constructor
  Mg3DTensor(
      Coo3DTensor<T> *coo_3dtensor,
      bool verbose = false)
  {
    Init(coo_3dtensor, verbose);
  }

  /// Destructor
  ~Mg3DTensor()
  {
    Clear();
  }
};

template <typename T>
static std::map<int32_t, CooMatrix<T> *> CooTracking;

template <typename T>
static std::map<int32_t, Coo3DTensor<T> *> Coo3DTracking;

/// matrix read wrapper: initiates file read only once.
/// assumption: read_input_sizes_2D() and read_input_2D() are called in order
///             and only once for each fileID/file.
template <typename T>
struct FileReaderWrapper
{
  std::string filename;
  int32_t ID;
  bool is3D;

  CooMatrix<T> *coo_matrix;
  Coo3DTensor<T> *coo_3dtensor;

  bool readFileNameStr(int32_t fileID)
  {
    char *pSparseInput;
    std::string envString;
    if (fileID >= 0 && fileID < 9999)
    {
      envString = "SPARSE_FILE_NAME" + std::to_string(fileID);
      pSparseInput = getenv(envString.c_str());
    }
    else if (fileID == 9999)
    {
      pSparseInput = getenv("SPARSE_FILE_NAME");
    }
    else
    {
      llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: SPARSE_FILE_NAME environmental variable is not set\n";
    }

    filename = pSparseInput; /// update

    return true;
  }

  void readMtxFile()
  {
    if (filename.find(".mtx") == std::string::npos)
    {
      llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: input file is not Market Matrix file\n";
    }

    /// init matrix read
    coo_matrix->InitMarket(filename);
  }

  void readTnsFile()
  {
    if (filename.find(".tns") == std::string::npos)
    {
      llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: input file is not TNS format\n";
    }

    /// init frostt file read
    coo_3dtensor->InitFrostt(filename);
  }

  FileReaderWrapper(int32_t fileID, bool tnsFile = false)
  {
    ID = fileID;

    if (CooTracking<T>.find(fileID) == CooTracking<T>.end() && !tnsFile) // not found, read the file
    {
      bool done = readFileNameStr(fileID);
      if (done && !filename.empty())
      { /// file is read here
        coo_matrix = new CooMatrix<T>();
        readMtxFile(); // 2D

        /// update hash-map
        CooTracking<T>[fileID] = coo_matrix;
      }
      else
      {
        fprintf(stderr, "No input specified.\n");
        assert(false);
      }

      is3D = false;
    }
    else if (CooTracking<T>.count(fileID) == 1)
    { /// re-use the old file read
      coo_matrix = CooTracking<T>[fileID];

      is3D = false;
    }

    if (Coo3DTracking<T>.find(fileID) == Coo3DTracking<T>.end() && tnsFile) // not found
    {
      bool done = readFileNameStr(fileID);
      if (done && !filename.empty())
      { /// file is read here
        coo_3dtensor = new Coo3DTensor<T>();
        readTnsFile(); // 3D

        /// update hash-map
        Coo3DTracking<T>[fileID] = coo_3dtensor;
      }
      else
      {
        fprintf(stderr, "No input specified.\n");
        assert(false);
      }

      is3D = true;
    }
    else if (Coo3DTracking<T>.count(fileID) == 1 && tnsFile)
    {
      coo_3dtensor = Coo3DTracking<T>[fileID];

      is3D = true;
    }
  }

  ~FileReaderWrapper()
  {
    /// Do nothing here, this is taken care of in Finalize() method.
  }

  void FileReaderWrapperFinalize()
  {

    if (is3D)
    {
      if (Coo3DTracking<T>.count(ID) == 1)
      {
        coo_3dtensor = Coo3DTracking<T>[ID];
      }
      else
      {
        /// not found!
        return;
      }
    }
    else
    { /// 2D
      if (CooTracking<T>.count(ID) == 1)
      {
        coo_matrix = CooTracking<T>[ID];
      }
      else
      {
        /// not found!
        return;
      }
    }

    if (is3D)
    {
      coo_3dtensor->num_nonzeros = 0; /// reset
      coo_3dtensor->Clear();

      /// update the hashMap
      delete coo_3dtensor;
      Coo3DTracking<T>.erase(ID);
    }
    else
    {
      coo_matrix->num_nonzeros = 0; /// reset
      coo_matrix->num_nonzeros_lowerTri = 0;
      coo_matrix->num_nonzeros_lowerTri_strict = 0;
      coo_matrix->num_nonzeros_upperTri = 0;
      coo_matrix->num_nonzeros_upperTri_strict = 0;
      coo_matrix->Clear();

      /// update the hashMap
      delete coo_matrix;
      CooTracking<T>.erase(ID);
    }
  }
};

/// helper func: get num of nonzeros based on selected matrix read
template <typename T>
uint64_t getNumNonZeros(CooMatrix<T> *coo_matrix, int32_t readMode)
{
  uint64_t NumNonZeros = -1;

  int selected_matrix_read = getMatrixReadOption(readMode);

  if (selected_matrix_read == LOWER_TRI_STRICT)
    NumNonZeros = coo_matrix->num_nonzeros_lowerTri_strict;
  else if (selected_matrix_read == LOWER_TRI)
    NumNonZeros = coo_matrix->num_nonzeros_lowerTri;
  else if (selected_matrix_read == UPPER_TRI_STRICT)
    NumNonZeros = coo_matrix->num_nonzeros_upperTri_strict;
  else if (selected_matrix_read == UPPER_TRI)
    NumNonZeros = coo_matrix->num_nonzeros_upperTri;
  else // DEFAULT
    NumNonZeros = coo_matrix->num_nonzeros;

  return NumNonZeros;
}

//===----------------------------------------------------------------------===//
/// Sparse Utility Functions
//===----------------------------------------------------------------------===//

/// Read input matrices based on the datatype
template <typename T>
void read_input_sizes_2D(int32_t fileID,
                         int32_t A1format, int32_t A1_tile_format,
                         int32_t A2format, int32_t A2_tile_format,
                         int sizes_rank, void *sizes_ptr, int32_t readMode)
{
  auto *desc_sizes = static_cast<StridedMemRefType<int64_t, 1> *>(sizes_ptr);

  int selected_matrix_read = getMatrixReadOption(readMode);
  FileReaderWrapper<T> FileReader(fileID); /// init of COO

  /// SparseFormatAttribute A1format: COO
  if (A1format == Compressed_nonunique && A2format == singleton)
  {
    /// get num-NNZs from coo_matrix struct.
    uint64_t NumNonZeros = getNumNonZeros(FileReader.coo_matrix, readMode);

    desc_sizes->data[0] = 2;           /// A1_pos
    desc_sizes->data[1] = NumNonZeros; /// A1_crd
    desc_sizes->data[2] = 0;           /// A1_tile_pos
    desc_sizes->data[3] = 0;           /// A1_tile_crd
    desc_sizes->data[4] = 1;           /// A2_pos
    desc_sizes->data[5] = NumNonZeros; /// A2_crd
    desc_sizes->data[6] = 0;           /// A2_tile_pos
    desc_sizes->data[7] = 0;           /// A2_tile_crd
    desc_sizes->data[8] = NumNonZeros;
    desc_sizes->data[9] = FileReader.coo_matrix->num_rows;
    desc_sizes->data[10] = FileReader.coo_matrix->num_cols;

    /*
    std::cout << "COO detail: \n"
              << "desc_sizes->data[0]: " << desc_sizes->data[0] << "\n"
              << "desc_sizes->data[1]: " << desc_sizes->data[1] << "\n"
              << "desc_sizes->data[2]: " << desc_sizes->data[2] << "\n"
              << "desc_sizes->data[3]: " << desc_sizes->data[3] << "\n"
              << "desc_sizes->data[4]: " << desc_sizes->data[4] << "\n"
              << "desc_sizes->data[5]: " << desc_sizes->data[5] << "\n"
              << "desc_sizes->data[6]: " << desc_sizes->data[6] << "\n";
    */
  }
  /// CSR
  else if (A1format == Dense && A2format == Compressed_unique)
  {
    /// get num-NNZs from coo_matrix struct.
    uint64_t NumNonZeros = getNumNonZeros(FileReader.coo_matrix, readMode);

    desc_sizes->data[0] = 1;                                   /// A1pos
    desc_sizes->data[1] = 1;                                   /// A1crd
    desc_sizes->data[2] = 0;                                   /// A1_tile_pos
    desc_sizes->data[3] = 0;                                   /// A1_tile_crd
    desc_sizes->data[4] = FileReader.coo_matrix->num_rows + 1; /// A2pos
    desc_sizes->data[5] = NumNonZeros;                         /// A2crd
    desc_sizes->data[6] = 0;                                   /// A2_tile_pos
    desc_sizes->data[7] = 0;                                   /// A2_tile_crd
    desc_sizes->data[8] = NumNonZeros;
    desc_sizes->data[9] = FileReader.coo_matrix->num_rows;
    desc_sizes->data[10] = FileReader.coo_matrix->num_cols;

    /*****************DEBUG******************/
    // std::cout << "CSR detail: \n"
    //           << "desc_sizes->data[0]: " << desc_sizes->data[0] << "\n"
    //           << "desc_sizes->data[1]: " << desc_sizes->data[1] << "\n"
    //           << "desc_sizes->data[2]: " << desc_sizes->data[2] << "\n"
    //           << "desc_sizes->data[3]: " << desc_sizes->data[3] << "\n"
    //           << "desc_sizes->data[4]: " << desc_sizes->data[4] << "\n"
    //           << "desc_sizes->data[5]: " << desc_sizes->data[5] << "\n"
    //           << "desc_sizes->data[6]: " << desc_sizes->data[6] << "\n";
    /*****************DEBUG******************/
  }
  /// CSC
  else if (A1format == Compressed_unique && A2format == Dense)
  {
    uint64_t NumNonZeros = 0;
    if (selected_matrix_read == DEFAULT)
      NumNonZeros = FileReader.coo_matrix->num_nonzeros;
    else
      llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: unsupported matrix format (CSC) for triangular reads.\n";

    desc_sizes->data[0] = FileReader.coo_matrix->num_cols + 1; /// A1pos
    desc_sizes->data[1] = NumNonZeros;                         /// A1crd
    desc_sizes->data[2] = 0;                                   /// A1_tile_pos
    desc_sizes->data[3] = 0;                                   /// A1_tile_crd
    desc_sizes->data[4] = 1;                                   /// A2pos
    desc_sizes->data[5] = 1;                                   /// A2crd
    desc_sizes->data[6] = 0;                                   /// A2_tile_pos
    desc_sizes->data[7] = 0;                                   /// A2_tile_crd
    desc_sizes->data[8] = NumNonZeros;
    desc_sizes->data[9] = FileReader.coo_matrix->num_rows;
    desc_sizes->data[10] = FileReader.coo_matrix->num_cols;

    /*****************DEBUG******************/
    // std::cout << "CSC detail: \n"
    //           << "desc_sizes->data[0]: " << desc_sizes->data[0] << "\n"
    //           << "desc_sizes->data[1]: " << desc_sizes->data[1] << "\n"
    //           << "desc_sizes->data[2]: " << desc_sizes->data[2] << "\n"
    //           << "desc_sizes->data[3]: " << desc_sizes->data[3] << "\n"
    //           << "desc_sizes->data[4]: " << desc_sizes->data[4] << "\n"
    //           << "desc_sizes->data[5]: " << desc_sizes->data[5] << "\n"
    //           << "desc_sizes->data[6]: " << desc_sizes->data[6] << "\n";
    /*****************DEBUG******************/
  }
  /// DCSR
  else if (A1format == Compressed_unique && A2format == Compressed_unique)
  {
    DcsrMatrix<T> dcsr_matrix(FileReader.coo_matrix);

    if (selected_matrix_read != DEFAULT)
      llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: unsupported matrix format (DCSR) for triangular reads.\n";

    desc_sizes->data[0] = dcsr_matrix.A1pos_size; /// A1pos
    desc_sizes->data[1] = dcsr_matrix.A1crd_size; /// A1crd
    desc_sizes->data[2] = 0;                      /// A1_tile_pos
    desc_sizes->data[3] = 0;                      /// A1_tile_crd
    desc_sizes->data[4] = dcsr_matrix.A2pos_size; /// A2pos
    desc_sizes->data[5] = dcsr_matrix.A2crd_size; /// A2crd
    desc_sizes->data[6] = 0;                      /// A2_tile_pos
    desc_sizes->data[7] = 0;                      /// A2_tile_crd
    desc_sizes->data[8] = dcsr_matrix.A2crd_size;
    desc_sizes->data[9] = dcsr_matrix.num_rows;
    desc_sizes->data[10] = dcsr_matrix.num_cols;
  }
  /// ELLPACK
  else if (A1format == Dense && A2format == singleton && A1_tile_format == Dense)
  {
    /// Load the ellpack matrixs

    EllpackMatrix<T> ellpack_matrix(FileReader.coo_matrix);
    int cols = ellpack_matrix.num_cols * ellpack_matrix.num_rows;

    /*
    desc_sizes->data[0] = 1;              // A1pos
    desc_sizes->data[1] = 1;              // A1crd
    desc_sizes->data[2] = 1;              // A2pos
    desc_sizes->data[3] = cols;           // A2crd
    desc_sizes->data[4] = 1;              // A1_tile_pos
    desc_sizes->data[5] = 1;              // A1_tile_crd
    desc_sizes->data[6] = 0;              // A2_tile_pos
    desc_sizes->data[7] = 0;              // A2_tile_crd
    desc_sizes->data[8] = cols;           // Controls count of value dimension
    desc_sizes->data[9] = FileReader.coo_matrix->num_rows;
    desc_sizes->data[10] = FileReader.coo_matrix->num_cols;
    */

    desc_sizes->data[0] = 1;    /// A1pos
    desc_sizes->data[1] = 1;    /// A1crd
    desc_sizes->data[2] = 1;    /// A1_tile_pos
    desc_sizes->data[3] = 1;    /// A1_tile_crd
    desc_sizes->data[4] = 1;    /// A2pos
    desc_sizes->data[5] = cols; /// A2crd
    desc_sizes->data[6] = 0;    /// A2_tile_pos
    desc_sizes->data[7] = 0;    /// A2_tile_crd
    desc_sizes->data[8] = cols; /// Controls count of value dimension
    desc_sizes->data[9] = FileReader.coo_matrix->num_rows;
    desc_sizes->data[10] = FileReader.coo_matrix->num_cols;

    /*****************DEBUG******************/
    // std::cout << "ELLPACK detail: \n"
    //           << "desc_sizes->data[0]: " << desc_sizes->data[0] << "\n"
    //           << "desc_sizes->data[1]: " << desc_sizes->data[1] << "\n"
    //           << "desc_sizes->data[2]: " << desc_sizes->data[2] << "\n"
    //           << "desc_sizes->data[3]: " << desc_sizes->data[3] << "\n"
    //           << "desc_sizes->data[4]: " << desc_sizes->data[4] << "\n"
    //           << "desc_sizes->data[5]: " << desc_sizes->data[5] << "\n"
    //           << "desc_sizes->data[6]: " << desc_sizes->data[6] << "\n";
    /*****************DEBUG******************/
  }
  /// BCSR
  else if (A1format == Dense && A2format == Compressed_nonunique && A1_tile_format == Dense && A2_tile_format == Dense)
  {
    puts("BCSR");
  }
  /// CSB
  else if (A1format == Compressed_unique && A2format == singleton && A1_tile_format == Dense && A2_tile_format == Dense)
  {
    puts("CSB");
  }
  else
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: unsupported matrix format\n";
  }
}

template <typename T, typename IndicesType>
void read_input_2D(int32_t fileID,
                   int32_t A1format, int32_t A1_tile_format,
                   int32_t A2format, int32_t A2_tile_format,
                   int A1pos_rank, void *A1pos_ptr,
                   int A1crd_rank, void *A1crd_ptr,
                   int A1tile_pos_rank, void *A1tile_pos_ptr,
                   int A1tile_crd_rank, void *A1tile_crd_ptr,
                   int A2pos_rank, void *A2pos_ptr,
                   int A2crd_rank, void *A2crd_ptr,
                   int A2tile_pos_rank, void *A2tile_pos_ptr,
                   int A2tile_crd_rank, void *A2tile_crd_ptr,
                   int Aval_rank, void *Aval_ptr,
                   int32_t readMode)
{

  auto *desc_A1pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A1pos_ptr);
  auto *desc_A1crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A1crd_ptr);
  auto *desc_A2pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A2pos_ptr);
  auto *desc_A2crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A2crd_ptr);
  auto *desc_A1tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A1tile_pos_ptr);
  auto *desc_A1tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A1tile_crd_ptr);
  auto *desc_A2tile_pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A2tile_pos_ptr);
  auto *desc_A2tile_crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A2tile_crd_ptr);
  auto *desc_Aval = static_cast<StridedMemRefType<T, 1> *>(Aval_ptr);

  /// For example, A2pos is not used for COO, but initialized with -1 to speficify that it is not used
  desc_A1pos->data[0] = -1;
  desc_A1crd->data[0] = -1;
  desc_A2pos->data[0] = -1;
  desc_A2crd->data[0] = -1;
  desc_A1tile_pos->data[0] = -1;
  desc_A1tile_crd->data[0] = -1;
  desc_A2tile_pos->data[0] = -1;
  desc_A2tile_crd->data[0] = -1;

  int selected_matrix_read = getMatrixReadOption(readMode);
  FileReaderWrapper<T> FileReader(fileID); /// init of COO

  /// SparseFormatAttribute A1format: COO
  if (A1format == Compressed_nonunique && A2format == singleton)
  {
    std::stable_sort(FileReader.coo_matrix->coo_tuples, FileReader.coo_matrix->coo_tuples + FileReader.coo_matrix->num_nonzeros, CooComparatorRow());

    desc_A1pos->data[0] = 0;
    uint64_t actual_num_nonzeros = 0;

    for (uint64_t i = 0; i < FileReader.coo_matrix->num_nonzeros; i++)
    {
      if (((selected_matrix_read == LOWER_TRI_STRICT) || (selected_matrix_read == LOWER_TRI)) && // filter lower triangular vals
          (FileReader.coo_matrix->coo_tuples[i].row > FileReader.coo_matrix->coo_tuples[i].col))
      {
        desc_A1crd->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].row;
        desc_A2crd->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].col;
        desc_Aval->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].val;
        actual_num_nonzeros++;
      }
      else if (((selected_matrix_read == UPPER_TRI_STRICT) || (selected_matrix_read == UPPER_TRI)) && /// filter upper triangular vals
               (FileReader.coo_matrix->coo_tuples[i].row < FileReader.coo_matrix->coo_tuples[i].col))
      {
        desc_A1crd->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].row;
        desc_A2crd->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].col;
        desc_Aval->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].val;
        actual_num_nonzeros++;
      }
      else if (((selected_matrix_read == UPPER_TRI) || (selected_matrix_read == LOWER_TRI)) && /// the diagonal vals go in both UPPER_TRI and LOWER_TRI
               (FileReader.coo_matrix->coo_tuples[i].row == FileReader.coo_matrix->coo_tuples[i].col))
      {
        desc_A1crd->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].row;
        desc_A2crd->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].col;
        desc_Aval->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].val;
        actual_num_nonzeros++;
      }
      else if (selected_matrix_read == DEFAULT) /// standard matrix vals
      {
        desc_A1crd->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].row;
        desc_A2crd->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].col;
        desc_Aval->data[actual_num_nonzeros] = FileReader.coo_matrix->coo_tuples[i].val;
        actual_num_nonzeros++;
      }

    } /// end-for
    desc_A1pos->data[1] = actual_num_nonzeros;

    FileReader.FileReaderWrapperFinalize(); /// clear coo_matrix
  }
  /// CSR
  else if (A1format == Dense && A2format == Compressed_unique)
  {
    CsrMatrix<T> csr_matrix(FileReader.coo_matrix, selected_matrix_read);

    uint64_t upperBound_NNZ = getNumNonZeros(FileReader.coo_matrix, readMode);
    FileReader.FileReaderWrapperFinalize(); /// clear coo_matrix

    // /*****************DEBUG******************/
    // std::cout << "CSR detail: " << csr_matrix.num_cols << ", " << csr_matrix.num_rows << ", " << csr_matrix.num_nonzeros << "\n";
    // std::cout << "row_offsets: ";
    // for (int i = 0; i < csr_matrix.num_rows + 1; i++)
    // {
    //   std::cout << csr_matrix.row_offsets[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "column_indices: ";
    // for (int i = 0; i < csr_matrix.num_nonzeros; i++)
    // {
    //   std::cout << csr_matrix.column_indices[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "values: ";
    // for (int i = 0; i < csr_matrix.num_nonzeros; i++)
    // {
    //   std::cout << csr_matrix.values[i] << " ";
    // }
    // std::cout << "\nfinished read_market\n";
    // /*****************DEBUG******************/

    desc_A1pos->data[0] = csr_matrix.num_rows;

    for (uint64_t i = 0; i < csr_matrix.num_rows + 1; i++)
    {
      desc_A2pos->data[i] = csr_matrix.row_offsets[i];
    }

    for (uint64_t i = 0; i < upperBound_NNZ; i++)
    {
      desc_A2crd->data[i] = csr_matrix.column_indices[i];
      desc_Aval->data[i] = csr_matrix.values[i];
    }
  }
  /// CSC
  else if (A1format == Compressed_unique && A2format == Dense)
  {
    CscMatrix<T> csc_matrix(FileReader.coo_matrix);
    FileReader.FileReaderWrapperFinalize(); /// clear coo_matrix

    // NOTE: we do not need to check readMode, since this has already been taken care of in read_sizes() call
    // /*****************DEBUG******************/
    // std::cout << "CSC detail: " << csc_matrix.num_rows << ", " << csc_matrix.num_cols << ", " << csc_matrix.num_nonzeros << "\n";
    // std::cout << "col_offsets: ";
    // for (int i = 0; i < csc_matrix.num_cols + 1; i++)
    // {
    //   std::cout << csc_matrix.col_offsets[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "row_indices: ";
    // for (int i = 0; i < csc_matrix.num_nonzeros; i++)
    // {
    //   std::cout << csc_matrix.row_indices[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "values: ";
    // for (int i = 0; i < csc_matrix.num_nonzeros; i++)
    // {
    //   std::cout << csc_matrix.values[i] << " ";
    // }
    // std::cout << "\nfinished read_market\n";
    /*****************DEBUG******************/

    desc_A2pos->data[0] = csc_matrix.num_cols;

    for (uint64_t i = 0; i < csc_matrix.num_cols + 1; i++)
    {
      desc_A1pos->data[i] = csc_matrix.col_offsets[i];
    }

    for (uint64_t i = 0; i < csc_matrix.num_nonzeros; i++)
    {
      desc_A1crd->data[i] = csc_matrix.row_indices[i];
      desc_Aval->data[i] = csc_matrix.values[i];
    }
  }
  /// DCSR
  else if (A1format == Compressed_unique && A2format == Compressed_unique)
  {
    DcsrMatrix<T> dcsr_matrix(FileReader.coo_matrix);
    FileReader.FileReaderWrapperFinalize(); /// clear coo_matrix

    /// NOTE: we do not need to check readMode, since this has already been taken care of in read_sizes() call

    /*****************DEBUG******************/
    // std::cout << "DCSR detail: " << dcsr_matrix.num_cols << ", " << dcsr_matrix.num_rows << ", " << dcsr_matrix.num_nonzeros << "\n";
    // std::cout << "A1pos: ";
    // for (int i = 0; i < dcsr_matrix.A1pos_size; i++)
    // {
    //   std::cout << dcsr_matrix.A1pos[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "A1crd: ";
    // for (int i = 0; i < dcsr_matrix.A1crd_size; i++)
    // {
    //   std::cout << dcsr_matrix.A1crd[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "A2pos: ";
    // for (int i = 0; i < dcsr_matrix.A2pos_size; i++)
    // {
    //   std::cout << dcsr_matrix.A2pos[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "A2crd: ";
    // for (int i = 0; i < dcsr_matrix.A2crd_size; i++)
    // {
    //   std::cout << dcsr_matrix.A2crd[i] << " ";
    // }

    // std::cout << "\n";
    // std::cout << "Aval: ";
    // for (int i = 0; i < dcsr_matrix.Aval_size; i++)
    // {
    //   std::cout << dcsr_matrix.Aval[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "finished read_market\n";
    /*****************DEBUG******************/

    for (uint64_t i = 0; i < dcsr_matrix.A1pos_size; i++)
    {
      desc_A1pos->data[i] = dcsr_matrix.A1pos[i];
    }

    for (uint64_t i = 0; i < dcsr_matrix.A1crd_size; i++)
    {
      desc_A1crd->data[i] = dcsr_matrix.A1crd[i];
    }

    for (uint64_t i = 0; i < dcsr_matrix.A2pos_size; i++)
    {
      desc_A2pos->data[i] = dcsr_matrix.A2pos[i];
    }

    for (uint64_t i = 0; i < dcsr_matrix.A2crd_size; i++)
    {
      desc_A2crd->data[i] = dcsr_matrix.A2crd[i];
      desc_Aval->data[i] = dcsr_matrix.Aval[i];
    }
  }
  /// ELLPACK
  else if (A1format == Dense && A2format == singleton && A1_tile_format == Dense)
  {
    EllpackMatrix<T> ellpack_matrix(FileReader.coo_matrix);
    FileReader.FileReaderWrapperFinalize();

    desc_A1pos->data[0] = ellpack_matrix.num_rows;
    desc_A1tile_pos->data[0] = ellpack_matrix.num_cols;

    for (uint64_t i = 0; i < ellpack_matrix.num_cols * ellpack_matrix.num_rows; i++)
    {
      desc_A2crd->data[i] = ellpack_matrix.col_crd[i];
      desc_Aval->data[i] = ellpack_matrix.Aval[i];
    }
  }
  /// BCSR
  /*else if (A1format == Dense && A2format == Compressed_nonunique && A3format == Dense && A4format == Dense)
  {
    puts("BCSR");
  }
  /// CSB
  else if (A1format == Compressed_unique && A2format == singleton && A3format == Dense && A4format == Dense)
  {
    puts("CSB");
  }*/
  else
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: unsupported matrix format\n";
  }
}

template <typename T>
void read_input_sizes_3D(int32_t fileID,
                         int32_t A1format, int32_t A1_tile_format,
                         int32_t A2format, int32_t A2_tile_format,
                         int32_t A3format, int32_t A3_tile_format,
                         int sizes_rank, void *sizes_ptr, int32_t readMode)
{
  /// TODO(gkestor): readMode is for future use.
  auto *desc_sizes = static_cast<StridedMemRefType<int64_t, 1> *>(sizes_ptr);

  FileReaderWrapper<T> FileReader(fileID, true); /// init of COO_3d_tensor

  if (A1format == Compressed_nonunique && A2format == singleton && A3format == singleton)
  {
    /// A1pos, A1crd, A1tile_pos, A1tile_crd
    /// A2pos, A2crd, A2tile_pos, A2tile_crd
    /// A3pos, A3crd, A3tile_pos, A3tile_crd
    /// Aval, I, J, K
    desc_sizes->data[0] = 2;                                      /// A1pos
    desc_sizes->data[1] = FileReader.coo_3dtensor->num_nonzeros;  /// A1crd
    desc_sizes->data[2] = 0;                                      /// A1_tile_pos
    desc_sizes->data[3] = 0;                                      /// A1_tile_crd
    desc_sizes->data[4] = 1;                                      /// A2pos
    desc_sizes->data[5] = FileReader.coo_3dtensor->num_nonzeros;  /// A2crd
    desc_sizes->data[6] = 0;                                      /// A2_tile_pos
    desc_sizes->data[7] = 0;                                      /// A2_tile_crd
    desc_sizes->data[8] = 1;                                      /// A3pos
    desc_sizes->data[9] = FileReader.coo_3dtensor->num_nonzeros;  /// A3crd
    desc_sizes->data[10] = 0;                                     /// A3_tile_pos
    desc_sizes->data[11] = 0;                                     /// A3_tile_crd
    desc_sizes->data[12] = FileReader.coo_3dtensor->num_nonzeros; /// Aval
    desc_sizes->data[13] = FileReader.coo_3dtensor->num_index_i;  /// I
    desc_sizes->data[14] = FileReader.coo_3dtensor->num_index_j;  /// J
    desc_sizes->data[15] = FileReader.coo_3dtensor->num_index_k;  /// K
  }
  /// CSF
  else if (A1format == Compressed_unique && A2format == Compressed_unique && A3format == Compressed_unique)
  {
    /// std::cout << "CSF format\n";
    Csf3DTensor<T> csf_3dtensor(FileReader.coo_3dtensor);

    desc_sizes->data[0] = csf_3dtensor.A1pos_size;
    desc_sizes->data[1] = csf_3dtensor.A1crd_size;
    desc_sizes->data[2] = 0;
    desc_sizes->data[3] = 0;
    desc_sizes->data[4] = csf_3dtensor.A2pos_size;
    desc_sizes->data[5] = csf_3dtensor.A2crd_size;
    desc_sizes->data[6] = 0;
    desc_sizes->data[7] = 0;
    desc_sizes->data[8] = csf_3dtensor.A3pos_size;
    desc_sizes->data[9] = csf_3dtensor.A3crd_size;
    desc_sizes->data[10] = 0;
    desc_sizes->data[11] = 0;
    desc_sizes->data[12] = csf_3dtensor.Aval_size;
    desc_sizes->data[13] = csf_3dtensor.num_index_i;
    desc_sizes->data[14] = csf_3dtensor.num_index_j;
    desc_sizes->data[15] = csf_3dtensor.num_index_k;
  }
  /// Mode-Generic
  else if (A1format == Compressed_nonunique && A2format == singleton && A3format == Dense)
  {
    /// std::cout << "Mode-Generic format\n";
    Mg3DTensor<T> mg_3dtensor(FileReader.coo_3dtensor);

    desc_sizes->data[0] = mg_3dtensor.A1pos_size;
    desc_sizes->data[1] = mg_3dtensor.A1crd_size;
    desc_sizes->data[2] = 0;
    desc_sizes->data[3] = 0;
    desc_sizes->data[4] = mg_3dtensor.A2pos_size;
    desc_sizes->data[5] = mg_3dtensor.A2crd_size;
    desc_sizes->data[6] = 0;
    desc_sizes->data[7] = 0;
    desc_sizes->data[8] = mg_3dtensor.A3pos_size;
    desc_sizes->data[9] = mg_3dtensor.A3crd_size;
    desc_sizes->data[10] = 0;
    desc_sizes->data[11] = 0;
    desc_sizes->data[12] = mg_3dtensor.Aval_size;
    desc_sizes->data[13] = mg_3dtensor.num_index_i;
    desc_sizes->data[14] = mg_3dtensor.num_index_j;
    desc_sizes->data[15] = mg_3dtensor.num_index_k;
  }
  else
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: unsupported tensor 3D format\n";
  }
}

template <typename T, typename IndicesType>
void read_input_3D(int32_t fileID,
                   int32_t A1format, int32_t A1_tile_format,
                   int32_t A2format, int32_t A2_tile_format,
                   int32_t A3format, int32_t A3_tile_format,
                   int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                   int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                   int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                   int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                   int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                   int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                   int Aval_rank, void *Aval_ptr, int32_t readMode)
{
  /// TODO(gkestor): readMode is for future use.

  auto *desc_A1pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A1pos_ptr);
  auto *desc_A1crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A1crd_ptr);
  auto *desc_A2pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A2pos_ptr);
  auto *desc_A2crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A2crd_ptr);
  auto *desc_A3pos = static_cast<StridedMemRefType<IndicesType, 1> *>(A3pos_ptr);
  auto *desc_A3crd = static_cast<StridedMemRefType<IndicesType, 1> *>(A3crd_ptr);
  auto *desc_Aval = static_cast<StridedMemRefType<T, 1> *>(Aval_ptr);

  FileReaderWrapper<T> FileReader(fileID, true); /// init of COO_3d_tensor

  if (A1format == Compressed_nonunique && A2format == singleton && A3format == singleton)
  {
    /// std::cout << "COO detail: " << coo_3dtensor.num_index_i << ", " << coo_3dtensor.num_index_j << ", " << coo_3dtensor.num_index_k << ", " << coo_3dtensor.num_nonzeros << "\n";
    std::stable_sort(FileReader.coo_3dtensor->coo_3dtuples, FileReader.coo_3dtensor->coo_3dtuples + FileReader.coo_3dtensor->num_nonzeros, Coo3DTensorComparator());
    desc_A1pos->data[0] = 0;
    desc_A1pos->data[1] = FileReader.coo_3dtensor->num_nonzeros;

    for (uint64_t i = 0; i < FileReader.coo_3dtensor->num_nonzeros; i++)
    {
      desc_A1crd->data[i] = FileReader.coo_3dtensor->coo_3dtuples[i].index_i;
      desc_A2crd->data[i] = FileReader.coo_3dtensor->coo_3dtuples[i].index_j;
      desc_A3crd->data[i] = FileReader.coo_3dtensor->coo_3dtuples[i].index_k;
      desc_Aval->data[i] = FileReader.coo_3dtensor->coo_3dtuples[i].val;
    }
  }
  /// CSF
  else if (A1format == Compressed_unique && A2format == Compressed_unique && A3format == Compressed_unique)
  {
    Csf3DTensor<T> csf_3dtensor(FileReader.coo_3dtensor);
    FileReader.FileReaderWrapperFinalize(); /// clear coo_3dtensor

    // Print
    // std::cout << "CSF detail: " << csf_3dtensor.num_index_i << ", " << csf_3dtensor.num_index_j << ", " << csf_3dtensor.num_index_k << ", " << csf_3dtensor.num_nonzeros << "\n";

    // std::cout << "A1pos: ";
    // for (int i = 0; i < csf_3dtensor.A1pos_size; i++)
    // {
    //   std::cout << csf_3dtensor.A1pos[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "A1crd: ";
    // for (int i = 0; i < csf_3dtensor.A1crd_size; i++)
    // {
    //   std::cout << csf_3dtensor.A1crd[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "A2pos: ";
    // for (int i = 0; i < csf_3dtensor.A2pos_size; i++)
    // {
    //   std::cout << csf_3dtensor.A2pos[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "A2crd: ";
    // for (int i = 0; i < csf_3dtensor.A2crd_size; i++)
    // {
    //   std::cout << csf_3dtensor.A2crd[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "A3pos: ";
    // for (int i = 0; i < csf_3dtensor.A3pos_size; i++)
    // {
    //   std::cout << csf_3dtensor.A3pos[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "A3crd: ";
    // for (int i = 0; i < csf_3dtensor.A3crd_size; i++)
    // {
    //   std::cout << csf_3dtensor.A3crd[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "Aval: ";
    // for (int i = 0; i < csf_3dtensor.Aval_size; i++)
    // {
    //   std::cout << csf_3dtensor.Aval[i] << " ";
    // }
    // std::cout << "\n";

    /// Fill data
    for (uint64_t i = 0; i < csf_3dtensor.A1pos_size; i++)
    {
      desc_A1pos->data[i] = csf_3dtensor.A1pos[i];
    }
    for (uint64_t i = 0; i < csf_3dtensor.A1crd_size; i++)
    {
      desc_A1crd->data[i] = csf_3dtensor.A1crd[i];
    }
    for (uint64_t i = 0; i < csf_3dtensor.A2pos_size; i++)
    {
      desc_A2pos->data[i] = csf_3dtensor.A2pos[i];
    }
    for (uint64_t i = 0; i < csf_3dtensor.A2crd_size; i++)
    {
      desc_A2crd->data[i] = csf_3dtensor.A2crd[i];
    }
    for (uint64_t i = 0; i < csf_3dtensor.A3pos_size; i++)
    {
      desc_A3pos->data[i] = csf_3dtensor.A3pos[i];
    }
    for (uint64_t i = 0; i < csf_3dtensor.A3crd_size; i++)
    {
      desc_A3crd->data[i] = csf_3dtensor.A3crd[i];
    }
    for (uint64_t i = 0; i < csf_3dtensor.Aval_size; i++)
    {
      desc_Aval->data[i] = csf_3dtensor.Aval[i];
    }
  }
  /// Mode-Generic
  else if (A1format == Compressed_nonunique && A2format == singleton && A3format == Dense)
  {

    Mg3DTensor<T> mg_3dtensor(FileReader.coo_3dtensor);
    FileReader.FileReaderWrapperFinalize(); /// clear coo_3dtensor

    // Print
    // std::cout << "ModeGeneric detail: " << mg_3dtensor.num_index_i << ", " << mg_3dtensor.num_index_j << ", " << mg_3dtensor.num_index_k << ", " << mg_3dtensor.num_nonzeros << "\n";

    // std::cout << "A1pos: ";
    // for(int i = 0; i < mg_3dtensor.A1pos_size; i++){
    //   std::cout << mg_3dtensor.A1pos[i] << " " ;
    // }
    // std::cout << "\n";
    // std::cout << "A1crd: ";
    // for(int i = 0; i < mg_3dtensor.A1crd_size; i++){
    //   std::cout << mg_3dtensor.A1crd[i] << " " ;
    // }
    // std::cout << "\n";
    // std::cout << "A2crd: ";
    // for(int i = 0; i < mg_3dtensor.A2crd_size; i++){
    //   std::cout << mg_3dtensor.A2crd[i] << " " ;
    // }
    // std::cout << "\n";
    // std::cout << "A3pos: ";
    // for(int i = 0; i < mg_3dtensor.A3pos_size; i++){
    //   std::cout << mg_3dtensor.A3pos[i] << " " ;
    // }
    // std::cout << "\n";
    // std::cout << "Aval: ";
    // for(int i = 0; i < mg_3dtensor.Aval_size; i++){
    //   std::cout << mg_3dtensor.Aval[i] << " " ;
    // }
    // std::cout << "\n";

    // Fill data
    for (uint64_t i = 0; i < mg_3dtensor.A1pos_size; i++)
    {
      desc_A1pos->data[i] = mg_3dtensor.A1pos[i];
    }
    for (uint64_t i = 0; i < mg_3dtensor.A1crd_size; i++)
    {
      desc_A1crd->data[i] = mg_3dtensor.A1crd[i];
    }
    for (uint64_t i = 0; i < mg_3dtensor.A2crd_size; i++)
    {
      desc_A2crd->data[i] = mg_3dtensor.A2crd[i];
    }
    for (uint64_t i = 0; i < mg_3dtensor.A3pos_size; i++)
    {
      desc_A3pos->data[i] = mg_3dtensor.A3pos[i];
    }
    for (uint64_t i = 0; i < mg_3dtensor.Aval_size; i++)
    {
      desc_Aval->data[i] = mg_3dtensor.Aval[i];
    }
  }
  else
  {
    llvm::errs() << __FILE__ << ":" << __LINE__ << "ERROR: unsupported tensor 3D format\n";
  }
}

/// Utility functions to read sparse matrices and fill in the pos and crd arrays per dimension
extern "C" void read_input_2D_f32_i32(int32_t fileID,
                                  int32_t A1format, int32_t A1_tile_format,
                                  int32_t A2format, int32_t A2_tile_format,
                                  int A1pos_rank, void *A1pos_ptr,
                                  int A1crd_rank, void *A1crd_ptr,
                                  int A1tile_pos_rank, void *A1tile_pos_ptr,
                                  int A1tile_crd_rank, void *A1tile_crd_ptr,
                                  int A2pos_rank, void *A2pos_ptr,
                                  int A2crd_rank, void *A2crd_ptr,
                                  int A2tile_pos_rank, void *A2tile_pos_ptr,
                                  int A2tile_crd_rank, void *A2tile_crd_ptr,
                                  int Aval_rank, void *Aval_ptr,
                                  int32_t readMode)
{
  read_input_2D<float, int32_t>(fileID,
                       A1format, A1_tile_format,
                       A2format, A2_tile_format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                       Aval_rank, Aval_ptr, readMode);
}

/// Utility functions to read sparse matrices and fill in the pos and crd arrays per dimension
extern "C" void read_input_2D_f32_i64(int32_t fileID,
                                  int32_t A1format, int32_t A1_tile_format,
                                  int32_t A2format, int32_t A2_tile_format,
                                  int A1pos_rank, void *A1pos_ptr,
                                  int A1crd_rank, void *A1crd_ptr,
                                  int A1tile_pos_rank, void *A1tile_pos_ptr,
                                  int A1tile_crd_rank, void *A1tile_crd_ptr,
                                  int A2pos_rank, void *A2pos_ptr,
                                  int A2crd_rank, void *A2crd_ptr,
                                  int A2tile_pos_rank, void *A2tile_pos_ptr,
                                  int A2tile_crd_rank, void *A2tile_crd_ptr,
                                  int Aval_rank, void *Aval_ptr,
                                  int32_t readMode)
{
  read_input_2D<float, int64_t>(fileID,
                       A1format, A1_tile_format,
                       A2format, A2_tile_format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                       Aval_rank, Aval_ptr, readMode);
}

extern "C" void read_input_2D_f64_i32(int32_t fileID,
                                  int32_t A1format, int32_t A1_tile_format,
                                  int32_t A2format, int32_t A2_tile_format,
                                  int A1pos_rank, void *A1pos_ptr,
                                  int A1crd_rank, void *A1crd_ptr,
                                  int A1tile_pos_rank, void *A1tile_pos_ptr,
                                  int A1tile_crd_rank, void *A1tile_crd_ptr,
                                  int A2pos_rank, void *A2pos_ptr,
                                  int A2crd_rank, void *A2crd_ptr,
                                  int A2tile_pos_rank, void *A2tile_pos_ptr,
                                  int A2tile_crd_rank, void *A2tile_crd_ptr,
                                  int Aval_rank, void *Aval_ptr,
                                  int32_t readMode)
{
  read_input_2D<double, int32_t>(fileID,
                        A1format, A1_tile_format,
                        A2format, A2_tile_format,
                        A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                        A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                        A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                        A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                        Aval_rank, Aval_ptr, readMode);
}

extern "C" void read_input_2D_f64_i64(int32_t fileID,
                                  int32_t A1format, int32_t A1_tile_format,
                                  int32_t A2format, int32_t A2_tile_format,
                                  int A1pos_rank, void *A1pos_ptr,
                                  int A1crd_rank, void *A1crd_ptr,
                                  int A1tile_pos_rank, void *A1tile_pos_ptr,
                                  int A1tile_crd_rank, void *A1tile_crd_ptr,
                                  int A2pos_rank, void *A2pos_ptr,
                                  int A2crd_rank, void *A2crd_ptr,
                                  int A2tile_pos_rank, void *A2tile_pos_ptr,
                                  int A2tile_crd_rank, void *A2tile_crd_ptr,
                                  int Aval_rank, void *Aval_ptr,
                                  int32_t readMode)
{
  read_input_2D<double, int64_t>(fileID,
                        A1format, A1_tile_format,
                        A2format, A2_tile_format,
                        A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                        A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                        A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                        A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                        Aval_rank, Aval_ptr, readMode);
}

extern "C" void read_input_3D_f32_i32(int32_t fileID,
                                  int32_t A1format, int32_t A1_tile_format,
                                  int32_t A2format, int32_t A2_tile_format,
                                  int32_t A3format, int32_t A3_tile_format,
                                  int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                  int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                  int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                  int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                  int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                  int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                                  int Aval_rank, void *Aval_ptr, int32_t readMode)
{

  read_input_3D<float, int32_t>(fileID,
                       A1format, A1_tile_format,
                       A2format, A2_tile_format,
                       A3format, A3_tile_format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                       A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                       A3tile_pos_rank, A3tile_pos_ptr, A3tile_crd_rank, A3tile_crd_ptr,
                       Aval_rank, Aval_ptr, readMode);
}

extern "C" void read_input_3D_f32_i64(int32_t fileID,
                                  int32_t A1format, int32_t A1_tile_format,
                                  int32_t A2format, int32_t A2_tile_format,
                                  int32_t A3format, int32_t A3_tile_format,
                                  int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                  int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                  int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                  int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                  int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                  int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                                  int Aval_rank, void *Aval_ptr, int32_t readMode)
{

  read_input_3D<float, int64_t>(fileID,
                       A1format, A1_tile_format,
                       A2format, A2_tile_format,
                       A3format, A3_tile_format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                       A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                       A3tile_pos_rank, A3tile_pos_ptr, A3tile_crd_rank, A3tile_crd_ptr,
                       Aval_rank, Aval_ptr, readMode);
}

extern "C" void read_input_3D_f64_i32(int32_t fileID,
                                  int32_t A1format, int32_t A1_tile_format,
                                  int32_t A2format, int32_t A2_tile_format,
                                  int32_t A3format, int32_t A3_tile_format,
                                  int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                  int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                  int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                  int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                  int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                  int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                                  int Aval_rank, void *Aval_ptr, int32_t readMode)
{
  read_input_3D<double, int32_t>(fileID,
                        A1format, A1_tile_format,
                        A2format, A2_tile_format,
                        A3format, A3_tile_format,
                        A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                        A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                        A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                        A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                        A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                        A3tile_pos_rank, A3tile_pos_ptr, A3tile_crd_rank, A3tile_crd_ptr,
                        Aval_rank, Aval_ptr, readMode);
}

extern "C" void read_input_3D_f64_i64(int32_t fileID,
                                  int32_t A1format, int32_t A1_tile_format,
                                  int32_t A2format, int32_t A2_tile_format,
                                  int32_t A3format, int32_t A3_tile_format,
                                  int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                  int A1tile_pos_rank, void *A1tile_pos_ptr, int A1tile_crd_rank, void *A1tile_crd_ptr,
                                  int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                  int A2tile_pos_rank, void *A2tile_pos_ptr, int A2tile_crd_rank, void *A2tile_crd_ptr,
                                  int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                  int A3tile_pos_rank, void *A3tile_pos_ptr, int A3tile_crd_rank, void *A3tile_crd_ptr,
                                  int Aval_rank, void *Aval_ptr, int32_t readMode)
{
  read_input_3D<double, int64_t>(fileID,
                        A1format, A1_tile_format,
                        A2format, A2_tile_format,
                        A3format, A3_tile_format,
                        A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                        A1tile_pos_rank, A1tile_pos_ptr, A1tile_crd_rank, A1tile_crd_ptr,
                        A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                        A2tile_pos_rank, A2tile_pos_ptr, A2tile_crd_rank, A2tile_crd_ptr,
                        A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                        A3tile_pos_rank, A3tile_pos_ptr, A3tile_crd_rank, A3tile_crd_ptr,
                        Aval_rank, Aval_ptr, readMode);
}

/// Utility functions to read metadata about the input matrices, such as the size of pos and crd array
extern "C" void read_input_sizes_2D_f32(int32_t fileID,
                                        int32_t A1format, int32_t A1_tile_format,
                                        int32_t A2format, int32_t A2_tile_format,
                                        int A1pos_rank, void *A1pos_ptr, int32_t readMode)
{
  read_input_sizes_2D<float>(fileID, A1format, A1_tile_format, A2format, A2_tile_format, A1pos_rank, A1pos_ptr, readMode);
}

extern "C" void read_input_sizes_2D_f64(int32_t fileID,
                                        int32_t A1format, int32_t A1_tile_format,
                                        int32_t A2format, int32_t A2_tile_format,
                                        int A1pos_rank, void *A1pos_ptr, int32_t readMode)
{
  read_input_sizes_2D<double>(fileID, A1format, A1_tile_format, A2format, A2_tile_format, A1pos_rank, A1pos_ptr, readMode);
}

/// Read 3D tensors
extern "C" void read_input_sizes_3D_f32(int32_t fileID,
                                        int32_t A1format, int32_t A1_tile_format,
                                        int32_t A2format, int32_t A2_tile_format,
                                        int32_t A3format, int32_t A3_tile_format,
                                        int A1pos_rank, void *A1pos_ptr, int32_t readMode)
{
  read_input_sizes_3D<float>(fileID,
                             A1format, A1_tile_format,
                             A2format, A2_tile_format,
                             A3format, A3_tile_format,
                             A1pos_rank, A1pos_ptr, readMode);
}

extern "C" void read_input_sizes_3D_f64(int32_t fileID,
                                        int32_t A1format, int32_t A1_tile_format,
                                        int32_t A2format, int32_t A2_tile_format,
                                        int32_t A3format, int32_t A3_tile_format,
                                        int A1pos_rank, void *A1pos_ptr, int32_t readMode)
{
  read_input_sizes_3D<double>(fileID,
                              A1format, A1_tile_format,
                              A2format, A2_tile_format,
                              A3format, A3_tile_format,
                              A1pos_rank, A1pos_ptr, readMode);
}

//===----------------------------------------------------------------------===//
///  Sort a vector within a range [first, last).
//===----------------------------------------------------------------------===//
extern "C" void _milr_ciface_comet_sort(UnrankedMemRefType<int64_t> *M, int64_t index_first, int64_t index_last)
{
  cometSortIndex(*M, index_first, index_last);
}

extern "C" void comet_sort_index(int64_t rank, void *ptr, int64_t index_first, int64_t index_last)
{
  UnrankedMemRefType<int64_t> descriptor = {rank, ptr};
  _milr_ciface_comet_sort(&descriptor, index_first, index_last);
}