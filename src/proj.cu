#define GRB_USE_APSPIE
#define private public
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#include "cli.h"

typedef graphblas::Matrix<float> Matrix;

int main(int argc, char** argv) {
  // --
  // CLI

  po::variables_map vm;

  parseArgsProj(argc, argv, vm);
  bool verbose = vm["proj-debug"].as<bool>();

  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  // ---
  // IO

  std::vector<graphblas::Index> rowidx, colidx;
  std::vector<float> val;
  graphblas::Index num_rows, num_cols;
  graphblas::Index num_edges;

  std::string X_path = vm["X"].as<std::string>();

  readMtx(X_path.c_str(), rowidx, colidx, val, num_rows, num_cols, num_edges, 1, false);
  Matrix X(num_rows, num_cols);
  X.build(&rowidx, &colidx, &val, num_edges, GrB_NULL);

  // --
  // Transpose

  cusparseHandle_t handle = 0;
  cusparseStatus_t status = cusparseCreate(&handle);

  int* tx_colidx;
  int* tx_rowptr;
  float* tx_val;
  cudaMalloc((void**)&tx_colidx,      num_edges * sizeof(int));
  cudaMalloc((void**)&tx_rowptr, (num_cols + 1) * sizeof(int));
  cudaMalloc((void**)&tx_val,         num_edges * sizeof(float));

  cusparseScsr2csc(
    handle,
    num_rows, num_cols, num_edges,
    X.matrix_.sparse_.d_csrVal_, X.matrix_.sparse_.d_csrRowPtr_, X.matrix_.sparse_.d_csrColInd_,
    tx_val, tx_colidx, tx_rowptr,
    CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO
  );

  Matrix tX(num_cols, num_rows);
  tX.build(tx_rowptr, tx_colidx, tx_val, num_edges);

  // --
  // Projection

  Matrix P(num_cols, num_cols);
  graphblas::mxm<float,float,float,float>(
    &P,
    GrB_NULL,
    GrB_NULL,
    graphblas::PlusMultipliesSemiring<float>(),
    &tX,
    &X,
    &desc
  );
}
