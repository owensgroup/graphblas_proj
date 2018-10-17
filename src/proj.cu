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

#define THREAD 1024

typedef graphblas::Matrix<float> Matrix;

__global__ void __fill_constant(float* d_x, float val, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n) {
    d_x[i] = val;
  }
}

int main(int argc, char** argv) {

  // --
  // CLI

  po::variables_map vm;

  parseArgsProj(argc, argv, vm);
  bool debug       = vm["proj-debug"].as<bool>();
  bool unweighted    = vm["unweighted"].as<bool>();
  bool print_results = vm["print-results"].as<bool>();

  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  // ---
  // IO

  std::vector<graphblas::Index> rowidx, colidx;
  std::vector<float> val;
  graphblas::Index num_rows, num_cols;
  graphblas::Index num_edges;

  std::string X_path = vm["X"].as<std::string>();
  if(debug) fprintf(stderr, "proj.cu: loading %s\n", X_path.c_str());
  readMtx(X_path.c_str(), rowidx, colidx, val, num_rows, num_cols, num_edges, 1, false);
  Matrix X(num_rows, num_cols);
  X.build(&rowidx, &colidx, &val, num_edges, GrB_NULL);
  if(debug) fprintf(stderr, "\tdone\n");

  // --
  // Transpose

  if(debug) fprintf(stderr, "proj.cu: computing transpose\n");

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

  if(debug) fprintf(stderr, "\tdone\n");

  // --
  // Change matrix edge weights

  int block = 1 + num_edges / THREAD;
  if(unweighted) {
    __fill_constant<<<block, THREAD>>>(X.matrix_.sparse_.d_csrVal_, 1.0f, num_edges);
    __fill_constant<<<block, THREAD>>>(tX.matrix_.sparse_.d_csrVal_, 1.0f, num_edges);
  }

  // --
  // Projection

  if(debug) fprintf(stderr, "proj.cu: computing projection\n");

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

  if(debug) fprintf(stderr, "\tdone\n");

  // --
  // Read results

  int proj_num_edges; P.nvals(&proj_num_edges);
  std::cerr << "proj_num_edges=" << proj_num_edges << std::endl;
  std::cerr << "num_cols=" << num_cols << std::endl;

  if(print_results) {
    int* h_proj_rowptr = (int*)malloc((num_cols + 1) * sizeof(int));
    int* h_proj_colidx = (int*)malloc(proj_num_edges * sizeof(int));
    float* h_proj_val  = (float*)malloc(proj_num_edges * sizeof(float));

    cudaMemcpy(h_proj_rowptr, P.matrix_.sparse_.d_csrRowPtr_, (num_cols + 1) * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_proj_colidx, P.matrix_.sparse_.d_csrColInd_, proj_num_edges * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_proj_val,    P.matrix_.sparse_.d_csrVal_,    proj_num_edges * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < num_cols; i++) {
      int start = h_proj_rowptr[i];
      int end   = h_proj_rowptr[i + 1];
      for(int offset = start; offset < end; offset++) {
        if(i != h_proj_colidx[offset]) { // Don't print self loops
          printf("%d %d %f\n", i, h_proj_colidx[offset], h_proj_val[offset]);
        }
      }
    }
    free(h_proj_rowptr);
    free(h_proj_colidx);
    free(h_proj_val);
  }

  // --
  // Free memory

  X.clear();
  tX.clear();
}
