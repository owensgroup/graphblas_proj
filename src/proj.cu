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
#include "timer.cuh"
#include "utils.cuh"

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
  bool debug         = vm["proj-debug"].as<bool>();
  bool unweighted    = vm["unweighted"].as<bool>();
  bool print_results = vm["print-results"].as<bool>();
  bool onto_cols     = vm["onto-cols"].as<bool>();
  int  num_chunks    = vm["num-chunks"].as<int>();
  bool chunked       = num_chunks > 0;

  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  // ---
  // IO

  std::vector<int> rowidx, colidx;
  std::vector<float> val;
  int num_rows, num_cols;
  int num_edges;

  std::string X_path = vm["X"].as<std::string>();
  if(debug) fprintf(stderr, "proj.cu: loading %s\n", X_path.c_str());
  readMtx(X_path.c_str(), rowidx, colidx, val, num_rows, num_cols, num_edges, 1, false);
  Matrix X(num_rows, num_cols);
  X.build(&rowidx, &colidx, &val, num_edges, GrB_NULL);
  if(debug) fprintf(stderr, "\tdone\n");


  // --
  // Transpose
  GpuTimer timer;
  timer.Start();

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
  int dim_out;

  if(onto_cols) {
    dim_out = num_cols;
  } else {
    dim_out = num_rows;
  }

  Matrix P(dim_out, dim_out);

  if(chunked) {
    if(onto_cols) {
      chunked_mxm(&P, &tX, &X, &desc, num_chunks);
    } else
      chunked_mxm(&P, &X, &tX, &desc, num_chunks);
    }
  } else {
    if(onto_cols) {
      easy_mxm(&P, &tX, &X, &desc);
    } else {
      easy_mxm(&P, &X, &tX, &desc);
    }
  }

  timer.Stop();
  if(debug) fprintf(stderr, "\tdone\n");

  // --
  // Read results

  int proj_num_edges; P.nvals(&proj_num_edges);
  std::cerr << "proj_num_edges          = " << proj_num_edges << std::endl;
  std::cerr << "dim_out                 = " << dim_out << std::endl;
  std::cerr << "proj_num_edges (noloop) = " << proj_num_edges - dim_out << std::endl;
  std::cerr << "timer                   = "  << timer.ElapsedMillis() << std::endl;

  if(print_results) {
    if(!chunked) P.matrix_.sparse_.gpuToCpu();

    int* h_proj_rowptr = P.matrix_.sparse_.h_csrRowPtr_;
    int* h_proj_colidx = P.matrix_.sparse_.h_csrColInd_;
    float* h_proj_val  = P.matrix_.sparse_.h_csrVal_;

    for(int i = 0; i < dim_out; i++) {
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
