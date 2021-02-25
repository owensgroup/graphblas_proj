
#include <iostream>
#include <cusparse.h>

#include "matmul.cuh"
#include "timer.cuh"

int nrows, ncols, nnz;

int* h_indptr;
int* h_indices;
float* h_data;

int* d_indptr;
int* d_indices;
float* d_data;

int* d_indptr_t;
int* d_indices_t;
float* d_data_t;

#define THREAD 1024

// --
// Helpers

__global__ void __fill_constant(float* d_x, float val, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n) d_x[i] = val;
}

void read_binary(std::string filename) {
  FILE* file = fopen(filename.c_str(), "rb");
  
  auto err = fread(&nrows, sizeof(int), 1, file);
  err = fread(&ncols, sizeof(int), 1, file);
  err = fread(&nnz,  sizeof(int), 1, file);

  std::cerr << "nrows : " << nrows << std::endl;
  std::cerr << "ncols : " << ncols << std::endl;
  std::cerr << "nnz   : " << nnz << std::endl;

  h_indptr  = (int*  )malloc((nrows + 1) * sizeof(int));
  h_indices = (int*  )malloc(nnz         * sizeof(int));
  h_data    = (float*)malloc(nnz         * sizeof(float));

  err = fread(h_indptr,  sizeof(int),   nrows + 1, file);
  err = fread(h_indices, sizeof(int),   nnz,      file);
  err = fread(h_data,    sizeof(float), nnz,      file);

  cudaMalloc((void**)&d_indptr,  (nrows + 1) * sizeof(int));
  cudaMalloc((void**)&d_indices, nnz         * sizeof(int));
  cudaMalloc((void**)&d_data,    nnz         * sizeof(float));

  cudaMemcpy(d_indptr,  h_indptr,  (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, nnz         * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data,    h_data,    nnz         * sizeof(float), cudaMemcpyHostToDevice);
  
  std::cout << "--" << std::endl;
  std::cout << "indptr  : ";
  for(int i = 0; i < nrows + 1; i++)
    std::cout << h_indptr[i] << " ";
  std::cout << std::endl;

  std::cout << "indices : ";
  for(int i = 0; i < nnz; i++)
    std::cout << h_indices[i] << " ";
  std::cout << std::endl;

  std::cout << "data    : ";
  for(int i = 0; i < nnz; i++)
    std::cout << h_data[i] << " ";
  std::cout << std::endl;
}


int main(int argc, char** argv) {
  read_binary(argv[1]);

  bool unweighted = true;
  
  GpuTimer t;
  t.start();
  
  // --
  // Transpose

  cudaDeviceSynchronize();
  
  cusparseHandle_t handle = 0;
  cusparseStatus_t status = cusparseCreate(&handle);

  cudaMallocManaged((void**)&d_indptr_t,  (ncols + 1) * sizeof(int));
  cudaMallocManaged((void**)&d_indices_t, nnz         * sizeof(int));
  cudaMallocManaged((void**)&d_data_t,    nnz         * sizeof(float));

  size_t buffer_size;
  cusparseCsr2cscEx2_bufferSize(
    handle,
    nrows, ncols, nnz,
    d_data, d_indptr, d_indices,
    d_data_t, d_indptr_t, d_indices_t,
    CUDA_R_32F,
    CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1,
    &buffer_size
  );
  
  char* buffer;
  cudaMalloc((void**)&buffer, sizeof(char)*buffer_size);

  cusparseCsr2cscEx2(
    handle,
    nrows, ncols, nnz,
    d_data, d_indptr, d_indices,
    d_data_t, d_indptr_t, d_indices_t,
    CUDA_R_32F,
    CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1,
    buffer
  );
  cusparseDestroy(handle);
  
  // free(buffer); // when to free?
  
  cudaDeviceSynchronize();

  std::cout << "----" << std::endl;
  
  std::cout << "indptr_t : ";
  for(int i = 0; i < ncols + 1; i++)
    std::cout << d_indptr_t[i] << " ";
  std::cout << std::endl;

  std::cout << "indices_t : ";
  for(int i = 0; i < nnz; i++)
    std::cout << d_indices_t[i] << " ";
  std::cout << std::endl;

  std::cout << "data_t    : ";
  for(int i = 0; i < nnz; i++)
    std::cout << d_data_t[i] << " ";
  std::cout << std::endl;
  
  // --
  // Change matrix edge weights

  // int block = 1 + nnz / THREAD;
  // if(unweighted) {
  //   __fill_constant<<<block, THREAD>>>(d_data,   1.0f, nnz);
  //   __fill_constant<<<block, THREAD>>>(d_data_t, 1.0f, nnz);
  // }

  // --
  // Projection

  int* p_indptr;
  int* p_indices;
  float* p_data;
  
  int p_nrows = -1;
  int p_ncols = -1;
  int p_nnz   = -1;
  
  cudaDeviceSynchronize();
  easy_mxm_legacy(
    ncols, nrows, nnz,
    d_indptr_t, d_indices_t, d_data_t,

    nrows, ncols, nnz,
    d_indptr, d_indices, d_data,
    
    p_nrows, p_ncols, p_nnz
    // p_indptr, p_indices, p_data
  );
  cudaDeviceSynchronize();
  
  t.stop();
  float elapsed = t.elapsed();
  
  std::cout << "elapsed : " << elapsed << std::endl;
  std::cout << "p_nrows : " << p_nrows << std::endl;
  std::cout << "p_ncols : " << p_ncols << std::endl;
  std::cout << "p_nnz   : " << p_nnz << std::endl;
  
  // // --
  // // Copy to host
  
  // int* h_p_indptr  = (int*  )malloc((p_nrows + 1) * sizeof(int));
  // int* h_p_indices = (int*  )malloc(p_nnz         * sizeof(int));
  // float* h_p_data  = (float*)malloc(p_nnz         * sizeof(int));
  
  // cudaMemcpy(h_p_indptr,  p_indptr,  (p_nrows + 1) * sizeof(int  ), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_p_indices, p_indices, p_nnz         * sizeof(int  ), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_p_data,    p_data,    p_nnz         * sizeof(float), cudaMemcpyDeviceToHost);
  
  // for(int i = 0; i < 10; i++)
  //   std::cout << h_p_indptr[i] << " ";
  // std::cout << std::endl;

  // for(int i = 0; i < 10; i++)
  //   std::cout << h_p_indices[i] << " ";
  // std::cout << std::endl;

  // for(int i = 0; i < 10; i++)
  //   std::cout << h_p_data[i] << " ";
  // std::cout << std::endl;
}
