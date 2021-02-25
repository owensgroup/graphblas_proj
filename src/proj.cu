
#include <iostream>
#include <cusparse.h>
#include "omp.h"
#include "nvToolsExt.h"

#include "matmul.cuh"
#include "timer.cuh"
#include "utils.cuh"

int nrows, ncols, nnz;

int* h_indptr;
int* h_indices;
float* h_data;

int* indptr;
int* indices;
float* data;

int* indptr_t;
int* indices_t;
float* data_t;

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

  cudaMalloc((void**)&indptr,  (nrows + 1) * sizeof(int));
  cudaMalloc((void**)&indices, nnz         * sizeof(int));
  cudaMalloc((void**)&data,    nnz         * sizeof(float));

  cudaMemcpy(indptr,  h_indptr,  (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(indices, h_indices, nnz         * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(data,    h_data,    nnz         * sizeof(float), cudaMemcpyHostToDevice);
}


int main(int argc, char** argv) {
  
  // --
  // MGPU setup
  
  int n_gpus = get_num_gpus();
  
  cudaStream_t* streams     = new cudaStream_t[n_gpus];
  cusparseHandle_t* handles = new cusparseHandle_t[n_gpus];

	for (int i = 0; i < n_gpus; i++) {
		cudaSetDevice(i);
		cudaStreamCreate(&(streams[i]));
    cusparseCreate(&(handles[i])); 
    cusparseSetStream(handles[i], streams[i]);
	}
  cudaSetDevice(0);
  
  // --
  // IO
  
  read_binary(argv[1]);

  bool unweighted = true;
  if(unweighted) {
    int block = 1 + nnz / THREAD;
    __fill_constant<<<block, THREAD>>>(data,   1.0f, nnz);
  }
  
  // --
  // Transpose (gpu0)

  cudaMallocManaged((void**)&indptr_t,  (ncols + 1) * sizeof(int));
  cudaMallocManaged((void**)&indices_t, nnz         * sizeof(int));
  cudaMallocManaged((void**)&data_t,    nnz         * sizeof(float));

  size_t buffer_size;
  cusparseCsr2cscEx2_bufferSize(
    handles[0],
    nrows, ncols, nnz,
    data, indptr, indices,
    data_t, indptr_t, indices_t,
    CUDA_R_32F,
    CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1,
    &buffer_size
  );
  
  char* buffer; cudaMalloc((void**)&buffer, sizeof(char) * buffer_size);

  cusparseCsr2cscEx2(
    handles[0],
    nrows, ncols, nnz,
    data, indptr, indices,
    data_t, indptr_t, indices_t,
    CUDA_R_32F,
    CUSPARSE_ACTION_NUMERIC,
    CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1,
    buffer
  );

  cudaDeviceSynchronize();

  int nrows_t = ncols;
  int ncols_t = nrows;

  // free(buffer); // when to free?
  
  // --
  // Copy data to gpus
  
  int** all_indptr    = (int**  )malloc(n_gpus * sizeof(int*  ));
  int** all_indices   = (int**  )malloc(n_gpus * sizeof(int*  ));
  float** all_data    = (float**)malloc(n_gpus * sizeof(float*));

  int** all_indptr_t  = (int**  )malloc(n_gpus * sizeof(int*  ));
  int** all_indices_t = (int**  )malloc(n_gpus * sizeof(int*  ));
  float** all_data_t  = (float**)malloc(n_gpus * sizeof(float*));
  
  nvtxRangePushA("copy");
  #pragma omp parallel for num_threads(n_gpus)
  for(int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    
    int* l_indptr; 
    int* l_indices;
    float* l_data;
    
    int* l_indptr_t; 
    int* l_indices_t;
    float* l_data_t;
    
    cudaMalloc(&l_indptr,  (nrows + 1) * sizeof(int  )); 
    cudaMalloc(&l_indices, nnz         * sizeof(int  ));
    cudaMalloc(&l_data,    nnz         * sizeof(float));
    
    cudaMalloc(&l_indptr_t,  (nrows_t + 1) * sizeof(int  )); 
    cudaMalloc(&l_indices_t, nnz           * sizeof(int  ));
    cudaMalloc(&l_data_t,    nnz           * sizeof(float));
    
    cudaMemcpy(l_indptr,  i  ndptr,  (nrows + 1) * sizeof(int  ), cudaMemcpyDeviceToDevice);
    cudaMemcpy(l_indices,   indices, nnz         * sizeof(int  ), cudaMemcpyDeviceToDevice);
    cudaMemcpy(l_data,      data,    nnz         * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(l_indptr_t,  indptr_t,  (nrows_t + 1) * sizeof(int  ), cudaMemcpyDeviceToDevice);
    cudaMemcpy(l_indices_t, indices_t, nnz           * sizeof(int  ), cudaMemcpyDeviceToDevice);
    cudaMemcpy(l_data_t,    data_t,    nnz           * sizeof(float), cudaMemcpyDeviceToDevice);
    
    all_indptr[i]    = l_indptr;
    all_indices[i]   = l_indices;
    all_data[i]      = l_data;
    
    all_indptr_t[i]  = l_indptr_t;
    all_indices_t[i] = l_indices_t;
    all_data_t[i]    = l_data_t; 
  }
  nvtxRangePop();
  cudaSetDevice(0);
  
  // --
  // Run on all GPUs
  
  GpuTimer t;
  t.start();

  nvtxRangePushA("work");
  #pragma omp parallel for num_threads(n_gpus)
  for(int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    
    int* p_indptr;
    int* p_indices;
    float* p_data;

    int p_nrows = -1;
    int p_ncols = -1;
    int p_nnz   = -1;
    
    easy_mxm(
      handles[i],
      nrows_t, ncols_t, nnz,
      all_indptr_t[i], all_indices_t[i], all_data_t[i],
      
      nrows, ncols, nnz,
      all_indptr[i], all_indices[i], all_data[i],
      
      p_nrows, p_ncols, p_nnz,
      p_indptr, p_indices, p_data
    );
  }
  
  for(int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
  cudaSetDevice(0);
  nvtxRangePop();
  
  t.stop();
  float elapsed = t.elapsed();
  
  std::cout << "elapsed : " << elapsed << std::endl;
}
