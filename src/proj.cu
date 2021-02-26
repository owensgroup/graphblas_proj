
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
// #define VERBOSE

// --
// Helpers

__global__ void __fill_constant(float* d_x, float val, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n) d_x[i] = val;
}

__global__ void __subtract(int * x, int c, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n) x[i] -= c;
}


void read_binary(std::string filename) {
  FILE* file = fopen(filename.c_str(), "rb");
  
  auto err = fread(&nrows, sizeof(int), 1, file);
  err = fread(&ncols, sizeof(int), 1, file);
  err = fread(&nnz,  sizeof(int), 1, file);

#ifdef VERBOSE
  std::cerr << "nrows : " << nrows << std::endl;
  std::cerr << "ncols : " << ncols << std::endl;
  std::cerr << "nnz   : " << nnz << std::endl;
#endif

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
  
  GpuTimer t;
  t.start();

  // --
  // Transpose (gpu0)

  // >>
  // cudaMalloc((void**)&indptr_t,  (ncols + 1) * sizeof(int));
  // cudaMalloc((void**)&indices_t, nnz         * sizeof(int));
  // cudaMalloc((void**)&data_t,    nnz         * sizeof(float));
  // --
  // This is uglier, but it speeds things up a little ... 
  cudaMallocManaged((void**)&indptr_t,  (ncols + 1) * sizeof(int));
  cudaMallocManaged((void**)&indices_t, nnz         * sizeof(int));
  cudaMallocManaged((void**)&data_t,    nnz         * sizeof(float));
  // <<

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
  
  int chunk_size = (nrows_t + n_gpus - 1) / n_gpus;
  
  int** all_indptr    = (int**  )malloc(n_gpus * sizeof(int*  ));
  int** all_indices   = (int**  )malloc(n_gpus * sizeof(int*  ));
  float** all_data    = (float**)malloc(n_gpus * sizeof(float*));

  int** chunked_indptr_t  = (int**  )malloc(n_gpus * sizeof(int*  ));
  int** chunked_indices_t = (int**  )malloc(n_gpus * sizeof(int*  ));
  float** chunked_data_t  = (float**)malloc(n_gpus * sizeof(float*));
  
  int* chunked_nrows  = (int*)malloc(n_gpus * sizeof(int));
  int* chunked_nnzs   = (int*)malloc(n_gpus * sizeof(int));
  
  nvtxRangePushA("copy");
  #pragma omp parallel for num_threads(n_gpus)
  for(int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    
    // Copy RHS    
    int* l_indptr; 
    int* l_indices;
    float* l_data;

    cudaMalloc(&l_indptr,    (nrows + 1)   * sizeof(int  )); 
    cudaMalloc(&l_indices,   nnz           * sizeof(int  ));
    cudaMalloc(&l_data,      nnz           * sizeof(float));

    cudaMemcpyAsync(l_indptr,    indptr,    (nrows + 1)   * sizeof(int  ), cudaMemcpyDeviceToDevice, streams[i]);
    cudaMemcpyAsync(l_indices,   indices,   nnz           * sizeof(int  ), cudaMemcpyDeviceToDevice, streams[i]);
    cudaMemcpyAsync(l_data,      data,      nnz           * sizeof(float), cudaMemcpyDeviceToDevice, streams[i]);

    // Prefetch LHS
    int* l_indptr_t; 
    int* l_indices_t;
    float* l_data_t;
    
    cudaMalloc(&l_indptr_t,  (nrows_t + 1) * sizeof(int  )); 
    cudaMalloc(&l_indices_t, nnz           * sizeof(int  ));
    cudaMalloc(&l_data_t,    nnz           * sizeof(float));
    
    cudaMemcpyAsync(l_indptr_t,  indptr_t,  (nrows_t + 1) * sizeof(int  ), cudaMemcpyDeviceToDevice, streams[i]);
    cudaMemcpyAsync(l_indices_t, indices_t, nnz           * sizeof(int  ), cudaMemcpyDeviceToDevice, streams[i]);
    cudaMemcpyAsync(l_data_t,    data_t,    nnz           * sizeof(float), cudaMemcpyDeviceToDevice, streams[i]);
    
    // Make chunks
    int* c_indptr_t;
    int* c_indices_t;
    float* c_data_t;

    int chunk_start = chunk_size * i;
    int chunk_end   = chunk_size * (i + 1);
    if(chunk_end > nrows_t) chunk_end = nrows_t;
    
    int chunk_start_offset;
    int chunk_end_offset;
    cudaMemcpyAsync(&chunk_start_offset, l_indptr_t + chunk_start, sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    cudaMemcpyAsync(&chunk_end_offset,   l_indptr_t + chunk_end,   sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    cudaStreamSynchronize(streams[i]);

    int chunk_rows = chunk_end - chunk_start;
    int chunk_nnz  = chunk_end_offset - chunk_start_offset;
    
    cudaMalloc((void**)&c_indptr_t,  (chunk_rows + 1) * sizeof(int));
    cudaMalloc((void**)&c_indices_t, chunk_nnz        * sizeof(int));
    cudaMalloc((void**)&c_data_t,    chunk_nnz        * sizeof(float));
    
    cudaMemcpyAsync(c_indptr_t,  l_indptr_t  + chunk_start,        (chunk_rows + 1) * sizeof(int),   cudaMemcpyDeviceToDevice, streams[i]);
    cudaMemcpyAsync(c_indices_t, l_indices_t + chunk_start_offset, chunk_nnz        * sizeof(int),   cudaMemcpyDeviceToDevice, streams[i]);
    cudaMemcpyAsync(c_data_t,    l_data_t    + chunk_start_offset, chunk_nnz        * sizeof(float), cudaMemcpyDeviceToDevice, streams[i]);

    int blocks = 1 + chunk_rows / 1024;
    if(chunk_start_offset > 0)
        __subtract<<<blocks, 1024, 0, streams[i]>>>(c_indptr_t, chunk_start_offset, chunk_rows + 1);

    all_indptr[i]        = l_indptr;
    all_indices[i]       = l_indices;
    all_data[i]          = l_data;
    chunked_indptr_t[i]  = c_indptr_t;
    chunked_indices_t[i] = c_indices_t;
    chunked_data_t[i]    = c_data_t; 
    chunked_nrows[i]     = chunk_rows;
    chunked_nnzs[i]      = chunk_nnz;
    
    cudaStreamSynchronize(streams[i]);
  }
  nvtxRangePop();

  cudaSetDevice(0);

  // --
  // Run on all GPUs

  nvtxRangePushA("work");
  
  int acc = 0;
  
  #pragma omp parallel for num_threads(n_gpus) reduction(+:acc)
  for(int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    
    int* p_indptr;
    int* p_indices;
    float* p_data;

    int p_nrows;
    int p_ncols;
    int p_nnz;
    
    easy_mxm(
      handles[i],
      chunked_nrows[i], ncols_t, chunked_nnzs[i],
      chunked_indptr_t[i], chunked_indices_t[i], chunked_data_t[i],
      
      nrows, ncols, nnz,
      all_indptr[i], all_indices[i], all_data[i],
      
      p_nrows, p_ncols, p_nnz,
      p_indptr, p_indices, p_data
    );
    
    acc += p_nnz;
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
  std::cout << "acc     : " << acc     << std::endl;
}
