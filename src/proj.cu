
#include <iostream>
#include <cusparse.h>
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


int easy_mxm(
    const int A_num_rows,
    const int A_num_cols,
    const int A_nnz,
    int* dA_csrOffsets,
    int* dA_columns,
    float* dA_values,

    const int B_num_rows,
    const int B_num_cols,
    const int B_nnz,    
    int* dB_csrOffsets,
    int* dB_columns,
    float* dB_values,
    
    int& C_num_rows,
    int& C_num_cols,
    int& C_nnz,
    
    int* &dC_csrOffsets,
    int* &dC_columns,
    float* &dC_values
) {

    // int* dC_csrOffsets;
    // int* dC_columns;
    // float* dC_values;

    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    
    void* dBuffer1 = NULL;
    void* dBuffer2 = NULL;
    
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
    
    cusparseCreate(&handle);
    
    // Create sparse matrices
    cusparseCreateCsr(
      &matA, A_num_rows, A_num_cols, A_nnz,
      dA_csrOffsets, dA_columns, dA_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    cusparseCreateCsr(
      &matB, B_num_rows, B_num_cols, B_nnz,
      dB_csrOffsets, dB_columns, dB_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    cusparseCreateCsr(
      &matC, A_num_rows, B_num_cols, 0,
      NULL, NULL, NULL,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    // matmul
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    cusparseSpGEMM_workEstimation(
      handle, opA, opB,
      &alpha, matA, matB, &beta, matC,
      computeType, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufferSize1, NULL
    );
    cudaMalloc((void**) &dBuffer1, bufferSize1);
    
    cusparseSpGEMM_workEstimation(
      handle, opA, opB,
      &alpha, matA, matB, &beta, matC,
      computeType, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufferSize1, dBuffer1
    );
    cusparseSpGEMM_compute(
      handle, opA, opB,
      &alpha, matA, matB, &beta, matC,
      computeType, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufferSize2, NULL
    );
    cudaMalloc((void**) &dBuffer2, bufferSize2);

    cusparseSpGEMM_compute(
      handle, opA, opB,
      &alpha, matA, matB, &beta, matC,
      computeType, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufferSize2, dBuffer2
    );
      
    // compute size of C
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
    std::cout << "C_num_rows1: " << C_num_rows1 << std::endl;
    std::cout << "C_num_cols1: " << C_num_cols1 << std::endl;
    std::cout << "C_nnz1: " << C_nnz1 << std::endl;
        
    // // allocate C
    // cudaMalloc((void**) &dC_csrOffsets, (A_num_rows + 1) * sizeof(int  ));
    // cudaMalloc((void**) &dC_columns,    C_nnz1           * sizeof(int  ));
    // cudaMalloc((void**) &dC_values,     C_nnz1           * sizeof(float));
    // cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);

    // // "copy" results to C
    // cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    // cusparseSpGEMM_destroyDescr(spgemmDesc);
    // cusparseDestroySpMat(matA);
    // cusparseDestroySpMat(matB);
    // cusparseDestroySpMat(matC);
    // cusparseDestroy(handle);
    
    // free(dBuffer1); // when to free?
    // free(dBuffer2); // when to free?
    
    C_num_rows = C_num_rows1;
    C_num_cols = C_num_cols1;
    C_nnz      = C_nnz1;
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
  h_indices = (int*  )malloc(nnz        * sizeof(int));
  h_data    = (float*)malloc(nnz        * sizeof(float));

  err = fread(h_indptr,  sizeof(int),   nrows + 1, file);
  err = fread(h_indices, sizeof(int),   nnz,      file);
  err = fread(h_data,    sizeof(float), nnz,      file);

  cudaMallocManaged(&d_indptr,  (nrows + 1) * sizeof(int));
  cudaMallocManaged(&d_indices, nnz         * sizeof(int));
  cudaMallocManaged(&d_data,    nnz         * sizeof(float));

  cudaMemcpy(d_indptr,  h_indptr,  (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, nnz         * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data,    h_data,    nnz         * sizeof(int), cudaMemcpyHostToDevice);
}


int main(int argc, char** argv) {
  read_binary(argv[1]);

  for(int i = 0; i < 10; i++)
    std::cout << d_indptr[i] << " ";
  std::cout << std::endl;

  for(int i = 0; i < 10; i++)
    std::cout << d_indices[i] << " ";
  std::cout << std::endl;

  for(int i = 0; i < 10; i++)
    std::cout << d_data[i] << " ";
  std::cout << std::endl;


  bool unweighted = true;
  bool onto_cols  = false;
  
  GpuTimer t;
  t.start();
  
  // --
  // Transpose

  cudaDeviceSynchronize();
  
  cusparseHandle_t handle = 0;
  cusparseStatus_t status = cusparseCreate(&handle);

  cudaMallocManaged((void**)&d_indptr_t,  (ncols + 1) * sizeof(int));
  cudaMallocManaged((void**)&d_indices_t, nnz        * sizeof(int));
  cudaMallocManaged((void**)&d_data_t,    nnz        * sizeof(float));

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

  // --
  // Change matrix edge weights

  // int block = 1 + nnz / THREAD;
  // if(unweighted) {
  //   __fill_constant<<<block, THREAD>>>(d_data,   1.0f, nnz);
  //   __fill_constant<<<block, THREAD>>>(d_data_t, 1.0f, nnz);
  // }

  // --
  // Projection

  // int dim_out = onto_cols ? ncols : nrows;

  int* p_indptr;
  int* p_indices;
  float* p_data;
  
  int p_nrows = -1;
  int p_ncols = -1;
  int p_nnz   = -1;
  
  easy_mxm(
    ncols, nrows, nnz,
    d_indptr_t, d_indices_t, d_data_t,

    nrows, ncols, nnz,
    d_indptr, d_indices, d_data,
    
    p_nrows, p_ncols, p_nnz,
    p_indptr, p_indices, p_data
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
