// #include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
// #include <cusparse.h>         // cusparseSpGEMM
// #include <stdio.h>            // printf
// #include <stdlib.h>           // EXIT_FAILURE

int easy_mxm(
    const int A_num_rows,
    const int A_num_cols,
    const int A_nnz,
    const int* dA_csrOffsets,
    const int* dA_columns,
    const float* dA_values,

    const int B_num_rows,
    const int B_num_cols,
    const int B_nnz,    
    const int* dB_csrOffsets,
    const int* dB_columns,
    const float* dB_values,

    int* dC_csrOffsets,
    int* dC_columns,
    float* dC_values
) {
  
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
    void* dBuffer2  = NULL;
    
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
    
    cusparseCreate(&handle);
    
    // Create sparse matrices
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
      dA_csrOffsets, dA_columns, dA_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
      dB_csrOffsets, dB_columns, dB_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
      NULL, NULL, NULL,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
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

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(
      handle, opA, opB,
      &alpha, matA, matB, &beta, matC,
      computeType, CUSPARSE_SPGEMM_DEFAULT,
      spgemmDesc, &bufferSize2, dBuffer2
    );
      
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
    
    // allocate matrix C
    cudaMalloc((void**),&dC_csrOffsets, (A_num_rows + 1) * sizeof(int  ));
    cudaMalloc((void**) &dC_columns,    C_nnz1           * sizeof(int  ));
    cudaMalloc((void**) &dC_values,     C_nnz1           * sizeof(float));
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);

    cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);
}
