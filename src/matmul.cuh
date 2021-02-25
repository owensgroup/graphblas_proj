// matmul.cuh

// void easy_mxm(
//     const int A_num_rows,
//     const int A_num_cols,
//     const int A_nnz,
//     int* dA_csrOffsets,
//     int* dA_columns,
//     float* dA_values,

//     const int B_num_rows,
//     const int B_num_cols,
//     const int B_nnz,    
//     int* dB_csrOffsets,
//     int* dB_columns,
//     float* dB_values,
    
//     int& C_num_rows,
//     int& C_num_cols,
//     int& C_nnz,
    
//     int* &dC_csrOffsets,
//     int* &dC_columns,
//     float* &dC_values
// ) {
//     // from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spgemm/spgemm_example.c

//     // int* dC_csrOffsets;
//     // int* dC_columns;
//     // float* dC_values;

//     float               alpha       = 1.0f;
//     float               beta        = 0.0f;
//     cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     cudaDataType        computeType = CUDA_R_32F;
    
//     //--------------------------------------------------------------------------
//     // CUSPARSE APIs
    
//     cusparseHandle_t     handle = NULL;
//     cusparseSpMatDescr_t matA, matB, matC;
    
//     void* dBuffer1 = NULL;
//     void* dBuffer2 = NULL;
    
//     size_t bufferSize1 = 0;
//     size_t bufferSize2 = 0;
    
//     cusparseCreate(&handle);
    
//     // Create sparse matrices
//     cusparseCreateCsr(
//       &matA, A_num_rows, A_num_cols, A_nnz,
//       dA_csrOffsets, dA_columns, dA_values,
//       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

//     cusparseCreateCsr(
//       &matB, B_num_rows, B_num_cols, B_nnz,
//       dB_csrOffsets, dB_columns, dB_values,
//       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
//     cusparseCreateCsr(
//       &matC, A_num_rows, B_num_cols, 0,
//       NULL, NULL, NULL,
//       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
//     // matmul
//     cusparseSpGEMMDescr_t spgemmDesc;
//     cusparseSpGEMM_createDescr(&spgemmDesc);

//     cusparseSpGEMM_workEstimation(
//       handle, opA, opB,
//       &alpha, matA, matB, &beta, matC,
//       computeType, CUSPARSE_SPGEMM_DEFAULT,
//       spgemmDesc, &bufferSize1, NULL
//     );
//     cudaMalloc((void**) &dBuffer1, bufferSize1);
    
//     cusparseSpGEMM_workEstimation(
//       handle, opA, opB,
//       &alpha, matA, matB, &beta, matC,
//       computeType, CUSPARSE_SPGEMM_DEFAULT,
//       spgemmDesc, &bufferSize1, dBuffer1
//     );
//     cusparseSpGEMM_compute(
//       handle, opA, opB,
//       &alpha, matA, matB, &beta, matC,
//       computeType, CUSPARSE_SPGEMM_DEFAULT,
//       spgemmDesc, &bufferSize2, NULL
//     );
//     cudaMalloc((void**) &dBuffer2, bufferSize2);

//     cusparseSpGEMM_compute(
//       handle, opA, opB,
//       &alpha, matA, matB, &beta, matC,
//       computeType, CUSPARSE_SPGEMM_DEFAULT,
//       spgemmDesc, &bufferSize2, dBuffer2
//     );
      
//     // compute size of C
//     int64_t C_num_rows1, C_num_cols1, C_nnz1;
//     cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
//     std::cout << "C_num_rows1: " << C_num_rows1 << std::endl;
//     std::cout << "C_num_cols1: " << C_num_cols1 << std::endl;
//     std::cout << "C_nnz1: " << C_nnz1 << std::endl;
        
//     // allocate C
//     cudaMalloc((void**) &dC_csrOffsets, (A_num_rows + 1) * sizeof(int  ));
//     cudaMalloc((void**) &dC_columns,    C_nnz1           * sizeof(int  ));
//     cudaMalloc((void**) &dC_values,     C_nnz1           * sizeof(float));
//     cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);

//     // "copy" results to C
//     cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

//     cusparseSpGEMM_destroyDescr(spgemmDesc);
//     cusparseDestroySpMat(matA);
//     cusparseDestroySpMat(matB);
//     cusparseDestroySpMat(matC);
//     cusparseDestroy(handle);
    
//     // free(dBuffer1); // when to free?
//     // free(dBuffer2); // when to free?
    
//     C_num_rows = C_num_rows1;
//     C_num_cols = C_num_cols1;
//     C_nnz      = C_nnz1;
// }


void easy_mxm(
    cusparseHandle_t handle,
  
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
  cudaMalloc((void**)&dC_csrOffsets, sizeof(int) * (A_num_rows + 1));
  
  int baseC, nnzC;
  csrgemm2Info_t info = NULL;
  size_t bufferSize   = 0;
  void *buffer        = NULL;
  
  int *nnzTotalDevHostPtr = &nnzC;
  
  float alpha = 1.0;
  float* beta = NULL;
  
  cusparseMatDescr_t descr;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  cusparseCreateCsrgemm2Info(&info);
  cudaDeviceSynchronize();
  
  cusparseStatus_t status;
  status = cusparseScsrgemm2_bufferSizeExt(
    handle,
    A_num_rows, B_num_cols, A_num_cols, 
    &alpha,
    descr, A_nnz, dA_csrOffsets, dA_columns,
    descr, B_nnz, dB_csrOffsets, dB_columns,
    beta,
    descr, B_nnz, dB_csrOffsets, dB_columns, // not used
    info, &bufferSize
  );
  
  cudaMalloc(&buffer, bufferSize);
  
  status = cusparseXcsrgemm2Nnz(
    handle,
    A_num_rows, B_num_cols, A_num_cols, 
    descr, A_nnz, dA_csrOffsets, dA_columns,
    descr, B_nnz, dB_csrOffsets, dB_columns,
    descr, 0,     dB_csrOffsets, dB_columns, // not used
    descr,        dC_csrOffsets, 
    nnzTotalDevHostPtr, info, buffer
  );
  
  if (NULL != nnzTotalDevHostPtr){
      nnzC = *nnzTotalDevHostPtr;
  } else {
      cudaMemcpy(&nnzC, dC_csrOffsets + A_num_rows, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, dC_csrOffsets, sizeof(int), cudaMemcpyDeviceToHost);
      nnzC -= baseC;
  }
  
  cudaMalloc((void**)&dC_columns, sizeof(int) * nnzC);
  cudaMalloc((void**)&dC_values, sizeof(double) * nnzC);
  
  cusparseScsrgemm2(
    handle,
    A_num_rows, B_num_cols, A_num_cols, 
    &alpha,
    descr, A_nnz, dA_values, dA_csrOffsets, dA_columns,
    descr, B_nnz, dB_values, dB_csrOffsets, dB_columns,
    beta,
    descr, B_nnz, dB_values, dB_csrOffsets, dB_columns,
    descr,        dC_values, dC_csrOffsets, dC_columns,
    info, buffer
  );

  cusparseDestroyCsrgemm2Info(info);
  
  C_num_rows = A_num_rows;
  C_num_cols = B_num_cols;
  C_nnz      = nnzC; 
}