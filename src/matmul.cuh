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
