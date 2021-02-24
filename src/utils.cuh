// utils.cuh

uint64_t easy_mxm(
    graphblas::Matrix<float>* out,
    graphblas::Matrix<float>* A,
    graphblas::Matrix<float>* B,
    graphblas::Descriptor* desc
)
{
    graphblas::mxm<float,float,float,float>(
        out,
        GrB_NULL,
        GrB_NULL,
        graphblas::PlusMultipliesSemiring<float>(),
        A,
        B,
        desc
    );
    
    int num_edges; out->nvals(&num_edges);
    
    if(num_edges < 0)
        std::cerr << "overflow!" << std::endl;
        
    return (uint64_t)num_edges;
}

__global__ void __subtract(int * x, int c, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n) {
        x[i] -= c;
    }
}

uint64_t chunked_mxm(
    int** h_chunked_ptr,
    int** h_chunked_ind,
    float** h_chunked_val,
    graphblas::Matrix<float>* A,
    graphblas::Matrix<float>* B,
    graphblas::Descriptor* desc,
    int num_chunks
)
{
    // if this is false, intermediate results get freed
    bool copy_to_host = true;

    uint64_t total_edges = 0;
    A->matrix_.sparse_.gpuToCpu();

    int A_nrows; A->nrows(&A_nrows);
    int A_ncols; A->ncols(&A_ncols);
    int B_ncols; B->ncols(&B_ncols);

    int chunk_size = (1 + A_nrows / num_chunks);

    for(int chunk = 0; chunk < num_chunks; chunk++) {
        std::cerr << "chunk=" << chunk << std::endl;

        int chunk_start        = chunk * chunk_size;
        int chunk_end          = min((chunk + 1) * chunk_size, A_nrows);
        int chunk_rows         = chunk_end - chunk_start;
        int chunk_start_offset = A->matrix_.sparse_.h_csrRowPtr_[chunk_start];
        int chunk_end_offset   = A->matrix_.sparse_.h_csrRowPtr_[chunk_end];
        int chunk_edges        = chunk_end_offset - chunk_start_offset;
        if(chunk_edges == 0) { continue; }

        int* d_chunk_ptr;
        int* d_chunk_ind;
        float* d_chunk_val;
        cudaMalloc((void**)&d_chunk_ptr, (chunk_rows + 1) * sizeof(int));
        cudaMalloc((void**)&d_chunk_ind, chunk_edges * sizeof(int));
        cudaMalloc((void**)&d_chunk_val, chunk_edges * sizeof(float));

        cudaMemcpy(d_chunk_ptr, A->matrix_.sparse_.d_csrRowPtr_ + chunk_start, (chunk_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_chunk_ind, A->matrix_.sparse_.d_csrColInd_ + chunk_start_offset, chunk_edges * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_chunk_val, A->matrix_.sparse_.d_csrVal_ + chunk_start_offset, chunk_edges * sizeof(float), cudaMemcpyDeviceToDevice);

        // Fix offsets
        int blocks = 1 + chunk_rows / 1024;
        if(chunk_start_offset > 0) {
            __subtract<<<blocks, 1024>>>(d_chunk_ptr, chunk_start_offset, chunk_rows + 1);
        }

        graphblas::Matrix<float> A_chunk(chunk_rows, A_ncols);
        A_chunk.build(d_chunk_ptr, d_chunk_ind, d_chunk_val, chunk_edges);

        // --
        // mxm

        graphblas::Matrix<float> C_chunk(chunk_rows, B_ncols);
        uint64_t C_chunk_edges = easy_mxm(&C_chunk, &A_chunk, B, desc);

        // --
        // Copy back to CPU

        if(copy_to_host) {
            cudaMallocHost((void**)&h_chunked_ptr[chunk], (chunk_rows + 1) * sizeof(int));
            cudaMallocHost((void**)&h_chunked_ind[chunk], C_chunk_edges * sizeof(int));
            cudaMallocHost((void**)&h_chunked_val[chunk], C_chunk_edges * sizeof(float));

            cudaMemcpy(h_chunked_ptr[chunk], C_chunk.matrix_.sparse_.d_csrRowPtr_, (chunk_rows + 1) * sizeof(int),
                cudaMemcpyDeviceToHost);
            cudaMemcpy(h_chunked_ind[chunk], C_chunk.matrix_.sparse_.d_csrColInd_, C_chunk_edges * sizeof(int),
                cudaMemcpyDeviceToHost);
            cudaMemcpy(h_chunked_val[chunk], C_chunk.matrix_.sparse_.d_csrVal_, C_chunk_edges * sizeof(float),
                cudaMemcpyDeviceToHost);
        }

        A_chunk.clear();
        C_chunk.clear();

        total_edges += C_chunk_edges;
    }

    return total_edges;
}


    // std::vector<int> rows, cols;
    // std::vector<float> vals;
    // for(int chunk = 0; chunk < num_chunks; chunk++) {
    //     int chunk_start = chunk * chunk_size;
    //     int chunk_end   = min((chunk + 1) * chunk_size, A_nrows);
    //     int chunk_rows  = chunk_end - chunk_start;
    //     for(int i = 0; i < chunk_rows; i++) {
    //         for(int offset = h_chunked_ptr[chunk][i]; offset < h_chunked_ptr[chunk][i + 1]; offset++) {
    //             // std::cerr << chunk_start + i << " " << h_chunked_ind[chunk][offset]
    //             //   << " " << h_chunked_val[chunk][offset] << std::endl;
    //             rows.push_back(chunk_start + i);
    //             cols.push_back(h_chunked_ind[chunk][offset]);
    //             vals.push_back(h_chunked_val[chunk][offset]);
    //         }
    //     }
    // }

    // out->build(&rows, &cols, &vals, vals.size(), GrB_NULL);