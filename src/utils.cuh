

void easy_mxm(
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
}

__global__ void __subtract(int * x, int c, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n) {
        x[i] -= c;
    }
}

void chunked_mxm(
    graphblas::Matrix<float>* out,
    graphblas::Matrix<float>* A,
    graphblas::Matrix<float>* B,
    graphblas::Descriptor* desc,
    int num_chunks
)
{
    A->matrix_.sparse_.gpuToCpu();
    B->matrix_.sparse_.gpuToCpu();

    int A_nrows; A->nrows(&A_nrows);
    int A_ncols; A->ncols(&A_ncols);
    int B_ncols; B->ncols(&B_ncols);

    int chunk_size = (1 + A_nrows / num_chunks);
    // std::cerr << "A_nrows=" << A_nrows << std::endl;
    // std::cerr << "chunk_size=" << chunk_size << std::endl;

    int** h_chunked_ptr   = new int*[num_chunks];
    int** h_chunked_ind   = new int*[num_chunks];
    float** h_chunked_val = new float*[num_chunks];

    for(int chunk = 0; chunk < num_chunks; chunk++) {
        int chunk_start = chunk * chunk_size;
        int chunk_end   = min((chunk + 1) * chunk_size, A_nrows);
        int chunk_rows  = chunk_end - chunk_start;

        // std::cerr << "chunk_rows  = "   << chunk_rows << std::endl;
        // std::cerr << "chunk_start = " << chunk_start << std::endl;
        // std::cerr << "chunk_end   = "   << chunk_end << std::endl;

        int chunk_start_offset = A->matrix_.sparse_.h_csrRowPtr_[chunk_start];
        int chunk_end_offset   = A->matrix_.sparse_.h_csrRowPtr_[chunk_end];
        int chunk_edges        = chunk_end_offset - chunk_start_offset;

        // std::cerr << "chunk_start_offset = " << chunk_start_offset << std::endl;
        // std::cerr << "chunk_end_offset   = " << chunk_end_offset << std::endl;
        // std::cerr << "chunk_edges        = " << chunk_edges << std::endl;
        if(chunk_edges == 0) {
            continue;
        }

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

        // // Debug
        // int* h_chunk_ptr   = (int*)malloc((chunk_rows + 1) * sizeof(int));
        // int* h_chunk_ind   = (int*)malloc(chunk_edges * sizeof(int));
        // float* h_chunk_val = (float*)malloc(chunk_edges * sizeof(float));
        // cudaMemcpy(h_chunk_ptr, d_chunk_ptr, (chunk_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_chunk_ind, d_chunk_ind, chunk_edges * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_chunk_val, d_chunk_val, chunk_edges * sizeof(float), cudaMemcpyDeviceToHost);

        // for(int i = 0; i < chunk_rows; i++) {
        //     for(int offset = h_chunk_ptr[i]; offset < h_chunk_ptr[i + 1]; offset++) {
        //         std::cerr << i << " " << offset << " " << h_chunk_ind[offset] << " " << h_chunk_val[offset] << std::endl;
        //     }
        // }

        // --
        // mxm

        graphblas::Matrix<float> A_chunk(chunk_rows, A_ncols);
        graphblas::Matrix<float> C_chunk(chunk_rows, B_ncols);
        A_chunk.build(d_chunk_ptr, d_chunk_ind, d_chunk_val, chunk_edges);
        easy_mxm(&C_chunk, &A_chunk, B, desc);

        // --
        // Copy back to CPU

        int C_chunk_edges; C_chunk.nvals(&C_chunk_edges);

        h_chunked_ptr[chunk] = (int*)malloc((chunk_rows + 1) * sizeof(int));
        h_chunked_ind[chunk] = (int*)malloc(C_chunk_edges * sizeof(int));
        h_chunked_val[chunk] = (float*)malloc(C_chunk_edges * sizeof(float));

        cudaMemcpy(h_chunked_ptr[chunk], C_chunk.matrix_.sparse_.d_csrRowPtr_, (chunk_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_chunked_ind[chunk], C_chunk.matrix_.sparse_.d_csrColInd_, C_chunk_edges * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_chunked_val[chunk], C_chunk.matrix_.sparse_.d_csrVal_, C_chunk_edges * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // --
    // Print for debugging

    std::vector<int> rows, cols;
    std::vector<float> vals;
    for(int chunk = 0; chunk < num_chunks; chunk++) {
        int chunk_start = chunk * chunk_size;
        int chunk_end   = min((chunk + 1) * chunk_size, A_nrows);
        int chunk_rows  = chunk_end - chunk_start;
        for(int i = 0; i < chunk_rows; i++) {
            for(int offset = h_chunked_ptr[chunk][i]; offset < h_chunked_ptr[chunk][i + 1]; offset++) {
                // std::cerr << chunk_start + i << " " << h_chunked_ind[chunk][offset] << " " << h_chunked_val[chunk][offset] << std::endl;
                rows.push_back(chunk_start + i);
                cols.push_back(h_chunked_ind[chunk][offset]);
                vals.push_back(h_chunked_val[chunk][offset]);
            }
        }
    }

    out->build(&rows, &cols, &vals, vals.size(), GrB_NULL);
}