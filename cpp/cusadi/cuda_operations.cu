#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <torch/extension.h>


// TODO: look at stackoverflow page, adjust for row_major and column major matrices.
// TODO: narrow scope, assume all matrices will be 2D at most (when is this not the case?)
    // output is always flattened to nnz x 1
    // work vectors will always be flattened to [[input_1_dim x 1], [input_2_dim x 1], ...]
// TODO: can bring casadi parallelization for MPC speciifcally? 

/************************* CUDA KERNEL FUNCTIONS *************************/

// Arithmetic operations
// Strided implementation of add_kernel
__global__ void add_strided_kernel(float __restrict__ *f_out, const unsigned int col_f_out,
                           const float __restrict__ *f_in1, const unsigned int col_f_in1,
                           const float __restrict__ *f_in2, const unsigned int col_f_in2,
                           const unsigned int num_rows, const unsigned int num_cols) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_rows) {
        f_out[idx*num_cols + col_f_out] = f_in1[idx*num_cols + col_f_in1]
            + f_in2[idx*num_cols + col_f_in2];
    }
}

// Accessor implementation of add_kernel
__global__ void add_accessor_kernel(torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> *f_out, const unsigned int col_f_out,
                                    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> *f_in1, const unsigned int col_f_in1,
                                    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> *f_in2, const unsigned int col_f_in2) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    printf("f_out size: %d\n", f_out.size(0));
    if (idx < f_out.size(0)) {
        f_out[idx][col_f_out] = f_in1[idx][col_f_in1] + f_in2[idx][col_f_in2];
    }
}

/************************* EXTERNAL WRAPPER FUNCTIONS *************************/
// TODO: remove error checking once functions have been tested
// TODO: unit tests for each function

extern "C" void op_add_strided(float *f_out, const unsigned int col_f_out,
                       const float *f_in1, const unsigned int col_f_in1,
                       const float *f_in2, const unsigned int col_f_in2,
                       const unsigned int num_rows, const unsigned int num_cols) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (num_rows + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    add_strided_kernel<<<gridSize, blockSize>>>(f_out, col_f_out, 
                                        f_in1, col_f_in1,
                                        f_in2, col_f_in2,
                                        num_rows, num_cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}

