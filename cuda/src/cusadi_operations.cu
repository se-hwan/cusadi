#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

/* AssignFromInput - Assigns a column of the matrix to a thread block
    * @param out - output matrix to assign column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - input matrix to assign column from
    * @param col_in - assignment column of input matrix
    * @param sz_in - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void AssignFromInput(float* __restrict__ out, const int col_out, const int sz_out,
                                float* __restrict__ in, const int col_in, const int sz_in,
                                const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        out[idx*sz_out + col_out] = in[idx*sz_in + col_in];
    }
}

/* AssignFromConst - Assigns a column of the output matrix to a const
    * @param out - output matrix to assign column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - constant to assign to column
    * @param batch_size - number of rows in input and output
*/
__device__ void AssignFromConst(float* __restrict__ out, const int col_out, const int sz_out,
                                const float in,
                                const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        out[idx*sz_out + col_out] = in;
    }
}

/* AddFromInput - Adds a column of the matrix to a thread block
    * @param out - output matrix to add column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - input matrix to add column from
    * @param col_in - assignment column of input matrix
    * @param sz_in - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void AddFromInput(float* __restrict__ out, const int col_out, const int sz_out,
                             float* __restrict__ in1, const int col_in1, const int sz_in1,
                             float* __restrict__ in2, const int col_in2, const int sz_in2,
                             const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] + in2[idx*sz_in2 + col_in2];
    }
}

/* SubtractFromInput - Subtracts a column of the matrix to a thread block
    * @param out - output matrix to subtract column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - input matrix to subtract column from
    * @param col_in - assignment column of input matrix
    * @param sz_in - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void SubtractFromInput(float* __restrict__ out, const int col_out, const int sz_out,
                                  float* __restrict__ in1, const int col_in1, const int sz_in1,
                                  float* __restrict__ in2, const int col_in2, const int sz_in2,
                                  const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] - in2[idx*sz_in2 + col_in2];
    }
}

