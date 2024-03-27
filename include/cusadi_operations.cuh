#ifndef _CUSADI_OPERATIONS_H_
#define _CUSADI_OPERATIONS_H_

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
/*
ARCHITECTURE:
__device__ void evaluate_fn_kernel() {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        codegen calls to __device__ atomic operations here
        need to pass in threadID to the operation functions
    }
}

extern "C" __global__ evaluateFunctionC() {
    need to launch a single kernel here which does all the computations,
    which have been codegened

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    evaluateFunction<<<gridSize, blockSize>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

*/



/***************************** CUSADI OPERATIONS ***********************************/

/* assignFromInput - Assigns a column of the matrix to a thread block
    * @param out - output matrix to assign column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - input matrix to assign column from
    * @param col_in - assignment column of input matrix
    * @param sz_in - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void assignFromInput(float* __restrict__ out, const int col_out, const int sz_out,
                                float* __restrict__ in, const int col_in, const int sz_in,
                                const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        out[idx*sz_out + col_out] = in[idx*sz_in + col_in];
    }
}

/* assignFromConst - Assigns a column of the output matrix to a const
    * @param out - output matrix to assign column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - constant to assign to column
    * @param batch_size - number of rows in input and output
*/
__device__ void assignFromConst(float* __restrict__ out, const int col_out, const int sz_out,
                                const float in,
                                const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        out[idx*sz_out + col_out] = in;
    }
}

/* addInputs - Adds a column of the matrix to a thread block
    * @param out - output matrix to add column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - input matrix to add column from
    * @param col_in - assignment column of input matrix
    * @param sz_in - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void addInputs(float* __restrict__ out, const int col_out, const int sz_out,
                             float* __restrict__ in1, const int col_in1, const int sz_in1,
                             float* __restrict__ in2, const int col_in2, const int sz_in2,
                             const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] + in2[idx*sz_in2 + col_in2];
    }
}

/* subtractInputs - Subtracts a column of the matrix to a thread block
    * @param out - output matrix to subtract column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - input matrix to subtract column from
    * @param col_in - assignment column of input matrix
    * @param sz_in - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void subtractInputs(float* __restrict__ out, const int col_out, const int sz_out,
                                  float* __restrict__ in1, const int col_in1, const int sz_in1,
                                  float* __restrict__ in2, const int col_in2, const int sz_in2,
                                  const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < sz_out) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] - in2[idx*sz_in2 + col_in2];
    }
}

/* negateInput - Subtracts a column of the matrix to a thread block
    * @param out - output matrix to subtract column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - input matrix to subtract column from
    * @param col_in - assignment column of input matrix
    * @param sz_in - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void negateInput(float* __restrict__ out, const int col_out, const int sz_out,
                             const float* __restrict__ in, const int col_in, const int sz_in,
                             const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = -in[idx*sz_in + col_in];
    }
}

/* multiplyInputs - Multiplies a column of the matrix to a thread block
    * @param out - output matrix to multiply column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in1 - input matrix to multiply column from
    * @param col_in1 - assignment column of input matrix
    * @param sz_in1 - column size of input matrix
    * @param in2 - input matrix to multiply column from
    * @param col_in2 - assignment column of input matrix
    * @param sz_in2 - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void multiplyInputs(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in1, const int col_in1, const int sz_in1,
                           const float* __restrict__ in2, const int col_in2, const int sz_in2,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] * in2[idx*sz_in2 + col_in2];
    }
}

/* divideInputs - Divides a column of the matrix to a thread block
    * @param out - output matrix to divide column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in1 - input matrix to divide column from
    * @param col_in1 - assignment column of input matrix
    * @param sz_in1 - column size of input matrix
    * @param in2 - input matrix to divide column from
    * @param col_in2 - assignment column of input matrix
    * @param sz_in2 - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void divideInputs(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in1, const int col_in1, const int sz_in1,
                           const float* __restrict__ in2, const int col_in2, const int sz_in2,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] / in2[idx*sz_in2 + col_in2];
    }
}

/* expInput - Exponentiates a column of the matrix to a thread block
    * @param out - output matrix to exponentiate column to
    * @param col_out - assignment column of output matrix
    * @param sz_out - column size of output matrix
    * @param in - input matrix to exponentiate column from
    * @param col_in - assignment column of input matrix
    * @param sz_in - column size of input matrix
    * @param batch_size - number of rows in input and output
*/
__device__ void expInput(float* __restrict__ out, const int col_out, const int sz_out,
                                  const float* __restrict__ in, const int col_in, const int sz_in,
                                  const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = expf(in[idx*sz_in + col_in]);
    }
}

__device__ void logInput(float* __restrict__ out, const int col_out, const int sz_out,
                         const float* __restrict__ in, const int col_in, const int sz_in,
                         const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = logf(in[idx*sz_in + col_in]);
    }
}

__device__ void sqrtInput(float* __restrict__ out, const int col_out, const int sz_out,
                          const float* __restrict__ in, const int col_in, const int sz_in,
                          const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = sqrtf(in[idx*sz_in + col_in]);
    }
}

__device__ void sqInput(float* __restrict__ out, const int col_out, const int sz_out,
                        const float* __restrict__ in, const int col_in, const int sz_in,
                        const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in[idx*sz_in + col_in] * in[idx*sz_in + col_in];
    }
}

__device__ void sinInput(float* __restrict__ out, const int col_out, const int sz_out,
                         const float* __restrict__ in, const int col_in, const int sz_in,
                         const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = sinf(in[idx*sz_in + col_in]);
    }
}

__device__ void cosInput(float* __restrict__ out, const int col_out, const int sz_out,
                         const float* __restrict__ in, const int col_in, const int sz_in,
                         const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = cosf(in[idx*sz_in + col_in]);
    }
}

__device__ void tanInput(float* __restrict__ out, const int col_out, const int sz_out,
                         const float* __restrict__ in, const int col_in, const int sz_in,
                         const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = tanf(in[idx*sz_in + col_in]);
    }
}


#endif // _CUSADI_OPERATIONS_H_