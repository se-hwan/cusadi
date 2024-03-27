#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// TODO: look at stackoverflow page, adjust for row_major and column major matrices.
// TODO: narrow scope, assume all matrices will be 2D at most (when is this not the case?)
    // output is always flattened to nnz x 1
    // work vectors will always be flattened to [[input_1_dim x 1], [input_2_dim x 1], ...]
// TODO: can bring casadi parallelization for MPC speciifcally? 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/************************* CUDA KERNEL FUNCTIONS *************************/

__global__ void assign_const_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                                    const float in,
                                    const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in;
    }
}

__global__ void assign_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                             const float* __restrict__ in, const int col_in, const int sz_in,
                             const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in[idx*sz_in + col_in];
    }
}

__global__ void negate_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                             const float* __restrict__ in, const int col_in, const int sz_in,
                             const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = -in[idx*sz_in + col_in];
    }
}

__global__ void add_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in1, const int col_in1, const int sz_in1,
                           const float* __restrict__ in2, const int col_in2, const int sz_in2,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] + in2[idx*sz_in2 + col_in2];
    }
}

__global__ void sub_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in1, const int col_in1, const int sz_in1,
                           const float* __restrict__ in2, const int col_in2, const int sz_in2,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] - in2[idx*sz_in2 + col_in2];
    }
}

__global__ void mul_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in1, const int col_in1, const int sz_in1,
                           const float* __restrict__ in2, const int col_in2, const int sz_in2,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] * in2[idx*sz_in2 + col_in2];
    }
}

__global__ void div_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in1, const int col_in1, const int sz_in1,
                           const float* __restrict__ in2, const int col_in2, const int sz_in2,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in1[idx*sz_in1 + col_in1] / in2[idx*sz_in2 + col_in2];
    }
}

__global__ void exp_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in, const int col_in, const int sz_in,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = expf(in[idx*sz_in + col_in]);
    }
}

__global__ void log_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in, const int col_in, const int sz_in,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = logf(in[idx*sz_in + col_in]);
    }
}

__global__ void sqrt_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                            const float* __restrict__ in, const int col_in, const int sz_in,
                            const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = sqrtf(in[idx*sz_in + col_in]);
    }
}

__global__ void sq_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                          const float* __restrict__ in, const int col_in, const int sz_in,
                          const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = in[idx*sz_in + col_in] * in[idx*sz_in + col_in];
    }
}

__global__ void sin_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in, const int col_in, const int sz_in,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = sinf(in[idx*sz_in + col_in]);
    }
}

__global__ void cos_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in, const int col_in, const int sz_in,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = cosf(in[idx*sz_in + col_in]);
    }
}

__global__ void tan_kernel(float* __restrict__ out, const int col_out, const int sz_out,
                           const float* __restrict__ in, const int col_in, const int sz_in,
                           const int batch_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batch_size) {
        out[idx*sz_out + col_out] = tanf(in[idx*sz_in + col_in]);
    }
}

/************************* EXTERNAL WRAPPER FUNCTIONS *************************/
// TODO: remove error checking once functions have been tested
// TODO: unit tests for each function
extern "C" {
    void cu_test() {
        printf("Hello from CUDA!\n");
    }

void cu_assign_const(float *out, const int col_out, const int sz_out,
                                const float in,
                                const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    assign_const_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out, in, batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_assign(float *out, const int col_out, const int sz_out,
                          const float *in, const int col_in, const int sz_in,
                          const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;

    assign_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                           in, col_in, sz_in, 
                                           batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_neg(float *out, const int col_out, const int sz_out,
                          const float *in, const int col_in, const int sz_in,
                          const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;

    negate_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                           in, col_in, sz_in, 
                                           batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_add(float *out, const int col_out, const int sz_out,
                       const float *in1, const int col_in1, const int sz_in1,
                       const float *in2, const int col_in2, const int sz_in2,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    add_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in1, col_in1, sz_in1,
                                        in2, col_in2, sz_in2,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_sub(float *out, const int col_out, const int sz_out,
                       const float *in1, const int col_in1, const int sz_in1,
                       const float *in2, const int col_in2, const int sz_in2,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    sub_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in1, col_in1, sz_in1,
                                        in2, col_in2, sz_in2,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_mul(float *out, const int col_out, const int sz_out,
                       const float *in1, const int col_in1, const int sz_in1,
                       const float *in2, const int col_in2, const int sz_in2,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    mul_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in1, col_in1, sz_in1,
                                        in2, col_in2, sz_in2,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_div(float *out, const int col_out, const int sz_out,
                       const float *in1, const int col_in1, const int sz_in1,
                       const float *in2, const int col_in2, const int sz_in2,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    div_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in1, col_in1, sz_in1,
                                        in2, col_in2, sz_in2,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_exp(float *out, const int col_out, const int sz_out,
                       const float *in, const int col_in, const int sz_in,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    exp_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in, col_in, sz_in,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_log(float *out, const int col_out, const int sz_out,
                       const float *in, const int col_in, const int sz_in,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    log_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in, col_in, sz_in,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_sqrt(float *out, const int col_out, const int sz_out,
                        const float *in, const int col_in, const int sz_in,
                        const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    sqrt_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                         in, col_in, sz_in,
                                         batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_sq(float *out, const int col_out, const int sz_out,
                      const float *in, const int col_in, const int sz_in,
                      const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    sq_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                       in, col_in, sz_in,
                                       batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_sin(float *out, const int col_out, const int sz_out,
                       const float *in, const int col_in, const int sz_in,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    sin_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in, col_in, sz_in,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_cos(float *out, const int col_out, const int sz_out,
                       const float *in, const int col_in, const int sz_in,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    cos_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in, col_in, sz_in,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void cu_tan(float *out, const int col_out, const int sz_out,
                       const float *in, const int col_in, const int sz_in,
                       const int batch_size) {
    int blockSize = 512; // This can be tuned for performance
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    tan_kernel<<<gridSize, blockSize>>>(out, col_out, sz_out,
                                        in, col_in, sz_in,
                                        batch_size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

}