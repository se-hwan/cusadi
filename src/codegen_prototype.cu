// AUTOMATICALLY GENERATED CODE FOR CUSADI

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "include/cusadi_operations.cuh"

__constant__ const int n_instr = 12067;
__constant__ const int n_in = 2;
__constant__ const int n_out = 1;
__constant__ const int nnz_in[] = {192,52};
__constant__ const int nnz_out[] = {1601};
__constant__ const int n_w = 226;

__device__ float* input_pointers[n_in];
__device__ float* output_pointers[n_out];

__global__ void evaluate_kernel (
        // const float *inputs[],
        // float *work,
        // float *outputs[],
        const float** inputs,
        float *work,
        float** outputs,
        const int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        work[idx*n_w + 0] = 5.0f;
        outputs[0][0] = 3.0f;
        work[idx * n_w + 0] = 1.0;
        outputs[0][idx * nnz_out[0] + 0] = work[idx * n_w + 0];
        outputs[0][idx * nnz_out[0] + 1] = work[idx * n_w + 0];
        work[idx * n_w + 1] = 0.05;
        work[idx * n_w + 2] = inputs[1][idx * nnz_in[1] + 13];
        work[idx * n_w + 3] = inputs[0][idx * nnz_in[0] + 2];
        work[idx * n_w + 4] = sin(work[idx * n_w + 3]);
        work[idx * n_w + 5] = inputs[0][idx * nnz_in[0] + 0];
        work[idx * n_w + 6] = sin(work[idx * n_w + 5]);
        work[idx * n_w + 7] = work[idx * n_w + 4] * work[idx * n_w + 6];
    }
}

__global__ void test_inputs_kernel (
        float* __restrict__ inputs[],
        const int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Device address of first input: %p\n", inputs[0]);
    printf("Device address of first input: %f\n", inputs[0][0]);
    // inputs[0][0] = 5.0;
    // printf("Inputs check: %p\n", inputs);
    // printf("pointer check: %p\n", input_pointers);
    if (idx < batch_size) {
        inputs[0][idx * nnz_in[0] + 0] = 5.0;
    }
}




/** EXTERNAL C FUNCTIONS **/

extern "C" {
    void evaluate(const float** inputs,
                  float *work,
                  float** outputs,
                  const int batch_size) {
        int blockSize = 512;
        int gridSize = (batch_size + blockSize - 1) / blockSize;
        evaluate_kernel<<<gridSize, blockSize>>>(inputs,
                                                 work,
                                                 outputs,
                                                 batch_size);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void test_inputs(float* inputs[],
                     const int batch_size) {
        int blockSize = 512;
        int gridSize = (batch_size + blockSize - 1) / blockSize;
        test_inputs_kernel<<<gridSize, blockSize>>>(inputs,
                                                    batch_size);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void checkPointer(const void* ptr) {
        cudaPointerAttributes attributes;
        cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);

        if (error == cudaSuccess) {
            if (attributes.type == cudaMemoryTypeDevice) {
                std::cout << "Pointer is in device memory.\n";
            } else if (attributes.type == cudaMemoryTypeHost) {
                std::cout << "Pointer is in host memory.\n";
            } else {
                std::cout << "Pointer type is unknown.\n";
            }
        } else {
            std::cout << "cudaPointerGetAttributes failed: " << cudaGetErrorString(error) << std::endl;
        }
    }
}

