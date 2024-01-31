#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>


// Trigonometric functions
__global__ void sin_kernel(float *f_out, const float *f_in, size_t size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        f_out[idx] = sinf(f_in[idx]);
    }
}

// Declaration of the function that invokes the CUDA kernel (defined in the .cu file)
void sin_test(float *f_out, const float *f_in, size_t size) {
    int blockSize = 256; // This can be tuned for performance
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    sin_kernel<<<gridSize, blockSize>>>(f_out, f_in, size);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    //     exit(1);
    // }    
}


// Simple cuBLAS function: vector addition
// Assumes a_device, c_device are already on GPU, c_device initialized with b_device
void add_vectors_cublas(cublasHandle_t handle, float *a_device, float *c_device, int n) {
    const float alpha = 1.0f;

    // Perform vector addition: c_device = alpha * a_device + c_device
    cublasSaxpy(handle, n, &alpha, a_device, 1, c_device, 1);
}

// Simple cuBLAS function: vector addition
// Assumes a_device, c_device are already on GPU, c_device initialized with b_device
void add_vectors_loop(cublasHandle_t handle, float *a_device, float *c_device, int n_loop, int n) {
    const float alpha = 1.0f;

    // Perform vector addition: c_device = alpha * a_device + c_device
    for (int i = 0; i < n_loop; i++)
        cublasSaxpy(handle, n, &alpha, a_device, 1, c_device, 1);
}