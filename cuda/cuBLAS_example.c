#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Simple cuBLAS function: vector addition
// Assumes a_device, c_device are already on GPU, c_device initialized with b_device
void add_vectors_cublas(cublasHandle_t handle, float *a_device, float *c_device, int n) {
    const float alpha = 1.0f;

    for (int i = 0; i < n; ++i) {
        printf("%f\n", a_device[i]);
    }
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