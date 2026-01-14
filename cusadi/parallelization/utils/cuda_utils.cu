#include <iostream>
#include <cuda_runtime.h>


#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(call)                                                                  \
{                                                                                               \
    cudaError_t err = call;                                                                     \
    if (err != cudaSuccess) {                                                                   \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)                                  \
                    << " at " << __FILE__ << ":" << __LINE__ << std::endl;                      \
        exit(EXIT_FAILURE);                                                                     \
    }                                                                                           \
}
#endif // CHECK_CUDA_ERROR


#ifndef CHECK_CUSPARSE_ERROR
#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
        } \
    } while(0);
#endif // CHECK_CUSPARSE_ERROR


// ! Print data on GPU memory
static __global__ void printGPUData(double* gpu_matrix, int env, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x; 
    if (row == env && col < cols) {
        int idx = env*cols + col;
        double value = gpu_matrix[idx];
        printf("Value at row %d, col %d: %f\n", row, col, value);
    }
}