#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/transform.h>
// #include <cmath>

void evaluate_casadi_function(int num_instructions, int batch_size,
                              float *a_device, float *c_device)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    for (int i = 0; i < num_instructions; i++) {
        cublasSaxpy(handle, batch_size, &alpha, a_device, 1, c_device, 1);
    }
        

    // for (int i = 0; i < num_instructions; i++) {
    //     int op = operations[i];

    //     switch (op) {
    //         case 1: // ADDITION
    //             cublasSaxpy(handle, batch_size, 1, work[:, ], 1, work[], 1);
    //             break;
    //         case 2: // SUBTRACTION
    //             cublasSaxpy(handle, batch_size, -1, , 1, , 1);
    //             break;
    //     }
    // }

    cublasDestroy(handle);
}


