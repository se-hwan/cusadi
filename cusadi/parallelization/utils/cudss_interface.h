#ifndef CUDSSINTERFACE_H
#define CUDSSINTERFACE_H

#include <stdio.h>
#include <iostream>
#include <cudss.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template<typename T> struct CudssTypeTraits;

template<> struct CudssTypeTraits<float> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_32F;
};

template<> struct CudssTypeTraits<double> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_64F;
};

template <typename T>
class cudssInterface {
    public:
        cudssHandle_t handle_solver;
        cudssConfig_t solverConfig;
        cudssData_t solverData;
        cudssStatus_t status = CUDSS_STATUS_SUCCESS;
        cudssMatrix_t x, b;
        cudssMatrix_t A;
        cudssMatrixType_t mtype     = CUDSS_MTYPE_SYMMETRIC;
        cudssMatrixViewType_t mview = CUDSS_MVIEW_LOWER;
        cudssIndexBase_t base       = CUDSS_BASE_ZERO;

        torch::Tensor Ap_tensor, Ai_tensor, Ax_tensor;
        torch::Tensor x_tensor, b_tensor;

        int batch_size = -1;
        int64_t n_rows = -1;
        int64_t n_cols = -1;
        int64_t nnz = -1;
        int*    Ap_csr_uniform; // Batched row pointers (int32)
        int*    Ai_csr_uniform; // Batched column indices (int32)
        T*    Ax_csr_uniform; // Batched matrix values (float64)
        T*    x_data_uniform; // Batched solution vector (float64)
        T*    b_data_uniform; // Batched RHS vector (float64)

        // IsaacGym variables
        torch::Tensor state;
        torch::Tensor params;

        // Host variables
        int*    A_num_rows; // Batched rows in A (int32)
        int*    A_num_cols; // Batched cols in A (int32)
        int*    A_nnz;      // Batched nnz in A (int32)
        int*    b_num_cols; // Batched cols in b (int32)

        // Device variables:
        int64_t*    Ap_csr; // Batched row pointers (int32)
        int64_t*    Ai_csr; // Batched column indices (int32)
        int64_t*    Ax_csr; // Batched matrix values (float64)
        int64_t*    x_data; // Batched solution vector (float64)
        int64_t*    b_data; // Batched RHS vector (float64)
        torch::Tensor x_0;  // Initial guess
        torch::Tensor x_k, y_k, z_k; // Iteration variables for ADMM solve

    cudssInterface(int batch_size);

    void loadPointers(
        const torch::Tensor& A_num_rows_tensor, // host
        const torch::Tensor& A_num_cols_tensor, // host
        const torch::Tensor& A_nnz_tensor,      // host
        const torch::Tensor& b_num_cols_tensor, // host
        const torch::Tensor& Ap_tensor,         // device
        const torch::Tensor& Ai_tensor,         // device
        const torch::Tensor& Ax_tensor,         // device
        const torch::Tensor& x_tensor,          // device
        const torch::Tensor& b_tensor);         // device
    void setupMatrices(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        const torch::Tensor& Ap_tensor,         // device
        const torch::Tensor& Ai_tensor,         // device
        const torch::Tensor& Ax_tensor,         // device
        const torch::Tensor& x_tensor,          // device
        const torch::Tensor& b_tensor);         // device
    void createMatricesUniform();
    void createMatrices();
    void factorizeSymbolic();
    void factorizeNumeric();
    void solveLinearSystem();
    void printConstraintMatrixData(const int env, const int cols);
    void printConstraintVectorData(const int env, const int cols);

    ~cudssInterface() {
        if (matrices_are_allocated) {
            printf("Cleaning up matrices in memory...\n");
            cudssMatrixDestroy(A);
            cudssMatrixDestroy(x);
            cudssMatrixDestroy(b);
            printf("Cleaned up matrices.\n");
        }
        printf("Cleaning up objects in memory...\n");
        cudssConfigDestroy(solverConfig);
        cudssDataDestroy(handle_solver, solverData);
        cudssDestroy(handle_solver);
        printf("Cleanup complete, all objects safely freed.\n");
    }


    private:
        cudaDataType_t data_type = CUDA_R_64F;
        bool matrices_are_allocated = false;
        // dim3 grid_dim(256, 1, 1);
        // dim3 block_dim;
        // torch::Tensor x_k, y_k, z_k;
        // torch::Tensor x_next, y_next, z_next;

};

#endif