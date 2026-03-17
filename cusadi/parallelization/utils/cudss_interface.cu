#include "cudss_interface.h"
#include "cuda_utils.cu"
#include "string.h"

// Documentation: https://docs.nvidia.com/cuda/cudss/index.html

template <typename T>
cudssInterface<T>::cudssInterface(int batch_size) {
    // printf("Creating cudssInterface object...\n");
    this->batch_size = batch_size;
    cudssCreate(&handle_solver);
    cudssConfigCreate(&solverConfig);
    cudssDataCreate(handle_solver, &solverData);
    // printf("Created cudssInterface object for %d batches.\n", batch_size);
}

template <typename T>
void cudssInterface<T>::loadPointers(const torch::Tensor& A_num_rows_tensor, // host
                  const torch::Tensor& A_num_cols_tensor, // host
                  const torch::Tensor& A_nnz_tensor,      // host
                  const torch::Tensor& b_num_cols_tensor, // host
                  const torch::Tensor& Ap_tensor,         // device
                  const torch::Tensor& Ai_tensor,         // device
                  const torch::Tensor& Ax_tensor,         // device
                  const torch::Tensor& x_tensor,          // device
                  const torch::Tensor& b_tensor)          // device
{
    data_type = CudssTypeTraits<T>::cuda_type;

    // printf("Loading pointers...\n");
    A_num_rows = A_num_rows_tensor.data_ptr<int>();
    A_num_cols = A_num_cols_tensor.data_ptr<int>();
    A_nnz = A_nnz_tensor.data_ptr<int>();
    b_num_cols = b_num_cols_tensor.data_ptr<int>();
    Ap_csr = Ap_tensor.data_ptr<int64_t>();
    Ai_csr = Ai_tensor.data_ptr<int64_t>();
    Ax_csr = Ax_tensor.data_ptr<int64_t>();
    x_data = x_tensor.data_ptr<int64_t>();
    b_data = b_tensor.data_ptr<int64_t>();
    // printf("Loaded pointers\n");
}

template <typename T>
void cudssInterface<T>::setupMatrices(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        const torch::Tensor& Ap_tensor,         // device
        const torch::Tensor& Ai_tensor,         // device
        const torch::Tensor& Ax_tensor,         // device
        const torch::Tensor& x_tensor,          // device
        const torch::Tensor& b_tensor)          // device
{
    // printf("Loading pointers...\n");
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    this->nnz = nnz;
    // Below need to be pointers on DEVICE (GPU)
    this->Ap_tensor = Ap_tensor;
    this->Ai_tensor = Ai_tensor;
    this->Ax_tensor = Ax_tensor;
    this->x_tensor = x_tensor;
    this->b_tensor = b_tensor;
    Ap_csr_uniform = Ap_tensor.data_ptr<int>();
    Ai_csr_uniform = Ai_tensor.data_ptr<int>();
    Ax_csr_uniform = Ax_tensor.data_ptr<T>();
    x_data_uniform = x_tensor.data_ptr<T>();
    b_data_uniform = b_tensor.data_ptr<T>();
    // printf("Loaded pointers\n");
}

template <typename T>
void cudssInterface<T>::createMatricesUniform() {
    CUDSS_CALL_AND_CHECK( // Create uniform batch of sparse A matrices
        cudssMatrixCreateCsr(&A, n_rows, n_cols, nnz,
                             Ap_tensor.data_ptr<int>(), NULL,
                             Ai_tensor.data_ptr<int>(), Ax_tensor.data_ptr<double>(),
                             CUDA_R_32I, data_type, mtype, mview, base),
        status, "cudssMatrixCreateUniformCsr");
    CUDSS_CALL_AND_CHECK( // Create uniform batch of dense x matrices
        cudssMatrixCreateDn(&x, n_rows, 1, n_rows, x_tensor.data_ptr<double>(),
                            data_type, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateUniformDense");
    CUDSS_CALL_AND_CHECK( // Create uniform batch of dense b matrices
        cudssMatrixCreateDn(&b, n_rows, 1, n_rows, b_tensor.data_ptr<double>(),
                            data_type, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateUniformDense");

    matrices_are_allocated = true;
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(solverConfig, CUDSS_CONFIG_UBATCH_SIZE, &batch_size, sizeof(batch_size)),
        status, "meow");
    // printf("Created matrices\n");
}

template <typename T>
void cudssInterface<T>::createMatrices() {
    // printf("Creating matrices...\n");
    // Sparse A matrices should 
    CUDSS_CALL_AND_CHECK( // Create batches of sparse A matrices
        cudssMatrixCreateBatchCsr(&A, batch_size, A_num_rows, A_num_cols, A_nnz,
                                  (void**)Ap_csr, NULL, (void**)Ai_csr, (void**)Ax_csr,
                                  CUDA_R_32I, data_type, mtype, mview, base),
        status, "cudssMatrixCreateBatchCsr");
    CUDSS_CALL_AND_CHECK( // Create batches of dense x matrices
        cudssMatrixCreateBatchDn(&x, batch_size, A_num_rows, b_num_cols, A_num_rows,
                                (void**)x_data, CUDA_R_32I, data_type, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateBatchDense");
    CUDSS_CALL_AND_CHECK( // Create batches of dense b matrices
        cudssMatrixCreateBatchDn(&b, batch_size, A_num_rows, b_num_cols, A_num_rows,
                                (void**)b_data, CUDA_R_32I, data_type, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateBatchDense");
    matrices_are_allocated = true;
    // printf("Created matrices\n");
}

template <typename T>
void cudssInterface<T>::factorizeSymbolic() {
    // printf("Symbolic factorization...\n");
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle_solver, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b),
        status, "cudssFactorizeSymbolic");
}

template <typename T>
void cudssInterface<T>::factorizeNumeric() {
    // printf("Numeric factorization...\n");
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle_solver, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b),
        status, "cudssFactorizeNumeric");
}

template <typename T>
void cudssInterface<T>::solveLinearSystem() {
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle_solver, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b),
        status, "ref");
}

template <typename T>
void cudssInterface<T>::printConstraintMatrixData(const int env, const int cols) {
    // printf("Printing constraint matrix data...\n");
    printGPUData<<<batch_size, 1024>>>((double*)Ax_csr_uniform, env, cols);
}

template <typename T>
void cudssInterface<T>::printConstraintVectorData(const int env, const int cols) {
    // printf("Printing constraint vector data...\n");
    printGPUData<<<batch_size, 1024>>>((double*)b_data_uniform, env, cols);
}








template class cudssInterface<float>;
template class cudssInterface<double>;