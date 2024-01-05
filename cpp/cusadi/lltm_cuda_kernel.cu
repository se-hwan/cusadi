#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void assign_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = in[n][in_idx];
  }
}

__global__ void const_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                             const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = in[in_idx];
  }
}

__global__ void add_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int col_out,
                           const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in_1, int col_in_1,
                           const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in_2, int col_in_2) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][col_out] = in_1[n][col_in_1] + in_2[n][col_in_2];
  }
}

__global__ void sub_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int col_out,
                           const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in_1, int col_in_1,
                           const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in_2, int col_in_2) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][col_out] = in_1[n][col_in_1] - in_2[n][col_in_2];
  }
}

__global__ void neg_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = -in[n][in_idx];
  }
}

__global__ void mul_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int col_out,
                           const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in_1, int col_in_1,
                           const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in_2, int col_in_2) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][col_out] = in_1[n][col_in_1] * in_2[n][col_in_2];
  }
}

__global__ void div_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int col_out,
                           const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in_1, int col_in_1,
                           const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in_2, int col_in_2) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][col_out] = in_1[n][col_in_1] / in_2[n][col_in_2];
  }
}

__global__ void sin_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = sinf(in[n][in_idx]);
  }
}

__global__ void cos_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = cosf(in[n][in_idx]);
  }
}

__global__ void tan_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = tanf(in[n][in_idx]);
  }
}

__global__ void exp_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = expf(in[n][in_idx]);
  }
}

__global__ void log_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = logf(in[n][in_idx]);
  }
}

__global__ void sqrt_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = sqrtf(in[n][in_idx]);
  }
}

__global__ void sq_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out, int out_idx,
                              const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> in, int in_idx) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size(0)){
    out[n][out_idx] = in[n][in_idx] * in[n][in_idx];
  }
}

__global__ void evaluate() {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < out.size()) {
    
  }
}




/* *************************************************************************** */

enum CasADiOperations {
    INPUT = 45,
    OUTPUT = 46,
    CONST = 44,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    NEG = 5,
    EXP = 6,
    LOG = 7,
    POW = 8,
    SQRT = 10,
    SQ = 11,
    SIN = 13,
    COS = 14,
    TAN = 15
};

void evaluateVectorizedCasADiFunctionCUDA(at::Tensor output,
                            at::Tensor work,
                            const std::vector<at::Tensor> input,
                            const at::Tensor operations,
                            const at::Tensor output_idx,
                            const at::Tensor input_idx,
                            const at::Tensor input_idx_lengths,
                            const at::Tensor const_instr,
                            const int num_instructions) {
  
  const int batch_size = output.size(0);
  const int state_size = output.size(1);
  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  int op = -1;
  int o_idx = -1;
  int i_idx = -1;
  int i_instr = 0;

  auto output_a = output.packed_accessor32<float,2>();
  auto work_a = work.packed_accessor32<float,2>();
  auto input_a = (input[0]).packed_accessor32<float,2>();
  printf("TESTJKIHDSALJKASHDKJLASHDKLASJDHAKSLJDHSAJK;D\n");
  for (int k = 0; k < num_instructions; k++) {
    op = operations[k].item<int>();
    o_idx = output_idx[k].item<int>();
    i_idx = input_idx[i_instr].item<int>();
    printf("Operation: %d\n" , op);
    switch (op) {
      case CONST:
        const_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                           const_instr.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), i_idx);
        break;
      case INPUT:
        // assign_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
        //                                    (input[i_idx]).packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx + 1);
        break;
      case OUTPUT:
        assign_kernel<<<blocks, threads>>>(output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                           work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      case ADD:
        add_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx + 1);
        break;
      case SUB:
        sub_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx + 1);
        break;
      case NEG:
        neg_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      case MUL:
        mul_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx + 1);
        break;
      case DIV:
        div_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx + 1);
        break;
      case SIN:
        sin_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      case COS:
        cos_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      case TAN: 
        sin_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      case SQ:
        sq_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                      work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      case SQRT:
        sqrt_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                         work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      case EXP:
        exp_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      case LOG:
        log_kernel<<<blocks, threads>>>(work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), o_idx,
                                        work.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), i_idx);
        break;
      default:
        std::cout << "Unexpected operation: " << op << "\n";
        std::cout << "Current instruction: " << k << "\n";
        break;
    }
    i_instr += input_idx_lengths[k + 1].item<int>();
  }
}

void vectorizedAddCUDA(at::Tensor output, int col_out,
                       at::Tensor in_1, int col_in_1,
                       at::Tensor in_2, int col_in_2) {
  const int threads = 1024;
  const int blocks = 512;
  add_kernel<<<blocks, threads>>>(output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), col_out,
                                  in_1.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), col_in_1,
                                  in_2.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), col_in_2);
}