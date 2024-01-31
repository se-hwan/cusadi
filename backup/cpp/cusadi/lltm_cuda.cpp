#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <array>
#include <pybind11/pybind11.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void evaluateVectorizedCasADiFunctionCUDA(at::Tensor output,
                            at::Tensor work,
                            const std::vector<at::Tensor> input,
                            const at::Tensor operations,
                            const at::Tensor output_idx,
                            const at::Tensor input_idx,
                            const at::Tensor input_idx_lengths,
                            const at::Tensor const_instr,
                            const int num_instructions);

void vectorizedAddCUDA(at::Tensor output, int col_out,
                       at::Tensor in_1, int col_in_1,
                       at::Tensor in_2, int col_in_2);

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
 
void evaluateCusADiFunction(at::Tensor output,
                            at::Tensor work,
                            const std::vector<at::Tensor> input,
                            const at::Tensor operations,
                            const at::Tensor output_idx,
                            const at::Tensor input_idx,
                            const at::Tensor input_idx_lengths,
                            const at::Tensor const_instr,
                            const int num_instructions) {
  CHECK_INPUT(output);
  CHECK_INPUT(work);
  CHECK_INPUT(operations);
  CHECK_INPUT(input_idx);
  CHECK_INPUT(input_idx_lengths);
  CHECK_INPUT(const_instr);
  return evaluateVectorizedCasADiFunctionCUDA(output,
                                              work,
                                              input,
                                              operations,
                                              output_idx,
                                              input_idx,
                                              input_idx_lengths,
                                              const_instr,
                                              num_instructions);
}

void test_casadi(std::vector<torch::Tensor> test, std::vector<float> &indices, int num_instructions) {
  // int len_const_instr = std::end(const_instr) - std::begin(const_instr);
  std::cout << "len_const_instr: " << std::endl;
  std::cout << INPUT << std::endl;

  (test[0])[0][0] = 1.0;
  (test[0]).index({"...", 0}) = (test[0]).index({"...", 1})/(test[0]).index({"...", 2});
  for (int k = 0; k < num_instructions; k++) {
    std::cout << k << "\n";
  }
}

void test_add(at::Tensor output, int col_out,
                      at::Tensor in_1, int col_in_1,
                      at::Tensor in_2, int col_in_2) {
  vectorizedAddCUDA(output, col_out, in_1, col_in_1, in_2, col_in_2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("evaluateCusADiFunction", &evaluateCusADiFunction, "LLTM backward");
  m.def("test_casadi", &test_casadi, "LLTM backward");
  m.def("test_add", &test_add, "LLTM backward");
  // py::class_<CusADiFunction> (m, "CusADiFunction")
  //   .def(py::init<>())
  //   .def("evaluate", &CusADiFunction::evaluate);
}