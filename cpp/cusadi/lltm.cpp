#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <array>
#include <pybind11/pybind11.h>

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
  int op = -1;
  int o_idx = -1;
  int i_idx = -1;
  int i_instr = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < num_instructions; k++) {
    op = operations[k].item<int>();
    o_idx = output_idx[k].item<int>();
    i_idx = input_idx[i_instr].item<int>();
    switch (op) {
      case CONST:
        work.index({"...", o_idx}) = const_instr[k].item<float>();
        break;
      case INPUT:
        work.index({"...", o_idx}) = (input[i_idx]).index({"...", i_idx + 1});
        break;
      case OUTPUT:{
        output.index({"...", o_idx}) = work.index({"...", i_idx});
        break;
      case ADD:
        work.index({"...", o_idx}) = work.index({"...", i_idx}) + work.index({"...", i_idx + 1});
        break;
      case SUB:
        work.index({"...", o_idx}) = work.index({"...", i_idx}) - work.index({"...", i_idx + 1});
        break;
      case NEG:
        work.index({"...", o_idx}) = -work.index({"...", i_idx});
        break;
      case MUL:
        work.index({"...", o_idx}) = work.index({"...", i_idx}) * work.index({"...", i_idx + 1});
        break;
      case DIV:
        work.index({"...", o_idx}) = work.index({"...", i_idx}) / work.index({"...", i_idx + 1});
        break;
      case SIN:
        work.index({"...", o_idx}) = torch::sin(work.index({"...", i_idx}));
        break;
      case COS:
        work.index({"...", o_idx}) = torch::cos(work.index({"...", i_idx}));
        break;
      case TAN: 
        work.index({"...", o_idx}) = torch::tan(work.index({"...", i_idx}));
        break;
      case SQ:
        work.index({"...", o_idx}) = work.index({"...", i_idx}) * work.index({"...", i_idx});
        break;
      case SQRT:
        work.index({"...", o_idx}) = torch::sqrt(work.index({"...", i_idx}));
        break;
      case EXP:
        work.index({"...", o_idx}) = torch::exp(work.index({"...", i_idx}));
        break;
      case LOG:
        work.index({"...", o_idx}) = torch::log(work.index({"...", i_idx}));
        break;
      default:
        std::cout << "Unexpected operation: " << op << "\n";
        std::cout << "Current instruction: " << k << "\n";
        exit(1);
        break;
    }
    i_instr += input_idx_lengths[k + 1].item<int>();
  }
  // auto stop = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  // std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
  }
}

void vectorizedAddCPP(at::Tensor output, int col_out,
                      at::Tensor in_1, int col_in_1,
                      at::Tensor in_2, int col_in_2) {
  output.index({"...", col_out}) = in_1.index({"...", col_in_1}) + in_2.index({"...", col_in_2});
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("evaluateCusADiFunction", &evaluateCusADiFunction, "LLTM backward");
  m.def("test_casadi", &test_casadi, "LLTM backward");
  m.def("vectorizedAddCPP", &vectorizedAddCPP, "LLTM backward");
  // py::class_<CusADiFunction> (m, "CusADiFunction")
  //   .def(py::init<>())
  //   .def("evaluate", &CusADiFunction::evaluate);
}