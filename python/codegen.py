from casadi import *
from CusadiOperations import OP_CUDA_DICT

f = casadi.Function.load("../inertial_quantities.casadi")
codegen_file = open(r"../src/codegen_prototype.cu", "w+")
codegen_strings = {}

# * Parse CasADi function
n_instr = f.n_instructions()
n_in = f.n_in()
n_out = f.n_out()
nnz_in = [f.nnz_in(i) for i in range(n_in)]
nnz_out = [f.nnz_out(i) for i in range(n_out)]
n_w = f.sz_w()

INSTR_LIMIT = n_instr

input_idx = []
input_idx_lengths = [0]
output_idx = []
output_idx_lengths = [0]
for i in range(INSTR_LIMIT):
    input_idx.extend(f.instruction_input(i))
    input_idx_lengths.append(len(f.instruction_input(i)))
    output_idx.extend(f.instruction_output(i))
    output_idx_lengths.append(len(f.instruction_output(i)))
operations = [f.instruction_id(i) for i in range(INSTR_LIMIT)]
const_instr = [f.instruction_constant(i) for i in range(INSTR_LIMIT)]

# * Codegen for const declarations and indices
codegen_strings['header'] = "// AUTOMATICALLY GENERATED CODE FOR CUSADI\n"
codegen_strings['includes'] = \
'''
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "include/cusadi_operations.cuh"
'''
codegen_strings["nnz_in"] = f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
codegen_strings["nnz_out"] = f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
codegen_strings["n_w"] = f"\n__constant__ int n_w = {n_w};\n"

# * Codegen for CUDA kernel
str_kernel = \
'''

__global__ void evaluate_kernel (
        const float *inputs[],
        float *work,
        float *outputs[],
        const int batch_size) {
'''

str_kernel += "\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;"
str_kernel += "\n    if (idx < batch_size) {"
o_instr = 0
i_instr = 0
for k in range(INSTR_LIMIT):
    op = operations[k]
    o_idx = output_idx[o_instr]
    i_idx = input_idx[i_instr]
    if op == OP_CONST:
        str_kernel += OP_CUDA_DICT[op] % (o_idx, const_instr[k])
    elif op == OP_INPUT:
        str_kernel += OP_CUDA_DICT[op] % (o_idx, i_idx, i_idx, input_idx[i_instr + 1])
    elif op == OP_OUTPUT:
        str_kernel += OP_CUDA_DICT[op] % (o_idx, o_idx, output_idx[o_instr + 1], i_idx)
    elif op == OP_SQ:
        str_kernel += OP_CUDA_DICT[op] % (o_idx, i_idx, i_idx)
    elif OP_CUDA_DICT[op].count("%d") == 3:
        str_kernel += OP_CUDA_DICT[op] % (o_idx, i_idx, input_idx[i_instr + 1])
    elif OP_CUDA_DICT[op].count("%d") == 2:
        str_kernel += OP_CUDA_DICT[op] % (o_idx, i_idx)
    else:
        raise Exception('Unknown CasADi operation: ' + str(op))
    o_instr += output_idx_lengths[k + 1]
    i_instr += input_idx_lengths[k + 1]

str_kernel += "\n    }"     # End of if statement
str_kernel += "\n}"         # End of kernel
codegen_strings['cuda_kernel'] = str_kernel

# * Codegen for C interface
codegen_strings['c_interface_header'] = """\nextern "C" {\n"""
codegen_strings['c_evaluation'] = \
'''
    void evaluate(const float *inputs[],
                  float *work,
                  float *outputs[],
                  const int batch_size) {
        int blockSize = 512;
        int gridSize = (batch_size + blockSize - 1) / blockSize;
        evaluate_kernel<<<gridSize, blockSize>>>(inputs,
                                                 work,
                                                 outputs,
                                                 batch_size);
    }

'''
# codegen_strings['c_getters'] = \
# '''
#     int get_n_in() {
#         return %d;
#     }

#     int get_n_out() {
#         return %d;
#     }

#     int get_nnz_in(int i) {
#         return nnz_in[i];
#     }

#     int get_nnz_out(int i) {
#         return nnz_out[i];
#     }

#     int get_n_w() {
#         return n_w;
#     }

# ''' % (n_in, n_out)


codegen_strings['c_interface_closer'] = """\n}"""

# * Write codegen to file
for cg_str in codegen_strings.values():
    codegen_file.write(cg_str)

codegen_file.close()