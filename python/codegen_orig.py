import ctypes
import torch
import time
from casadi import *
from evaluateCasADiPython import evaluateCasADiPython

f = casadi.Function.load("../test.casadi")

codegen_file = open(r"../src/codegen_prototype_orig.cu", "w+")
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
        str_kernel += f"\n        work[idx * n_w + {o_idx}] = {const_instr[k]};"
    else:
        if op == OP_INPUT:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = inputs[{i_idx}][idx * nnz_in[{i_idx}] + {input_idx[i_instr + 1]}];"
        elif op == OP_OUTPUT:
            str_kernel += f"\n        outputs[{o_idx}][idx * nnz_out[{o_idx}] + {output_idx[o_instr + 1]}] = work[idx * n_w + {i_idx}];"
        elif op == OP_ADD:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = work[idx * n_w + {i_idx}] + work[idx * n_w + {input_idx[i_instr + 1]}];"
        elif op == OP_SUB:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = work[idx * n_w + {i_idx}] - work[idx * n_w + {input_idx[i_instr + 1]}];"
        elif op == OP_NEG:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = -work[idx * n_w + {i_idx}];"
        elif op == OP_MUL:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = work[idx * n_w + {i_idx}] * work[idx * n_w + {input_idx[i_instr + 1]}];"
        elif op == OP_DIV:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = work[idx * n_w + {i_idx}] / work[idx * n_w + {input_idx[i_instr + 1]}];"
        elif op == OP_SIN:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = sinf(work[idx * n_w + {i_idx}]);"
        elif op == OP_COS:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = cosf(work[idx * n_w + {i_idx}]);"
        elif op == OP_TAN:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = tanf(work[idx * n_w + {i_idx}]);"
        elif op == OP_SQ:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = work[idx * n_w + {i_idx}] * work[idx * n_w + {i_idx}];"
        elif op == OP_SQRT:
            str_kernel += f"\n        work[idx * n_w + {o_idx}] = sqrt(work[idx * n_w + {i_idx}]);"
        else:
            raise Exception('Unknown CasADi operation: ' + str(op))
    o_instr += output_idx_lengths[k + 1]
    i_instr += input_idx_lengths[k + 1]

str_kernel += "\n    }"     # End of if statement
str_kernel += "\n}"         # End of kernel
codegen_strings['cuda_kernel'] = str_kernel

# * Codegen for C interface
codegen_strings['c_interface'] = '''

extern "C" {
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
}

'''

# * Write codegen to file
for cg_str in codegen_strings.values():
    codegen_file.write(cg_str)

codegen_file.close()