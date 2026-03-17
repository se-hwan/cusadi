import os
import re
import textwrap
import casadi as ca
from casadi import *

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODEGEN_DIR = os.path.join(CURRENT_DIR, "codegen")


def get_codegen_kernel_names():
    return sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(CODEGEN_DIR)
        if filename.endswith(".cu")
    )

def build_pybind(fn, precision="float"):
    remaining_bindings = pybind_codegen(fn, precision)
    print(f"Pybind complete for {fn.name()}")
    print(f"Binding written to {os.path.join(CODEGEN_DIR, f'bindings.cpp')}")
    return remaining_bindings

def build_kernel(fn, batch_size=1, precision="float", dynamic_batching=False):
    assert precision in ["float", "double"], \
        "Precision must be either 'float' or 'double'"
    if dynamic_batching:
        assert batch_size > 0, "Batch size must be greater than 0"
    print_function_info(fn, batch_size, precision, dynamic_batching)
    cuda_codegen(fn, batch_size, precision, dynamic_batching)
    print(f"CUDA codegen complete for {fn.name()}.")
    print(f"Kernel written to {os.path.join(CODEGEN_DIR, f'{fn.name()}.cu')}")

def print_function_info(f, batch_size, precision, dynamic_batching):
    print("Generating CUDA code for CasADi function: ", f.name())
    print("     Dynamic batching: ", dynamic_batching)
    print("     Number of instructions: ", f.n_instructions())
    print("     Number of inputs: ", f.n_in())
    print("     Number of outputs: ", f.n_out())
    print("     Number of work variables: ", f.sz_w())
    print("     Batch size: ", batch_size)
    print("     Precision: ", precision)

def cuda_codegen(f, batch_size, precision, dynamic_batching):
    f_name = f.name()
    codegen_filepath = os.path.join(CODEGEN_DIR, f"{f_name}.cu")
    codegen_file = open(codegen_filepath, "w+")
    
    codegen_string = ""
    codegen_string += get_cuda_header(f)
    codegen_string += get_kernel(f, batch_size, precision, dynamic_batching)
    codegen_string += get_c_interface(f, precision)
    codegen_file.write(codegen_string)
    codegen_file.close()

def get_cuda_header(f):
    # * Codegen for const declarations and indices
    n_w = f.sz_w()
    n_in = f.n_in()
    n_out = f.n_out()
    nnz_in = [f.nnz_in(i) for i in range(n_in)]
    nnz_out = [f.nnz_out(i) for i in range(n_out)]
    str_header = "// AUTOMATICALLY GENERATED CODE FOR CUSADI\n"
    str_header = ""
    str_header += "#include <cuda_runtime.h>\n"
    str_header += "#include \"../utils/cuda_utils.cu\"\n\n"
    str_header += "#include <math.h>\n"
    str_header += "#include <limits.h>\n"
    str_header += f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
    str_header += f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
    str_header += f"\n__constant__ int n_w = {n_w};\n"
    return str_header

def get_kernel(f, batch_size, precision, dynamic_batching):
    # * Parse CasADi function
    f_name = f.name()
    n_instr = f.n_instructions()
    n_in = f.n_in()
    n_out = f.n_out()
    input_idx = []
    input_idx_lengths = [0]
    output_idx = []
    output_idx_lengths = [0]

    INSTR_LIMIT = n_instr
    for i in range(INSTR_LIMIT):
        input_idx.extend(f.instruction_input(i))
        input_idx_lengths.append(len(f.instruction_input(i)))
        output_idx.extend(f.instruction_output(i))
        output_idx_lengths.append(len(f.instruction_output(i)))
    operations = [f.instruction_id(i) for i in range(INSTR_LIMIT)]
    const_instr = [f.instruction_constant(i) for i in range(INSTR_LIMIT)]

    str_kernel =   f"__global__ void evaluate_{f_name} (\n"
    str_kernel +=  f"       const int batch_size,\n"
    for i in range(n_in):
        str_kernel += f"       const {precision} *input_{i},\n"
    for i in range(n_out):
        str_kernel += f"       {precision} *output_{i},\n"
    str_kernel +=  f"       {precision} *work)" + " {\n\n"
    str_kernel +=  f"   [[maybe_unused]] const {precision} inf = std::numeric_limits<{precision}>::infinity();\n\n"
    str_kernel +=   "   int tID = blockIdx.x * blockDim.x + threadIdx.x;\n"
    str_kernel +=   "   if (tID < batch_size) {\n"

    if dynamic_batching:
        offset = 1
        from .kernel_operations import OP_CUDA_DICT as CUDA_OPS
    else:
        offset = batch_size
        from .kernel_operations import OP_CUDA_DICT_COALESCED as CUDA_OPS

    o_instr = 0
    i_instr = 0
    for k in range(INSTR_LIMIT):
        op = operations[k]
        o_idx = output_idx[o_instr]
        i_idx = input_idx[i_instr]
        if op == OP_CONST:
            str_kernel += CUDA_OPS[op] % (offset*o_idx, const_instr[k])
        elif op == OP_INPUT:
            str_kernel += CUDA_OPS[op] % (offset*o_idx, i_idx, i_idx, input_idx[i_instr + 1])
        elif op == OP_OUTPUT:
            str_kernel += CUDA_OPS[op] % (o_idx, o_idx, output_idx[o_instr + 1], offset*i_idx)
        elif op == OP_SQ:
            str_kernel += CUDA_OPS[op] % (offset*o_idx, offset*i_idx, offset*i_idx)
        elif CUDA_OPS[op].count("%d") == 3:
            str_kernel += CUDA_OPS[op] % (offset*o_idx, offset*i_idx, offset*input_idx[i_instr + 1])
        elif CUDA_OPS[op].count("%d") == 2:
            str_kernel += CUDA_OPS[op] % (offset*o_idx, offset*i_idx)
        else:
            raise Exception('Unknown CasADi operation: ' + str(op))
        o_instr += output_idx_lengths[k + 1]
        i_instr += input_idx_lengths[k + 1]
    str_kernel += "\n    }"           # End of if statement
    str_kernel += "\n}\n\n"           # End of kernel

    # # ! ORIGINAL
    # from .kernel_operations import OP_CUDA_DICT_COALESCED as CUDA_OPS
    # o_instr = 0
    # i_instr = 0
    # offset = 1 #if batch_size == 0 else batch_size
    # for k in range(INSTR_LIMIT):
    #     op = operations[k]
    #     o_idx = output_idx[o_instr]
    #     i_idx = input_idx[i_instr]
    #     if op == OP_CONST:
    #         str_kernel += CUDA_OPS[op] % (offset*o_idx, const_instr[k])
    #     elif op == OP_INPUT:
    #         str_kernel += CUDA_OPS[op] % (offset*o_idx, i_idx, i_idx, input_idx[i_instr + 1])
    #     elif op == OP_OUTPUT:
    #         str_kernel += CUDA_OPS[op] % (o_idx, o_idx, output_idx[o_instr + 1], offset*i_idx)
    #     elif op == OP_SQ:
    #         str_kernel += CUDA_OPS[op] % (offset*o_idx, offset*i_idx, offset*i_idx)
    #     elif CUDA_OPS[op].count("%d") == 3:
    #         str_kernel += CUDA_OPS[op] % (offset*o_idx, offset*i_idx, offset*input_idx[i_instr + 1])
    #     elif CUDA_OPS[op].count("%d") == 2:
    #         str_kernel += CUDA_OPS[op] % (offset*o_idx, offset*i_idx)
    #     else:
    #         raise Exception('Unknown CasADi operation: ' + str(op))
    #     o_instr += output_idx_lengths[k + 1]
    #     i_instr += input_idx_lengths[k + 1]
    # str_kernel += "\n    }"           # End of if statement
    # str_kernel += "\n}\n\n"           # End of kernel
    
    return str_kernel

def get_c_interface(f, precision):
    f_name = f.name()
    n_in = f.n_in()
    n_out = f.n_out()
    str_c_interface = f"extern \"C\" void launch_{f_name}_kernel (\n"
    str_c_interface += "        const int batch_size,\n"
    for i in range(n_in):
        str_c_interface += f"        const {precision}* input_{i},\n"
    for i in range(n_out):
        str_c_interface += f"        {precision}* output_{i},\n"
    str_c_interface += f"        {precision}* work) {{\n"
    str_c_interface += "    int block_size = 32;\n"
    str_c_interface += "    int grid = (batch_size + block_size - 1) / block_size;\n"
    str_c_interface += f"    evaluate_{f_name}<<<grid, block_size>>>(\n"
    str_c_interface += "        batch_size,\n"
    for i in range(n_in):
        str_c_interface += f"        input_{i},\n"
    for i in range(n_out):
        str_c_interface += f"        output_{i},\n"
    str_c_interface += "        work);\n"
    str_c_interface += "}\n\n"
    return str_c_interface

def _default_bindings_header():
    return "#pragma once\n#include <torch/extension.h>\n\n\n"

def _default_bindings_cpp():
    return '#include "bindings.h"\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n\n}\n'

def _generate_header_binding_block(f_name, n_in, n_out, precision):
    fn_signature = f"extern \"C\" void launch_{f_name}_kernel(\n"
    fn_signature += "    const int batch_size,\n"
    for i in range(n_in):
        fn_signature += f"    const {precision}* input_{i},\n"
    for i in range(n_out):
        fn_signature += f"    {precision}* output_{i},\n"
    fn_signature += f"    {precision}* work);\n\n"

    fn_binding = f"void {f_name}_binding(\n"
    fn_binding += "    const int batch_size,\n"
    for i in range(n_in):
        fn_binding += f"    const torch::Tensor& input_{i},\n"
    for i in range(n_out):
        fn_binding += f"    torch::Tensor& output_{i},\n"
    fn_binding += "    torch::Tensor& work) {\n"
    fn_binding += f"    launch_{f_name}_kernel(\n"
    fn_binding += "        batch_size,\n"
    for i in range(n_in):
        fn_binding += f"        input_{i}.data_ptr<{precision}>(),\n"
    for i in range(n_out):
        fn_binding += f"        output_{i}.data_ptr<{precision}>(),\n"
    fn_binding += f"        work.data_ptr<{precision}>());\n"
    fn_binding += "}\n\n"
    return fn_signature + fn_binding

def _parse_bindings_header(content):
    preamble_match = re.match(r'(?P<preamble>.*?)(?=extern "C" void launch_|\Z)', content, re.DOTALL)
    preamble = preamble_match.group("preamble") if preamble_match else _default_bindings_header()
    if not preamble.strip():
        preamble = _default_bindings_header()

    block_pattern = re.compile(
        r'extern "C" void launch_(?P<name>\w+)_kernel\s*\('
        r'(?P<c_params>.*?)'
        r'\);\s*'
        r'void (?P=name)_binding\s*\('
        r'(?P<binding_params>.*?)'
        r'\)\s*\{'
        r'(?P<body>.*?)'
        r'\n\}\s*',
        re.DOTALL,
    )

    ordered_names = []
    bindings = {}
    for match in block_pattern.finditer(content):
        name = match.group("name")
        ordered_names.append(name)
        bindings[name] = {
            "name": name,
            "n_in": len(re.findall(r'\binput_\d+\b', match.group("c_params"))),
            "n_out": len(re.findall(r'\boutput_\d+\b', match.group("c_params"))),
            "block": match.group(0).strip() + "\n\n",
        }

    return preamble, ordered_names, bindings

def _parse_bindings_cpp(content):
    module_pattern = re.compile(r'm\.def\("(?P<name>\w+)", &(?P=name)_binding\);')
    return [match.group("name") for match in module_pattern.finditer(content)]

def _ordered_unique(names):
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered

def _render_bindings_header(preamble, names, bindings):
    header = preamble.rstrip() + "\n\n"
    for name in names:
        if name in bindings:
            header += bindings[name]["block"]
    return header.rstrip() + "\n"

def _render_bindings_cpp(names):
    lines = ['#include "bindings.h"', "", "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", ""]
    for name in names:
        lines.append(f'    m.def("{name}", &{name}_binding);')
    lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)

def pybind_codegen(f, precision):
    """Generate Python bindings for the CasADi function."""
    f_name = f.name()
    n_in = f.n_in()
    n_out = f.n_out()
    kernel_names = set(get_codegen_kernel_names())
    bindings_header_path = os.path.join(CODEGEN_DIR, "bindings.h")
    bindings_cpp_path = os.path.join(CODEGEN_DIR, "bindings.cpp")
    header_content = _default_bindings_header()
    if os.path.exists(bindings_header_path):
        with open(bindings_header_path, "r") as header_file:
            header_content = header_file.read()

    cpp_content = _default_bindings_cpp()
    if os.path.exists(bindings_cpp_path):
        with open(bindings_cpp_path, "r") as cpp_file:
            cpp_content = cpp_file.read()

    preamble, header_order, header_bindings = _parse_bindings_header(header_content)
    cpp_order = _parse_bindings_cpp(cpp_content)

    header_bindings = {
        name: binding for name, binding in header_bindings.items() if name in kernel_names
    }

    if f_name in kernel_names:
        current_binding = header_bindings.get(f_name)
        if current_binding is None or current_binding["n_in"] != n_in or current_binding["n_out"] != n_out:
            header_bindings[f_name] = {
                "name": f_name,
                "n_in": n_in,
                "n_out": n_out,
                "block": _generate_header_binding_block(f_name, n_in, n_out, precision),
            }

    ordered_names = _ordered_unique(
        [name for name in cpp_order if name in kernel_names]
        + [name for name in header_order if name in kernel_names]
        + ([f_name] if f_name in kernel_names else [])
    )

    header_names = [name for name in ordered_names if name in header_bindings]
    header_output = _render_bindings_header(preamble, header_names, header_bindings)
    with open(bindings_header_path, "w") as header_file:
        header_file.write(header_output)

    cpp_names = [name for name in header_names if name in kernel_names]
    cpp_output = _render_bindings_cpp(cpp_names)
    with open(bindings_cpp_path, "w") as cpp_file:
        cpp_file.write(cpp_output)

    return cpp_names


# # TODO: return time taken here too
# def get_pytorch_binding(f, precision):
#     f_name = f.name()
#     n_in = f.n_in()
#     n_out = f.n_out()
#     str_binding =            "#ifndef CPP_COMPILATION\n"
#     str_binding +=           "#include <torch/extension.h>\n"
#     str_binding +=          f"void launch_{f_name} (\n"
#     str_binding +=           "        const int batch_size,\n"
#     for i in range(n_in):
#         str_binding +=      f"        const torch::Tensor& input_{i},\n"
#     for i in range(n_out):
#         str_binding +=      f"        torch::Tensor& output_{i},\n"
#     str_binding +=           "        torch::Tensor& work) {\n"
#     str_binding +=           "    int block_size = 32;\n"
#     str_binding +=           "    int extra_block = (batch_size % block_size > 0) ? 1 : 0;\n"
#     str_binding +=          f"    dim3 grid_dim(batch_size/block_size + extra_block, 1, 1);\n"
#     str_binding +=           "    dim3 block_dim(block_size, 1, 1);\n"
#     str_binding +=          f"    evaluate_{f_name}<<<grid_dim, block_dim>>>(\n"
#     str_binding +=           "        batch_size,\n"
#     for i in range(n_in):
#         str_binding +=      f"        static_cast<{precision}*>(input_{i}.data_ptr()),\n"
#     for i in range(n_out):
#         str_binding +=      f"        static_cast<{precision}*>(output_{i}.data_ptr()),\n"
#     str_binding +=          f"        static_cast<{precision}*>(work.data_ptr()));\n"
#     str_binding +=           "    CHECK_CUDA_ERROR(cudaPeekAtLastError());\n"
#     str_binding +=           "    CHECK_CUDA_ERROR(cudaDeviceSynchronize());\n"
#     str_binding += "}\n"
#     str_binding += "#endif\n"
#     return str_binding

#     # ! Optimization thoughts:
#     # * 1. [COMPLETE] Global coalescing for work vector
#     # ! 2. If work vector size is small enough, could we load this into shared memory?
#     # !     - For each block, compute TOTAL_SHMEM_SIZE/(double * sz_w) --> number of threads/environments it is responsible for
#     # !     - Then, stride the shared memory so that each env. writes/reads from a different location (no sync should be needed)
#     # !     - We could also preload all indices into shmem/registers, so we don't have to compute indices in the kernel
#     # ! 3. If sz_w is too big, then have to use coalesced global memory
#     # !     - 48 kb limit, with doubles, this is ~6000 elements
#     # !     - if sz_w is 200, this means 30 environments per block
#     # ! 4. For 3090, 82 SMs, 64 KB shared memory per block, 48 KB L1 cache per SM
#     # !     - Each SM has maximum of 16 threadblocks, 1536 threads
#     # !     - This means we could have 96 threads/block, 16 blocks/SM --> 1312 blocks concurrently for full occupancy
#     # ! 5. We could group operations that can be done in parallel (I/O copy), but there's no pattern...
