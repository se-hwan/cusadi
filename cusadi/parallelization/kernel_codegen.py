import os
import textwrap
import casadi as ca
from casadi import *

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODEGEN_DIR = os.path.join(CURRENT_DIR, "codegen")
FUNCTION_DIR = os.path.join(CURRENT_DIR, "casadi_fns")

def get_functions(fn_names='all'):
    casadi_fns = []
    for filename in os.listdir(FUNCTION_DIR):
        if filename.endswith(".casadi"):
            fn_filepath = os.path.join(FUNCTION_DIR, filename)
            try:
                fn = ca.Function.load(fn_filepath)
                if fn_names == 'all' or fn.name() in fn_names:
                    casadi_fns.append(fn)
                    print(f"Loaded CasADi function: {fn.name()} ({fn.n_instructions()} instructions)")
            except Exception as e:
                print(f"Error loading {fn_filepath}: {e}")
    return casadi_fns

def build_pybind(fn, precision="float"):
    pybind_codegen(fn, precision)
    print(f"Pybind complete for {fn.name()}")
    print(f"Binding written to {os.path.join(CODEGEN_DIR, f'bindings.cpp')}")

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
    codegen_string += get_c_interface(f)
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
        from .kernel_operations import OP_CUDA_DICT_COALESCED_v2 as CUDA_OPS
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

def get_c_interface(f):
    f_name = f.name()
    n_in = f.n_in()
    n_out = f.n_out()
    str_c_interface = f"extern \"C\" void launch_{f_name}_kernel (\n"
    str_c_interface += "        const int batch_size,\n"
    for i in range(n_in):
        str_c_interface += f"        const float* input_{i},\n"
    for i in range(n_out):
        str_c_interface += f"        float* output_{i},\n"
    str_c_interface += "        float* work) {\n"
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

def pybind_codegen(f, precision):
    """Generate Python bindings for the CasADi function."""
    f_name = f.name()
    n_in = f.n_in()
    n_out = f.n_out()
    
    # Generate bindings.h - completely overwrite the file
    bindings_header_path = os.path.join(CODEGEN_DIR, "bindings.h")
    with open(bindings_header_path, 'r') as f:
        header_content = f.read()

    # Build function signature
    fn_signature = f"extern \"C\" void launch_{f_name}_kernel(\n"
    fn_signature += "    const int batch_size,\n"
    for i in range(n_in):
        fn_signature += f"    const float* input_{i},\n"
    for i in range(n_out):
        fn_signature += f"    float* output_{i},\n"
    fn_signature += "    float* work);\n\n"
    
    #### PYBIND CPP ####    
    fn_binding = f"void {f_name}_binding(\n"
    fn_binding += "    const int batch_size,\n"
    for i in range(n_in): # Input parameters
        fn_binding += f"    const torch::Tensor& input_{i},\n"
    for i in range(n_out): # Output parameters
        fn_binding += f"    torch::Tensor& output_{i},\n"
    fn_binding += "    torch::Tensor& work) {\n"
    fn_binding += f"    launch_{f_name}_kernel(\n"
    fn_binding += f"        batch_size,\n"
    
    # Function call parameters
    for i in range(n_in):
        fn_binding += f"        input_{i}.data_ptr<{precision}>(),\n"
    for i in range(n_out):
        fn_binding += f"        output_{i}.data_ptr<{precision}>(),\n"
    fn_binding += f"        work.data_ptr<{precision}>());\n"
    fn_binding += "}\n\n"

    with open(bindings_header_path, "a") as f:
        if fn_signature not in header_content:
            f.write(fn_signature)
        if fn_binding not in header_content:
            f.write(fn_binding)

    # Pybind module
    bindings_cpp_path = os.path.join(CODEGEN_DIR, "bindings.cpp")
    with open(bindings_cpp_path, "r") as f:
        cpp_content = f.read()
    module_header = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"

    def find_line_number(file, search_string):
        with open(file, "r") as f:
            for index, line in enumerate(f):
                if search_string in line:
                    return index
                    break
        return -1

    line_number = find_line_number(bindings_cpp_path, module_header)
    module_code = f"\n    m.def(\"{f_name}\", &{f_name}_binding);"
    if module_code not in cpp_content:
        cpp_content = cpp_content.splitlines(keepends=True)
        cpp_content.insert(line_number+1, module_code)
        with open(bindings_cpp_path, "w") as f:
                f.writelines(cpp_content)


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
