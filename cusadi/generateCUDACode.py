import textwrap
from casadi import *
from cusadi import *

def generateCUDACode(f, filepath=None):
    print("Generating CUDA code for CasADi function: ", f.name())
    if filepath is None:
        codegen_filepath = os.path.join(CUSADI_ROOT_DIR, "codegen", f"{f.name()}.cu")
    else:
        codegen_filepath = filepath
    codegen_file = open(codegen_filepath, "w+")
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
    codegen_strings['includes'] = textwrap.dedent(
    '''
    #include <cuda_runtime.h>
    #include <cmath>
    #include <iostream>
    ''')
    codegen_strings["nnz_in"] = f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
    codegen_strings["nnz_out"] = f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
    codegen_strings["n_w"] = f"\n__constant__ int n_w = {n_w};\n"

    # * Codegen for CUDA kernel
    str_kernel = textwrap.dedent(
    '''

    __global__ void evaluate_kernel (
            const float *inputs[],
            float *work,
            float *outputs[],
            const int batch_size) {
    ''')

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

    str_kernel += "\n    }"         # End of if statement
    str_kernel += "\n}\n"           # End of kernel
    codegen_strings['cuda_kernel'] = str_kernel

    # * Codegen for C interface
    codegen_strings['c_interface_header'] = """\nextern "C" {\n\n"""
    codegen_strings['c_evaluation'] = textwrap.dedent(
    '''
        void evaluate(const float *inputs[],
                    float *work,
                    float *outputs[],
                    const int batch_size) {
            int blockSize = 256;
            int gridSize = (batch_size + blockSize - 1) / blockSize;
            evaluate_kernel<<<gridSize, blockSize>>>(inputs,
                                                    work,
                                                    outputs,
                                                    batch_size);
        }
    ''')
    codegen_strings['c_interface_closer'] = """\n}"""

    # * Write codegen to file
    for cg_str in codegen_strings.values():
        codegen_file.write(cg_str)
    codegen_file.close()
    print("CUDA codegen complete for CasADi function: ", f.name())


def generateCMakeLists(casadi_fns):
    cmake_filepath = os.path.join(CUSADI_ROOT_DIR, "CMakeLists.txt")
    cmake_file = open(cmake_filepath, "w+")
    cmake_strings = {}

    cmake_strings['version'] = "cmake_minimum_required(VERSION 3.15)\n"
    cmake_strings['project'] = "project(CusADi)\n"
    cmake_strings['packages'] = textwrap.dedent(
    """
    # Find CUDA package
    include(CheckLanguage)
    check_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    message("CUDA found")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86)
    endif()
    message(${CMAKE_CUDA_ARCHITECTURES})

    # Set C++ standard
    set(CMAKE_CXX_STANDARD 11)

    # Set CUDA flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_86 --use_fast_math)  # Adjust architecture as needed

    """)

    # Set sources for each CasADi function
    str_libraries = ""
    str_sources = ""
    for f in casadi_fns:
        print(f.name())
        fn_source_name = f.name().upper() + "_SOURCE"
        fn_filepath = f"codegen/{f.name()}.cu"
        str_sources += f"set({fn_source_name} {fn_filepath})\n"
        str_libraries += f"add_library({f.name()} SHARED ${{{fn_source_name}}})\n"
        str_libraries += f"target_link_libraries({f.name()})\n"

    cmake_strings['sources'] = str_sources
    cmake_strings['include'] = textwrap.dedent(
    """
    # Include directories for your header files
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    """)

    # Add and link libraries for each CasADi function
    cmake_strings['libraries'] = str_libraries

    # * Write codegen to file
    for cmake_str in cmake_strings.values():
        cmake_file.write(cmake_str)