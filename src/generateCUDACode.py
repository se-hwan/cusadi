import textwrap
from casadi import *
from src import *

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

def generateCUDACodeDouble(f, filepath=None, benchmarking=True, debug_mode=True):
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
    #include <math.h>
    #include <iostream>
    ''')
    codegen_strings["nnz_in"] = f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
    codegen_strings["nnz_out"] = f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
    codegen_strings["n_w"] = f"\n__constant__ int n_w = {n_w};\n"
    codegen_strings["error_check"] = textwrap.dedent(
    r'''
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    }
    ''')


    # * Codegen for CUDA kernel
    str_kernel = textwrap.dedent(
    '''

    __global__ void evaluate_kernel (
            const double *inputs[],
            double *work,
            double *outputs[],
            const int batch_size) {
    ''')
    str_kernel +=  "\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    str_kernel +=  "\n    int env_idx = idx * n_w;"
    str_kernel +=  "\n    if (idx < batch_size) {"
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
    codegen_strings['c_interface_header'] = """\n\nextern "C" {\n"""
    codegen_strings['c_evaluation'] = textwrap.dedent(
    '''
        float evaluate(const double *inputs[],
                    double *work,
                    double *outputs[],
                    const int batch_size) {
            int blockSize = 384;
            int gridSize = (batch_size + blockSize - 1) / blockSize;
            float time;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            evaluate_kernel<<<gridSize, blockSize>>>(inputs,
                                                    work,
                                                    outputs,
                                                    batch_size);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
    ''')
    if debug_mode:
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaPeekAtLastError());"
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaDeviceSynchronize());"
    codegen_strings['c_evaluation'] += "\n    return time;"
    codegen_strings['c_evaluation'] += "\n}"
    codegen_strings['c_interface_closer'] = """\n\n}"""

    # * Write codegen to file
    for cg_str in codegen_strings.values():
        codegen_file.write(cg_str)
    codegen_file.close()
    print("CUDA codegen complete for CasADi function: ", f.name())

def generateCUDACodeFloat(f, filepath=None, benchmarking=True, debug_mode=True):
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
    #include <math.h>
    #include <iostream>
    ''')
    codegen_strings["nnz_in"] = f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
    codegen_strings["nnz_out"] = f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
    codegen_strings["n_w"] = f"\n__constant__ int n_w = {n_w};\n"
    codegen_strings["error_check"] = textwrap.dedent(
    r'''
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    }
    ''')


    # * Codegen for CUDA kernel
    str_kernel = textwrap.dedent(
    '''

    __global__ void evaluate_kernel (
            const float *inputs[],
            float *work,
            float *outputs[],
            const int batch_size) {
    ''')
    str_kernel += f"\n    float work_env[{n_w}];"
    str_kernel +=  "\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    str_kernel +=  "\n    int env_idx = idx * n_w;"
    str_kernel +=  "\n    if (idx < batch_size) {"
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
    codegen_strings['c_interface_header'] = """\n\nextern "C" {\n"""
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
    ''')
    if debug_mode:
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaPeekAtLastError());"
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaDeviceSynchronize());\n}"
    else:
        codegen_strings['c_evaluation'] += "\n}"
    codegen_strings['c_interface_closer'] = """\n\n}"""

    # * Write codegen to file
    for cg_str in codegen_strings.values():
        codegen_file.write(cg_str)
    codegen_file.close()
    print("CUDA codegen complete for CasADi function: ", f.name())

def generateCUDACodeOrig(f, filepath=None, benchmarking=True, debug_mode=True):
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
    #include <math.h>
    #include <iostream>
    ''')
    codegen_strings["nnz_in"] = f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
    codegen_strings["nnz_out"] = f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
    codegen_strings["n_w"] = f"\n__constant__ int n_w = {n_w};\n"
    codegen_strings["error_check"] = textwrap.dedent(
    r'''
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    }
    ''')


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
            str_kernel += OP_CUDA_DICT_ORIG[op] % (o_idx, const_instr[k])
        elif op == OP_INPUT:
            str_kernel += OP_CUDA_DICT_ORIG[op] % (o_idx, i_idx, i_idx, input_idx[i_instr + 1])
        elif op == OP_OUTPUT:
            str_kernel += OP_CUDA_DICT_ORIG[op] % (o_idx, o_idx, output_idx[o_instr + 1], i_idx)
        elif op == OP_SQ:
            str_kernel += OP_CUDA_DICT_ORIG[op] % (o_idx, i_idx, i_idx)
        elif OP_CUDA_DICT_ORIG[op].count("%d") == 3:
            str_kernel += OP_CUDA_DICT_ORIG[op] % (o_idx, i_idx, input_idx[i_instr + 1])
        elif OP_CUDA_DICT_ORIG[op].count("%d") == 2:
            str_kernel += OP_CUDA_DICT_ORIG[op] % (o_idx, i_idx)
        else:
            raise Exception('Unknown CasADi operation: ' + str(op))
        o_instr += output_idx_lengths[k + 1]
        i_instr += input_idx_lengths[k + 1]

    str_kernel += "\n    }"         # End of if statement
    str_kernel += "\n}\n"           # End of kernel
    codegen_strings['cuda_kernel'] = str_kernel

    # * Codegen for C interface
    codegen_strings['c_interface_header'] = """\n\nextern "C" {\n"""
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
    ''')
    if debug_mode:
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaPeekAtLastError());"
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaDeviceSynchronize());\n}"
    else:
        codegen_strings['c_evaluation'] += "\n}"
    codegen_strings['c_interface_closer'] = """\n\n}"""

    # * Write codegen to file
    for cg_str in codegen_strings.values():
        codegen_file.write(cg_str)
    codegen_file.close()
    print("CUDA codegen complete for CasADi function: ", f.name())

def generateCUDACodeV3(f, filepath=None, benchmarking=True, debug_mode=True):
    # * Parse CasADi function
    print("Generating CUDA code for CasADi function: ", f.name())
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

    # * Determine number of .cu files to generate
    SPLIT_CODEGEN = False
    N_INSTR_PER_SUBFILE = 200000
    n_files = 1
    codegen_subfile_names = []
    codegen_subfiles = []
    if n_instr > 250000:
        print("Function contains more than 250,000 instructions.")
        print("Splitting function into multiple CUDA kernels.")
        SPLIT_CODEGEN = True
        n_files = n_instr // N_INSTR_PER_SUBFILE + 1
        for i in range(n_files):
            codegen_subfilepath = os.path.join(CUSADI_ROOT_DIR, "codegen", f"{f.name()}v3_p{i:02}.cu")
            codegen_subfile_names.append(f"{f.name()}v3_p{i:02}")
            codegen_subfiles.append(open(codegen_subfilepath, "w+"))
    
    if filepath is None:
        codegen_filepath = os.path.join(CUSADI_ROOT_DIR, "codegen", f"{f.name()}v3.cu")
    else:
        codegen_filepath = filepath
    codegen_file = open(codegen_filepath, "w+")
    codegen_strings = {}


    # * Codegen for const declarations and indices
    appendCUDAHeaderCode(codegen_strings, nnz_in, nnz_out, n_w, codegen_subfile_names)

    # * Codegen for CUDA function header
    str_kernel = getCUDAFunctionHeader(n_w)

    if SPLIT_CODEGEN:
        generateCUDASubfileCode(codegen_subfiles, codegen_subfile_names, f,
                                operations, const_instr,
                                output_idx, input_idx,
                                output_idx_lengths, input_idx_lengths)
        for i in range(n_files):
            str_kernel += f"\n          {f.name()}v3_p{i:02}(inputs, work_env, outputs, idx, env_idx, nnz_in, nnz_out, n_w);"
    else:
        o_instr = 0
        i_instr = 0
        for k in range(INSTR_LIMIT):
            op = operations[k]
            o_idx = output_idx[o_instr]
            i_idx = input_idx[i_instr]
            if op == OP_CONST:
                str_kernel += OP_CUDA_DICT_V2[op] % (o_idx, const_instr[k])
            elif op == OP_INPUT:
                str_kernel += OP_CUDA_DICT_V2[op] % (o_idx, i_idx, i_idx, input_idx[i_instr + 1])
            elif op == OP_OUTPUT:
                str_kernel += OP_CUDA_DICT_V2[op] % (o_idx, o_idx, output_idx[o_instr + 1], i_idx)
            elif op == OP_SQ:
                str_kernel += OP_CUDA_DICT_V2[op] % (o_idx, i_idx, i_idx)
            elif OP_CUDA_DICT_V2[op].count("%d") == 3:
                str_kernel += OP_CUDA_DICT_V2[op] % (o_idx, i_idx, input_idx[i_instr + 1])
            elif OP_CUDA_DICT_V2[op].count("%d") == 2:
                str_kernel += OP_CUDA_DICT_V2[op] % (o_idx, i_idx)
            else:
                raise Exception('Unknown CasADi operation: ' + str(op))
            o_instr += output_idx_lengths[k + 1]
            i_instr += input_idx_lengths[k + 1]

    str_kernel += "\n    }"         # End of if statement
    str_kernel += "\n}\n"           # End of kernel
    codegen_strings['cuda_kernel'] = str_kernel

    # * Codegen for C interface
    appendCInterfaceCode(codegen_strings, debug_mode)

    # * Write codegen to file
    for cg_str in codegen_strings.values():
        codegen_file.write(cg_str)
    codegen_file.close()
    print("CUDA codegen complete for CasADi function: ", f.name())

def generateCUDASubfileCode(codegen_subfiles, codegen_subfile_names, f,
                            operations, const_instr,
                            output_idx, input_idx,
                            output_idx_lengths, input_idx_lengths):
    N_INSTR_PER_SUBFILE = 200000
    n_in = f.n_in()
    n_out = f.n_out()
    nnz_in = [f.nnz_in(i) for i in range(n_in)]
    nnz_out = [f.nnz_out(i) for i in range(n_out)]
    n_w = f.sz_w()
    n_instr = len(operations)
    subfile_code_strings = ""
    instr_idx = 0
    o_instr = 0
    i_instr = 0
    for i in range(len(codegen_subfiles)):
        subfile_code_strings = ""
        subfile_code_strings += textwrap.dedent(
        '''
        #include <cuda_runtime.h>
        #include <math.h>
        ''')
        # subfile_code_strings += f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
        # subfile_code_strings += f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
        # subfile_code_strings += f"\n__constant__ int n_w = {n_w};\n\n"
        subfile_code_strings += textwrap.dedent(
        f"""
        __device__ void {codegen_subfile_names[i]}(const float *inputs[], float *work_env,
            float *outputs[], int idx, int env_idx, const int *nnz_in, const int *nnz_out, const int n_w)
        """)
        subfile_code_strings += "{\n"
        for k in range(instr_idx, min(instr_idx + N_INSTR_PER_SUBFILE, n_instr)):
        # for k in range(N_INSTR_PER_SUBFILE):
            op = operations[k]
            o_idx = output_idx[o_instr]
            i_idx = input_idx[i_instr]
            if op == OP_CONST:
                subfile_code_strings += OP_CUDA_DICT_V2[op] % (o_idx, const_instr[k])
            elif op == OP_INPUT:
                subfile_code_strings += OP_CUDA_DICT_V2[op] % (o_idx, i_idx, i_idx, input_idx[i_instr + 1])
            elif op == OP_OUTPUT:
                subfile_code_strings += OP_CUDA_DICT_V2[op] % (o_idx, o_idx, output_idx[o_instr + 1], i_idx)
            elif op == OP_SQ:
                subfile_code_strings += OP_CUDA_DICT_V2[op] % (o_idx, i_idx, i_idx)
            elif OP_CUDA_DICT_V2[op].count("%d") == 3:
                subfile_code_strings += OP_CUDA_DICT_V2[op] % (o_idx, i_idx, input_idx[i_instr + 1])
            elif OP_CUDA_DICT_V2[op].count("%d") == 2:
                subfile_code_strings += OP_CUDA_DICT_V2[op] % (o_idx, i_idx)
            else:
                raise Exception('Unknown CasADi operation: ' + str(op))
            o_instr += output_idx_lengths[k + 1]
            i_instr += input_idx_lengths[k + 1]
        subfile_code_strings += "\n}\n"           # End of kernel
        codegen_subfiles[i].write(subfile_code_strings)
        codegen_subfiles[i].close()
        instr_idx = k + 1

def appendCUDAHeaderCode(codegen_strings, nnz_in, nnz_out, n_w, codegen_subfile_names):
    codegen_strings['header'] = "// AUTOMATICALLY GENERATED CODE FOR CUSADI\n"
    codegen_strings['includes'] = textwrap.dedent(
    '''
    #include <cuda_runtime.h>
    #include <math.h>
    #include <iostream>
    ''')

    subfile_include_strings = ""
    for i in range(len(codegen_subfile_names)):
        subfile_include_strings += f"#include \"{codegen_subfile_names[i]}.cu\"\n"
    
    codegen_strings['includes'] += subfile_include_strings
    codegen_strings["nnz_in"] = f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
    codegen_strings["nnz_out"] = f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
    codegen_strings["n_w"] = f"\n__constant__ int n_w = {n_w};\n"
    codegen_strings["error_check"] = textwrap.dedent(
    r'''
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    }
    ''')

def getCUDAFunctionHeader(n_w):
    str_kernel = textwrap.dedent(
    '''

    __global__ void evaluate_kernel (
            const float *inputs[],
            float *work,
            float *outputs[],
            const int batch_size) {
    ''')
    str_kernel += f"\n    float work_env[{n_w}];"
    str_kernel +=  "\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    str_kernel +=  "\n    int env_idx = idx * n_w;"
    str_kernel +=  "\n    if (idx < batch_size) {"
    return str_kernel

def appendCInterfaceCode(codegen_strings, debug_mode):
    codegen_strings['c_interface_header'] = """\n\nextern "C" {\n"""
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
    ''')
    if debug_mode:
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaPeekAtLastError());"
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaDeviceSynchronize());\n}"
    else:
        codegen_strings['c_evaluation'] += "\n}"
    codegen_strings['c_interface_closer'] = """\n\n}"""
