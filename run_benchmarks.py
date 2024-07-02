import os, sys
import torch
import numpy as np
import scipy
from casadi import *
from cusadi import *

# fn_1e6 (function with million instructions) is available in CUSADI_BENCHMARK_DIR, but not included in benchmark
# Compilation can take hours, both for .c and parallelized CUDA .cu file
# Change the EVALUATE_1E6 flag to True to include fn_1e6 in the benchmark

EVALUATE_1E6 = False
REBUILD_CUDA_CODEGEN = False
N_ENVS_SWEEP = [1, 5, 10, 50, 100, 250, 500, 1000, 5000, 10000]
N_EVALS = 20

# Load functions for benchmarking
fn_filepath_1e1 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e1.casadi")
fn_filepath_1e2 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e2.casadi")
fn_filepath_1e3 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e3.casadi")
fn_filepath_1e4 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e4.casadi")
fn_filepath_1e5 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e5.casadi")
fn_filepath_1e6 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e6.casadi")
fn_1e1 = casadi.Function.load(fn_filepath_1e1)
fn_1e2 = casadi.Function.load(fn_filepath_1e2)
fn_1e3 = casadi.Function.load(fn_filepath_1e3)
fn_1e4 = casadi.Function.load(fn_filepath_1e4)
fn_1e5 = casadi.Function.load(fn_filepath_1e5)
fn_1e6 = casadi.Function.load(fn_filepath_1e6)
benchmark_casadi_fns = [fn_1e1, fn_1e2, fn_1e3, fn_1e4, fn_1e5, fn_1e6]
N_INSTRUCTIONS = []

for f in benchmark_casadi_fns:
    print("Function loaded with ", f.n_instructions(), " instructions")
    N_INSTRUCTIONS.append(f.n_instructions())

    # Generate Cusadi functions for benchmarking
    if (not os.path.isfile(f"{CUSADI_CODEGEN_DIR}/{f.name()}.cu")):
        generateCUDACode(f)

    # Generate Pytorch functions for benchmarking
    if (not os.path.isfile(f"{CUSADI_BENCHMARK_DIR}/{f.name()}_PT.py")):
        generatePytorchCode(f, f"{CUSADI_BENCHMARK_DIR}/{f.name()}_PT.py")

    # Generate CPU compiled functions
    if (not os.path.isfile(f"{CUSADI_BENCHMARK_DIR}/{f.name()}.so")):
        c_filepath = f"{CUSADI_BENCHMARK_DIR}/{f.name()}.c"
        so_filepath = f"{CUSADI_BENCHMARK_DIR}/{f.name()}.so"
        f.generate(f"{f.name()}.c")
        os.system(f"mv {f.name()}.c {c_filepath}")
        os.system(f"gcc -fPIC -shared -O3 -march=native {c_filepath} -o {so_filepath}")
        os.system(f"rm {c_filepath}")

    # Generate CPU parallelized functions for benchmarking
    for N_ENVS in N_ENVS_SWEEP:
        if (not os.path.isfile(f"{CUSADI_BENCHMARK_DIR}/{f.name()}_mapped_{N_ENVS}.so") and
            N_ENVS < 10000):
            print("Generating parallelized CPU code for ", f.name(), " with ", N_ENVS, " environments")
            f_mapped_name = f"{f.name()}_mapped_{N_ENVS}"
            c_filepath = f"{CUSADI_BENCHMARK_DIR}/{f_mapped_name}.c"
            so_filepath = f"{CUSADI_BENCHMARK_DIR}/{f_mapped_name}.so"
            f_mapped = f.map(f"{f_mapped_name}", "openmp", N_ENVS, [], [], {})
            f_mapped.generate(f"{f_mapped_name}.c")
            os.system(f"mv {f_mapped_name}.c {c_filepath}")
            os.system(f"gcc -fPIC -shared -O3 -fopenmp -march=native {c_filepath} -o {so_filepath}")
            os.system(f"rm {c_filepath}")

if (REBUILD_CUDA_CODEGEN):
    generateCMakeLists(benchmark_casadi_fns)
    os.system(f"cd {CUSADI_BUILD_DIR} && cmake .. && make -j")

t_1e1 = {}; t_1e2 = {}; t_1e3 = {}; t_1e4 = {}; t_1e5 = {}; t_1e6 = {}
time_zero_array = np.zeros((len(N_ENVS_SWEEP), N_EVALS))
benchmark_sizes = ["n_1e1", "n_1e2", "n_1e3", "n_1e4", "n_1e5", "n_1e6"]
method_names = ["cusadi", "pytorch", "serial_cpu", "parallel_cpu"]
for method in method_names:
    t_1e1[method] = time_zero_array.copy()
    t_1e2[method] = time_zero_array.copy()
    t_1e3[method] = time_zero_array.copy()
    t_1e4[method] = time_zero_array.copy()
    t_1e5[method] = time_zero_array.copy()
    t_1e6[method] = time_zero_array.copy()
t_data = [t_1e1, t_1e2, t_1e3, t_1e4, t_1e5, t_1e6]
benchmark_data = dict(zip(benchmark_casadi_fns, t_data))
if not EVALUATE_1E6:
    benchmark_data.pop(fn_1e6)

def main():
    sys.setrecursionlimit(10000)

    for fn, time in benchmark_data.items():
        for i in range(len(N_ENVS_SWEEP)):
            print("Running benchmarks for ", N_ENVS_SWEEP[i], " environments...")
            N_ENVS = N_ENVS_SWEEP[i]
            inputs_GPU = [torch.rand(N_ENVS, fn.nnz_in(i), device='cuda',
                          dtype=torch.float32).contiguous() for i in range(fn.n_in())]
            inputs_CPU = [torch.transpose(inputs_GPU[i], 0, 1).cpu().detach().numpy()
                          for i in range(fn.n_in())]
            outputs_GPU = [torch.zeros(N_ENVS, fn.nnz_out(i), device='cuda',
                           dtype=torch.float32).contiguous() for i in range(fn.n_out())]
            work_GPU = torch.zeros(N_ENVS, fn.sz_w(), device='cuda', dtype=torch.float32).contiguous()
            fn_cusadi = CusadiFunction(fn, N_ENVS)
            fn_CPU = casadi.external(fn.name(), f"{CUSADI_BENCHMARK_DIR}/{fn.name()}.so")
            fn_CPU_parallel = casadi.external(f"{fn.name()}_mapped_{N_ENVS}",
                                              f"{CUSADI_BENCHMARK_DIR}/{fn.name()}_mapped_{N_ENVS}.so")
            for j in range(N_EVALS):
                _, time["cusadi"][i, j] = timeFunction(lambda:
                    runCusadiBenchmark(fn_cusadi, inputs_GPU))
                _, time["pytorch"][i, j] = timeFunction(lambda: 
                    runPytorchBenchmark(fn, outputs_GPU, inputs_GPU, work_GPU))
                _, time["serial_cpu"][i, j] = timeFunction(lambda:
                    runSerialCPUBenchmark(fn_CPU, inputs_CPU))
                _, time["parallel_cpu"][i, j] = timeFunction(lambda:
                    runParallelCPUBenchmark(fn_CPU_parallel, inputs_CPU))
    data_MATLAB = {}
    for i in range(len(benchmark_sizes)):
        data_MATLAB[benchmark_sizes[i]] = t_data[i]
    data_MATLAB["N_ENVS_SWEEP"] = np.array(N_ENVS_SWEEP)
    data_MATLAB["N_INSTRUCTIONS"] = np.array(N_INSTRUCTIONS)
    scipy.io.savemat(f"{CUSADI_DATA_DIR}/benchmark_data.mat", data_MATLAB)

def timeFunction(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def runCusadiBenchmark(fn, inputs):
    fn.evaluate(inputs)

def runPytorchBenchmark(fn, outputs, inputs, work):
    eval(f"{fn.name()}_PT.evaluate_{fn.name()}(outputs, inputs, work)")

def runSerialCPUBenchmark(fn, inputs):
    for i in range(inputs[0].shape[1]):
        inputs_np = [inputs[j][:, i] for j in range(len(inputs))]
        fn.call(inputs_np)

def runParallelCPUBenchmark(fn, inputs):
    fn.call(inputs)


if __name__ == "__main__":
    main()