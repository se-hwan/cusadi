import os, sys
import torch
import time
import numpy as np
import scipy
from casadi import *
from src import *
import subprocess

REBUILD_CUDA_CODEGEN = True
N_ENVS_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
N_EVALS = 20

# Load functions for benchmarking
fn_filepath_1e1 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e1.casadi")
fn_filepath_1e2 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e2.casadi")
fn_filepath_1e3 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e3.casadi")
fn_filepath_1e4 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e4.casadi")
fn_filepath_1e5 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e5.casadi")
fn_1e1 = casadi.Function.load(fn_filepath_1e1)
fn_1e2 = casadi.Function.load(fn_filepath_1e2)
fn_1e3 = casadi.Function.load(fn_filepath_1e3)
fn_1e4 = casadi.Function.load(fn_filepath_1e4)
fn_1e5 = casadi.Function.load(fn_filepath_1e5)
benchmark_casadi_fns = [fn_1e1, fn_1e2, fn_1e3, fn_1e4, fn_1e5]
N_INSTRUCTIONS = []

for f in benchmark_casadi_fns:
    print("Function loaded with ", f.n_instructions(), " instructions")
    N_INSTRUCTIONS.append(f.n_instructions())

    # Generate Cusadi functions for benchmarking
    if (not os.path.isfile(f"{CUSADI_CODEGEN_DIR}/{f.name()}.cu")):
        generateCUDACodeDouble(f)

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

if (REBUILD_CUDA_CODEGEN):
    generateCMakeLists(benchmark_casadi_fns)
    os.system(f"cd {CUSADI_BUILD_DIR} && cmake .. && make -j")

t_1e1 = {}; t_1e2 = {}; t_1e3 = {}; t_1e4 = {}; t_1e5 = {};
time_zero_array = np.zeros((len(N_ENVS_SWEEP), N_EVALS))
benchmark_sizes = ["n_1e1", "n_1e2", "n_1e3", "n_1e4", "n_1e5"]
method_names = ["cusadi", "pytorch",
                "serial_cpu", "serial_cpu_transfer",
                "parallel_cpu", "parallel_cpu_transfer"]
for method in method_names:
    t_1e1[method] = time_zero_array.copy()
    t_1e2[method] = time_zero_array.copy()
    t_1e3[method] = time_zero_array.copy()
    t_1e4[method] = time_zero_array.copy()
    t_1e5[method] = time_zero_array.copy()
t_data = [t_1e1, t_1e2, t_1e3, t_1e4, t_1e5]
benchmark_data = dict(zip(benchmark_casadi_fns, t_data))

def main():
    sys.setrecursionlimit(10000)

    for fn, time in benchmark_data.items():
        fn_name = fn.name()
        fn_path = f"{CUSADI_BENCHMARK_DIR}/{fn_name}.so"
        for i in range(len(N_ENVS_SWEEP)):
            print("Running benchmarks for ", N_ENVS_SWEEP[i], " environments...")
            N_ENVS = N_ENVS_SWEEP[i]
            inputs_GPU = [torch.rand(N_ENVS, fn.nnz_in(i), device='cuda',
                          dtype=torch.double).contiguous() for i in range(fn.n_in())]
            outputs_GPU = [torch.zeros(N_ENVS, fn.nnz_out(i), device='cuda',
                           dtype=torch.double).contiguous() for i in range(fn.n_out())]
            work_GPU = torch.zeros(N_ENVS, fn.sz_w(), device='cuda', dtype=torch.double).contiguous()
            fn_cusadi = CusadiFunction(fn, N_ENVS)
            for j in range(N_EVALS):
                time["cusadi"][i, j] = runCusadiBenchmark(fn_cusadi, inputs_GPU)
                _, time["pytorch"][i, j] = timeFunction(lambda: 
                    runPytorchBenchmark(fn, outputs_GPU, inputs_GPU, work_GPU))
                time["serial_cpu"][i, j] = runSerialCPUBenchmark(fn_name, fn_path, N_ENVS)
                time["parallel_cpu"][i, j] = runParallelCPUBenchmark(fn_name, fn_path, N_ENVS)
                time["serial_cpu_transfer"][i, j] = runSerialCPUBenchmarkWithTransfer(fn_name, fn_path, N_ENVS)
                time["parallel_cpu_transfer"][i, j] = runParallelCPUBenchmarkWithTransfer(fn_name, fn_path, N_ENVS)
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
    return fn.eval_time

def runPytorchBenchmark(fn, outputs, inputs, work):
    eval(f"torch.vmap({fn.name()}_PT.evaluate_{fn.name()}, out_dims=None)(outputs, inputs, work)")

def runSerialCPUBenchmark(fn_name, fn_path, N_ENVS):
    result_serial = subprocess.run([
        "./evaluate_serial_cpu", fn_name, fn_path, str(N_ENVS)],
        cwd = CUSADI_BENCHMARK_DIR,
        capture_output=True,
        text=True)
    return float(result_serial.stdout.strip())

def runParallelCPUBenchmark(fn_name, fn_path, N_ENVS):
    result_parallel = subprocess.run([
        "./evaluate_parallel_cpu", fn_name, fn_path, str(N_ENVS)],
        cwd = CUSADI_BENCHMARK_DIR,
        capture_output=True,
        text=True)
    return float(result_parallel.stdout.strip())

def runSerialCPUBenchmarkWithTransfer(fn_name, fn_path, N_ENVS):
    result_serial = subprocess.run([
        "./evaluate_serial_cpu_transfer", fn_name, fn_path, str(N_ENVS)],
        cwd = CUSADI_BENCHMARK_DIR,
        capture_output=True,
        text=True)
    return float(result_serial.stdout.strip())

def runParallelCPUBenchmarkWithTransfer(fn_name, fn_path, N_ENVS):
    result_parallel = subprocess.run([
        "./evaluate_parallel_cpu_transfer", fn_name, fn_path, str(N_ENVS)],
        cwd = CUSADI_BENCHMARK_DIR,
        capture_output=True,
        text=True)
    return float(result_parallel.stdout.strip())

if __name__ == "__main__":
    main()