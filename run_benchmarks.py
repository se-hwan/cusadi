import os, sys
import argparse
import torch
import numpy as np
import scipy
from casadi import *
from cusadi import *

# N_ENVS_SWEEP = [1, 5, 10, 50, 100, 250, 500, 1000, 5000, 10000]
N_ENVS_SWEEP = [1, 10]
N_EVALS = 20

# Load functions for benchmarking
# fn_filepath_1e0 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e0.casadi")
# fn_filepath_1e1 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e1.casadi")
# fn_filepath_1e2 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e2.casadi")
fn_filepath_1e3 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e3.casadi")
# fn_filepath_1e4 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e4.casadi")
# fn_filepath_1e5 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e5.casadi")
# fn_filepath_1e6 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e6.casadi")
fn_1e3 = casadi.Function.load(fn_filepath_1e3)
# fn_1e4 = casadi.Function.load(fn_filepath_1e4)
# fn_1e5 = casadi.Function.load(fn_filepath_1e5)
# fn_1e6 = casadi.Function.load(fn_filepath_1e6)
benchmark_casadi_fns = [fn_1e3]  # [fn_1e0, fn_1e1, fn_1e2, fn_1e3, fn_1e4, fn_1e5, fn_1e6]

for f in benchmark_casadi_fns:
    print("Function loaded with ", f.n_instructions(), " instructions")

    # Generate Pytorch functions for benchmarking
    generatePytorchCode(f, f"{CUSADI_BENCHMARK_DIR}/{f.name()}_PT.py")

    # Generate CPU compiled functions
    c_filepath = f"{CUSADI_BENCHMARK_DIR}/{f.name()}.c"
    so_filepath = f"{CUSADI_BENCHMARK_DIR}/{f.name()}.so"
    f.generate(f"{f.name()}.c")
    os.system(f"mv {f.name()}.c {c_filepath}")
    os.system(f"gcc -fPIC -shared -O3 -march=native {c_filepath} -o {so_filepath}")
    os.system(f"rm {c_filepath}")

    # Generate CPU parallelized functions for benchmarking
    for N_ENVS in N_ENVS_SWEEP:
        print("Generating parallelized CPU code for ", f.name(), " with ", N_ENVS, " environments")
        f_mapped_name = f"{f.name()}_mapped_{N_ENVS}"
        c_filepath = f"{CUSADI_BENCHMARK_DIR}/{f_mapped_name}.c"
        so_filepath = f"{CUSADI_BENCHMARK_DIR}/{f_mapped_name}.so"
        f_mapped = f.map(f"{f_mapped_name}", "openmp", N_ENVS, [], [], {})
        f_mapped.generate(f"{f_mapped_name}.c")
        os.system(f"mv {f_mapped_name}.c {c_filepath}")
        os.system(f"gcc -fPIC -shared -O3 -fopenmp -march=native {c_filepath} -o {so_filepath}")
        os.system(f"rm {c_filepath}")

t_1e3 = {}; t_1e4 = {}; t_1e5 = {}; t_1e6 = {}
time_zero_array = np.zeros((len(N_ENVS_SWEEP), N_EVALS))
benchmark_sizes = ["1e0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]
method_names = ["cusadi", "pytorch", "serial_cpu", "parallel_cpu"]
for s in method_names:
    t_1e3[s] = time_zero_array.copy()
    t_1e4[s] = time_zero_array.copy()
    t_1e5[s] = time_zero_array.copy()
    t_1e6[s] = time_zero_array.copy()
t_data = [t_1e3] # [t_1e3, t_1e4, t_1e5, t_1e6]
benchmark_data = dict(zip(benchmark_casadi_fns, t_data))

def main(args):
    # benchmark_data = dict(zip([fn_1e3, fn_1e4, fn_1e5, fn_1e6],
    #                           [t_1e3, t_1e4, t_1e5, t_1e6]))
    sys.setrecursionlimit(10000)

    for fn, t_data in benchmark_data.items():
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
                _, t_data["cusadi"][i, j] = timeFunction(lambda:
                    runCusadiBenchmark(fn_cusadi, inputs_GPU))
                _, t_data["pytorch"][i, j] = timeFunction(lambda: 
                    runPytorchBenchmark(fn, outputs_GPU, inputs_GPU, work_GPU))
                _, t_data["serial_cpu"][i, j] = timeFunction(lambda:
                    runSerialCPUBenchmark(fn_CPU, inputs_CPU))
                _, t_data["parallel_cpu"][i, j] = timeFunction(lambda:
                    runParallelCPUBenchmark(fn_CPU_parallel, inputs_CPU))
    print(t_1e3)
    data_MATLAB = dict(zip(benchmark_sizes, t_data))
    scipy.io.savemat(f"{CUSADI_BENCHMARK_DIR}/benchmark_data.mat", data_MATLAB)

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

def printParserArguments(parser, args):
    # Print out all arguments, descriptions, and default values in a formatted manner
    print(f"\n{'Argument':<10} {'Description':<80} {'Default':<10} {'Current Value':<10}")
    print("=" * 120)
    for action in parser._actions:
        if action.dest == 'help':
            continue
        arg_strings = ', '.join(action.option_strings)
        description = action.help or 'No description'
        default = action.default if action.default is not argparse.SUPPRESS else 'No default'
        current_value = getattr(args, action.dest, default)
        print(f"{arg_strings:<10} {description:<80} {default:<10} {current_value:<10}")
    print()

def setupParser():
    parser = argparse.ArgumentParser(description='Script to evaluate Cusadi function and check error')
    parser.add_argument('--fn', type=str, dest='fn_name', default='test',
                        help='Function name in cusadi/casadi_functions, defaults to "test"')
    parser.add_argument('--num_envs', type=int, dest='n_envs', default=4000,
                        help='Number of instances to evaluate in parallel, default to 4000')
    return parser

if __name__ == "__main__":
    parser = setupParser()
    args = parser.parse_args()
    printParserArguments(parser, args)
    main(args)


