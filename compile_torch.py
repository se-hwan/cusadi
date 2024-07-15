import os, sys
import argparse
import torch
import numpy as np
from casadi import *
from src import *
from src.benchmark_functions import *

N_ENVS = 4000
N_ITERS = 20

def timeFunction(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

sys.setrecursionlimit(10000)
fn_filepath = os.path.join(CUSADI_FUNCTION_DIR, "test.casadi")
fn = casadi.Function.load(fn_filepath)
inputs_GPU = [torch.rand(N_ENVS, fn.nnz_in(i), device='cuda',
              dtype=torch.float32).contiguous() for i in range(fn.n_in())]
outputs_GPU = [torch.zeros(N_ENVS, fn.nnz_out(), device='cuda') for i in range(fn.n_out())]
work_GPU = torch.zeros(N_ENVS, fn.sz_w(), device='cuda')

compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        _, compile_time = timeFunction(lambda: eval_pt.evaluate_fn_1e3(outputs_GPU, inputs_GPU, work_GPU))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
print("~" * 10)

