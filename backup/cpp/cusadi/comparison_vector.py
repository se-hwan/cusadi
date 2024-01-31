import math
import torch
import time
from casadi import *

# Our module!
import lltm_cpp
import lltm_cuda

assert torch.cuda.is_available()
device = torch.device("cuda")  # device object representing GPU

#### BENCHMARKING ####

N_ENVS = 4096
N_RUNS = 1000

input_1 = torch.rand(N_ENVS, 192, device=device, dtype=torch.float32)
input_2 = torch.rand(N_ENVS, 52, device=device, dtype=torch.float32)
output = torch.zeros(N_ENVS, 10, device=device, dtype=torch.float32)

col_input_1 = 3
col_input_2 = 1
col_output = 2

# * Evaluate Pytorch add in Python
t_pytorch = 0
for i in range(N_RUNS):
    t0 = time.time()
    output[:, col_output] = input_1[:, col_input_1] + input_2[:, col_input_2]
    torch.cuda.synchronize()
    t_pytorch += time.time() - t0
print("Time taken for Pytorch natively: ", t_pytorch/N_RUNS, " s")


# * Evaluation with C++
t_cpp = 0
for i in range(N_RUNS):
    t0 = time.time()
    lltm_cpp.vectorizedAddCPP(output, col_output,
                              input_1, col_input_1,
                              input_2, col_input_2)
    torch.cuda.synchronize()
    t_cpp += time.time() - t0
print("Time taken for C++: ", t_cpp/N_RUNS, " s")


# * Evaluation with CUDA
t_cuda = 0
for i in range(N_RUNS):
    t0 = time.time()
    lltm_cuda.test_add(output, col_output,
                                input_1, col_input_1,
                                input_2, col_input_2)
    torch.cuda.synchronize()
    t_cuda += time.time() - t0
print("Time taken for CUDA: ", t_cuda/N_RUNS, " s")