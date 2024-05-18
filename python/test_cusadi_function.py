import ctypes
import torch
import time
import numpy
from casadi import *
from CusadiFunction import CusADiFunction

N_ENVS = 4000
f = casadi.Function.load("../test.casadi")

input_tensors = [torch.rand(N_ENVS, f.nnz_in(i), device='cuda', dtype=torch.float32).contiguous()
                 for i in range(f.n_in())]

test = CusADiFunction(f, N_ENVS)
test.evaluate(input_tensors)

print(test.outputs_sparse)

output_numpy = numpy.zeros((N_ENVS, f.nnz_out()))
for n in range(N_ENVS):
    inputs_np = [input_tensors[i][n, :].cpu().numpy() for i in range(f.n_in())]
    output_numpy[n, :] = (f.call(inputs_np))[0].nonzeros()

print(output_numpy)