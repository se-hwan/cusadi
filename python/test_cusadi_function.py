import ctypes
import torch
import time
import numpy
from casadi import *
from CusadiFunction import CusADiFunction

N_ENVS = 4000
f = casadi.Function.load("../inertial_quantities.casadi")
print("Function has %d arguments" % f.n_in())
print("Function has %d outputs" % f.n_out())

input_tensors = [torch.rand(N_ENVS, f.nnz_in(i), device='cuda', dtype=torch.float32).contiguous()
                 for i in range(f.n_in())]

test = CusADiFunction(f, N_ENVS)
test.evaluate(input_tensors)

output_numpy = [numpy.zeros((N_ENVS, f.nnz_out(i))) for i in range(f.n_out())]
for n in range(N_ENVS):
    inputs_np = [input_tensors[i][n, :].cpu().numpy() for i in range(f.n_in())]
    for i in range(f.n_out()):
        output_numpy[i][n, :] = f.call(inputs_np)[i].nonzeros()



print("CUSASDI AND CASADI EVALUATION COMPARISON:")
for i in range(f.n_out()):
    print(f"Output {i}:")
    print("     CusADi:")
    print(test.outputs_sparse[i][0])
    print("     CasADi:")
    print(output_numpy[i][0])
