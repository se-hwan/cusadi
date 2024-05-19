import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

import torch
import numpy
from casadi import *
from CusadiFunction import CusADiFunction

N_ENVS = 4000
f = casadi.Function.load("test.casadi")
print("Function has %d arguments" % f.n_in())
print("Function has %d outputs" % f.n_out())

input_tensors = [torch.rand(N_ENVS, f.nnz_in(i), device='cuda', dtype=torch.float32).contiguous()
                 for i in range(f.n_in())]

libcusadi_path = os.path.join(current_dir, "../../build/libcusadi.so")
test = CusADiFunction(f, N_ENVS, libcusadi_path)
test.evaluate(input_tensors)

output_numpy = [numpy.zeros((N_ENVS, f.nnz_out(i))) for i in range(f.n_out())]
for n in range(N_ENVS):
    inputs_np = [input_tensors[i][n, :].cpu().numpy() for i in range(f.n_in())]
    for i in range(f.n_out()):
        output_numpy[i][n, :] = f.call(inputs_np)[i].nonzeros()

print("CUSASDI AND CASADI EVALUATION COMPARISON:")
for i in range(f.n_out()):
    error_norm = numpy.linalg.norm(test.outputs_sparse[i].cpu().numpy() - output_numpy[i])
    print(f"Output {i} error norm:", error_norm)
