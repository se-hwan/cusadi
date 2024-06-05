import sys, os
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(TEST_DIR)
CUSADI_ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.append(CUSADI_ROOT_DIR)
import torch
import numpy
from casadi import *
from cusadi.CusadiFunction import CusadiFunction

N_ENVS = 20000
f = casadi.Function.load("eigvals.casadi")
print(f)
print("Function name: ", f.name())
print("Function has %d arguments" % f.n_in())
print("Function has %d outputs" % f.n_out())

for i in range(f.n_in()):
    print("Input %d has %d nonzeros" % (i, f.nnz_in(i)))

input_tensors = [torch.rand(N_ENVS, f.nnz_in(i), device='cuda', dtype=torch.float32).contiguous()
                 for i in range(f.n_in())]

# print("INPUTS:")
# for i in range(f.n_in()):
#     print("CUSADI:  ", input_tensors[i])

test = CusadiFunction(f, N_ENVS)
print("Evaluating...")
test.evaluate(input_tensors)
print("Evaluation complete")

output_numpy = [numpy.zeros((N_ENVS, f.nnz_out(i))) for i in range(f.n_out())]
for n in range(N_ENVS):
    inputs_np = [input_tensors[i][n, :].cpu().numpy() for i in range(f.n_in())]
    for i in range(f.n_out()):
        output_numpy[i][n, :] = f.call(inputs_np)[i].nonzeros()

# print("OUTPUTS:")
# for i in range(f.n_out()):
#     print("CUSADI:  ", test.outputs_sparse[i])
#     print("CASADI:  ", output_numpy[i])

# print("Work tensor: ", test._work_tensor)

print("CUSADI AND CASADI EVALUATION COMPARISON:")
for i in range(f.n_out()):
    error_norm = numpy.linalg.norm(test.outputs_sparse[i].cpu().numpy() - output_numpy[i])
    print(f"Output {i} error norm:", error_norm)
