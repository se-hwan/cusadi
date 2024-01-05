from casadi import *
import numpy
import torch
import time
from eval_casadi import eval_casadi_torch
from eval_casadi import eval_casadi_torch_hashed

f = Function.load('test.casadi')

N_ENVS = 4000
N_RUNS = 25
compute_device = 'cuda'

avg_time = 0
avg_time_hashed = 0
for i in range(N_RUNS):
    input_val_cpu = [torch.ones((N_ENVS, 192, 1), device=compute_device),
                    torch.rand((N_ENVS, 52, 1), device=compute_device)]
    [out_cpu, t] = eval_casadi_torch(f, input_val_cpu, compute_device)
    [out_cpu, t_hashed] = eval_casadi_torch_hashed(f, input_val_cpu, compute_device)
    avg_time += t
    avg_time_hashed += t_hashed

print("Computation time for ", N_ENVS, " environments on CPU: ", avg_time/N_RUNS)
print("Computation time for ", N_ENVS, " environments on CPU with hashing: ", avg_time_hashed/N_RUNS)