import os, sys
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
sys.path.append(ROOT_DIR)

import torch
import scipy
import math
from src import *
from casadi import *
import matplotlib.pyplot as plt

N_ENVS = 4000
N_HORIZON = 12
EYE_3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]

# Inputs for MIT Humanoid SRBM MPC function
#   1. u0                   [144]
#   2. mu_0                 [1]
#   3. Parameters_dyn       [77]
#   4. Parameters_fix       [54]

dt_MPC = [0.025]
g = [9.81]
I = [0.4626, 0.0014, 0.0040, 0.0014, 0.3037, -0.0017, 0.0040, -0.0017, 0.2454]
m = [24.8885]
t_stance = [0.2]
t_swing = [0.1]
z_des = [0.6]
p_limit = [0.3]
raibert_heuristic = [1]
centrifugal_heuristic = [0]
r_body_to_hip = [-0.005653, -0.082, -0.05735,  -0.005653, 0.082, -0.05735]
Q_x = [1, 1, 1,  5, 5, 50,  0.01, 0.01, 0.01,  0.2, 0.2, 0.1]
Q_u = [1e-5]*12
l_heel = [0.031295129]
l_toe = [0.076138708]
mu = [0.6]
f_max = [2000]
f_min = [1]

u_0 = [0, 0, m[0]*g[0]/2, 0, 0, 0, 0, 0, m[0]*g[0]/2, 0, 0, 0] * N_HORIZON
u_0_cusadi = torch.tensor(u_0, device='cuda', dtype=torch.double).repeat(N_ENVS, 1)

mu_0 = [100]
mu_0_cusadi = torch.tensor(mu_0, device='cuda', dtype=torch.double).repeat(N_ENVS, 1)

x_0 = [1,0,0,0,1,0,0,0,1, 3,0,0.6, 0,0,0, 0,0,0]
x_des = [1, 0, 0, 0, 1, 0, 0, 0, 1, 3.0047, 0, 0.6, 0, 0, 0, 1, 0, 0]
p_f = [3.0125772, -0.0801625, -0.146271, 3.0125772, 0.0801625, -0.146271]
contact_prediction = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
p_foot_frame = [0, 0, 0, 0, 0, 0]
R_foot_yaw = [0, 0]
param_dyn = x_0 + x_des + p_f + contact_prediction + p_foot_frame + R_foot_yaw + [0];
param_dyn_cusadi = torch.tensor(param_dyn, device='cuda', dtype=torch.double).repeat(N_ENVS, 1)

param_fixed = dt_MPC + g + I + m + t_stance + t_swing + z_des + p_limit \
    + raibert_heuristic + centrifugal_heuristic + r_body_to_hip + Q_x \
    + Q_u + [1] + l_heel + l_toe + mu + f_max + f_min
param_fixed_cusadi = torch.tensor(param_fixed, device='cuda', dtype=torch.double).repeat(N_ENVS, 1)

fn_inputs_CPU = [u_0, mu_0, param_dyn, param_fixed]
fn_inputs_cusadi = [u_0_cusadi, mu_0_cusadi, param_dyn_cusadi, param_fixed_cusadi]


# print("Param fixed:", param_fixed)


# Function loading and evaluation
fn_filepath = os.path.join(CUSADI_FUNCTION_DIR, "SRBM_MPC_5.casadi")
# fn_filepath = os.path.join(CUSADI_FUNCTION_DIR, "test_conditional.casadi")
f = casadi.Function.load(fn_filepath)
print("Evaluating function:", f.name())
print("Function has %d arguments" % f.n_in())
print("Function has %d outputs" % f.n_out())

fn_cusadi = CusadiFunction(f, N_ENVS)
import time
start = time.time()
fn_cusadi.evaluate(fn_inputs_cusadi)
print("Time taken for evaluation:", time.time() - start)
print(f.call(fn_inputs_CPU))
print(fn_cusadi.outputs_sparse[0])
print("Function evaluation complete.")