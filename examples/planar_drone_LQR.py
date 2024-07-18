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

N_ENVS_PER_INIT = 50
N_ENVS = 3 * N_ENVS_PER_INIT
fn_drone_LQR = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "drone_sim_LQR_sweep.casadi"))
step_drone_LQR = CusadiFunction(fn_drone_LQR, N_ENVS)

t_start = 0
t_end = 15
dt = 0.001
N_steps = int((t_end - t_start) / dt)
t = torch.linspace(t_start, t_end, N_steps)
F_lim = 50*torch.ones((N_ENVS, 1), device='cuda', dtype=torch.float32)

data_MATLAB = {}
data_MATLAB["t"] = t.cpu().numpy()

Q_default = torch.tensor([1, 1, 1, 1, 1, 1], device='cuda', dtype=torch.float32).repeat(N_ENVS, 1)
R_default = torch.tensor([1, 1], device='cuda', dtype=torch.float32).repeat(N_ENVS, 1)
mass_default = torch.tensor([1], device='cuda', dtype=torch.float32).repeat(N_ENVS, 1)

initial_states = torch.vstack([
    torch.tensor((-2, -2, 0, 0, 0, 0), device='cuda', dtype=torch.float32).repeat(N_ENVS_PER_INIT, 1),
    torch.tensor((1, -1, 0, 0, 0, 0), device='cuda', dtype=torch.float32).repeat(N_ENVS_PER_INIT, 1),
    torch.tensor((-1, 2, 0, 0, 0, 0), device='cuda', dtype=torch.float32).repeat(N_ENVS_PER_INIT, 1),
    ])

# BASELINE:
data_baseline = {}
state_tensor = initial_states.clone()
traj_baseline = torch.zeros((N_steps, N_ENVS, 6), device='cuda', dtype=torch.float32)
traj_baseline[0, :, :] = initial_states
for i in range(1, N_steps):
    fn_input = [state_tensor, Q_default, R_default, F_lim, mass_default]
    step_drone_LQR.evaluate(fn_input)
    traj_baseline[i, :, :] = step_drone_LQR.outputs_sparse[0]
    state_tensor = traj_baseline[i, :, :]
data_baseline["traj_baseline"] = traj_baseline.cpu().numpy()

# EXAMPLE 1: Sweep over Q_x
data_Q_x = {}
state_tensor = initial_states.clone()
traj_Q_x = torch.zeros((N_steps, N_ENVS, 6), device='cuda', dtype=torch.float32)
traj_Q_x[0, :, :] = initial_states
Q_x_sweep = Q_default.clone()
Q_x_sweep[:, 0] = torch.logspace(math.log10(0.05), math.log10(20), N_ENVS_PER_INIT,
                                 device='cuda', dtype=torch.float32).repeat(1, 3)
for i in range(1, N_steps):
    fn_input = [state_tensor, Q_x_sweep, R_default, F_lim, mass_default]
    step_drone_LQR.evaluate(fn_input)
    traj_Q_x[i, :, :] = step_drone_LQR.outputs_sparse[0]
    state_tensor = traj_Q_x[i, :, :]
data_Q_x["traj_Q_x"] = traj_Q_x.cpu().numpy()
data_Q_x["Q_sweep"] = Q_x_sweep.cpu().numpy()

# EXAMPLE 2: Sweep over m
data_mass = {}
state_tensor = initial_states.clone()
traj_mass = torch.zeros((N_steps, N_ENVS, 6), device='cuda', dtype=torch.float32)
traj_mass[0, :, :] = initial_states
mass_sweep = mass_default.clone()
mass_sweep[:, 0] = torch.logspace(math.log10(0.25), math.log10(4), N_ENVS_PER_INIT,
                                  device='cuda', dtype=torch.float32).repeat(1, 3)
for i in range(1, N_steps):
    fn_input = [state_tensor, Q_default, R_default, F_lim, mass_sweep]
    step_drone_LQR.evaluate(fn_input)
    traj_mass[i, :, :] = step_drone_LQR.outputs_sparse[0]
    state_tensor = traj_mass[i, :, :]
data_mass["traj_mass"] = traj_mass.cpu().numpy()
data_mass["mass_sweep"] = mass_sweep.cpu().numpy()

# EXAMPLE 3: Sweep over R
data_R = {}
state_tensor = initial_states.clone()
traj_R = torch.zeros((N_steps, N_ENVS, 6), device='cuda', dtype=torch.float32)
traj_R[0, :, :] = initial_states
R_sweep = R_default.clone()
R_sweep[:, 0] = torch.logspace(math.log10(0.01), math.log10(100), N_ENVS_PER_INIT,
                               device='cuda', dtype=torch.float32).repeat(1, 3)
for i in range(1, N_steps):
    fn_input = [state_tensor, Q_default, R_sweep, F_lim, mass_default]
    step_drone_LQR.evaluate(fn_input)
    traj_R[i, :, :] = step_drone_LQR.outputs_sparse[0]
    state_tensor = traj_R[i, :, :]
data_R["traj_R"] = traj_R.cpu().numpy()
data_R["R_sweep"] = R_sweep.cpu().numpy()

# SAVE DATA
data_MATLAB["baseline"] = data_baseline
data_MATLAB["Q_x"] = data_Q_x
data_MATLAB["R"] = data_R
data_MATLAB["mass"] = data_mass
data_MATLAB["F_lim"] = F_lim.cpu().numpy()
data_MATLAB["Q_default"] = Q_default.cpu().numpy()
data_MATLAB["R_default"] = R_default.cpu().numpy()
data_MATLAB["mass_default"] = mass_default.cpu().numpy()
scipy.io.savemat(f"{CUSADI_DATA_DIR}/drone_LQR_data.mat", data_MATLAB)