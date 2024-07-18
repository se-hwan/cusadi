import os, sys
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
sys.path.append(ROOT_DIR)

import torch
import scipy
from src import *
from casadi import *

N_ENVS = 30000
fn_drone_ROA = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "drone_sim_LQR_ROA.casadi"))
step_drone_ROA = CusadiFunction(fn_drone_ROA, N_ENVS)

t_start = 0
t_end = 10
dt = 0.001
N_steps = int((t_end - t_start) / dt)
t = torch.linspace(t_start, t_end, N_steps)

data_MATLAB = {}
data_MATLAB["t"] = t.cpu().numpy()

# State: [x, y, theta, x_dot, y_dot, theta_dot]
tmp = torch.ones((N_ENVS, 1), device='cuda', dtype=torch.float32)
F_lim_sweep = [10*tmp, 20*tmp, 30*tmp, 40*tmp, 50*tmp]
v_max = 20
omega_max = 5

for k in range(len(F_lim_sweep)):
    data_traj = {}
    trajectory_tensor = torch.zeros((N_steps, N_ENVS, 6), device='cuda', dtype=torch.float32)
    omg_mag = omega_max * (2*torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32) - 1)
    vel_angle = torch.pi * (2*torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32) - 1)
    vel_mag = v_max * (2*torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32) - 1)
    v_x = vel_mag * torch.cos(vel_angle)
    v_y = vel_mag * torch.sin(vel_angle)
    initial_state_tensor = torch.hstack([
        torch.zeros((N_ENVS, 3), device='cuda', dtype=torch.float32),
        v_x, v_y, omg_mag])
    state_tensor = initial_state_tensor.clone()
    trajectory_tensor[0, :, :] = state_tensor

    for i in range(1, N_steps):
        fn_input = [state_tensor, F_lim_sweep[k]]
        step_drone_ROA.evaluate(fn_input)
        trajectory_tensor[i, :, :] = step_drone_ROA.outputs_sparse[0]
        state_tensor = trajectory_tensor[i, :, :]
    
    init_lin_mtm = torch.norm(trajectory_tensor[0, :, 3:5], dim=-1)
    init_ang_mtm = trajectory_tensor[0, :, 5]
    successes = (torch.norm(trajectory_tensor[-1, :, :], dim=1) < 1e-3).squeeze()
    failures = (torch.logical_not(successes)).squeeze()
    data_traj["F_lim"] = F_lim_sweep[k].cpu().numpy()
    # data_traj["trajectory"] = trajectory_tensor.cpu().numpy()
    data_traj["init_state"] = initial_state_tensor.cpu().numpy()
    data_traj["init_lin_mtm"] = init_lin_mtm.cpu().numpy()
    data_traj["init_ang_mtm"] = init_ang_mtm.cpu().numpy()
    data_traj["successes"] = successes.cpu().numpy()
    data_traj["failures"] = failures.cpu().numpy()
    data_MATLAB[f"F_lim_{k}"] = data_traj

scipy.io.savemat(f"{CUSADI_DATA_DIR}/drone_ROA_data.mat", data_MATLAB)