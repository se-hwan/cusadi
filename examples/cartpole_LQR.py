import os, sys
from casadi import *

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
sys.path.append(ROOT_DIR)

from cusadi import *
import numpy as np

N_ENVS = 4000
step_cartpole = CusadiFunction("step_cartpole", N_ENVS)

t_start = 0
t_end = 10
# t_sim = 

# Parallel simulations


    # Monte Carlo estimation of LQR region of atrt

    # Randomize across dynamic parameters, fix initial condition


    # Simulations with process noise

