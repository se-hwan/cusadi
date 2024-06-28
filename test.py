import os, sys
import torch
import numpy as np
import scipy
from casadi import *
from cusadi import *


# Load functions for benchmarking
fn_filepath_1e6 = os.path.join(CUSADI_BENCHMARK_DIR, "fn_1e6.casadi")
fn_1e6 = casadi.Function.load(fn_filepath_1e6)
N_INSTRUCTIONS = []

generateCUDACodeV3(fn_1e6)

