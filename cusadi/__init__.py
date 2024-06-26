import os

CUSADI_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CUSADI_CLASS_DIR = os.path.join(CUSADI_ROOT_DIR, "cusadi")
CUSADI_FUNCTION_DIR = os.path.join(CUSADI_ROOT_DIR, "cusadi", "casadi_functions")
CUSADI_BENCHMARK_DIR = os.path.join(CUSADI_ROOT_DIR, "cusadi", "benchmark_functions")
CUSADI_SCRIPT_DIR = os.path.join(CUSADI_ROOT_DIR, "scripts")
CUSADI_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build")
CUSADI_CODEGEN_DIR = os.path.join(CUSADI_ROOT_DIR, "codegen")

from .CusadiFunction import CusadiFunction
from .CusadiOperations import OP_CUDA_DICT, OP_PYTORCH_DICT
from .generateCUDACode import generateCUDACode, generateCMakeLists
from .generatePytorchCode import generatePytorchCode
from .benchmark_functions import *