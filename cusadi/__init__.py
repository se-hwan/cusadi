from importlib import import_module
import os

# List the submodules you want exposed under mpc_suite.*
_submodules = [
    "visualization",
    "optimization",
    "utils",
    "models",
    "parallelization",
]

# Dynamically import each submodule and add it to the package namespace
for _name in _submodules:
    module = import_module(f"{__name__}.{_name}")
    globals()[_name] = module

# Define constant filepaths
CUSADI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CUSADI_DIR)
PARALLEL_DIR = os.path.join(CUSADI_DIR, 'parallelization')
FUNCTION_DIR = os.path.join(PARALLEL_DIR, 'casadi_fns')
CODEGEN_DIR = os.path.join(PARALLEL_DIR, 'codegen')

# Clean up
del import_module, _submodules, _name, module