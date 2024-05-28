import os

CUSADI_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CUSADI_CLASS_DIR = os.path.join(CUSADI_ROOT_DIR, "cusadi")
CUSADI_FUNCTION_DIR = os.path.join(CUSADI_ROOT_DIR, "cusadi", "casadi_functions")
CUSADI_SCRIPT_DIR = os.path.join(CUSADI_ROOT_DIR, "scripts")
CUSADI_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build")
CUSADI_CODEGEN_DIR = os.path.join(CUSADI_ROOT_DIR, "codegen")