import os

CUSADI_CLASS_DIR = os.path.dirname(os.path.abspath(__file__))
CUSADI_ROOT_DIR = os.path.join(CUSADI_CLASS_DIR, "..", "cusadi")
CUSADI_SCRIPT_DIR = os.path.join(CUSADI_ROOT_DIR, "scripts")
CUSADI_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build")
CUSADI_CODEGEN_DIR = os.path.join(CUSADI_ROOT_DIR, "codegen")