import ctypes
import torch
import time
from casadi import *
from evaluateCasADiPython import evaluateCasADiPython

f = casadi.Function.load("../test.casadi")

# Load the shared library
libcusadi = ctypes.CDLL('../build/libcusadi.so')
libcusadi.evaluate.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),         # Inputs
    ctypes.POINTER(ctypes.c_float),                         # Work vector
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),         # Outputs
    ctypes.c_int                                            # Number of outputs
]

assert torch.cuda.is_available()
device = torch.device("cuda")  # device object representing GPU
print(device)

def castAsCPointer(ptr, type='float'):
    if type == 'int':
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))
    elif type == 'float':
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))

# Single evaluation test
N_ENVS = 4096

print("Num inputs: ", f.n_in())
print("Num nnz for each input: ", [f.nnz_in(i) for i in range(f.n_in())])
print("Num outputs: ", f.n_out())
print("Num nnz for each output: ", [f.nnz_out(i) for i in range(f.n_out())])

input_1_pt = torch.ones((N_ENVS, 192), device='cuda', dtype=torch.float32).contiguous()
input_2_pt = torch.ones((N_ENVS, 52), device='cuda', dtype=torch.float32).contiguous()
input_1_ptr = castAsCPointer(input_1_pt.data_ptr())
input_2_ptr = castAsCPointer(input_2_pt.data_ptr())
input_ptrs = torch.zeros(f.n_in(), device='cuda', dtype=torch.int64).contiguous()
input_ptrs[0] = input_1_pt.data_ptr()
input_ptrs[1] = input_2_pt.data_ptr()
fn_input = castAsCPointer(input_ptrs.data_ptr(), 'int')

work_tensor = torch.zeros(N_ENVS, f.sz_w(), device='cuda', dtype=torch.float32).contiguous()
fn_work = ctypes.cast((work_tensor.data_ptr()), ctypes.POINTER(ctypes.c_float))

outputs = [torch.zeros(N_ENVS, f.nnz_out(i), device='cuda', dtype=torch.float32) for i in range(f.n_out())]
outputs_ptrs = [castAsCPointer(outputs[i].data_ptr()) for i in range(f.n_out())]
fn_output = (ctypes.POINTER(ctypes.c_float) * f.n_out())(*outputs_ptrs)

def evaluateCodegen():
    for i in range(f.n_out()):
        outputs[i] *= 0
    # work_tensor *= 0
    libcusadi.evaluate(fn_input,
                       fn_work,
                       fn_output,
                       N_ENVS)

test_input = torch.zeros(2, device='cuda', dtype=torch.int64).contiguous()
test_input[0] = input_1_pt.data_ptr()
test_input[1] = input_2_pt.data_ptr()
test_input_ptr = castAsCPointer(test_input.data_ptr(), 'int')
libcusadi.test_inputs(fn_input, N_ENVS)
print(input_1_pt)

# print(work_tensor)
# print(outputs[0])
# evaluateCodegen()
# print(work_tensor)
# print(outputs[0])