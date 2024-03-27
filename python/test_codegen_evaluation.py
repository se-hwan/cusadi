import ctypes
import torch
import time
from casadi import *
from evaluateCasADiPython import evaluateCasADiPython

f = casadi.Function.load("../test.casadi")

# Load the shared library
libcusadi = ctypes.CDLL('../build/libcusadi.so')
# libcusadi.evaluate.argtypes = [
#     ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),         # Inputs
#     ctypes.POINTER(ctypes.c_float),                         # Work vector
#     ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),         # Outputs
#     ctypes.c_int                                            # Number of outputs
# ]

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

input_1_pt = torch.rand((N_ENVS, 192), device='cuda', dtype=torch.float32).contiguous()
input_2_pt = torch.rand((N_ENVS, 52), device='cuda', dtype=torch.float32).contiguous()
input_1_ptr = castAsCPointer(input_1_pt.data_ptr())
input_2_ptr = castAsCPointer(input_2_pt.data_ptr())
input_ptrs = torch.zeros(f.n_in(), device='cuda', dtype=torch.int64).contiguous()
input_ptrs[0] = input_1_pt.data_ptr()
input_ptrs[1] = input_2_pt.data_ptr()
fn_input = castAsCPointer(input_ptrs.data_ptr(), 'int')

work_tensor = torch.zeros(N_ENVS, f.sz_w(), device='cuda', dtype=torch.float32).contiguous()
fn_work = ctypes.cast((work_tensor.data_ptr()), ctypes.POINTER(ctypes.c_float))

outputs = [torch.zeros(N_ENVS, f.nnz_out(i), device='cuda', dtype=torch.float32) for i in range(f.n_out())]
output_ptrs = torch.zeros(f.n_out(), device='cuda', dtype=torch.int64).contiguous()
for i in range(f.n_out()):
    output_ptrs[i] = outputs[i].data_ptr()
fn_output = castAsCPointer(output_ptrs.data_ptr(), 'int')

def evaluateCodegen():
    for i in range(f.n_out()):
        outputs[i] *= 0
    # work_tensor *= 0
    libcusadi.evaluate(fn_input,
                       fn_work,
                       fn_output,
                       N_ENVS)


input_1_np = input_1_pt[0, :].cpu().numpy()
input_2_np = input_2_pt[0, :].cpu().numpy()
input_np = [input_1_np, input_2_np]
out_np = f.call(input_np)
out_np = out_np[0].nonzeros()

evaluateCodegen()
print(work_tensor)
print(outputs[0][0, :].cpu().numpy())
print(sum(outputs[0][0, :].cpu().numpy()))
print(sum(out_np))
print("Error: ", sum(out_np - outputs[0][0, :].cpu().numpy()))