import ctypes
import torch
import time
import cupy as cp

# Load the shared library
libcublas = ctypes.CDLL('./build/libcuda_operations.so')

# Define argument and return types for the C function
libcublas.add_cuda.argtypes = [ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.c_int]
libcublas.add_cuda.restype = None


def ptr_to_float_pointer(ptr):
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))

# Python wrapper function
def add_vectors(c_ptr, a_ptr, b_ptr, n):
    a_float_ptr = ptr_to_float_pointer(a_ptr)
    b_float_ptr = ptr_to_float_pointer(b_ptr)
    c_float_ptr = ptr_to_float_pointer(c_ptr)
    libcublas.add_cuda(c_float_ptr, a_float_ptr, b_float_ptr, n)

# Python wrapper function
def add_vectors_loop(handle, a_ptr, c_ptr, n_loop, n):
    a_float_ptr = ptr_to_float_pointer(a_ptr)
    c_float_ptr = ptr_to_float_pointer(c_ptr)
    libcublas.add_vectors_loop(handle, a_float_ptr, c_float_ptr, n_loop, n)



# Example data on GPU
N_ENVS = 4096
a_device = torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32)
b_device = torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32)
c_device = b_device.clone()
c_empty = torch.zeros_like(c_device)

handle = ctypes.c_void_p(torch.cuda.current_blas_handle())



start = cp.cuda.Event()
end = cp.cuda.Event()

N_RUNS = 10

t_pytorch = 0
t0 = time.time()
for i in range(N_RUNS):
    start.record()
    c = torch.add(a_device, b_device)
    end.record()
    end.synchronize()
    t_pytorch += cp.cuda.get_elapsed_time(start, end)/1000
print("Pytorch wall time: ", time.time()-t0)
print("Pytorch add time: ", t_pytorch/N_RUNS)


t_cublas = 0
t0 = time.time()
for i in range(N_RUNS):
    start.record()
    add_vectors(c_empty.data_ptr(), a_device.data_ptr(), c_device.data_ptr(), N_ENVS)
    end.record()
    end.synchronize()
    t_cublas += cp.cuda.get_elapsed_time(start, end)/1000
print("CUBLAS wall time: ", time.time()-t0)
print("CUBLAS time: ", t_cublas/N_RUNS)