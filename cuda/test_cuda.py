from cuBLAS_bindings import add_vectors, add_vectors_loop
import torch
import time
import cupy as cp
import ctypes

start = cp.cuda.Event()
end = cp.cuda.Event()

# Example data on GPU
N_ENVS = 4096
a_device = torch.rand((N_ENVS, 5), device='cuda', dtype=torch.float32)
b_device = torch.rand((N_ENVS, 5), device='cuda', dtype=torch.float32)
c_device = b_device.clone()

a_device = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda', dtype=torch.float32)
c_device = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda', dtype=torch.float32)

handle = ctypes.c_void_p(torch.cuda.current_blas_handle())
print(a_device[0:3, :])
print(c_device[0:3, :])
add_vectors(handle, a_device.data_ptr(), c_device.data_ptr(), 2)
print(c_device[0:3, :])


t_pytorch = 0
t0 = time.time()
start.record()
add_vectors_loop(handle, a_device.data_ptr(), c_device.data_ptr(), 10000, 2)
end.record()
end.synchronize()
t_pytorch += cp.cuda.get_elapsed_time(start, end)/1000
print("cublas looped wall time: ", time.time()-t0)

t_pytorch = 0
t0 = time.time()
for i in range(10000):
    start.record()
    c = torch.add(a_device, b_device)
    end.record()
    end.synchronize()
    t_pytorch += cp.cuda.get_elapsed_time(start, end)/1000
print("Pytorch wall time: ", time.time()-t0)
print("Pytorch add time: ", t_pytorch)


t_cublas = 0
t0 = time.time()
for i in range(10000):
    start.record()
    add_vectors(handle, a_device.data_ptr(), c_device.data_ptr(), N_ENVS)
    end.record()
    end.synchronize()
    t_cublas += cp.cuda.get_elapsed_time(start, end)/1000
print("CUBLAS wall time: ", time.time()-t0)
print("CUBLAS time: ", t_cublas)

t0 = time.time()
for i in range(N_ENVS):
    c[i] = a_device[i] + b_device[i]
t1 = time.time()
print("Pytorch add time serial: ", t1-t0)


# from cuBLAS_bindings import evaluate_casadi_function
# t0 = time.time()
# evaluate_casadi_function(10000, N_ENVS, a_device.data_ptr(), c_device.data_ptr())
# t1 = time.time()
# print("cublas casadi time: ", t1-t0)

# t0 = time.time()
# for i in range(10000):
#     c = torch.add(a_device, b_device)
# t1 = time.time()
# print("python for loop: ", t1-t0)


# t0 = time.time()
# for i in range(10000):
#     add_vectors(a_device.data_ptr(), c_device.data_ptr(), N_ENVS)
# t1 = time.time()
# print("cublas serial: ", t1-t0)


# # # Copy result back to host for display
# # print("Result a:", a_device)
# # print("Result b:", b_device)
# # print("Result c:", c_device)
