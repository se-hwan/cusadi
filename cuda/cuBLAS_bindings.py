import ctypes
import torch

# Load the shared library
libcublas = ctypes.CDLL('./libcuBLAS_example.so')

# Define argument and return types for the C function
libcublas.add_vectors_cublas.argtypes = [ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.c_int]
libcublas.add_vectors_cublas.restype = None

# Define argument and return types for the C function
libcublas.add_vectors_loop.argtypes = [ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.c_int,
                                         ctypes.c_int]
libcublas.add_vectors_loop.restype = None

# Load the shared library
libcasadi = ctypes.CDLL('./libcasadi_eval.so')

# Define argument and return types for the C function
libcasadi.evaluate_casadi_function.argtypes = [ctypes.c_int, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float)]
libcasadi.evaluate_casadi_function.restype = None


def ptr_to_float_pointer(ptr):
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))

# Python wrapper function
def add_vectors(handle, a_ptr, c_ptr, n):
    a_float_ptr = ptr_to_float_pointer(a_ptr)
    c_float_ptr = ptr_to_float_pointer(c_ptr)
    libcublas.add_vectors_cublas(handle, a_float_ptr, c_float_ptr, n)

# Python wrapper function
def add_vectors_loop(handle, a_ptr, c_ptr, n_loop, n):
    a_float_ptr = ptr_to_float_pointer(a_ptr)
    c_float_ptr = ptr_to_float_pointer(c_ptr)
    libcublas.add_vectors_loop(handle, a_float_ptr, c_float_ptr, n_loop, n)

# Python wrapper function
def evaluate_casadi_function(num_instr, batch_size, a_ptr, c_ptr):
    a_float_ptr = ptr_to_float_pointer(a_ptr)
    c_float_ptr = ptr_to_float_pointer(c_ptr)
    libcasadi.evaluate_casadi_function(num_instr, batch_size, a_float_ptr, c_float_ptr)
