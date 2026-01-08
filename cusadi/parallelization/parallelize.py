from .kernel_codegen import get_functions, build_kernel, build_pybind
from .utils.jit_kernels import compile_and_load_kernels

def parallelize(fn_names='all', batch_size=1, precision='float'):
    casadi_fns = get_functions(fn_names)
    for fn in casadi_fns:
        build_kernel(fn, batch_size=batch_size, precision=precision)
        build_pybind(fn, precision=precision)
    compile_and_load_kernels(fn_names)