import os
import glob
from torch.utils.cpp_extension import load

# TODO: detect GPU architecture
def compile_and_load_kernels(fn_names='all'):
    parallel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    codegen_dir = os.path.join(parallel_dir, 'codegen')
    project_dir = os.path.dirname(os.path.dirname(parallel_dir))
    print("Loading JIT kernels from: ", codegen_dir)
    print("Build directory: ", f"{project_dir}/build")

    sources = [os.path.join(codegen_dir, 'bindings.cpp')]
    if fn_names == 'all':
        sources.extend(glob.glob(codegen_dir + '/*.cu'))
    else:
        for fn_name in fn_names:
            sources.append(os.path.join(codegen_dir, f'{fn_name}.cu'))

    load(name='cusadi_kernels',
         sources=sources,
         extra_cflags=['-O3', '-march=native'],
         extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_86'],
         verbose=True,
         is_python_module=True,
         build_directory=f"{project_dir}/build"
    )