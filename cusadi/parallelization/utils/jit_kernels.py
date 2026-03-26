import os
import glob
import sys
import importlib
import torch
from torch.utils.cpp_extension import load

# TODO: detect GPU architecture
def compile_and_load_kernels(kernel_names):
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        print(f"\n**********Compiling kernels for detected CUDA architecture {major}.{minor}**********")
    parallel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    codegen_dir = os.path.join(parallel_dir, 'codegen')
    project_dir = os.path.dirname(os.path.dirname(parallel_dir))
    build_dir = os.path.join(project_dir, 'build_kernels')
    
    # Make build directory if it doesn't exist:
    if not os.path.exists(f"{build_dir}"):
        os.makedirs(f"{build_dir}")
    print("Loading JIT kernels from: ", codegen_dir)
    print("Build directory: ", f"{build_dir}")

    kernel_sources = []
    for name in sorted(kernel_names):
        kernel_sources.append(os.path.join(codegen_dir, f'{name}.cu'))
    
    if len(kernel_sources) == 0:
        print("No kernel sources found. Skipping JIT compilation.")
        return None
    else:
        kernel_sources.append(os.path.join(codegen_dir, 'bindings.cpp'))
        module = load(name='cusadi_kernels',
                      sources=kernel_sources,
                      extra_cflags=['-O3',
                                    # '-march=native'
                                    ],
                      extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_86'],
                    #   extra_cuda_cflags=['-O3', '-arch=sm_86'],
                      verbose=True,
                      is_python_module=True,
                      build_directory=f"{project_dir}/build_kernels"
                      )
        # torch may bump the loaded name to cusadi_kernels_vN when inputs change.
        # Keep a stable alias so `import cusadi_kernels` always points to latest.
        sys.modules['cusadi_kernels'] = module
        importlib.invalidate_caches()
        return module

def compile_and_load_cudss():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        print(f"\n**********Compiling cuDSS for detected CUDA architecture {major}.{minor}**********")
    parallel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.dirname(os.path.dirname(parallel_dir))
    build_dir = os.path.join(project_dir, 'build_cudss')
    # Make build directory if it doesn't exist:
    if not os.path.exists(f"{build_dir}"):
        os.makedirs(f"{build_dir}")

    cudss_source = [os.path.join(parallel_dir, 'utils', 'cudss_binding.cpp'),
                    os.path.join(parallel_dir, 'utils', 'cudss_interface.cu'),
                    os.path.join(parallel_dir, 'utils', 'cuda_utils.cu'),]
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    load(name='cudss',
         sources=cudss_source,
         extra_cflags=['-O3', '-march=native'],
         extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_86'],
         extra_include_paths=[f"{cuda_home}/include",],
         extra_ldflags=[f"-L{cuda_home}/lib64", "-lcudss"],
         verbose=True,
         is_python_module=True,
         build_directory=f"{project_dir}/build_cudss"
    )
