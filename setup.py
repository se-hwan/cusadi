from setuptools import setup, find_packages
import os
import glob
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# TODO: detect GPU architecture
def get_extensions():
    ext_modules = [] # C++ modules for Pybind
    compile_args = {
        'cxx': ['-O3', '-march=native', '-fPIC', '-Wno-attributes', '-lineinfo'],
        'nvcc': ['-O3', '--use_fast_math', '-arch=sm_86', '--threads=4'],
    } # ! TODO: add additional architectures and options for different graphics cards
    link_args=['-Wl,--no-as-needed', '-lcuda', '-lcudss']

    # Cusadi functions
    PARALLEL_DIR = 'cusadi/parallelization/'
    include_dirs=[os.path.abspath(PARALLEL_DIR + 'utils')]
    cusadi_sources = [*glob.glob(PARALLEL_DIR + 'codegen/*.cu'),
                      *glob.glob(PARALLEL_DIR + 'codegen/*.cpp')]
    cudss_sources = [PARALLEL_DIR + '/utils/cudss_interface.cu']
    cusadi_extension = CppExtension(name='cusadi.kernels',
                                     sources=cusadi_sources, 
                                     extra_compile_args=compile_args,
                                     extra_link_args=link_args,
                                     include_dirs=include_dirs,)
    cudss_extension = CUDAExtension(name='cusadi.cudss',
                                    sources=cudss_sources,
                                    extra_compile_args=compile_args,
                                    extra_link_args=link_args,
                                    include_dirs=include_dirs,)
    ext_modules.append(cusadi_extension)
    # ext_modules.append(cudss_extension)
    return ext_modules

setup(
    name="cusadi",
    version="2.0",
    author="Se Hwan Jeon",
    author_email="sehwan@mit.edu",
    description="MPC control suite with GPU acceleration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # ext_modules=get_extensions(),
    # cmdclass={"build_ext": BuildExtension},
    url="https://github.com/se-hwan/cusadi",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

# GPU parallelization things:

    # try:
    #     import glob
    #     import yaml
    #     from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    # except ImportError:
    #     raise RuntimeError("Building this package requires torch. Please install torch first.")

    # # Read and parse build config
    # with open("build_config.yaml", "r") as f:
    #     build_config = yaml.safe_load(f)
    # enabled_systems = build_config.get("systems", [])
    # if enabled_systems is None:
    #     enabled_systems = []

    # ext_modules = [] # C++ modules for Pybind
    # compile_args = {
    #     'cxx': ['-g', '-O3', '-march=native', '-fPIC', '-Wno-attributes',],
    #     'nvcc': ['-O3', '--use_fast_math', '-arch=sm_86'],
    # } # ! TODO: add additional architectures and options for different graphics cards
    # link_args=['-Wl,--no-as-needed', '-lcuda', '-lcudss']
    # include_dirs=[os.path.abspath('gpu_mpc/utils')] 

    ############################################
    ############### CusADi build ###############
    ############################################
    # if 'cusadi' in build_config:
    #     cusadi_config = build_config['cusadi']
    #     build_cudss_interface = cusadi_config['build_cudss_interface']
    #     build_cusadi_tests = cusadi_config['build_tests'] # ! TODO

    #     cusadi_sources = []
    #     for system in enabled_systems:
    #         kernel_sources = glob.glob(f'gpu_mpc/systems/{system}/codegen/*.cu')
    #         binding_sources = glob.glob(f'gpu_mpc/systems/{system}/codegen/*.cpp')
    #         cusadi_sources.extend(kernel_sources)
    #         cusadi_sources.extend(binding_sources)
    #     cusadi_extension = CUDAExtension(name='gpu_mpc.cusadi.kernels', sources=cusadi_sources, 
    #                                      extra_compile_args=compile_args, extra_link_args=link_args,
    #                                      include_dirs=include_dirs,)
    #     ext_modules.append(cusadi_extension)
    #     if build_cudss_interface: # ! For now, assuming cudss is tied to cusadi. May not always be the case
    #         cudss_sources = ['gpu_mpc/utils/cudss/cudssInterface_binding.cpp',
    #                          'gpu_mpc/utils/cudss/cudssInterface.cu']
    #         cudss_extension = CUDAExtension(name='gpu_mpc.cusadi.cudss', sources=cudss_sources,
    #                                         extra_compile_args=compile_args, extra_link_args=link_args,
    #                                         include_dirs=include_dirs,)
    #         ext_modules.append(cudss_extension)
            
    # # JAX
    # if 'jaxadi' in build_config:
    #     jaxadi_config = build_config['jaxadi']
    # # Taichi
    # if 'taisadi' in build_config:
    #     taisadi_config = build_config['taisadi']