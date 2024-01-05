from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      headers=['lltm.h'],
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
)