cmake_minimum_required(VERSION 3.15)

# Set project name
project(MyProject)

# Find CUDA package
include(CheckLanguage)
check_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  message("CUDA found")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86)
endif()
message(${CMAKE_CUDA_ARCHITECTURES})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Set source files
set(CUDA_SOURCES
    src/cuda_operations.cu
)

set(C_SOURCES
    src/main.c)

# Compile the CUDA file separately
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_86 --use_fast_math)  # Adjust architecture as needed


# Include directories for your header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Compile the C file separately
add_library(cuda_libs SHARED ${CUDA_SOURCES})
add_library(c_libs SHARED ${C_SOURCES})

target_link_libraries(c_libs cuda_libs)