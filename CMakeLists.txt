cmake_minimum_required(VERSION 3.15)
project(CusADi)

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

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_86 --use_fast_math)  # Adjust architecture as needed

set(INERTIAL_QUANTITIES_SOURCE codegen/inertial_quantities.cu)
set(TEST_SOURCE codegen/test.cu)
set(CHECK_SOURCE codegen/check.cu)
set(EIGVALS_SOURCE codegen/eigvals.cu)

# Include directories for your header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(inertial_quantities SHARED ${INERTIAL_QUANTITIES_SOURCE})
target_link_libraries(inertial_quantities)
add_library(test SHARED ${TEST_SOURCE})
target_link_libraries(test)
add_library(check SHARED ${CHECK_SOURCE})
target_link_libraries(check)
add_library(eigvals SHARED ${EIGVALS_SOURCE})
target_link_libraries(eigvals)
