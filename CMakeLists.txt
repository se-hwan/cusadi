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
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xptxas -O1 -arch=sm_86 --use_fast_math)  # Adjust architecture as needed

# set(FN_1E1_SOURCE codegen/fn_1e1.cu)
# set(FN_1E2_SOURCE codegen/fn_1e2.cu)
# set(FN_1E3_SOURCE codegen/fn_1e3.cu)
# set(FN_1E4_SOURCE codegen/fn_1e4.cu)
# set(FN_1E5_SOURCE codegen/fn_1e5.cu)
# set(FN_1E6_SOURCE codegen/fn_1e6v2.cu)
set(FN_1E6_V3_SOURCE codegen/fn_1e6v3_p00.cu
                     codegen/fn_1e6v3_p01.cu
                     codegen/fn_1e6v3_p02.cu 
                     codegen/fn_1e6v3_p03.cu 
                     codegen/fn_1e6v3_p04.cu 
                     codegen/fn_1e6v3_p05.cu )
set (FN_1E6_MAIN codegen/fn_1e6v3.cu)

# Include directories for your header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add_library(fn_1e1 SHARED ${FN_1E1_SOURCE})
# target_link_libraries(fn_1e1)
# add_library(fn_1e2 SHARED ${FN_1E2_SOURCE})
# target_link_libraries(fn_1e2)
# add_library(fn_1e3 SHARED ${FN_1E3_SOURCE})
# target_link_libraries(fn_1e3)
# add_library(fn_1e4 SHARED ${FN_1E4_SOURCE})
# target_link_libraries(fn_1e4)
# add_library(fn_1e5 SHARED ${FN_1E5_SOURCE})
# target_link_libraries(fn_1e5)
# add_library(fn_1e6v2 SHARED ${FN_1E6_SOURCE})
# target_link_libraries(fn_1e6v2)

add_library(fn_1e6v3_src SHARED ${FN_1E6_V3_SOURCE})
add_library(fn_1e6v3_main SHARED ${FN_1E6_MAIN})
target_link_libraries(fn_1e6v3_main fn_1e6v3_src)