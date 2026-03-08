cmake_minimum_required(VERSION 3.18)
project(nf4_dequant CUDA CXX)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 架构
# 本地 4070Ti 用 89，A100 用 80
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)  # 默认 A100
endif()

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 主程序
add_executable(nf4_dequant
    src/main.cpp
    src/utils.cpp
    src/kernel.cu
)

# 编译选项
target_compile_options(nf4_dequant PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
    >
)

# 如果需要调试，取消下面的注释
# target_compile_options(nf4_dequant PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-G>
# )


find_package(CUDAToolkit REQUIRED)

target_include_directories(nf4_dequant PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(nf4_dequant PRIVATE CUDA::cudart)