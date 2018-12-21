cmake_minimum_required(VERSION 3.12)
project(TD3_GPU_CUDA LANGUAGES CUDA C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(TD3_GPU_CUDA_1
        cuda_td_1.cu wb.h dummy.c)
add_executable(TD3_GPU_CUDA_2
        cuda_td_2.cu wb.h dummy.c)
add_executable(TD3_GPU_CUDA_3
        cuda_td_3.cu wb.h dummy.c)
add_executable(TD3_GPU_CUDA_4
        cuda_td_4.cu wb.h dummy.c)
add_executable(TD3_GPU_CUDA_5
        cuda_td_5.cu wb.h dummy.c)
add_executable(cuda_pattern_td_1
        cuda_pattern_td_1.cu wb.h dummy.c)
add_executable(cuda_pattern_td_2
        cuda_pattern_td_2.cu wb.h dummy.c)
add_executable(cuda_pattern_td_3
        cuda_pattern_td_3.cu wb.h dummy.c)