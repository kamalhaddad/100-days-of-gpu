# Enable CUDA support
find_package(CUDAToolkit REQUIRED)

# Set CUDA runtime library
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify the executable and source files
add_executable(day02 matAdd.cu)

# Set CUDA architecture (modify based on your GPU, e.g., sm_75 for RTX 2080)
set_target_properties(day01 PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 80)

# Link CUDA libraries (if needed)
target_link_libraries(day02 PRIVATE CUDA::cuda_driver)
