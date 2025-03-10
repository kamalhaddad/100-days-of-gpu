# Enable CUDA support
find_package(CUDAToolkit REQUIRED)

# Set CUDA runtime library
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the build type to Debug
set(CMAKE_BUILD_TYPE Debug)

# Enable debugging symbols for both host and device code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")   # For host code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # For device code

# Specify the executable and source files
add_executable(day01 vecAdd.cu)

# Set CUDA architecture (modify based on your GPU, e.g., sm_75 for RTX 2080)
set_target_properties(day01 PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 80)

# Link CUDA libraries (if needed)
target_link_libraries(day01 PRIVATE CUDA::cuda_driver)
