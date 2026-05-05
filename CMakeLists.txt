cmake_minimum_required(VERSION 3.24)

project(CUDACyclone LANGUAGES CXX CUDA)

add_executable(CUDACyclone
    CUDACyclone.cu
    CUDAHash.cu
)

target_compile_features(CUDACyclone PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(CUDACyclone PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

set(CUDACYCLONE_KNOWN_ARCHS 75 80 86 87 88 89 90 100 103 110 120 121)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDACYCLONE_KNOWN_ARCHS}" CACHE STRING "CUDA architectures" FORCE)
endif()
set_property(TARGET CUDACyclone PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

target_compile_options(CUDACyclone PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --ptxas-options=-O3>
)

target_link_libraries(CUDACyclone PRIVATE cudadevrt)

if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(CUDACyclone PRIVATE Threads::Threads)
endif()
