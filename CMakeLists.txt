cmake_minimum_required(VERSION 3.24)

project(CUDACyclone LANGUAGES CXX CUDA)

set(CUDACYCLONE_CPU_AVX2_SOURCES
    cpu_avx2/Cyclone.cpp
    cpu_avx2/SECP256K1.cpp
    cpu_avx2/Int.cpp
    cpu_avx2/IntGroup.cpp
    cpu_avx2/IntMod.cpp
    cpu_avx2/Point.cpp
    cpu_avx2/ripemd160_avx2.cpp
    cpu_avx2/p2pkh_decoder.cpp
    cpu_avx2/sha256_avx2.cpp
    cpu_avx2/Random.cpp
    cpu_avx2/Timer.cpp
)

add_executable(CUDACyclone
    CUDACyclone.cu
    CUDAHash.cu
    ${CUDACYCLONE_CPU_AVX2_SOURCES}
)

target_compile_features(CUDACyclone PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(CUDACyclone PRIVATE CUDACYCLONE_EMBEDDED_CPU_WORKER)
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

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CUDACyclone PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_definitions(CUDACyclone PRIVATE WIN64)
    target_compile_options(CUDACyclone PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /arch:AVX2>
    )
    target_link_libraries(CUDACyclone PRIVATE Advapi32)
else()
    target_compile_options(CUDACyclone PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -mavx2 -mbmi2 -madx -funroll-loops>
    )
endif()

if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(CUDACyclone PRIVATE Threads::Threads)
endif()

option(CUDACYCLONE_BUILD_CPU_AVX2 "Build the standalone AVX2 CPU search backend" OFF)

if(CUDACYCLONE_BUILD_CPU_AVX2)
    add_executable(CycloneCPU_avx2 ${CUDACYCLONE_CPU_AVX2_SOURCES})
    target_compile_features(CycloneCPU_avx2 PRIVATE cxx_std_17)

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(CycloneCPU_avx2 PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(MSVC)
        target_compile_definitions(CycloneCPU_avx2 PRIVATE WIN64)
        target_compile_options(CycloneCPU_avx2 PRIVATE /O2 /arch:AVX2)
        target_link_libraries(CycloneCPU_avx2 PRIVATE Advapi32)
    else()
        target_compile_options(CycloneCPU_avx2 PRIVATE -O3 -mavx2 -mbmi2 -madx -funroll-loops)
    endif()
endif()
