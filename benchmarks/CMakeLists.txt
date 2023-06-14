cmake_minimum_required(VERSION 3.14)
project(matrixmul_benchmark)
enable_testing()

include(FetchContent)

## Project-wide setup
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Externally provided libraries
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main)

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)

add_executable(matrixmul_benchmark benchmark.cpp)
target_link_libraries(
    matrixmul_benchmark 
    benchmark::benchmark
    MatrixMul
)

target_include_directories (matrixmul_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../lib)