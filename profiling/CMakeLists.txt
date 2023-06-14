cmake_minimum_required(VERSION 3.14)
project(matrixmul_profiling)

add_executable(
  matrixmul_profiling
  profiling.cpp
)

target_link_libraries(
  matrixmul_profiling
  MatrixMul
)

target_include_directories (matrixmul_profiling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../lib)