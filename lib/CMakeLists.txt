# CMakeList.txt : CMake project for FastMatrixMul, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

add_library(MatrixMul "MatrixMul.cpp" "MatrixMul.h")

target_include_directories (MatrixMul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET MatrixMul PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
