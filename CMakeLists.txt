# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("MatrixMul")

IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release)
ENDIF()

# Include sub-projects.
add_subdirectory ("lib")
add_subdirectory ("benchmarks")
add_subdirectory ("tests")
add_subdirectory ("profiling")
