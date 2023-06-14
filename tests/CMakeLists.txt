cmake_minimum_required(VERSION 3.14)
project(matrixmul_test)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows : Prevent overriding the parent project's compiler/linker settings
set( gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  matrixmul_test
  test.cpp
)
target_link_libraries(
  matrixmul_test
  GTest::gtest_main
  GTest::gmock_main
  MatrixMul
)

target_include_directories (matrixmul_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

include(GoogleTest)
gtest_discover_tests(matrixmul_test)