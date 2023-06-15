cmake_minimum_required(VERSION 3.14)
project(matrixmul_profiling)

set(source_files 
    cache_optimized_array_matrix_mul.cpp 
    naive_array_matrix_mul.cpp
)

add_custom_target(matrixmul_profiling)

# Loop through each source file
foreach(source_file IN LISTS source_files)
    # Get the name of the file without the extension (e.g. 'a' from src/a.cpp'
    get_filename_component(target_name ${source_file} NAME_WE)
    
    # Create an executable with the above name, building the above source
    add_executable("${target_name}" "${source_file}")

    target_link_libraries(
        "${target_name}"
        MatrixMul
    )

    target_include_directories ("${target_name}" PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../lib)
    add_dependencies(matrixmul_profiling "${target_name}")
endforeach()