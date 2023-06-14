#pragma once

#include <cstdint>
#include <vector>
#include <immintrin.h>
#include <memory>

std::vector<int_fast64_t> naive_vector_matrix_mul(std::vector<int_fast64_t> A, std::vector<int_fast64_t> B, uint_fast32_t n);

std::unique_ptr<int_fast64_t[]> naive_array_matrix_mul(int_fast64_t* A, int_fast64_t* B, uint_fast32_t n);