#pragma once

#include <cstdint>
#include <vector>
#include <immintrin.h>

std::vector<int_fast64_t> naive_matrix_mul(std::vector<int_fast64_t> A, std::vector<int_fast64_t> B, uint_fast32_t n);