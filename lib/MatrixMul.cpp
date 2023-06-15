
#include "MatrixMul.h"


std::vector<int_fast64_t> naive_vector_matrix_mul(std::vector<int_fast64_t> A, std::vector<int_fast64_t> B, uint_fast32_t n) {
	std::vector<int_fast64_t> outputMatrix;
	outputMatrix.reserve(n * n);
	for (size_t y = 0; y < n; y++) {
		for (size_t x = 0; x < n; x++) {

			int_fast64_t sum = 0;

			for (size_t k = 0; k < n; k++) {
				int_fast64_t outputElem = A[y * n + k] * B[k * n + x];

				sum += outputElem;
			}

			outputMatrix.push_back(sum);
		}
	}

	return outputMatrix;
}

std::unique_ptr<int_fast64_t[]> naive_array_matrix_mul(int_fast64_t* A, int_fast64_t* B, uint_fast32_t n) {
	auto outputMatrix = std::make_unique<int_fast64_t[]>(n*n);
	for (size_t y = 0; y < n; y++) {
		for (size_t x = 0; x < n; x++) {

			int_fast64_t sum = 0;

			for (size_t k = 0; k < n; k++) {
				int_fast64_t outputElem = A[y * n + k] * B[k * n + x];

				sum += outputElem;
			}

			outputMatrix[y * n + x] = sum;
		}
	}
	return outputMatrix;
}

std::unique_ptr<int_fast64_t[]> cache_optimized_array_matrix_mul(int_fast64_t* A, int_fast64_t* B, uint_fast32_t n) {
	auto outputMatrix = std::make_unique<int_fast64_t[]>(n * n);

	for (size_t y = 0; y < n; y++) {
		for (size_t k = 0; k < n; k++) {
			for (size_t x = 0; x < n; x++) {
				outputMatrix[y * n + x] += A[y * n + k] * B[k * n + x];
			}
		}
	}
	return outputMatrix;
}