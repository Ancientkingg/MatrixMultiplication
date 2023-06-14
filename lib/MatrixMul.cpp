
#include "MatrixMul.h"


std::vector<int_fast64_t> naive_vector_matrix_mul(std::vector<int_fast64_t> A, std::vector<int_fast64_t> B, uint_fast32_t n) {
	std::vector<int_fast64_t> outputMatrix;
	outputMatrix.reserve(n * n);
	for (size_t column = 0; column < n; column++) {
		for (size_t row = 0; row < n; row++) {
			int_fast64_t sum = 0;
			for (size_t k = 0; k < n; k++) {
				int_fast64_t outputElem = A[column * n + k] * B[k * n + row];
				
				sum += outputElem;
			}

			outputMatrix.push_back(sum);
		}
	}

	return outputMatrix;
}

std::unique_ptr<int_fast64_t[]> naive_array_matrix_mul(int_fast64_t* A, int_fast64_t* B, uint_fast32_t n) {
	auto outputMatrix = std::make_unique<int_fast64_t[]>(n*n);
	for (size_t column = 0; column < n; column++) {
		for (size_t row = 0; row < n; row++) {
			int_fast64_t sum = 0;
			for (size_t k = 0; k < n; k++) {
				int_fast64_t outputElem = A[column * n + k] * B[k * n + row];

				sum += outputElem;
			}

			outputMatrix[column * n + row] = sum;
		}
	}
	return outputMatrix;
}

std::unique_ptr<int_fast64_t[]> cache_optimized_array_matrix_mul(int_fast64_t* A, int_fast64_t* B, uint_fast32_t n) {
	auto outputMatrix = std::make_unique<int_fast64_t[]>(n * n);

	for (size_t column = 0; column < n; column++) {
		for (size_t k = 0; k < n; k++) {

			for (size_t row = 0; row < n; row++) {
				outputMatrix[column * n + row] += A[column * n + k] * B[k * n + row];
			}
		}
	}
	return outputMatrix;
}