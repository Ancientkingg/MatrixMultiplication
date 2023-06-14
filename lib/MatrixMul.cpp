
#include "MatrixMul.h"




std::vector<int_fast64_t> naive_matrix_mul(std::vector<int_fast64_t> A, std::vector<int_fast64_t> B, uint_fast32_t n) {
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
