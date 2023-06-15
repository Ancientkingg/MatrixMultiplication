
#include "MatrixMul.h"

#include <cstdlib>

#include "matrix_size.h"

int main() {

	std::unique_ptr<int_fast64_t[]> A = std::make_unique<int_fast64_t[]>(SIZE * SIZE);
	std::unique_ptr<int_fast64_t[]> B = std::make_unique<int_fast64_t[]>(SIZE * SIZE);

	// Generate matrices with random elements of size `SIZE`
	for (size_t i = 0; i < SIZE * SIZE; i++) {
		A[i] = rand();
		B[i] = rand();
	}

	static volatile auto x = cache_optimized_array_matrix_mul(A.get(), B.get(), SIZE);

	return 0;
}