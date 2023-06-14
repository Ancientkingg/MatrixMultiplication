
#include "MatrixMul.h"

#include <cstdlib>


int main() {
	constexpr size_t size = 1024;

	std::unique_ptr<int_fast64_t[]> A = std::make_unique<int_fast64_t[]>(size * size);
	std::unique_ptr<int_fast64_t[]> B = std::make_unique<int_fast64_t[]>(size * size);

	// Generate matrices with random elements of size `size`
	for (size_t i = 0; i < size * size; i++) {
		A[i] = rand();
		B[i] = rand();
	}

	static volatile auto x = naive_array_matrix_mul(A.get(), B.get(), size);

	return 0;
}