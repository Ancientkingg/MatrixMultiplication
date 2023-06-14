
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "MatrixMul.h"

#include <vector>
#include <cstdint>

TEST(MatrixMulTest, 2x2IdentityMatrixMul) {
	std::vector<int_fast64_t> A = { 1,0,0,1 };
	std::vector<int_fast64_t> B = { 1,0,0,1 };

	std::vector<int_fast64_t> C = { 1,0,0,1 };

	ASSERT_EQ(naive_matrix_mul(A, B, 2), C);
}

TEST(NaiveMatrixMulTest, 2x2MatrixMul) {
	std::vector<int_fast64_t> A = { 1,2,3,4 };
	std::vector<int_fast64_t> B = { 5,6,7,8 };

	std::vector<int_fast64_t> C = { 19,22,43,50 };

	ASSERT_EQ(naive_matrix_mul(A, B, 2), C);
}

TEST(NaiveMatrixMultest, 3x3MatrixMul) {
	std::vector<int_fast64_t> A = { 1,2,3,4,5,6,7,8,9 };
	std::vector<int_fast64_t> B = { 10,11,12,13,14,15,16,17,18 };

	std::vector<int_fast64_t> C = { 84,90,96,201,216,231,318,342,366 };

	ASSERT_EQ(naive_matrix_mul(A, B, 3), C);
}