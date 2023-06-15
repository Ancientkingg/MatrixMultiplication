
#include <benchmark/benchmark.h>

#include "MatrixMul.h"

#include <cstdlib>

#define MATRIX_SIZES Arg(2)->Arg(3)->Arg(16)->Arg(64)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)

static void BM_naive_vector_matrix_mul(benchmark::State& state) {
	std::vector<int_fast64_t> A(state.range(0) * state.range(0), 0);
	std::vector<int_fast64_t> B(state.range(0) * state.range(0), 0);

	// Generate matrices with random elements of size `state.range(0) * state.range(0)`
	generate(A.begin(), A.end(), rand);
	generate(B.begin(), B.end(), rand);

	for (auto _ : state) {
		benchmark::DoNotOptimize(naive_vector_matrix_mul(A, B, state.range(0)));
	}
}

BENCHMARK(BM_naive_vector_matrix_mul)->MATRIX_SIZES;

static void BM_naive_array_matrix_mul(benchmark::State& state) {
	std::unique_ptr<int_fast64_t[]> A = std::make_unique<int_fast64_t[]>(state.range(0) * state.range(0));
	std::unique_ptr<int_fast64_t[]> B = std::make_unique<int_fast64_t[]>(state.range(0) * state.range(0));

	// Generate matrices with random elements of size `state.range(0) * state.range(0)`
	for (size_t i = 0; i < state.range(0) * state.range(0); i++) {
		A[i] = rand();
		B[i] = rand();
	}

	for (auto _ : state) {
		benchmark::DoNotOptimize(naive_array_matrix_mul(A.get(), B.get(), state.range(0)));
	}
}

BENCHMARK(BM_naive_array_matrix_mul)->MATRIX_SIZES;

static void BM_cache_optimized_array_matrix_mul(benchmark::State& state) {
	std::unique_ptr<int_fast64_t[]> A = std::make_unique<int_fast64_t[]>(state.range(0) * state.range(0));
	std::unique_ptr<int_fast64_t[]> B = std::make_unique<int_fast64_t[]>(state.range(0) * state.range(0));

	// Generate matrices with random elements of size `state.range(0) * state.range(0)`
	for (size_t i = 0; i < state.range(0) * state.range(0); i++) {
		A[i] = rand();
		B[i] = rand();
	}

	for (auto _ : state) {
		benchmark::DoNotOptimize(cache_optimized_array_matrix_mul(A.get(), B.get(), state.range(0)));
	}
}

BENCHMARK(BM_cache_optimized_array_matrix_mul)->MATRIX_SIZES;

BENCHMARK_MAIN();