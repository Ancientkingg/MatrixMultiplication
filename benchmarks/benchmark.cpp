
#include <benchmark/benchmark.h>

#include "MatrixMul.h"

#include <cstdlib>

static void BM_naive_vector_matrix_mul(benchmark::State& state) {
	std::vector<int_fast64_t> A(state.range(0) * state.range(0), 0);
	std::vector<int_fast64_t> B(state.range(0) * state.range(0), 0);

	// Generate matrices with random elements of size `size`
	generate(A.begin(), A.end(), rand);
	generate(B.begin(), B.end(), rand);

	for (auto _ : state) {
		benchmark::DoNotOptimize(naive_vector_matrix_mul(A, B, state.range(0)));
	}
}

BENCHMARK(BM_naive_vector_matrix_mul)->Arg(3)->Arg(10)->Arg(100)->Arg(1000);

static void BM_naive_array_matrix_mul(benchmark::State& state) {
	std::unique_ptr<int_fast64_t[]> A = std::make_unique<int_fast64_t[]>(state.range(0) * state.range(0));
	std::unique_ptr<int_fast64_t[]> B = std::make_unique<int_fast64_t[]>(state.range(0) * state.range(0));

	// Generate matrices with random elements of size `size`
	for (size_t i = 0; i < state.range(0) * state.range(0); i++) {
		A[i] = rand();
		B[i] = rand();
	}

	for (auto _ : state) {
		benchmark::DoNotOptimize(naive_array_matrix_mul(A.get(), B.get(), state.range(0)));
	}
}

BENCHMARK(BM_naive_array_matrix_mul)->Arg(3)->Arg(10)->Arg(100)->Arg(1000);


static void BM_cache_optimized_array_matrix_mul(benchmark::State& state) {
	std::unique_ptr<int_fast64_t[]> A = std::make_unique<int_fast64_t[]>(state.range(0) * state.range(0));
	std::unique_ptr<int_fast64_t[]> B = std::make_unique<int_fast64_t[]>(state.range(0) * state.range(0));

	// Generate matrices with random elements of size `size`
	for (size_t i = 0; i < state.range(0) * state.range(0); i++) {
		A[i] = rand();
		B[i] = rand();
	}

	for (auto _ : state) {
		benchmark::DoNotOptimize(cache_optimized_array_matrix_mul(A.get(), B.get(), state.range(0)));
	}
}

BENCHMARK(BM_naive_array_matrix_mul)->Arg(3)->Arg(10)->Arg(100)->Arg(1000);

BENCHMARK_MAIN();