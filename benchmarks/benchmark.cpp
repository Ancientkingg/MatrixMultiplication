
#include <benchmark/benchmark.h>

#include "MatrixMul.h"

#include <cstdlib>

static void BM_naive_matrix_mul(benchmark::State& state) {
	std::vector<int_fast64_t> A(state.range(0) * state.range(0), 0);
	std::vector<int_fast64_t> B(state.range(0) * state.range(0), 0);

	// Generate matrices with random elements of size `size`
	generate(A.begin(), A.end(), rand);
	generate(B.begin(), B.end(), rand);

	for (auto _ : state) {
		benchmark::DoNotOptimize(naive_matrix_mul(A, B, state.range(0)));
	}
}

BENCHMARK(BM_naive_matrix_mul)->Arg(3)->Arg(10)->Arg(100)->Arg(1000);

BENCHMARK_MAIN();