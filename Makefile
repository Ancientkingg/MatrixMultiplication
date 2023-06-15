benchmark:
	mkdir -p benchmark_out && \
	cd benchmark_out && \
	cmake .. && \
	cmake --build . --target matrixmul_benchmark && \
	cd benchmarks && \
	./matrixmul_benchmark 

test:
	mkdir -p test_out && \
	cd test_out && \
	cmake .. && \
	cmake --build . --target matrixmul_test && \
	cd tests && \
	./matrixmul_test
	
clean:
	rm -rf benchmark_out test_out