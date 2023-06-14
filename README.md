# Matrix Multiplication

This repository explores multiple implementations of performing a squared matrix multiplication and documents the performance of each implementation.
The	implementations are written in C++ with CMAKE as the build system to offer portability.
The Google [benchmark](https://github.com/google/benchmark) and [test](https://github.com/google/test) frameworks are used to evaluate the performance of the implementations.
To profile the memory and cache usage of each implementation, the implementations are cross-compiled to linux and [Valgrind](https://valgrind.org/) in WSL is used to profile the executables manually.

## Implementations

### Naive

The naive implementation is the most straightforward implementation of matrix multiplication.
It iterates over all elements of the result matrix and calculates the value by summing up the products of the corresponding row and column of the input matrices.
Note that in my implementation the input matrices are one dimensional row-major as opposed to the usual two dimensions row-major in a matrix.

Two implementations are provided, one using a vector and one using a raw array.
