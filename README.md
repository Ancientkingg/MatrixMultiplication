# Matrix Multiplication

This repository explores multiple implementations of performing a squared matrix multiplication and documents the performance of each implementation.

The	implementations are written in C++ with [CMAKE](https://cmake.org/) as the build system to offer portability.
The Google [benchmark](https://github.com/google/benchmark) and [test](https://github.com/google/test) frameworks are used to evaluate the performance of the implementations.
To profile the memory and cache usage of each implementation, the implementations are cross-compiled to linux and [Valgrind](https://valgrind.org/) in WSL is used to profile the executables manually.

## Implementations

### Naive

The naive implementation is the most straightforward implementation of matrix multiplication.
It iterates over all elements of the result matrix and calculates the value by summing up the products of the corresponding row and column of the input matrices.

In Python the algorithm for multiplying two square row-major matrices with each other would look like the following:
```python
for i in range(n):
	for j in range(n):
		for k in range(n):
			outputMatrix[i][j] += leftMatrix[i][k] * rightMatrix[k][j]
```

*Note that in my implementation the input matrices are one dimensional row-major as opposed to the usual two dimensions row-major in a matrix.*

Two implementations are provided, one using a **vector** and one using a **raw array**:

```cpp
// Vector implementation
std::vector<int_fast64_t> naive_vector_matrix_mul(std::vector<int_fast64_t> A, std::vector<int_fast64_t> B, uint_fast32_t n);

// Array implementation
std::unique_ptr<int_fast64_t[]> naive_array_matrix_mul(int_fast64_t* A, int_fast64_t* B, uint_fast32_t n);
```

Inspecting the assembly in compiler explorer shows that both the [vector implementation](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgBYBQogFUAztgAKAD24AGfgCsp5eiyahUAUgBMAIUtXyKxqiIEh1ZpgDC6egFc2TEAtydwAZAiZsADk/ACNsUkCAVnIAB3QlYhcmL19/QNT0zKEwiOi2OISLZIdsJyyRIhZSIhy/AKCauqEGpqISqNj4pPtG5ta8jtG%2B8IHyoaqASnt0H1JUTi5LAGZw1F8cAGozLY9UJSJCYWPcMy0AQW3d/ewjk4A3WpJSa9uHix2mHsfIdjh4CGw2OEiKRwgA6JA/e6PQHPV4eDhsMgAT0RD3u50wIBAHycZFBUIA%2BjQWOcAGzSClEa4HVgED4UklfClsFjQgjqbk%2BegQAlEzlkk6U6l0hlMra4A53cgHUXEz4SsHCKk0oj0xnMuwHHxSnVbCyMlkLI4AdhsSK0AE5VeLvpKtdLdbLmSsiCkfEQALK8mHqY52vEOn1%2BwPB/mw0jYFSkD4QJgHABUlrDv1uDsEpAOIoIAC9sBasa8ACIHLRhg4V0Esus46y2K1mW05x25/OFjKli3qKs1utDxtMUe2Ns28Ndh253MmmUWpR%2BYe1raz%2B3z7uO3tFgdEA4Aa3XddP4/PU%2Bs7c724Xu4dS89Fqj/twjDYw4eiSsFczaa2CeZiJNWmY2L%2Bp4AUc1gHKGoHZvaEYPjuTprlOWzVm%2BRAftgbCIchj4dpWXaEah2FBny6iwn6ShIBSMQsKgx4in4CwEShxFzlxSG5gmRCrGmFGxqGm45talZcEs9DcIk/ABFwOjkOg3AeNeVgqisawvNsfDkEQ2hSUsx4gIkWiGNw0j8GwpnmQpSkqVw/BKCA5kGYpUnkHAsAoNg6i1P6ZCUNQTTAEoqjGNg9AiEg6AAO4KXpGBsCkDC8lkEURNFsUJYZ5DJaljAJMAPAWDw%2BXoClDDxJE7AbPwBXVaQADy/oxfF9n8H5tR3KQYXcF1/moA0lwDXIwhiBInAyONigqBoeX6OVRgmGg6mGAQMQuZASzoCkzhCC5TnKR8pAwjg20QEsShaesBgEuEmVRe1uW8Pw0LYBselxaQLApNwfDSbJ8l5Y5uBDYFBbqAAHLSAC09IHMAqCoAcpWwjwhZqa21jKvgxBkDBWw8As72GQsSxINgLA4AkV0WVwVnkDZZnkJ1yljS5bnk8ZtkM1sIMeRzx3uToSzecgaCVYV8TBRAjVFWgK0ldIWjmTgbwEOsABqBDYHFzUpMwAP8HQ9BEPELkQDEeUxOETRYib5B26wpBYs1MS6J8TvJRwwjNUw9CO0LOAxD4wAeBI9BHXpOA8iYkghwQCZ1B8R1Kd1qD%2Bl9/BQlFeX0Jtv1u14OB5XyNlvUsNBGGFuv64bxtvbNk2SDNgjCMoaiaELS2GMYpjrYXW3wLt%2B1ZEdcPNVsBxwzQHpwzyRBILPPJrAimFIDScVRfQBz8Ogp3nYmo/2FFnxZG4TCeN4bQGKEMxlBUBhpBkB3ZLfeTla/RRMP0T9DHKp0d%2BPQxifwCEA8%2BXQmCgOmKUQYCQgFTHGBAkYvR/4IKkNdW601yBxWMEQQ2RAACSTBBAmyBlwOSbNQbcDgrDBG0gDh7AHmjaQsItAcMLPjL4RMSZkw8hTcgVMaZDHpiZVmMlGaCwcpzVy%2BkeYMwsDIg%2BY1RZGXIKdDIrhpBAA%3D%3D%3D)
and the [array implementation](https://godbolt.org/#g:!((g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:18,endLineNumber:4,positionColumn:18,positionLineNumber:4,selectionStartColumn:18,selectionStartLineNumber:4,startColumn:18,startLineNumber:4),source:'%23include+%3Ccstdint%3E%0A%23include+%3Cvector%3E%0A%23include+%3Cimmintrin.h%3E%0A%23include+%3Cmemory%3E%0A%0Astd::unique_ptr%3Cint_fast64_t%5B%5D%3E+naive_array_matrix_mul(int_fast64_t*+A,+int_fast64_t*+B,+uint_fast32_t+n)+%7B%0A%09auto+outputMatrix+%3D+std::make_unique%3Cint_fast64_t%5B%5D%3E(n*n)%3B%0A%09for+(size_t+y+%3D+0%3B+y+%3C+n%3B+y%2B%2B)+%7B%0A%09%09for+(size_t+x+%3D+0%3B+x+%3C+n%3B+x%2B%2B)+%7B%0A%0A%09%09%09int_fast64_t+sum+%3D+0%3B%0A%0A%09%09%09for+(size_t+k+%3D+0%3B+k+%3C+n%3B+k%2B%2B)+%7B%0A%09%09%09%09int_fast64_t+outputElem+%3D+A%5By+*+n+%2B+k%5D+*+B%5Bk+*+n+%2B+x%5D%3B%0A%0A%09%09%09%09sum+%2B%3D+outputElem%3B%0A%09%09%09%7D%0A%0A%09%09%09outputMatrix%5By+*+n+%2B+x%5D+%3D+sum%3B%0A%09%09%7D%0A%09%7D%0A%09return+outputMatrix%3B%0A%7D'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',m:50,n:'0',o:'',s:0,t:'0'),(g:!((h:executor,i:(argsPanelShown:'1',compilationPanelShown:'0',compiler:g121,compilerName:'',compilerOutShown:'0',execArgs:'',execStdin:'',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'',overrides:!(),source:1,stdinPanelShown:'1',tree:'1',wrap:'1'),l:'5',n:'0',o:'Executor+x86-64+gcc+12.1+(C%2B%2B,+Editor+%231)',t:'0')),header:(),l:'4',m:50,n:'0',o:'',s:0,t:'0')),k:50,l:'3',n:'0',o:'',t:'0'),(g:!((h:compiler,i:(compiler:clang1400,deviceViewOpen:'1',filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-O3+-ffast-math+-march%3Dhaswell+',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1,wantOptInfo:'1'),l:'5',n:'0',o:'+x86-64+clang+14.0.0+(Editor+%231)',t:'0')),header:(),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4)
both enable some form of vectorization in their inner loops.

