
#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>
#include <cassert>
#include <algorithm> //for std::sort
#include <cstddef>  // std::size_t type
#include <random>   // for std::mt19937, std::uniform_int_distribution
#include <vector>


#include "cuda_common.h"
#include "GEMM.h"

//Adapted from a2
std::vector<int> build_matrix(size_t n)
{
    std::vector<int> matrix(n * n);

    std::size_t random_seed = 20241008;  // use magic seed
    std::mt19937 rng(random_seed);     // use mersenne twister generator
    std::uniform_int_distribution<> distrib(0, 100); //change these to modify value range

    for (size_t i = 0; i < n * n; ++i) {
        matrix[i] = distrib(rng);
    }

    return matrix;
}

void printMatrix(const std::vector<int>& matrix, size_t n) {
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}

/*
Benchmark for GEMM
*/
void GEMM_test(std::vector<int> matrix, size_t n)
{
   
   
    constexpr size_t threads = 32;
    int blocks = (n + threads - 1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    int* d_input;
    int* d_output;
    
    cudaMalloc(&d_input, n * n * sizeof(int));
    cudaMalloc(&d_output, n * n * sizeof(int));
    
    cudaMemcpy(d_input, matrix.data(), n * n * sizeof(int), cudaMemcpyHostToDevice); //host to device
    
    auto const test_start = std::chrono::high_resolution_clock::now();

    csc485b::a4::gpu::run_GEMM << < BLOCKS, THREADS >> > (d_input, d_output, n);

    auto const end = std::chrono::high_resolution_clock::now();

    std::vector<int> result(n * n);
    cudaMemcpy(result.data(), d_output, n * n * sizeof(int), cudaMemcpyDeviceToHost); //Device to Host

    std::cout << "GEMM Time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end -test_start).count()
        << " us"
        << std::endl;

    printMatrix(result,n);

    //cleanup
    cudaFree(d_input);
    cudaFree(d_output);

}


int main()
{
    
    int n = 4;
    
    std::vector<int> matrix = build_matrix(n);

    std::cout << "Generated Matrix:\n";
    printMatrix(matrix, n);

    //run tests
    GEMM_test(matrix, n);

    return EXIT_SUCCESS;

}

