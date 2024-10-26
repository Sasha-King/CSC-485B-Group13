/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>
#include <cassert>

#include "dense_graph.h"
#include "sparse_graph.h"

#include "data_generator.h"
#include "data_types.h"


void testMatMul(const csc485b::a2::DenseGraph& g, const csc485b::a2::DenseGraph& output)
{
    for (int row = 0; row < g.n; row++)
    {
        for (int col = 0; col < g.n; col++)
        {
            csc485b::a2::node_t sum = 0;
            for (int k = 0; k < g.n; k++)
            {
                sum += g.adjacencyMatrix[row * g.n + k] * g.adjacencyMatrix[k * g.n + col];
            }

            //std::cout << "expected: " << sum << " recieved: " << output.adjacencyMatrix[row * g.n + col] << std::endl;
            //std::cout << "rip" << std::endl; 
            assert(output.adjacencyMatrix[row * g.n + col] == sum);
        }
    }
}


 /**
  * Runs timing tests on a CUDA graph implementation.
  * Consists of independently constructing the graph and then
  * modifying it to its two-hop neighbourhood.
  */
template < typename DeviceGraph >
void run(DeviceGraph g, csc485b::a2::edge_t const* d_edges, std::size_t m)
{
    cudaDeviceSynchronize();
    auto const build_start = std::chrono::high_resolution_clock::now();

    // this code doesn't work yet!
    int threads_build = 1024;
    int blocks_build = (m + threads_build - 1) / threads_build;
    csc485b::a2::gpu::build_graph << < blocks_build, threads_build >> > (g, d_edges, m);
    cudaDeviceSynchronize();


    std::vector< csc485b::a2::node_t > host_matrix_built(g.n * g.n);
    csc485b::a2::DenseGraph built_graph{ g.n, host_matrix_built.data() };
    cudaMemcpy(built_graph.adjacencyMatrix, g.adjacencyMatrix, sizeof(csc485b::a2::node_t) * g.n * g.n, cudaMemcpyDeviceToHost);




    auto const reachability_start = std::chrono::high_resolution_clock::now();
    // neither does this!
    constexpr size_t threads = 32;
    int blocks = (g.n + threads - 1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);



    csc485b::a2::node_t* d_matrix;
    cudaMalloc((void**)&d_matrix, sizeof(csc485b::a2::node_t) * g.n * g.n);
    csc485b::a2::DenseGraph d_output{ g.n, d_matrix }; // FOR DEVICE



    csc485b::a2::gpu::two_hop_reachability <<< BLOCKS, THREADS >>> (g, d_output);

    cudaDeviceSynchronize();

    std::vector< csc485b::a2::node_t > host_matrix(g.n * g.n);
    csc485b::a2::DenseGraph output{ g.n, host_matrix.data() }; // DEVICE TO HOST
    cudaMemcpy(output.adjacencyMatrix, d_output.adjacencyMatrix, sizeof(csc485b::a2::node_t) * g.n * g.n, cudaMemcpyDeviceToHost);


    auto const end = std::chrono::high_resolution_clock::now();

    std::cout << "Build time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(reachability_start - build_start).count()
        << " us"
        << std::endl;

    std::cout << "Reachability time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - reachability_start).count()
        << " us"
        << std::endl;

    testMatMul(built_graph, output);
    cudaFree(d_matrix);
}

std::vector<csc485b::a2::node_t> cpu_adjacency_matrix_dense(size_t n, csc485b::a2::edge_list_t edge_list, std::size_t m)
{
    std::vector<csc485b::a2::node_t> matrix(n * n);
    for (int i = 0; i < m; ++i)
    {
        unsigned int x = edge_list[i].x;
        unsigned int y = edge_list[i].y;

        matrix[x * n + y] = 1;
        matrix[y * n + x] = 1;
    }
    return matrix;
}

void cpu_matrix_mul(std::vector<csc485b::a2::node_t> adjMatrix, csc485b::a2::node_t* adjMatrixResult, int n)
{
    std::vector<csc485b::a2::node_t> matrix(n * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            csc485b::a2::node_t sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += adjMatrix[i * n + k] * adjMatrix[k * n + j];
            }

            assert(sum == adjMatrixResult[i * n + j]);
        }
    }
}


void testBuildGraph(csc485b::a2::DenseGraph g, size_t m, csc485b::a2::edge_list_t edge_list)
{
    size_t n = g.n;
    std::vector<csc485b::a2::node_t> matrix(n * n);
    for (int i = 0; i < m; ++i)
    {
        size_t x = edge_list[i].x;
        size_t y = edge_list[i].y;
        matrix[x * n + y] = 1;
        matrix[y * n + x] = 1;
    }

    for (int i = 0; i < n * n; ++i)
    {
        assert(matrix[i] == g.adjacencyMatrix[i]);
    }
    return;
}
/**
 * Allocates space for a dense graph and then runs the test code on it.
 */
void run_dense(csc485b::a2::edge_t const* d_edges, std::size_t n, std::size_t m, csc485b::a2::edge_list_t graph)
{
    using namespace csc485b;

    // allocate device DenseGraph
    a2::node_t* d_matrix;
    cudaMalloc((void**)&d_matrix, sizeof(a2::node_t) * n * n);
    a2::DenseGraph d_dg{ n, d_matrix };

    run(d_dg, d_edges, m);


    
    // check output?
    std::vector< a2::node_t > host_matrix(d_dg.matrix_size());
    a2::DenseGraph dg{ n, host_matrix.data() };
    cudaMemcpy(dg.adjacencyMatrix, d_dg.adjacencyMatrix, sizeof(a2::node_t) * d_dg.matrix_size(), cudaMemcpyDeviceToHost);
    //std::copy(host_matrix.cbegin(), host_matrix.cend(), std::ostream_iterator< a2::node_t >(std::cout, " "));

    testBuildGraph(dg, m, graph);
    //testMatMul(dg);


    cudaFree(d_matrix);
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 */
void run_sparse(csc485b::a2::edge_t const* d_edges, std::size_t n, std::size_t m)
{
    using namespace csc485b;

    // allocate device SparseGraph
    a2::node_t* d_offsets, * d_neighbours;
    cudaMalloc((void**)&d_offsets, sizeof(a2::node_t) * n);
    cudaMalloc((void**)&d_neighbours, sizeof(a2::node_t) * m);
    a2::SparseGraph d_sg{ n, m, d_offsets, d_neighbours };

    //run(d_sg, d_edges, m);

    // clean up
    cudaFree(d_neighbours);
    cudaFree(d_offsets);
}

int main()
{
    using namespace csc485b;

    // Create input
    std::size_t constexpr n = 1 << 12;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph(n, n * expected_degree);
    std::size_t const m = graph.size();


    a2::edge_t* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(a2::edge_t) * m);
    cudaMemcpyAsync(d_edges, graph.data(), sizeof(a2::edge_t) * m, cudaMemcpyHostToDevice);

    // run your code!
    run_dense(d_edges, n, m, graph);

    

    //run_sparse(d_edges, n, m);

    return EXIT_SUCCESS;
}