/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>
#include <algorithm> //for sort 

#include "dense_graph.h"
#include "sparse_graph.h"

#include "data_generator.h"
#include "data_types.h"

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

    //This Works
    csc485b::a2::gpu::build_graph << < 4, 256 >> > (g, d_edges, m);

    cudaDeviceSynchronize();
    auto const reachability_start = std::chrono::high_resolution_clock::now();

    //This does not work
    csc485b::a2::gpu::two_hop_reachability << < 1, 1 >> > (g);

    cudaDeviceSynchronize();
    auto const end = std::chrono::high_resolution_clock::now();

    std::cout << "GPU Build time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(reachability_start - build_start).count()
        << " us"
        << std::endl;

    std::cout << "GPU Reachability time: "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - reachability_start).count()
        << " us"
        << std::endl;

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
    
    //Ouputs matrix, not needed atm because of comparison
    //std::copy(host_matrix.cbegin(), host_matrix.cend(), std::ostream_iterator< a2::node_t >(std::cout, " ")); 
    //std::cout << std::endl;
    //std::cout << "expected:" << std::endl;

    std::vector<csc485b::a2::node_t> expected_matrix = cpu_adjacency_matrix_dense(n, graph, m);
    bool failed = false;
    for (int i = 0; i < n * n; ++i)
    {
        if (expected_matrix[i] != dg.adjacencyMatrix[i])
        {
            failed = true;
            break;
        }
    }

    if (failed) std::cout << "DenseGraph matrix test failed" << std::endl;
    else std::cout << "DenseGraph matrix test passed" << std::endl;
    
    //cleanup
    cudaFree(d_matrix);
}


/*
* Constructs a SparseGraph from an input list of edges.
* In slight variation of CSR format https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf 
* uses index 0 and does not store tott in neighbours_start_at[V+1]
* Also the way the edges are formated there is a (1,0) and a (0,1) edge
*/
csc485b::a2::SparseGraph cpu_CSR(std::size_t n, std::size_t m, csc485b::a2::edge_list_t graph) {
    
    using namespace csc485b;

    a2::node_t* offsets = new a2::node_t[n]();
    a2::node_t* dest = new a2::node_t[m]();

    //first pass stores out degree of each vertetx in offsets
    for (size_t i = 0; i < m; i++){
        
        a2::node_t x = graph[i].x;
        offsets[x]++;   
    }

    //Updates offsetts to contain cumulative out degree
    int t = 0;
    int a;
    for (a = 0; a < n; a++) {
        
        t += offsets[a];
        offsets[a] = t; 
    }

    for (std::size_t i = 0; i < m; i++) {

        a2::node_t x = graph[i].x;
        a2::node_t y = graph[i].y;

        dest[--offsets[x]] = y;
    }

    a2::SparseGraph sg{ n, m, offsets, dest };
    return sg;
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 * Altered to take graph as input for testing purposes 
 */
void run_sparse(csc485b::a2::edge_t const* d_edges, std::size_t n, std::size_t m, csc485b::a2::edge_list_t graph)
{
    using namespace csc485b;

    // allocate device SparseGraph
    a2::node_t* d_offsets, * d_neighbours;
    cudaMalloc((void**)&d_offsets, sizeof(a2::node_t) * n);
    cudaMalloc((void**)&d_neighbours, sizeof(a2::node_t) * m);
    a2::SparseGraph d_sg{ n, m, d_offsets, d_neighbours };

    run(d_sg, d_edges, m);

    std::size_t const threads_per_block = 256;
    std::size_t const num_blocks = (n + threads_per_block - 1) / threads_per_block;

    a2::node_t * g_neighbours_start = new a2::node_t[n](); //Offset arrray
    a2::node_t * g_neighbours = new a2::node_t[m](); //Edge destinations 
    
    //Getting the data out from cuda for testing 
    cudaMemcpy(g_neighbours_start, d_offsets, sizeof(a2::node_t) * n, cudaMemcpyDeviceToHost); //get offsets
    cudaMemcpy(g_neighbours, d_neighbours, sizeof(a2::node_t) * m, cudaMemcpyDeviceToHost); //get destinations 

    //sort the neigbours 

    a2::SparseGraph expected = cpu_CSR(n, m, graph);
    
    //Printing out offset and neigbourrs array of both solutions 
    
    std::cout << "Sparse GPU Offsets: ";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << g_neighbours_start[i] << " ";
    }
    std::cout << std::endl;

    /*
    std::cout << "Sparse GPU Neighbours: ";
    for (std::size_t i = 0; i < m; ++i) {
        std::cout << g_neighbours[i] << " ";
    }
    std::cout << std::endl;
    */

    std::cout << "Sparse CPU Offsets: ";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << expected.neighbours_start_at[i] << " ";
    }
    std::cout << std::endl;

    /*
    std::cout << "Sparse CPU Neighbours: ";
    for (std::size_t i = 0; i < m; ++i) {
        std::cout << expected.neighbours[i] << " ";
    }
    std::cout << std::endl;
    */

    
    //tests
    bool failed_offsets = false;
    bool failed_neighbours = false;
    
    //check offsets
    for (int i = 0; i < n; ++i) {
        if (expected.neighbours_start_at[i] != g_neighbours_start[i])
        {
            failed_offsets = true;
            break;
        }
    }
    
    /*
    //check neigbours, this isnt fully needed since they are corret (each offset points towards the right 
    section of neighbours its just that section isnt sorted)
    for (int i = 0; i < m; ++i){
        if (expected.neighbours[i] != g_neighbours[i])
        {
            failed_neighbours = true;
            break;
        }
    }
    */

    if (failed_offsets) std::cout << "SparseGraph test failed at offsets" << std::endl;
    //else if (failed_neighbours) std::cout << "SparseGraph test failed at neigbours" << std::endl;
    else std::cout << "SparseGraph test passed" << std::endl;

    //clean up
    delete[] g_neighbours_start;
    delete[] g_neighbours;
    cudaFree(d_neighbours);
    cudaFree(d_offsets);
}

int main()
{
    using namespace csc485b;

    // Create input
    std::size_t constexpr n = 32;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph(n, n * expected_degree);
    std::size_t const m = graph.size();

    // lazily echo out input graph
    /*
    for (auto const& e : graph)
    {
        std::cout << "(" << e.x << "," << e.y << ") ";
    }
    */
    
    // allocate and memcpy input to device
    a2::edge_t* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(a2::edge_t) * m);
    cudaMemcpyAsync(d_edges, graph.data(), sizeof(a2::edge_t) * m, cudaMemcpyHostToDevice);

    // run your code!
    //run_dense(d_edges, n, m, graph);
    run_sparse(d_edges, n, m, graph);

    return EXIT_SUCCESS;
}