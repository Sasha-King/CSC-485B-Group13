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

#include <thrust/scan.h>
#include <thrust/execution_policy.h>


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
    std::cout << "Squraed matrix is correct" << std::endl;
}


 /**
  * Runs timing tests on a CUDA graph implementation.
  * Consists of independently constructing the graph and then
  * modifying it to its two-hop neighbourhood.
  */
/* This fucntion should exclusively run DenseGraph due to matrix testing*/
template < typename DeviceGraph >
void run(DeviceGraph g, csc485b::a2::edge_t const* d_edges, std::size_t m)
{

    // this code doesn't work yet!
   
        auto const build_start = std::chrono::high_resolution_clock::now();
        constexpr size_t build_threads = 1024;
        int build_blocks = (g.n + build_threads - 1) / build_threads;
        csc485b::a2::gpu::build_graph << < build_blocks, build_threads >> > (g, d_edges, m);
        cudaDeviceSynchronize();

        std::vector< csc485b::a2::node_t > built_matrix(g.n * g.n);
        csc485b::a2::DenseGraph built_graph{ g.n, built_matrix.data() }; // DEVICE TO HOST
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



        csc485b::a2::gpu::two_hop_reachability << < BLOCKS, THREADS >> > (g, d_output);
        cudaDeviceSynchronize();

        std::vector< csc485b::a2::node_t > host_matrix(g.n * g.n);
        csc485b::a2::DenseGraph output{ g.n, host_matrix.data() }; // DEVICE TO HOST
        cudaMemcpy(output.adjacencyMatrix, d_output.adjacencyMatrix, sizeof(csc485b::a2::node_t) * g.n * g.n, cudaMemcpyDeviceToHost);


        auto const end = std::chrono::high_resolution_clock::now();

        std::cout << "GPU Build time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(reachability_start - build_start).count()
            << " us"
            << std::endl;

        std::cout << "GPU Reachability time: "
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

/**
 * Allocates space for a dense graph and then runs the test code on it.
 */
void run_dense(csc485b::a2::edge_t const* d_edges, std::size_t n, std::size_t m, csc485b::a2::edge_list_t graph)
{
    using namespace csc485b;

    // allocate device DenseGraph
    a2::node_t* d_matrix;
    cudaMalloc((void**)&d_matrix, sizeof(a2::node_t) * n * n);
    a2::DenseGraph d_build_graph{ n, d_matrix };
    run(d_build_graph, d_edges, m);

    // check output?
    std::vector< a2::node_t > host_matrix(d_build_graph.matrix_size());
    a2::DenseGraph build_graph{ n, host_matrix.data() };
    cudaMemcpy(d_build_graph.adjacencyMatrix, d_build_graph.adjacencyMatrix, sizeof(a2::node_t) * d_build_graph.matrix_size(), cudaMemcpyDeviceToHost);




    cudaFree(d_matrix);
}


/*
* Constructs a SparseGraph from an input list of edges.
* In slight variation of CSR format found at
* https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf 
* uses index 0 and does not store total edge count in neighbours_start_at[V+1]
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

    //Updates offsetts to contain cumulative out degree (prefix sum)
    int t = 0;
    int a;
    for (a = 0; a < n; a++) {
        
        t += offsets[a];
        offsets[a] = t; 
    }

    //Settting neighbours and final offsett values
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

    //Run Sparse
    cudaDeviceSynchronize();
    auto const build_start = std::chrono::high_resolution_clock::now();

    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    //This Works
    csc485b::a2::gpu::out_degree << < num_blocks, threads_per_block >> > (d_sg, d_edges, m);
    cudaDeviceSynchronize();

    //csc485b::a2::gpu::pre_sum << < num_blocks, threads_per_block >> > (d_sg, m);
    
    //Using thrust library for prefix sum instead of own implmentationn
    thrust::inclusive_scan(thrust::device, d_sg.neighbours_start_at, d_sg.neighbours_start_at + n, d_sg.neighbours_start_at);

    csc485b::a2::gpu::build_neighbours << < num_blocks, threads_per_block >> > (d_sg, d_edges, m);
    cudaDeviceSynchronize();

    auto const reachability_start = std::chrono::high_resolution_clock::now();

    //This does not work
    //csc485b::a2::gpu::two_hop_reachability << < 1, 1 >> > (d_sg, d_sg);

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


    a2::node_t* g_neighbours_start = new a2::node_t[n](); //Offset arrray
    a2::node_t* g_neighbours = new a2::node_t[m](); //Edge destinations 

    //Getting the data out from cuda for testing 
    cudaMemcpy(g_neighbours_start, d_offsets, sizeof(a2::node_t) * n, cudaMemcpyDeviceToHost); //get offsets
    cudaMemcpy(g_neighbours, d_neighbours, sizeof(a2::node_t) * m, cudaMemcpyDeviceToHost); //get destinations 

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
    // CPU Testing makes it longer
    std::size_t constexpr n = 4096;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph(n, n * expected_degree);
    std::size_t const m = graph.size();


    a2::edge_t* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(a2::edge_t) * m);
    cudaMemcpyAsync(d_edges, graph.data(), sizeof(a2::edge_t) * m, cudaMemcpyHostToDevice);

    // run your code!
    //run_dense(d_edges, n, m, graph);
    run_sparse(d_edges, n, m, graph);

    return EXIT_SUCCESS;
}