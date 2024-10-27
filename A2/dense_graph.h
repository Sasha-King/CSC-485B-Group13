/**
 * The file in which you will implement your DenseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

#define FULL_MASK 0xffffffff //mask

namespace csc485b {
namespace a2      {

/**
 * A DenseGraph is optimised for a graph in which the number of edges
 * is close to n(n-1). It is represented using an adjacency matrix.
 */
struct DenseGraph
{
  std::size_t n; /**< Number of nodes in the graph. */
  node_t * adjacencyMatrix; /** Pointer to an n x n adj. matrix */

  /** Returns number of cells in the adjacency matrix. */
  __device__ __host__ __forceinline__
  std::size_t matrix_size() const { return n * n; }
};

namespace gpu {

/**
 * Constructs a DenseGraph from an input edge list of m edges.
 *
 * @pre The pointers in DenseGraph g have already been allocated.
 */
__global__
void build_graph(DenseGraph g, edge_t const* edge_list, std::size_t m)
{
    unsigned int th_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = g.n;

    for (unsigned int i = th_id; i < m; i += gridDim.x * blockDim.x) {
        unsigned int x = edge_list[i].x;
        unsigned int y = edge_list[i].y;

        g.adjacencyMatrix[x * n + y] = 1;
        g.adjacencyMatrix[y * n + x] = 1;
    }

    return;
}

//Warp primitives https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
__device__ void squareMatrix(const node_t* input, node_t* output, size_t N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //__shared__ node_t a[1024];
    //__shared__ node_t b[1024];

    int tileSize = blockDim.x; // Should be 32 

    //Matrix multitply
    if (row < N && col < N){
        int sum = 0;
        
        //This stuff should be able to be sped up 
        for (int tileOffset = 0; tileOffset < (tileSize + N - 1) / tileSize; ++tileOffset) {
                 
            int a = input[row * N + (tileOffset * tileSize + threadIdx.x)]; //row

            int b = input[(tileOffset * tileSize + threadIdx.y) * N + col]; //col 

            // Accumulate results within the warp using shuffle
            for (int i = 0; i < tileSize; i++) {
                
                //unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < N); //This just seems to slow it down since output is still correct 

                //Syncs threads and "collects" the lane, FULL_MASK works better here then active (id assume its because all threads are generaly active) 
                // Might be able to be made faster just not sure how 
                int a_lane = __shfl_sync(FULL_MASK, a, i, tileSize); 
                int b_lane = __shfl_sync(FULL_MASK, b, i, tileSize); 
                       
                sum += a_lane * b_lane;
                
            }    
      
            //Shared memory code
            /* 
            a[threadIdx.y * tileSize + threadIdx.x] = input[row * N + (tileOffset * tileSize + threadIdx.x)];
            b[threadIdx.y * tileSize + threadIdx.x] = input[(tileOffset * tileSize + threadIdx.y) * N + col];

            __syncthreads();

            for (int i = 0; i < tileSize; i++)
            {
                sum += a[threadIdx.y * tileSize + i] * b[i * tileSize + threadIdx.x];
            }
            __syncthreads();
            */          
        }
        output[row * N + col] = fminf(fmaxf(sum, 0), 1); // clamp between 0, 1
    }
}

//This is so simple I dont think warps would help much 
__device__ void removeSelfLoops(node_t* matrix, const size_t& N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        if (row == col) matrix[row * N + col] = 0;
    }
    return;
}

/**
  * Repopulates the adjacency matrix as a new graph that represents
  * the two-hop neighbourhood of input graph g
  */
__global__
void two_hop_reachability( DenseGraph g, DenseGraph output)
{
    squareMatrix(g.adjacencyMatrix, output.adjacencyMatrix, g.n);
    removeSelfLoops(output.adjacencyMatrix, g.n);
    return;
}

        } // namespace gpu
    } // namespace a2
} // namespace csc485b
