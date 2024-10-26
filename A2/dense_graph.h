/**
 * The file in which you will implement your DenseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

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

/**
  * Repopulates the adjacency matrix as a new graph that represents
  * the two-hop neighbourhood of input graph g
  */
__global__
void two_hop_reachability( DenseGraph g, DenseGraph output)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ node_t a[1024];
    __shared__ node_t b[1024];

    size_t n = g.n;
    int tileSize = blockDim.x;

    //Matrix multitply
    if (row < n && col < n)
    {
        int sum = 0;

        for (int tileOffset = 0; tileOffset < (tileSize + n - 1) / tileSize; ++tileOffset) {
            a[threadIdx.y * tileSize + threadIdx.x] = g.adjacencyMatrix[row * n + (tileOffset * tileSize + threadIdx.x)];
            b[threadIdx.y * tileSize + threadIdx.x] = g.adjacencyMatrix[(tileOffset * tileSize + threadIdx.y) * n + col];

            __syncthreads();

            
            for (int i = 0; i < tileSize; i++)
            {
                sum += a[threadIdx.y * tileSize + i] * b[i * tileSize + threadIdx.x];
            }
            __syncthreads();
        }
        output.adjacencyMatrix[row * n + col] = sum;
    }


    //Set diagonals 

     //Clamp values


    return;
}

        } // namespace gpu
    } // namespace a2
} // namespace csc485b
