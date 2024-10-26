/**
 * The file in which you will implement your SparseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2      {

/*
* A SparseGraph is optimised for a graph in which the number of edges
* is close to cn, for a small constanct c. It is represented in CSR format.
*/
struct SparseGraph
{
    std::size_t n; /**< Number of nodes in the graph. */
    std::size_t m; /**< Number of edges in the graph. */
    node_t * neighbours_start_at; /** Pointer to an n=|V| offset array */
    node_t * neighbours; /** Pointer to an m=|E| array of edge destinations */
};

namespace gpu {

/*
* Constructs a SparseGraph from an input list of edges using the GPU.
* In slight variation of CSR format https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf (uses index 0 and does not store E in neighbours_start_at[V+1])
* Also the way the edges are formated there is a (1,0) and a (0,1) edge
* @pre The pointers in SparseGraph g have already been allocated.
*/
__global__ 
void build_graph( SparseGraph g, edge_t const * edge_list, std::size_t m )
{
    unsigned int th_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step One, store out-degree of each vertex in N (neighbours_start_at) 
    // First pass over input
    for (size_t i = th_id; i < g.m; i += gridDim.x * blockDim.x){
        node_t x = edge_list[i].x;
        atomicAdd(&g.neighbours_start_at[x], 1); //stores out degree of each node
    }
    __syncthreads();
    
    //WIP This shouldnt take this long but im tired 
    //This will cause it to fail over multiple blocks (m > 256 it breaks) 
    if (th_id == 0) {
        int temp = 0;
        int a;
        for (a = 0; a < g.n; a++) {
            temp += g.neighbours_start_at[a];
            g.neighbours_start_at[a] = temp;
        }
    }
    __syncthreads();

    //Settting neighbours and final offsett values
    for (unsigned int i = th_id; i < g.m; i += gridDim.x * blockDim.x) {
        
        node_t x = edge_list[i].x;
        node_t y = edge_list[i].y;
        
       unsigned int pos = atomicSub(&g.neighbours_start_at[x], 1); //atomic add returns orginal x before incremenation
       g.neighbours[pos-1] = y;
       
    } 
}


__global__
void two_hop_reachability( SparseGraph g, SparseGraph output )
{
    // IMPLEMENT ME!
    // algorithm unknown
    return;
}

        } // namespace gpu
    } // namespace a2
} // namespace csc485b