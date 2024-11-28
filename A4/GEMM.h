/**
 * The file in which you will implement your DenseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
    namespace a4 {

        namespace gpu {

        __device__ void squareMatrix(const node_t* input, node_t* output, size_t N)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            __shared__ node_t a[1024];
            __shared__ node_t b[1024];

            int tileSize = blockDim.x;

            //Matrix multitply
            if (row < N && col < N)
            {
                int sum = 0;

                for (int tileOffset = 0; tileOffset < (tileSize + N - 1) / tileSize; ++tileOffset) {
                    a[threadIdx.y * tileSize + threadIdx.x] = input[row * N + (tileOffset * tileSize + threadIdx.x)];
                    b[threadIdx.y * tileSize + threadIdx.x] = input[(tileOffset * tileSize + threadIdx.y) * N + col];

                    __syncthreads();

                    for (int i = 0; i < tileSize; i++)
                    {
                        sum += a[threadIdx.y * tileSize + i] * b[i * tileSize + threadIdx.x];
                    }
                    __syncthreads();
                }
                output[row * N + col] = fminf(fmaxf(sum, 0), 1); // clamp between 0, 1
            }
        }

        } // namespace gpu
    } // namespace a4
} // namespace csc485b