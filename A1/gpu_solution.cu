/**
 * The file in which you will implement your GPU solutions!
 */

#include "algorithm_choices.h"

#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl

#include "cuda_common.h"

namespace csc485b {
    namespace a1 {
        namespace gpu {

            /**
             * The CPU baseline benefits from warm caches because the data was generated on
             * the CPU. Run the data through the GPU once with some arbitrary logic to
             * ensure that the GPU cache is warm too and the comparison is more fair.
             */
            __global__
                void warm_the_gpu(element_t* data, std::size_t invert_at_pos, std::size_t num_elements)
            {
                int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

                // We know this will never be true, because of the data generator logic,
                // but I doubt that the compiler will figure it out. Thus every element
                // should be read, but none of them should be modified.
                if (th_id < num_elements && data[th_id] > num_elements * 100)
                {
                    ++data[th_id]; // should not be possible.
                }
            }

            /*
            * Implementation based of https://en.wikipedia.org/wiki/Bitonic_sorter
            */
            __device__ void bitonic_sort(element_t* data, unsigned int subarray_offset, unsigned int subarray_size, unsigned int th_id, bool direction)
            {
                __shared__ element_t array_chunk[1024];
                array_chunk[threadIdx.x] = data[th_id];
                __syncthreads();

                
                for (unsigned int step = 2; step <= subarray_size; step <<= 1)
                {
                    for (unsigned int substep = step >> 1; substep > 0; substep >>= 1)
                    {
                        unsigned int index = substep ^ threadIdx.x;
                        if (index > threadIdx.x)
                        { 
                            if (((index & step) == 0 && array_chunk[threadIdx.x] > array_chunk[index]) ||
                                ((index & step) != 0 && array_chunk[threadIdx.x] < array_chunk[index])) //Sort the subarray
                            {
                                element_t temp = array_chunk[threadIdx.x];
                                array_chunk[threadIdx.x] = array_chunk[index];
                                array_chunk[index] = temp;
                            }

                            if ((((index & step) != 0 && array_chunk[threadIdx.x] > array_chunk[index]) ||
                                ((index & step) == 0 && array_chunk[threadIdx.x] < array_chunk[index])) && direction) //Sort the subarray
                            {
                                element_t temp = array_chunk[threadIdx.x];
                                array_chunk[threadIdx.x] = array_chunk[index];
                                array_chunk[index] = temp;
                            }
                        }
                        __syncthreads();
                    }
                }
                data[th_id] = array_chunk[threadIdx.x];

            }

            /*
            Based of of: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/ for reversing.
            Reverses one block
            */
            __device__ void reverse_at(element_t* data, unsigned int th_id, std::size_t invert_at_pos, std::size_t n)
            {
                __shared__ element_t smem[1024];
                int t = threadIdx.x;
                int tr = n - 1 - (th_id - invert_at_pos);
                smem[t] = data[th_id];
                __syncthreads();

                if (th_id >= invert_at_pos)
                {
                    data[th_id] = smem[tr];
                }
            }

            /**
            * Your solution. Should match the CPU output.
            */
            __global__ void opposing_sort(element_t* data, std::size_t invert_at_pos, std::size_t num_elements)
            {

                // Data is the global memory, don't forget.

                int const th_id = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int chunk_offset = blockIdx.x * blockDim.x;
                unsigned int chunk_size = blockDim.x;
                unsigned int num_chunks = num_elements / chunk_size;

                chunk_size = chunk_size < num_elements ? chunk_size : num_elements;
                if (th_id < num_elements)
                {
                    bitonic_sort(data, chunk_offset, chunk_size, th_id, 0);
                    __syncthreads();

                    reverse_at(data, th_id, invert_at_pos, num_elements);
                }
                // We need to sync across blocks now. 

                // Sort the array like usual for the last 1/4th of the array just reverse the order. This could be done really quickly
               
            }

           
            /**
             * Performs all the logic of allocating device vectors and copying host/input
             * vectors to the device. Times the opposing_sort() kernel with wall time,
             * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
             */
            void run_gpu_soln(std::vector< element_t > data, std::size_t switch_at, std::size_t n)
            {
                // Kernel launch configurations. Feel free to change these.
                // This is set to maximise the size of a thread block on a T4, but it hasn't
                // been tuned. It's not known if this is optimal.
                std::size_t const threads_per_block = 1024;
                std::size_t const num_blocks = 1;//(n + threads_per_block - 1) / threads_per_block;

                // Allocate arrays on the device/GPU
                element_t* d_data;
                cudaMalloc((void**)&d_data, sizeof(element_t) * n);
                CHECK_ERROR("Allocating input array on device");

                // Copy the input from the host to the device/GPU
                cudaMemcpy(d_data, data.data(), sizeof(element_t) * n, cudaMemcpyHostToDevice);
                CHECK_ERROR("Copying input array to device");

                // Warm the cache on the GPU for a more fair comparison
                warm_the_gpu << < num_blocks, threads_per_block >> > (d_data, switch_at, n);

                // Time the execution of the kernel that you implemented
                auto const kernel_start = std::chrono::high_resolution_clock::now();
                opposing_sort << < num_blocks, threads_per_block >> > (d_data, switch_at, n);
                auto const kernel_end = std::chrono::high_resolution_clock::now();
                CHECK_ERROR("Executing kernel on device");

                // After the timer ends, copy the result back, free the device vector,
                // and echo out the timings and the results.
                cudaMemcpy(data.data(), d_data, sizeof(element_t) * n, cudaMemcpyDeviceToHost);
                CHECK_ERROR("Transferring result back to host");
                cudaFree(d_data);
                CHECK_ERROR("Freeing device memory");

                std::cout << "GPU Solution time: "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
                    << " ns" << std::endl;

                for (auto const x : data) std::cout << x << " "; std::cout << std::endl;
            }

        } // namespace gpu
    } // namespace a1
} // namespace csc485b