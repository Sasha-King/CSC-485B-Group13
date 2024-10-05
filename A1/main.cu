#include <cstddef>  // std::size_t type
#include <iostream> // std::cout, std::endl
#include <vector>

#include "algorithm_choices.h"
#include "data_generator.h"
#include "data_types.h"
#include "cuda_common.h"

int main()
{
    std::size_t const data_size_limit = pow(2, 20);

    element_t* data_cpu = new element_t[data_size_limit];

    for (int i = 0; i <= 20; ++i)
    {
        
        std::size_t size = pow(2, i);
        std::cout << "pow(2, " << i << ") ";
        std::size_t const switch_at = 3 * (i >> 2);
        auto data = csc485b::a1::generate_uniform<element_t>(size);

        csc485b::a1::cpu::run_cpu_baseline(data, switch_at, size);

        memcpy(data_cpu, data.data(), sizeof(element_t) * size);

        csc485b::a1::gpu::run_gpu_soln(data, switch_at, size);

        
        for (std::size_t j = 0; j < size; ++j)
        {
            if (data[j] != data_cpu[j])
            {
                std::cout << "Mismatch at index " << j << ": expected " << data_cpu[j] << ", got " << data[j] << std::endl;
                return EXIT_SUCCESS;
            }
        }
    }

    return EXIT_SUCCESS;
}