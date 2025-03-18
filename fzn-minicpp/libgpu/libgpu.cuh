#pragma once

#ifdef __NVCC__
#include <cstdint>
#include <cassert>
#include <cuda_runtime_api.h>

namespace Gpu
{
    namespace Parallel
    {
        __device__
        inline void getBeginEnd(uint32_t * begin, uint32_t * end, uint32_t index, uint32_t workers, uint32_t jobs)
        {
            uint32_t const jobsPerWorker = (jobs + workers - 1) / workers; // Fast ceil integer division
            *begin = jobsPerWorker * index;
            *end = min(jobs, *begin + jobsPerWorker);
        }
    }

    namespace Memory
    {
        template<typename T>
        T * mallocStd(uint32_t size)
        {
            void * memory = std::malloc(size);
            assert(memory != nullptr);
            return reinterpret_cast<T *>(memory);
        }

        template<typename T>
        T * mallocHost(uint32_t size)
        {
            void * memory = nullptr;
            cudaError_t status = cudaMallocHost(&memory, size);
            assert(status == cudaSuccess);
            assert(memory != nullptr);
            return reinterpret_cast<T *>(memory);
        }

        template<typename T>
        T * mallocDevice(uint32_t size)
        {
            void * memory = nullptr;
            cudaError_t status = cudaMalloc(&memory, size);
            assert(status == cudaSuccess);
            assert(memory != nullptr);
            return reinterpret_cast<T *>(memory);
        }

        template<typename T>
        T * mallocManaged(uint32_t size)
        {
            void * memory = nullptr;
            cudaError_t status = cudaMallocManaged(&memory, size);
            assert(status == cudaSuccess);
            assert(memory != nullptr);
            return reinterpret_cast<T *>(memory);
        }

        __device__ inline
        uint32_t getSharedMemorySize()
        {
            uint32_t size;
            asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(size));
            return size;
        }
    }
}
#endif