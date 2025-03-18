#pragma once

#include <cstdint>
#include <cassert>

// Compile with C++ only compiler
#ifndef __NVCC__
#define __host__
#define __device__
#endif


class LinearAllocator
{
    public:
        uint32_t static constexpr DefaultAlign = 4; // 32-bit aligned

    private:
        std::uintptr_t const begin;
        std::uintptr_t current;
        std::uintptr_t const end;

    public:
        __host__ __device__ inline LinearAllocator(void * memory, std::uint32_t size);
        template<typename T>
        __host__ __device__ inline T * allocate(std::uint32_t size = sizeof(T), std::uint32_t align = alignof(T));
        __host__ __device__ inline void clear() {current = begin;};
        __host__ __device__ inline void * getMemory() const {return reinterpret_cast<void *>(begin);};
        __host__ __device__ inline void * getFreeMemory() const {return reinterpret_cast<void *>(current);};
        __host__ __device__ inline std::uint32_t getFreeMemorySize() const {return static_cast<std::uint32_t>(end - current);};
        __host__ __device__ inline std::uint32_t getUsedMemorySize() const {return static_cast<std::uint32_t>(current - begin);};
        __host__ __device__ inline std::uint32_t getTotalMemorySize() const {return static_cast<std::uint32_t>(end - begin);};
};

__host__ __device__
LinearAllocator::LinearAllocator(void * memory, std::uint32_t size) :
        begin(reinterpret_cast<std::uintptr_t>(memory)),
        current(begin),
        end(begin + static_cast<std::uintptr_t>(size))
{
    assert(begin < end);
}

template<typename T>
__host__ __device__
T * LinearAllocator::allocate(std::uint32_t size, std::uint32_t align)
{
    std::uintptr_t memory = current;
    std::uint32_t const offset = memory % align;
    if (offset != 0)
    {
        memory += align - offset;
    }
    current = memory + size;
    assert(current < end); // We always have some extra space
    return reinterpret_cast<T *>(memory);
}
