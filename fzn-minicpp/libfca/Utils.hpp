#pragma once

#include <cmath>
#include <type_traits>

// Compile with C++ only compiler
#ifndef __NVCC__
#define __host__
#define __device__
#endif

namespace Fca
{
    namespace Math
    {
        template<typename T1, typename T2>
        __host__ __device__ inline
        double log(T1 const base, T2 const number)
        {
            static_assert(std::is_arithmetic_v<T1>);
            static_assert(std::is_arithmetic_v<T2>);
            return log2(static_cast<double>(number)) / log2(static_cast<double>(base));
        }

        template<typename T1, typename T2>
        __host__ __device__ inline
        double pow(T1 const base, T2 const exponent)
        {
            static_assert(std::is_arithmetic_v<T1>);
            static_assert(std::is_arithmetic_v<T2>);
            return std::pow(static_cast<double>(base), static_cast<double>(exponent));
        }

        template<typename T1, typename T2>
        __host__ __device__ inline
        double division(T1 const a, T2 const b)
        {
            static_assert(std::is_arithmetic_v<T1>);
            static_assert(std::is_arithmetic_v<T2>);
            return static_cast<double>(a) / static_cast<double>(b);
        }

        template<typename T>
        __host__ __device__ inline
        uint32_t ceilDivPosInt(T const a, T const b)
        {
            static_assert(std::is_integral_v<T>);
            //assert(a >= 0);
            //assert(b > 0);
            auto const _a = static_cast<uint32_t>(a);
            auto const _b = static_cast<uint32_t>(b);
            return (a + b - 1) / b;
        }

        template<typename T1, typename T2>
        __host__ __device__ inline
        uint32_t roundUpPosInt(T1 const a, T2 const b)
        {
            auto const _a = static_cast<uint32_t>(a);
            auto const _b = static_cast<uint32_t>(b);
            return ceilDivPosInt(_a, _b) * _b;
        }

        __host__ __device__ inline
        uint32_t getMod32(uint32_t const a)
        {
            return a & 31u;
        }

        __host__ __device__ inline
        uint32_t getDiv32(uint32_t const a)
        {
            return a >> 5u;
        }
    }

    namespace Bits
    {
        __host__ __device__ inline
        uint32_t getPopCount32(uint32_t a)
        {
            #ifdef __CUDA_ARCH__
            return __popc(a);
            #else
            return __builtin_popcount(a);
            #endif

        }
    }
}