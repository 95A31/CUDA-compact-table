#pragma once

#include <cstdint>

// Compile with C++ only compiler
#ifndef __NVCC__
#define __host__
#define __device__
#endif

namespace Fca
{
    // Aliases
    using i8 = int8_t;
    using u8 = uint8_t;
    using i16 = int16_t;
    using u16 = uint16_t;
    using i32 = int32_t;
    using u32 = uint32_t;
    using i64 = int64_t;
    using u64 = uint64_t;
    struct alignas(16) u128 { u32 w0, w1, w2, w3;};
    struct alignas(32) u256 { u32 w0, w1, w2, w3, w4, w5, w6, w7;};
    struct alignas(64) u512 { u32 w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15; };

    template <typename T>
    __host__ __device__
    inline void setZero(T * const a) {
        auto * const _a = (u64 *) a;
        constexpr u32 count = sizeof(T) / sizeof(u64);
        #pragma unroll
        for (u32 i = 0; i < count; i += 1) _a[i] = 0;
    }

    template <typename T>
    __host__ __device__
    inline bool isZero(T * const a) {
        auto * const _a = (u64 *) a;
        constexpr u32 count = sizeof(T) / sizeof(u64);
        u64 result = 0;
        #pragma unroll
        for (u32 i = 0; i < count; i += 1) result |= _a[i];
        return result == 0;
    }

    template <typename T>
    __host__ __device__
    inline void bitwiseOr(T * const dst, T const * const mask) {
        auto * const _dst = (u64 *) dst;
        auto const * const _mask = (u64 const *) mask;
        constexpr u32 count = sizeof(T) / sizeof(u64);
        #pragma unroll
        for (u32 i = 0; i < count; i += 1) _dst[i] |= _mask[i];
    }

    template <typename T>
    __host__ __device__
    inline void bitwiseAnd(T * const dst, T const * const mask) {
        auto * const _dst = (u64 *) dst;
        auto const * const _mask = (u64 const *) mask;
        constexpr u32 count = sizeof(T) / sizeof(u64);
        #pragma unroll
        for (u32 i = 0; i < count; i += 1) _dst[i] &= _mask[i];
    }

    template <typename T>
    __host__ __device__
    inline bool isAndNotZero(T const * const a, T const * const b) {
        auto const * const _a = (u64 const *) a;
        auto const * const _b = (u64 const *) b;
        constexpr u32 count = sizeof(T) / sizeof(u64);
        u64 result = 0;
        #pragma unroll
        for (u32 i = 0; i < count; i += 1) result |= (_a[i] & _b[i]);
        return result != 0;
    }


}
