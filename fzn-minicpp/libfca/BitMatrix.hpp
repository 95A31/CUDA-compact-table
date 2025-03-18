/*
 * fzn-minicpp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * fzn-minicpp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c) 2022. by Fabio Tardivo
 */

#pragma once

#include <libfca/Types.hpp>

namespace Fca
{
    class BitMatrix
    {
        // Members
        protected:
            u32 const rows;
            u32 const cols;
            u32 * const data;

        // Functions
        public:
            __host__ __device__ inline BitMatrix(u32 rows, u32 cols, u32 * data);
            __host__ __device__ inline static void set(u32 rows, u32 cols, u32 * data, u32 rIdx, u32 cIdx, bool value);
            __host__ __device__ inline static bool get(u32 rows, u32 cols, u32 const * data, u32 rIdx, u32 cIdx);
            __host__ __device__ inline static u32 getDataSize(u32 rows, u32 cols);
            __host__ __device__ inline static void print(u32 rows, u32 cols, u32 const * data, char endl = '\n');

            template<typename T>
            __host__ __device__ inline static T * getRowAs(u32 rows, u32 cols, u32 * data, u32 rIdx);

    };

    __host__ __device__
    BitMatrix::BitMatrix(u32 rows, u32 cols, u32 * data) :
           rows(rows),
           cols(cols),
           data(data)
    {
        assert(rows > 0);
        assert(cols > 0);
        assert(cols % 32 == 0);
    }


    __host__ __device__ inline
    void BitMatrix::set(u32 rows, u32 cols, u32 * data, u32 rIdx, u32 cIdx, bool value)
    {
        assert(rIdx < rows);
        assert(cIdx < cols);

        u64 const _rIdx = rIdx;
        u64 const _cIdx = cIdx;
        u64 const _cols = cols;
        u64 const bIdx = _rIdx * _cols + _cIdx;
        u32 const bWord = bIdx / 32;
        u32 const bOffset = bIdx % 32;
        u32 const wMask = 1 << (31 - bOffset);
        if (value)
        {
           data[bWord] |= wMask;
        }
        else
        {
            data[bWord] &= ~wMask;
        }
    }

    __host__ __device__ inline
    bool BitMatrix::get(u32 rows, u32 cols, u32 const * data, u32 rIdx, u32 cIdx)
    {
        assert(rIdx < rows);
        assert(cIdx < cols);

        u64 const _rIdx = rIdx;
        u64 const _cIdx = cIdx;
        u64 const _cols = cols;
        u64 const bIdx = _rIdx * _cols + _cIdx;
        u32 const bWord = bIdx / 32;
        u32 const bOffset = bIdx % 32;
        u32 const wMask = 1 << (31 - bOffset);
        return (data[bWord] & wMask) != 0;
    }


    __host__ __device__ inline
    u32 BitMatrix::getDataSize(u32 rows, u32 cols)
    {
        assert(rows > 0);
        assert(cols > 0);
        assert(cols % 32 == 0);
        u64 const _rows = rows;
        u64 const _cols = cols;
        return (_rows * _cols) / 8; // Size in bytes (8-bits)
    }

    __host__ __device__ inline
    void BitMatrix::print(u32 rows, u32 cols, u32 const * data, char endl)
    {
        assert(cols % 32 == 0);
        for (u32 rIdx = 0; rIdx < rows; rIdx += 1)
        {
            for (u32 cIdx = 0; cIdx < cols; cIdx += 1)
            {
                printf("%c ", get(rows, cols, data, rIdx, cIdx) ? '1' : '0');
            }
            printf("%c", endl);
        }
#ifndef __CUDA_ARCH__
        fflush(stdout);
#endif
    }

    template<typename T>
    __host__ __device__ inline
    T * BitMatrix::getRowAs(u32 rows, u32 cols, u32 * data, u32 rIdx)
    {
        u32 const wIdx = (cols / 32) * rIdx;
        assert(reinterpret_cast<std::uintptr_t>(data + wIdx) % alignof(T) == 0);
        return reinterpret_cast<T*>(data + wIdx);
    }
}