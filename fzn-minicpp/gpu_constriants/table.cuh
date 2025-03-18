#pragma once

#include <libgpu/libgpu.cuh>

#include "global_constraints/table.hpp"

class TableGPU : public Table
{
    private:
        InstanceData * instData_d;
        InstanceData * instData_h;

        LinearAllocator * readOnlyAlloc_h;
        LinearAllocator * readOnlyAlloc_d;

        LinearAllocator * inputOutputAlloc_h;
        LinearAllocator * inputOutputAlloc_d;

        void * beginOutputMem_h;
        void * beginOutputMem_d;
        Fca::u32 outputMemSize;

        // CUDA
        Fca::u32 smCount;
        cudaStream_t cuStream;

    public:
        TableGPU(std::vector<var<int>::Ptr> & vars, std::vector<std::vector<int>> & tuples);
        void post() override;
        void propagate() override;
    protected:
        void allocateInstanceData() override;
};

__global__ void updateDomainsKernel(Table::InstanceData id, Fca::u32 domainsMemSize, Fca::u32 domainsInfoMemSize);

