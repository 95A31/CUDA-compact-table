#pragma once

#include <libgpu/libgpu.cuh>

#include "global_constraints/table.hpp"

class TableGPU : public Table
{
    public:
        Fca::u32 static constexpr BlockSize = 32;
        Fca::u32 static constexpr nWordsPerBlock = BlockSize / 32;

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
        cudaStream_t cuStream;

    public:
        TableGPU(std::vector<var<int>::Ptr> & vars, std::vector<std::vector<int>> & tuples);
        void post() override;
        void propagate() override;
    protected:
        void allocateInstanceData() override;
        // void initPropagateLowLatency();
        // void propagateBase();
};

__global__ void updateDomainsKernel(Table::InstanceData id);
// __global__ void calcIntervalsKernel(Fca::i32 nActivities, Cumulative::StartInterval const * si_d, Fca::i32 const * p_d, Fca::i32 * nIntervals_d, Cumulative::Interval * i_d);
// __global__ void resetConsistencyKernel(bool * isConsistent_d);
// __global__ void updateBoundsKernel(Fca::i32 nActivities, Fca::i32 const * h_d, Fca::i32 const * p_d, Fca::i32 c, Fca::i32 * nIntervals_d, Cumulative::Interval const * i_d, Cumulative::StartInterval * si_d, bool * isConsistent_d);
