#include "table.cuh"

#include <libfca/BitMatrix.hpp>
#include <libfca/Array.hpp>
#include <libfca/Timer.hpp>
#include <libfca/Utils.hpp>
#include <libgpu/libgpu.cuh>

#include <numeric>

using namespace std;
using namespace Fca;
using namespace Gpu::Memory;
using namespace Gpu::Parallel;
using namespace Fca::Math;
using namespace Fca::Bits;


TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) :
    Table(vars, tuples)
{
    assert(sizeof(InstanceData) <= 265); // We pass it to the kernel by value!

    // CUDA initialization
    cudaDeviceProp cu_prop;
    cudaGetDeviceProperties(&cu_prop, 0);
    smCount = cu_prop.multiProcessorCount;
    cudaStreamCreate(&cuStream);
}

void TableGPU::allocateInstanceData()
{
    instData_h = mallocHost<InstanceData>(sizeof(InstanceData));
    instData_d = mallocHost<InstanceData>(sizeof(InstanceData)); // Host is correct

    u32 const readOnlyMemSize =
        LinearAllocator::DefaultAlign + tuplesMemSize;

    readOnlyAlloc_h = new LinearAllocator(mallocHost<void>(readOnlyMemSize), readOnlyMemSize);
    instData_h->tuples = readOnlyAlloc_h->allocate<i32>(tuplesMemSize);

    readOnlyAlloc_d = new LinearAllocator(mallocDevice<void>(readOnlyMemSize), readOnlyMemSize);
    instData_d->tuples = readOnlyAlloc_d->allocate<i32>(tuplesMemSize);

    u32 const inputOutputMemSize =
        LinearAllocator::DefaultAlign + changedVarsMemSize +
        LinearAllocator::DefaultAlign + unfixedVarsMemSize +
        LinearAllocator::DefaultAlign + domainsInfoMemSize +
        LinearAllocator::DefaultAlign + sizeof(u32) + // nValidTuples
        LinearAllocator::DefaultAlign + validTuplesMemSize +
        LinearAllocator::DefaultAlign + domainsMemSize;

    inputOutputAlloc_h = new LinearAllocator(mallocHost<void>(inputOutputMemSize), inputOutputMemSize);
    instData_h->changedVars = inputOutputAlloc_h->allocate<u32>(changedVarsMemSize);
    instData_h->unfixedVars = inputOutputAlloc_h->allocate<u32>(unfixedVarsMemSize);
    instData_h->domainsInfo = inputOutputAlloc_h->allocate<DomainsInfo>(domainsInfoMemSize);
    instData_h->nValidTuples = inputOutputAlloc_h->allocate<u32>(sizeof(u32));
    instData_h->validTuples = inputOutputAlloc_h->allocate<u32>(validTuplesMemSize);
    instData_h->domains = inputOutputAlloc_h->allocate<u32>(domainsMemSize);

    inputOutputAlloc_d = new LinearAllocator(mallocDevice<void>(inputOutputMemSize), inputOutputMemSize);
    instData_d->changedVars = inputOutputAlloc_d->allocate<u32>(changedVarsMemSize);
    instData_d->unfixedVars = inputOutputAlloc_d->allocate<u32>(unfixedVarsMemSize);
    instData_d->domainsInfo = inputOutputAlloc_d->allocate<DomainsInfo>(domainsInfoMemSize);
    instData_d->nValidTuples = inputOutputAlloc_d->allocate<u32>(sizeof(u32));
    instData_d->validTuples = inputOutputAlloc_d->allocate<u32>(validTuplesMemSize);
    instData_d->domains = inputOutputAlloc_d->allocate<u32>(domainsMemSize);

    beginOutputMem_h = instData_h->nValidTuples;
    beginOutputMem_d = instData_d->nValidTuples;
    void const * const endOutputMem_h = inputOutputAlloc_h->getFreeMemory();
    outputMemSize = reinterpret_cast<std::uintptr_t>(endOutputMem_h) - reinterpret_cast<std::uintptr_t>(beginOutputMem_h);
}

void TableGPU::post()
{
    for (auto const & v : vars)
    {
        v->propagateOnDomainChange(this);
    }

    allocateInstanceData();
    tmpValidTuples = static_cast<u32*>(malloc(validTuplesMemSize));

    initializeInstanceData(instData_h);

    instData_d->nVars = instData_h->nVars;
    instData_d->nTuples = instData_h->nTuples;
    instData_d->maxWordsInDomain = instData_h->maxWordsInDomain;

    cudaMemcpyAsync(
        readOnlyAlloc_d->getMemory(),
        readOnlyAlloc_h->getMemory(),
        readOnlyAlloc_h->getUsedMemorySize(),
        cudaMemcpyHostToDevice,
        cuStream);

    // Initialize last sizes
    for (u32 vIdx = 0; vIdx < instData_h->nVars; vIdx += 1)
    {
        lastSize.set(vIdx, INT_MAX);
    }

    propagate();
}
void TableGPU::propagate()
{
    updateInstanceData(instData_h);
    instData_d->nChangedVars = instData_h->nChangedVars;
    instData_d->nUnfixedVars = instData_h->nUnfixedVars;

    updateValidTuples(instData_h);

    Timer::begin("TableGPU::updateDomains");
    if (*instData_h->nValidTuples > 0 and instData_h->nUnfixedVars > 0)
    {
        clearDomains(instData_h, instData_h->domains);

        cudaMemcpyAsync(
            inputOutputAlloc_d->getMemory(),
            inputOutputAlloc_h->getMemory(),
            inputOutputAlloc_h->getUsedMemorySize(),
            cudaMemcpyHostToDevice,
            cuStream);

        u32 const gridSizeX = instData_h->nUnfixedVars;
        u32 const gridSizeY = ceilDivPosInt(*instData_h->nValidTuples, 128u);
        u32 const sharedMemSize =
            LinearAllocator::DefaultAlign + domainsMemSize +
            LinearAllocator::DefaultAlign + domainsInfoMemSize;
        dim3 gridSize(gridSizeX,gridSizeY,1);

        updateDomainsKernel<<<gridSize, 32, sharedMemSize, cuStream>>>(*instData_d, domainsMemSize, domainsInfoMemSize);

        cudaMemcpyAsync(
            beginOutputMem_h,
            beginOutputMem_d,
            outputMemSize,
            cudaMemcpyDeviceToHost,
            cuStream);

        cudaStreamSynchronize(cuStream);
    }
    Timer::end("TableGPU::updateDomains");
    filterDomains(instData_h);
}

__global__
void updateDomainsKernel(Table::InstanceData instData, u32 const domainsMemSize, u32 const domainsInfoMemSize)
{
    assert(blockDim.x == 32);

    __shared__ u32 * domains_s;
    __shared__ Table::DomainsInfo * domainsInfo_s;
    extern __shared__ u32 sMem[];

    if (*instData.nValidTuples > 0)
    {
        // Copy instance data in registers
        Table::InstanceData instData_r = instData;

        if (threadIdx.x == 0)
        {
            LinearAllocator allocator(sMem, getSharedMemorySize());
            domains_s = allocator.allocate<u32>(domainsMemSize);
            domainsInfo_s = allocator.allocate<Table::DomainsInfo>(domainsInfoMemSize);
        }
        __syncwarp();

        //for (u32 uvIdx = 0; uvIdx < instData_r.nUnfixedVars; uvIdx += 1)
        u32 const uvIdx = blockIdx.x;
        {
            u32 const vIdx = instData_r.unfixedVars[uvIdx];
            auto dInfo = instData.domainsInfo[vIdx];
            domainsInfo_s[uvIdx] = dInfo;

            u32 const firstWordIdx = dInfo.firstWordIdx;
            u32 const nWords = dInfo.nWords;
            for (u32 wIdx = threadIdx.x; wIdx < nWords; wIdx += 32)
            {
                domains_s[firstWordIdx + wIdx] = 0;
            }
            __syncwarp();
        }

        u32 bvtIdx, evtIdx;
        getBeginEnd(&bvtIdx,&evtIdx, blockIdx.y, gridDim.y, *instData_r.nValidTuples);
        for (u32 vtIdx = bvtIdx; vtIdx < evtIdx; vtIdx += 1)
        {
            u32 const tIdx = instData_r.validTuples[vtIdx];
            //for (u32 uvIdx = threadIdx.x; uvIdx < instData_r.nUnfixedVars; uvIdx += 32)
            {
                u32 const vIdx = instData_r.unfixedVars[uvIdx];
                i32 const val = instData_r.tuples[(tIdx * instData_r.nVars) + vIdx];
                u32 const valIdx = val - domainsInfo_s[uvIdx].firstBitValue;
                u32 const valWordIdx = getDiv32(valIdx);
                u32 const valMask = 1 << 31 - getMod32(valIdx);
                domains_s[domainsInfo_s[uvIdx].firstWordIdx + valWordIdx] |= valMask;
            }
            __syncwarp();
        }

        // Write domains in global
        //for (u32 uvIdx = 0; uvIdx < instData_r.nUnfixedVars; uvIdx += 1)
        {
            u32 const firstWordIdx = domainsInfo_s[uvIdx].firstWordIdx;
            u32 const nWords = domainsInfo_s[uvIdx].nWords;
            for (u32 wIdx = threadIdx.x; wIdx < nWords; wIdx += 32)
            {
                atomicOr(instData_r.domains + firstWordIdx + wIdx, domains_s[firstWordIdx + wIdx]) ;
            }
        }
    }
}