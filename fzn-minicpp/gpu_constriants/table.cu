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
using namespace Fca::Math;
using namespace Fca::Bits;


TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) :
    Table(vars, tuples)
{
    assert(sizeof(InstanceData) <= 265); // We pass it to the kernel by value!

    // CUDA initialization
    cudaDeviceProp cu_prop;
    cudaGetDeviceProperties(&cu_prop, 0);
    cudaStreamCreate(&cuStream);
}

void TableGPU::allocateInstanceData()
{
    instData_h = mallocHost<InstanceData>(sizeof(InstanceData));
    instData_d = mallocHost<InstanceData>(sizeof(InstanceData)); // Host is correct

    u32 const readOnlyMemSize =
        BigWordAlign + supportMemSize;

    readOnlyAlloc_h = new LinearAllocator(mallocHost<void>(readOnlyMemSize), readOnlyMemSize);
    instData_h->supports = readOnlyAlloc_h->allocate<u32>(supportMemSize,  BigWordAlign);

    readOnlyAlloc_d = new LinearAllocator(mallocDevice<void>(readOnlyMemSize), readOnlyMemSize);
    instData_d->supports = readOnlyAlloc_d->allocate<u32>(supportMemSize,  BigWordAlign);

    u32 const inputOutputMemSize =
        LinearAllocator::DefaultAlign + changedVarsMemSize +
        LinearAllocator::DefaultAlign + unfixedVarsMemSize +
        LinearAllocator::DefaultAlign + domainsInfoMemSize +
        LinearAllocator::DefaultAlign + sizeof(u32) + // someValidTuple
        BigWordAlign + validTuplesMemSize +
        LinearAllocator::DefaultAlign + domainsMemSize;

    inputOutputAlloc_h = new LinearAllocator(mallocHost<void>(inputOutputMemSize), inputOutputMemSize);
    instData_h->changedVars = inputOutputAlloc_h->allocate<u32>(changedVarsMemSize);
    instData_h->unfixedVars = inputOutputAlloc_h->allocate<u32>(unfixedVarsMemSize);
    instData_h->domainsInfo = inputOutputAlloc_h->allocate<DomainsInfo>(domainsInfoMemSize);
    instData_h->someValidTuple = inputOutputAlloc_h->allocate<u32>(sizeof(u32));
    instData_h->validTuples = inputOutputAlloc_h->allocate<u32>(validTuplesMemSize, BigWordAlign);
    instData_h->domains = inputOutputAlloc_h->allocate<u32>(domainsMemSize);

    inputOutputAlloc_d = new LinearAllocator(mallocDevice<void>(inputOutputMemSize), inputOutputMemSize);
    instData_d->changedVars = inputOutputAlloc_d->allocate<u32>(changedVarsMemSize);
    instData_d->unfixedVars = inputOutputAlloc_d->allocate<u32>(unfixedVarsMemSize);
    instData_d->domainsInfo = inputOutputAlloc_d->allocate<DomainsInfo>(domainsInfoMemSize);
    instData_d->someValidTuple = inputOutputAlloc_d->allocate<u32>(sizeof(u32));
    instData_d->validTuples = inputOutputAlloc_d->allocate<u32>(validTuplesMemSize, BigWordAlign);
    instData_d->domains = inputOutputAlloc_d->allocate<u32>(domainsMemSize);

    beginOutputMem_h = instData_h->someValidTuple;
    beginOutputMem_d = instData_d->someValidTuple;
    void const * const endOutputMem_h = inputOutputAlloc_h->getFreeMemory();
    outputMemSize = reinterpret_cast<std::uintptr_t>(endOutputMem_h) - reinterpret_cast<std::uintptr_t>(beginOutputMem_h);
}

void TableGPU::post()
{
    for (auto const & v : vars)
    {
        v->propagateOnBoundChange(this);
    }

    allocateInstanceData();
    tmpMask = static_cast<u32*>(aligned_alloc(BigWordAlign, tmpMaskMemSize));

    initializeInstanceData(instData_h);

    instData_d->nVars = instData_h->nVars;
    instData_d->nTuples = instData_h->nTuples;
    instData_d->supportsCols = instData_h->supportsCols;
    instData_d->supportsRows = instData_h->supportsRows;
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

    // Initialize valid tuples
    u32 const nWords = validTuples.size();
    for (u32 wIdx = 0; wIdx < nWords; wIdx += 1)
    {
        validTuples.set(wIdx, UINT_MAX);
    }

    propagate();
}
void TableGPU::propagate()
{
    updateInstanceData(instData_h);
    updateValidTuples(instData_h);

    if (*instData_h->someValidTuple and instData_h->nUnfixedVars != 0)
    {
        Timer::begin("TableGPU::updateDomains");

        cudaMemcpyAsync(
            inputOutputAlloc_d->getMemory(),
            inputOutputAlloc_h->getMemory(),
            inputOutputAlloc_h->getUsedMemorySize(),
            cudaMemcpyHostToDevice,
            cuStream);

        u32 const gridSizeX = ceilDivPosInt(instData_h->maxWordsInDomain, nWordsPerBlock);
        u32 const gridSizeY = instData_h->nUnfixedVars;
        u32 const sharedMemSize = 0;
        dim3 gridSize(gridSizeX,gridSizeY,1);

        updateDomainsKernel<<<gridSize, BlockSize, sharedMemSize, cuStream>>>(*instData_d);

        cudaMemcpyAsync(
            beginOutputMem_h,
            beginOutputMem_d,
            outputMemSize,
            cudaMemcpyDeviceToHost,
            cuStream);

        cudaStreamSynchronize(cuStream);

        Timer::end("TableGPU::updateDomains");
    }

    filterDomains(instData_h);
}

__global__
void updateDomainsKernel(Table::InstanceData instData)
{
    if (*instData.someValidTuple)
    {
        // Copy instance data in registers
        Table::InstanceData instData_r = instData;

        // Retrieve variable information
        u32 const varIdx = instData_r.unfixedVars[blockIdx.y];
        Table::DomainsInfo dInfo_r = instData_r.domainsInfo[varIdx];

        // Copy domains words in shared
        u32 const firstWordIdx = dInfo_r.firstWordIdx + TableGPU::nWordsPerBlock * blockIdx.x;
        u32 const wIdx = getDiv32(threadIdx.x);
        u32 domWord_r = instData_r.domains[firstWordIdx + wIdx];

        u32 const laneIdx = getMod32(threadIdx.x);
        if (domWord_r != 0)
        {
            // Check values
            i32 const firstValue = dInfo_r.firstBitValue + TableGPU::nWordsPerBlock * 32 * blockIdx.x;
            for (u32 valIdx = 0; valIdx < 32; valIdx += 1)
            {
                i32 const val = firstValue + wIdx * 32 + valIdx;
                bool isSupported = false;
                u32 bMask = 1 << 31 - valIdx;
                bool const isPresent = domWord_r & bMask;
                if (isPresent)
                {
                    // Check support
                    auto const * const validTuplesBW = reinterpret_cast<Table::BigWordType*>(instData_r.validTuples);
                    u32 const rIdx = dInfo_r.firstWordIdx * 32 + val - dInfo_r.firstBitValue;
                    u32 const nBigWordsSupportsRow = instData_r.supportsCols / Table::BigWordBits;
                    auto const * const supportsRowBW = BitMatrix::getRowAs<Table::BigWordType>(instData_r.supportsRows, instData_r.supportsCols, instData_r.supports, rIdx);
                    for (u32 bwIdx = laneIdx; bwIdx < nBigWordsSupportsRow and (not isSupported); bwIdx += 32)
                    {
                        auto const validTuplesWordBW = validTuplesBW[bwIdx];
                        auto const supportsWordBW = supportsRowBW[bwIdx];
                        isSupported = __reduce_or_sync(__activemask(), isAndNotZero(&validTuplesWordBW, &supportsWordBW));
                    }
                }
                isSupported = __reduce_or_sync(__activemask(), isSupported); // DO NOT REMOVE!
                domWord_r = isSupported ? domWord_r : domWord_r & ~bMask;
            }
            if (laneIdx == 0)
            {
                 instData_r.domains[firstWordIdx + wIdx] = domWord_r;
            }
        }
    }
}
