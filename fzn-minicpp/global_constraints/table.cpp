#include "table.hpp"

#include <libfca/BitMatrix.hpp>
#include <libfca/Timer.hpp>
#include <libfca/Utils.hpp>
#include <libgpu/libgpu.cuh>

#include <climits>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace Fca;
using namespace Fca::Math;

Table::Table(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) :
    Constraint(vars[0]->getSolver()),
    vars(vars),
    tuples(tuples),
    validTuples(vars[0]->getSolver()->getStateManager(), vars[0]->getSolver()->getStore(), getDiv32(getSupportsCols(tuples))),
    lastSize(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),vars.size())
{
    calculateInstanceDataMemSize();
}

void Table::calculateInstanceDataMemSize()
{
    u32 const nVars = static_cast<u32>(vars.size());
    u32 const supportsCols = getSupportsCols(tuples);
    u32 const supportsRows = getSupportsRows(vars);
    supportMemSize = BitMatrix::getDataSize(supportsRows, supportsCols);
    changedVarsMemSize = sizeof(u32) * nVars;
    unfixedVarsMemSize = sizeof(u32) * nVars;
    domainsInfoMemSize = sizeof(DomainsInfo) * nVars;
    validTuplesMemSize = supportsCols / 8; // Bits -> Bytes
    domainsMemSize = supportsRows / 8; // Bits -> Bytesq
    tmpMaskMemSize = supportsCols / 8; // Bits -> Bytes

}

void Table::allocateInstanceData()
{
    instData = static_cast<InstanceData*>(malloc(sizeof(InstanceData)));

    u32 const allocatorMemSize =
        BigWordAlign + supportMemSize +
        LinearAllocator::DefaultAlign + changedVarsMemSize +
        LinearAllocator::DefaultAlign + unfixedVarsMemSize +
        LinearAllocator::DefaultAlign + domainsInfoMemSize +
        LinearAllocator::DefaultAlign + sizeof(u32) + // someValidTuple
        BigWordAlign + validTuplesMemSize +
        LinearAllocator::DefaultAlign + domainsMemSize;

    auto * const allocator = new LinearAllocator(malloc(allocatorMemSize), allocatorMemSize);
    instData->supports = allocator->allocate<u32>(supportMemSize, BigWordAlign);
    instData->changedVars = allocator->allocate<u32>(changedVarsMemSize);
    instData->unfixedVars = allocator->allocate<u32>(unfixedVarsMemSize);
    instData->domainsInfo = allocator->allocate<DomainsInfo>(domainsInfoMemSize);
    instData->someValidTuple = allocator->allocate<u32>(sizeof(u32));
    instData->validTuples = allocator->allocate<u32>(validTuplesMemSize, BigWordAlign);
    instData->domains = allocator->allocate<u32>(domainsMemSize);
}

void Table::post()
{
    for (auto const & v : vars)
    {
        v->propagateOnBoundChange(this);
    }

    allocateInstanceData();
    tmpMask = static_cast<u32*>(aligned_alloc(BigWordAlign, tmpMaskMemSize));

    initializeInstanceData(instData);

    // Initialize last sizes
    for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
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

void Table::propagate()
{
    updateInstanceData(instData);
    updateValidTuples(instData);
    updateDomains(instData);
    filterDomains(instData);
}

Fca::u32 Table::getSupportsRows(std::vector<var<int>::Ptr> vars) const
{
    return accumulate(
        vars.begin(),
        vars.end(),
        0u,
        [&](u32 const a, var<int>::Ptr const & v) -> u32 {return a + v->getBitDomainWords() * 32;}); // Domains uses 32-bits words
}

Fca::u32 Table::getSupportsCols(std::vector<std::vector<int>> const &tuples) const
{
    return roundUpPosInt(tuples.size(), BigWordBits); // Bytes -> bits
}

void Table::initializeInstanceData(InstanceData * instData)
{
    // Basic data
    instData->nVars = static_cast<u32>(vars.size());
    instData->nTuples = static_cast<u32>(tuples.size());
    instData->supportsCols =  getSupportsCols(tuples);
    instData->supportsRows = getSupportsRows(vars);
    instData->maxWordsInDomain = std::transform_reduce(
        vars.begin(),
        vars.end(),
        0,
        [&](u32 a, u32 b) {return std::max(a, b);},
        [&](var<int>::Ptr const & v) {return v->getBitDomainWords();});

    // Domains info
    u32 firstWordIdx = 0;
    for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
    {
        auto const & var = vars[vIdx];
        auto & dInfo = instData->domainsInfo[vIdx];
        dInfo.firstWordIdx = firstWordIdx;
        dInfo.nWords = var->getBitDomainWords(); // Domains uses 32-bits words
        dInfo.firstBitValue = var->getBitDomainSmallestValue();
        dInfo.min = var->min();
        dInfo.max = var->max();
        firstWordIdx += dInfo.nWords;
    }
    // Supports
    u32 const nBigWords = (instData->supportsCols / BigWordBits) * instData->supportsRows;
    auto * const supportsBW = reinterpret_cast<BigWordType*>(instData->supports);
    for (u32 wIdx = 0; wIdx < nBigWords; wIdx += 1)
    {
        setZero(supportsBW + wIdx);
    }
    for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
    {
        auto const & var = vars[vIdx];
        auto const & dInfo = instData->domainsInfo[vIdx];
        for (u32 tIdx = 0; tIdx < instData->nTuples; tIdx += 1)
        {
            i32 const val = tuples.at(tIdx).at(vIdx);
            u32 const rIdx = (dInfo.firstWordIdx * 32) + val - dInfo.firstBitValue;
            if (var->contains(val))
            {
                BitMatrix::set(instData->supportsRows, instData->supportsCols, instData->supports, rIdx, tIdx, true);
            }
        }
    }
}

void Table::updateInstanceData(InstanceData * instData)
{
    // Update variables
    instData->nChangedVars = 0;
    instData->nUnfixedVars = 0;
    for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
    {
        u32 const vSize = vars[vIdx]->size();
        if (vSize != lastSize[vIdx])
        {
            instData->changedVars[instData->nChangedVars] = vIdx;
            instData->nChangedVars += 1;
        }
        if (vSize != 1)
        {
            instData->unfixedVars[instData->nUnfixedVars] = vIdx;
            instData->nUnfixedVars += 1;
        }
    }

    // Update valid tuples
    u32 nWords = validTuples.size();
    for (u32 wIdx = 0; wIdx < nWords; wIdx += 1)
    {
        instData->validTuples[wIdx] = validTuples[wIdx];
    }

    // Update domains info
    for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
    {
        auto const & var = vars[vIdx];
        auto & dInfo = instData->domainsInfo[vIdx];
        dInfo.min = var->min();
        dInfo.max = var->max();
        var->dumpBitDomainWords(instData->domains + dInfo.firstWordIdx);
    }
}

void Table::updateValidTuples(InstanceData * instData)
{
    Timer::begin("Table::updateValidTuples");

    // Update
    u32 const nBigWords = instData->supportsCols / BigWordBits;
    auto * const tmpMaskBW = reinterpret_cast<BigWordType*>(tmpMask);
    *instData->someValidTuple = true;
    for (u32 cvIdx = 0; cvIdx < instData->nChangedVars and *instData->someValidTuple; cvIdx += 1)
    {
       // Clear mask
        for (u32 wIdx = 0; wIdx < nBigWords; wIdx += 1)
        {
            setZero(tmpMaskBW + wIdx);
        }

        // Supports -> mask
        u32 const vIdx = instData->changedVars[cvIdx];
        DomainsInfo const & dInfo = instData->domainsInfo[vIdx];
        for (i32 val = dInfo.min; val <= dInfo.max; val += 1)
        {
            u32 const vOffset = val - dInfo.firstBitValue;
            u32 const bIdx = getMod32(vOffset);
            u32 const wMask = getMask32(bIdx);
            u32 const wIdx = dInfo.firstWordIdx + getDiv32(vOffset);
            bool const contains = instData->domains[wIdx] & wMask;
            if (contains)
            {
                u32 const rIdx = wIdx * 32 + bIdx;
                auto const * const supportsRowBW = BitMatrix::getRowAs<BigWordType>(instData->supportsRows, instData->supportsCols, instData->supports, rIdx);
                for (u32 bwIdx = 0; bwIdx < nBigWords; bwIdx += 1)
                {
                    bitwiseOr(tmpMaskBW + bwIdx, supportsRowBW + bwIdx);
                }
            }
        }

        // Mask -> validTuples
        auto * const validTuplesBW = reinterpret_cast<BigWordType*>(instData->validTuples);
        bool someValidTuple = false;
        for (u32 bwIdx = 0; bwIdx < nBigWords; bwIdx += 1)
        {
            bitwiseAnd(validTuplesBW + bwIdx, tmpMaskBW + bwIdx);
            someValidTuple = someValidTuple or (not isZero(validTuplesBW + bwIdx));
        }
        *instData->someValidTuple = someValidTuple;
    }
    Timer::end("Table::updateValidTuples");
}

void Table::updateDomains(InstanceData * instData)
{
    Timer::begin("Table::updateDomains");

    u32 const nBigWords = instData->supportsCols / BigWordBits;
    auto * const validTuplesBW = reinterpret_cast<BigWordType*>(instData->validTuples);
    if (*instData->someValidTuple)
    {
        for (u32 uvIdx = 0; uvIdx < instData->nUnfixedVars; uvIdx += 1)
        {
            // ValidTuples -> domains
            u32 const vIdx = instData->unfixedVars[uvIdx];
            DomainsInfo & dInfo = instData->domainsInfo[vIdx];
            for (i32 val = dInfo.min; val <= dInfo.max; val += 1)
            {
                u32 const vOffset = val - dInfo.firstBitValue;
                u32 const bIdx = getMod32(vOffset);
                u32 const wMask = getMask32(bIdx);
                u32 const wIdx = dInfo.firstWordIdx + getDiv32(vOffset);
                bool const contains = instData->domains[wIdx] & wMask;
                if (contains)
                {
                    u32 const rIdx = wIdx * 32 + bIdx;
                    bool someSupport = false;
                    auto const * const supportsRowBW = BitMatrix::getRowAs<BigWordType>(instData->supportsRows, instData->supportsCols, instData->supports, rIdx);
                    for (u32 bwIdx = 0; bwIdx < nBigWords and (not someSupport); bwIdx += 1)
                    {
                        someSupport = someSupport or isAndNotZero(supportsRowBW + bwIdx, validTuplesBW + bwIdx);
                    }
                    if (not someSupport)
                    {
                        instData->domains[wIdx] &= ~wMask;
                    }
                }
            }
        }
    }
    Timer::end("Table::updateDomains");
}

void Table::filterDomains(InstanceData * instData)
{
    if (*instData->someValidTuple)
    {
        Timer::begin("Table::filterDomains");
        
        // Filter domains
        for (u32 uvIdx = 0; uvIdx < instData->nUnfixedVars; uvIdx += 1) // For each variable
        {
            u32 const vIdx = instData->unfixedVars[uvIdx];
            auto const var = vars[vIdx];
            auto const & dInfo = instData->domainsInfo[vIdx];

            for (i32 val = dInfo.min; val <= dInfo.max; val += 1) // For each value
            {
                if (var->containsBase(val))
                {
                    u32 const vOffset = val - dInfo.firstBitValue;
                    u32 const bIdx = getMod32(vOffset);
                    u32 const wMask = getMask32(bIdx);
                    u32 const wIdx = dInfo.firstWordIdx + getDiv32(vOffset);
                    bool const contains = instData->domains[wIdx] & wMask;
                    if (not contains)
                    {
                        var->remove(val);
                    }
                }
            }
        }

        // Update valid tuples
        u32 const nWords = validTuples.size();
        for (u32 wIdx = 0; wIdx < nWords; wIdx += 1)
        {
            validTuples.set(wIdx, instData->validTuples[wIdx]);
        }

        // Update last sizes
        for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
        {
            lastSize.set(vIdx, vars[vIdx]->size());
        }
        Timer::end("Table::filterDomains");
    }
    else
    {
        failNow();
    }
}