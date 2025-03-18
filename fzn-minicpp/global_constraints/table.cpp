#include "table.hpp"

#include <libfca/Array.hpp>
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
    nValidTuples(vars[0]->getSolver()->getStateManager(), tuples.size()),
    validTuples(vars[0]->getSolver()->getStateManager(), vars[0]->getSolver()->getStore(), tuples.size()),
    lastSize(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),vars.size())
{
    calculateInstanceDataMemSize();
}

void Table::calculateInstanceDataMemSize()
{
    u32 const nVars = static_cast<u32>(vars.size());
    u32 const nTuples = static_cast<u32>(tuples.size());

    tuplesMemSize = sizeof(i32) * nVars * nTuples;
    changedVarsMemSize = sizeof(u32) * nVars;
    unfixedVarsMemSize = sizeof(u32) * nVars;
    domainsInfoMemSize = sizeof(DomainsInfo) * nVars;
    validTuplesMemSize = sizeof(i32) * nTuples;

    domainsMemSize = 0;
    for (u32 vIdx = 0; vIdx < nVars; vIdx += 1)
    {
        domainsMemSize += vars[vIdx]->getBitDomainWords() * 4; // Domains uses 32-bits (4 bytes) words
    }
}

void Table::allocateInstanceData()
{
    instData = static_cast<InstanceData*>(malloc(sizeof(InstanceData)));

    u32 const allocatorMemSize =
        LinearAllocator::DefaultAlign + tuplesMemSize +
        LinearAllocator::DefaultAlign + changedVarsMemSize +
        LinearAllocator::DefaultAlign + unfixedVarsMemSize +
        LinearAllocator::DefaultAlign + domainsInfoMemSize +
        LinearAllocator::DefaultAlign + sizeof(u32) + // nValidTuple
        LinearAllocator::DefaultAlign + validTuplesMemSize +
        LinearAllocator::DefaultAlign + domainsMemSize;

    auto * const allocator = new LinearAllocator(malloc(allocatorMemSize), allocatorMemSize);
    instData->tuples = allocator->allocate<i32>(tuplesMemSize);
    instData->changedVars = allocator->allocate<u32>(changedVarsMemSize);
    instData->unfixedVars = allocator->allocate<u32>(unfixedVarsMemSize);
    instData->domainsInfo = allocator->allocate<DomainsInfo>(domainsInfoMemSize);
    instData->nValidTuples = allocator->allocate<u32>(sizeof(u32));
    instData->validTuples = allocator->allocate<u32>(validTuplesMemSize);
    instData->domains = allocator->allocate<u32>(domainsMemSize);
}

void Table::post()
{
    for (auto const & v : vars)
    {
        v->propagateOnDomainChange(this);
    }

    allocateInstanceData();
    tmpValidTuples = static_cast<u32*>(malloc(validTuplesMemSize));
    tmpDomains = static_cast<u32*>(malloc(domainsMemSize));

    initializeInstanceData(instData);

    // Initialize last sizes
    for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
    {
        lastSize.set(vIdx, UINT32_MAX);
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

void Table::initializeInstanceData(InstanceData * instData)
{
    // Basic data
    instData->nVars = static_cast<u32>(vars.size());
    instData->nTuples = static_cast<u32>(tuples.size());
    instData->maxWordsInDomain = 0;
    for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
    {
        u32 const nWords = vars[vIdx]->getBitDomainWords();
        instData->maxWordsInDomain = max(instData->maxWordsInDomain, nWords);
    }

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

    // Tuples
    for (u32 tIdx = 0; tIdx < instData->nTuples; tIdx += 1)
    {
        auto const & tuple = tuples.at(tIdx);
        for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
        {
            instData->tuples[tIdx * instData->nVars + vIdx] = tuple.at(vIdx);
        }
    }

    // Initialize valid tuples
    *instData->nValidTuples = instData->nTuples;
    for (u32 tIdx = 0; tIdx < instData->nTuples; tIdx += 1)
    {
        validTuples.set(tIdx, tIdx);
    }
}

void Table::updateInstanceData(InstanceData * instData)
{
    // Update variables
    instData->nChangedVars = 0;
    instData->nUnfixedVars = 0;
    for (u32 vIdx = 0; vIdx < instData->nVars; vIdx += 1)
    {
        u32 const dSize = vars[vIdx]->size();
        if (dSize != lastSize.get(vIdx))
        {
            instData->changedVars[instData->nChangedVars] = vIdx;
            instData->nChangedVars += 1;
        }
        if (dSize != 1)
        {
            instData->unfixedVars[instData->nUnfixedVars] = vIdx;
            instData->nUnfixedVars += 1;
        }
    }

    // Update valid tuples
    *instData->nValidTuples = nValidTuples;
    for (u32 tIdx = 0; tIdx < *instData->nValidTuples; tIdx += 1)
    {
        instData->validTuples[tIdx] = validTuples.get(tIdx);
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

void Table::clearDomains(InstanceData * instData, Fca::u32 * domains)
{
    u32 const nWords = domainsMemSize / 4;

    for (u32 uvIdx = 0; uvIdx < instData->nUnfixedVars; uvIdx += 1)
    {
        u32 const vIdx = instData->unfixedVars[uvIdx];
        DomainsInfo & dInfo = instData->domainsInfo[vIdx];
        for (u32 wIdx = 0; wIdx < dInfo.nWords; wIdx += 1)
        {
            domains[dInfo.firstWordIdx + wIdx] = 0;
        }
    }

}

void Table::updateValidTuples(InstanceData * instData)
{
    Timer::begin("Table::updateValidTuples");

    if (instData->nChangedVars > 0)
    {
        //printf("Changed Vars = %d\n", instData->nChangedVars);
        u32 validTuplesCount = 0;
        for (u32 vtIdx = 0; vtIdx < *instData->nValidTuples; vtIdx += 1)
        {
            u32 const tIdx = instData->validTuples[vtIdx];
            auto const * tuple = instData->tuples + (tIdx * instData->nVars);
            bool isValid = true;
            for (u32 cvIdx = 0; cvIdx < instData->nChangedVars and isValid; cvIdx += 1)
            {
                u32 const vIdx = instData->changedVars[cvIdx];
                isValid = isValid and vars.at(vIdx)->contains(tuple[vIdx]);
            }
            if (isValid)
            {
                tmpValidTuples[validTuplesCount] = tIdx;
                validTuplesCount += 1;
            }
        }
        *instData->nValidTuples = validTuplesCount;
        for (int vtIdx = 0; vtIdx < *instData->nValidTuples; vtIdx += 1)
        {
            instData->validTuples[vtIdx] = tmpValidTuples[vtIdx];
        }
    }
    Timer::end("Table::updateValidTuples");
}

void Table::updateDomains(InstanceData * instData)
{
    Timer::begin("Table::updateDomains");

    if (*instData->nValidTuples > 0 and instData->nUnfixedVars > 0)
    {
        clearDomains(instData, tmpDomains);

        for (u32 vtIdx = 0; vtIdx < *instData->nValidTuples; vtIdx += 1)
        {
            u32 const tIdx = instData->validTuples[vtIdx];
            auto const * tuple = instData->tuples + (tIdx * instData->nVars);
            for (u32 uvIdx = 0; uvIdx < instData->nUnfixedVars; uvIdx += 1)
            {
                // valid tuples -> domains
                u32 const varIdx = instData->unfixedVars[uvIdx];
                DomainsInfo & dInfo = instData->domainsInfo[varIdx];
                i32 const val = tuple[varIdx];
                u32 const valIdx = val - dInfo.firstBitValue;
                u32 const valWord = getDiv32(valIdx);
                u32 const valMask = 1 << 31 - getMod32(valIdx);
                tmpDomains[dInfo.firstWordIdx + valWord] |= valMask;
            }
        }

        for (u32 uvIdx = 0; uvIdx < instData->nUnfixedVars; uvIdx += 1)
        {
            u32 const vIdx = instData->unfixedVars[uvIdx];
            DomainsInfo & dInfo = instData->domainsInfo[vIdx];
            for (u32 wIdx = 0; wIdx < dInfo.nWords; wIdx += 1)
            {
                instData->domains[dInfo.firstWordIdx + wIdx] = tmpDomains[dInfo.firstWordIdx + wIdx];
            }
        }
    }
    Timer::end("Table::updateDomains");
}

void Table::filterDomains(InstanceData * instData)
{
    if (*instData->nValidTuples > 0)
    {
        Timer::begin("Table::filterDomains");

        // Filter domains
        for (u32 uvIdx = 0; uvIdx < instData->nUnfixedVars; uvIdx += 1) // For each variable
        {
            u32 const vIdx = instData->unfixedVars[uvIdx];
            auto const var = vars[vIdx];
            auto const & dInfo = instData->domainsInfo[vIdx];
            var->loadBitDomainWords(instData->domains + dInfo.firstWordIdx);
            // for (i32 val = dInfo.min; val <= dInfo.max; val += 1) // For each value
            // {
            //     if (var->containsBase(val))
            //     {
            //         u32 const valIdx = val - dInfo.firstBitValue;
            //         u32 const valWordIdx = getDiv32(valIdx);
            //         u32 const valMask = 1 << 31 - getMod32(valIdx);
            //         bool const contains = instData->domains[dInfo.firstWordIdx + valWordIdx] & valMask;
            //         if (not contains)
            //         {
            //             printf("Removing val %d from var %u\n", val, vIdx);
            //             var->remove(val);
            //         }
            //     }
            // }
        }

        // Update valid tuples
        nValidTuples = *instData->nValidTuples;
        for (int vtIdx = 0; vtIdx < *instData->nValidTuples; vtIdx +=1)
        {
            validTuples.set(vtIdx, instData->validTuples[vtIdx]);
        }

        // Update last sizes
        for (u32 uvIdx = 0; uvIdx < instData->nUnfixedVars; uvIdx += 1)
        {
            u32 const vIdx = instData->unfixedVars[uvIdx];
            lastSize.set(vIdx, vars[vIdx]->size());
        }
        Timer::end("Table::filterDomains");
    }
    else
    {
        failNow();
    }
}