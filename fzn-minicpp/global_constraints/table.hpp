#pragma once


#include <libminicpp/varitf.hpp>
#include <libminicpp/trailArray.hpp>
#include <libminicpp/constraint.hpp>

#include <libfca/Types.hpp>
#include <libfca/LinearAllocator.cuh>

class Table : public Constraint
{
    public:
        using BigWordType = Fca::u512;
        Fca::u32 static constexpr BigWordBits = sizeof(BigWordType) * 8;
        Fca::u32 static constexpr BigWordAlign = alignof(BigWordType);

        struct DomainsInfo
        {
            Fca::u32 firstWordIdx;
            Fca::i32 firstBitValue;
            Fca::u32 nWords;
            Fca::i32 min;
            Fca::i32 max;
        };

        struct InstanceData
        {
            // Read only
            Fca::u32 nVars;
            Fca::u32 nTuples;
            Fca::u32 supportsCols;
            Fca::u32 supportsRows;
            Fca::u32 maxWordsInDomain;
            Fca::u32 * supports;

            // Input
            Fca::u32 nChangedVars;
            Fca::u32 nUnfixedVars;
            Fca::u32 * changedVars;
            Fca::u32 * unfixedVars;
            DomainsInfo * domainsInfo;

            // Input/Output
            Fca::u32 * someValidTuple; // Bool
            Fca::u32 * validTuples;
            Fca::u32 * domains;
        };

    protected:
        InstanceData * instData;

        std::vector<var<int>::Ptr> vars;
        std::vector<std::vector<int>> tuples;
        TrailArray<unsigned int> validTuples;
        TrailArray<unsigned int> lastSize;

        Fca::u32 supportMemSize;
        Fca::u32 changedVarsMemSize;
        Fca::u32 unfixedVarsMemSize;
        Fca::u32 domainsInfoMemSize;
        Fca::u32 validTuplesMemSize;
        Fca::u32 domainsMemSize;

        Fca::u32 tmpMaskMemSize;
        Fca::u32 * tmpMask;

    public:
        Table(std::vector<var<int>::Ptr> & vars,  std::vector<std::vector<int>> & tuples);
        void post() override;
        void propagate() override;
    protected:
        Fca::u32 getSupportsRows(std::vector<var<int>::Ptr> vars) const;
        Fca::u32 getSupportsCols(std::vector<std::vector<int>> const & tuples) const;
        void calculateInstanceDataMemSize();
        virtual void allocateInstanceData();
        void initializeInstanceData(InstanceData * instData);
        void updateInstanceData(InstanceData * instData);
        void updateValidTuples(InstanceData * instData);
        void updateDomains(InstanceData * instData);
        void filterDomains(InstanceData * instData);
};


