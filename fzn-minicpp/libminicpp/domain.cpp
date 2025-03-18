/*
 * mini-cp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2018. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 */

#include "domain.hpp"

#include <cstring>

#include "fail.hpp"
#include "mathUtils.h"
#include "bitsUtils.h"
#include <iostream>
#include <libfca/Utils.hpp>

#include "bitset.hpp"
BitDomain::BitDomain(Trailer::Ptr eng,Storage::Ptr store,int min, int max)
    : _min(eng,min),
      _max(eng,max),
      _sz(eng,max - min + 1),
      _smallest_val(floorDivision(min, 32) * 32),
      _words_count(ceilDivision(max - _smallest_val + 1, 32))
{
    // Words are considered 32 aligned, from left to right, bits are considered left to right.
    _dom = (trail<unsigned int>*)store->allocate(sizeof(trail<unsigned int>) * _words_count); // allocate storage from stack allocator

    for(int i = 0; i < _words_count; i++)
       new (_dom+i) trail<unsigned int>(eng,0xffffffff);  // placement-new for each reversible.

    //Adjust first and last words
    _dom[0] = _dom[0] & getRightFilledMask32(_min % 32);
    _dom[_words_count - 1] = _dom[_words_count - 1] & getLeftFilledMask32(_max % 32);
}

int BitDomain::count(int from, int to) const
{
    int minIdx = from  - _smallest_val;
    int minWordIdx = minIdx / 32;
    int minBitIdx = minIdx % 32;
    unsigned int minWordMask = getRightFilledMask32(minBitIdx);

    int maxIdx = to - _smallest_val;
    int maxWordIdx = maxIdx / 32;
    int maxBitIdx = maxIdx % 32;
    unsigned int maxWordMask = getLeftFilledMask32(maxBitIdx);

    int count = 0;
    if (minWordIdx == maxWordIdx)
    {
        count += getPopCount(_dom[minWordIdx] & minWordMask & maxWordMask);
    }
    else
    {
        count += getPopCount(_dom[minWordIdx] & minWordMask);
        for(int wordIdx = minWordIdx + 1; wordIdx < maxWordIdx; wordIdx += 1)
        {
            count += getPopCount(_dom[wordIdx]);
        }
        count += getPopCount(_dom[maxWordIdx] & maxWordMask);
    }
    return count;
}

int BitDomain::findMin(int from) const
{
    assert(_sz > 0);
    
    int minIdx = from  - _smallest_val;
    int minWordIdx = minIdx / 32;
    int minBitIdx = minIdx % 32;
    unsigned int minWordmask = getRightFilledMask32(minBitIdx);

    int maxIdx = _max - _smallest_val;
    int maxWordIdx = maxIdx / 32;
    int maxBitIdx = maxIdx % 32;
    unsigned int maxWordMask = getLeftFilledMask32(maxBitIdx);

    if (minWordIdx == maxWordIdx)
    {
        return _smallest_val + (32 * minWordIdx) + getLeftmostOneIndex32(_dom[minWordIdx] & minWordmask & maxWordMask);
    }
    else
    {
        if((_dom[minWordIdx] & minWordmask) != 0)
        {
            return _smallest_val + (32 * minWordIdx) + getLeftmostOneIndex32(_dom[minWordIdx] & minWordmask);
        }
        for (int wordIdx = minWordIdx + 1; wordIdx < maxWordIdx; wordIdx += 1)
        {
            if(_dom[wordIdx] != 0)
            {
                return _smallest_val + (32 * wordIdx) + getLeftmostOneIndex32(_dom[wordIdx]);
            }
        }
        return _smallest_val + (32 * maxWordIdx) + getLeftmostOneIndex32(_dom[maxWordIdx] & maxWordMask);
    }
}
int BitDomain::findMax(int to) const
{
    assert(_sz > 0);

    int minIdx = _min  - _smallest_val;
    int minWordIdx = minIdx / 32;
    int minBitIdx = minIdx % 32;
    unsigned int minWordMask = getRightFilledMask32(minBitIdx);

    int maxIdx = to - _smallest_val;
    int maxWordIdx = maxIdx / 32;
    int maxBitIdx = maxIdx % 32;
    unsigned int maxWordMask = getLeftFilledMask32(maxBitIdx);

    if (minWordIdx == maxWordIdx)
    {
        return _smallest_val + (32 * minWordIdx) + getRightmostOneIndex32(_dom[minWordIdx] & minWordMask & maxWordMask);
    }
    else
    {
        if((_dom[maxWordIdx] & maxWordMask) != 0)
        {
            return _smallest_val + (32 * maxWordIdx) + getRightmostOneIndex32(_dom[maxWordIdx] & maxWordMask);
        }
        for (int wordIdx = maxWordIdx - 1; wordIdx > minWordIdx; wordIdx -= 1)
        {
            if (_dom[wordIdx] != 0)
            {
                return _smallest_val + (32 * wordIdx) + getRightmostOneIndex32(_dom[wordIdx]);
            }
        }
        return _smallest_val + (32 * minWordIdx) + getRightmostOneIndex32(_dom[minWordIdx] & minWordMask);
    }
}


void BitDomain::dumpWords(unsigned int *words)
{
    for (int wIdx = 0; wIdx < _words_count; wIdx += 1)
    {
        words[wIdx] = _dom[wIdx];
    }
}

void BitDomain::loadWords(unsigned int * words, IntNotifier & x)
{

    int minIdx = _min - _smallest_val;
    int minWordIdx = minIdx / 32;
    int minBitIdx = minIdx % 32;
    unsigned int minWordMask = getRightFilledMask32(minBitIdx);

    int maxIdx = _max - _smallest_val;
    int maxWordIdx = maxIdx / 32;
    int maxBitIdx = maxIdx % 32;
    unsigned int maxWordMask = getLeftFilledMask32(maxBitIdx);

    unsigned int currentWord;
    if (minWordIdx == maxWordIdx)
    {
        currentWord = words[minWordIdx] & minWordMask & maxWordMask;
        _dom[minWordIdx] = currentWord;
    }
    else
    {
        currentWord = words[minWordIdx] & minWordMask;
        _dom[minWordIdx] = currentWord;

        for (int wIdx = minWordIdx + 1; wIdx < maxWordIdx; wIdx += 1)
        {
            currentWord =  words[wIdx];
            _dom[wIdx] = currentWord;
        }

        currentWord =  words[maxWordIdx] & maxWordMask;
        _dom[maxWordIdx] = currentWord;
    }

    int const newSize = count(_min, _max);
    if (newSize != 0)
    {
        bool sizeChanged = _sz != newSize;
        if (sizeChanged)
        {
            _sz = newSize;
            x.change();
            if (_sz == 1)
            {
                x.bind();
            }
        }
        int newMin = findMin(_min);
        int newMax= findMax(_max);
        bool minChanged = newMin != _min;
        bool maxChanged = newMax != _max;
        if (minChanged)
        {
            _min = newMin;
            x.changeMin();
        }
        if (maxChanged)
        {
            _max = newMax;
            x.changeMax();
        }
    }
    else
    {
        x.empty();
    }
}

void BitDomain::setZero(int at)
{
    at = at - _smallest_val;
    int atWordIdx = at / 32;
    int atBitIdx = at % 32;
    _dom[atWordIdx] = _dom[atWordIdx] & ~getMask32(atBitIdx);
}

void BitDomain::assign(int v,IntNotifier& x)  // removeAllBut(v,x)
{
    if (_sz == 1 && v == _min)
        return;
    if (v < _min || v > _max || !GETBIT(v))
    {
        _sz = 0;
        x.empty();
        return;
    }
    bool minChanged = _min != v;
    bool maxChanged = _max != v;
    _min = v;
    _max = v;
    _sz  = 1;
    x.bind();
    x.change();
    if (minChanged) x.changeMin();
    if (maxChanged) x.changeMax();
}

void BitDomain::remove(int v,IntNotifier& x)
{
    /*
    if(member(v))
    {
        bool minChanged = v == _min;
        bool maxChanged = v == _max;           

        if (minChanged) 
        {               
            _min = findMin(_min + 1);
            x.changeMin();
        } 
        else if (maxChanged) 
        {
            _max = findMax(_max - 1);
            x.changeMax();
        }

        setZero(v);
        x.change(); 

        _sz -= 1;
        if (_sz == 0)
        {
            x.empty();      
        } 
        else if (_sz == 1)
        {
            x.bind();
        }           
    }
     */


    if (v < _min || v > _max)
        return;
    if (_min.value() == _max.value())
        x.empty();
    bool minChanged = v == _min;
    bool maxChanged = v == _max;
    if (minChanged) {
        _sz = _sz - 1;
       _min = findMin(_min + 1);
        x.changeMin();
        if (_sz == 1) x.bind();
        if (_sz == 0) x.empty();
        x.change();
    } else if (maxChanged) {
        _sz = _sz - 1;
        _max = findMax(_max - 1);
        x.changeMax();
        if (_sz == 1) x.bind();
        if (_sz == 0) x.empty();
        x.change();
    } else if (member(v)) {
        setZero(v);
        _sz = _sz - 1;
        if (_sz == 1) x.bind();
        if (_sz == 0) x.empty();
        x.change();
    }

}

void BitDomain::removeBelow(int newMin,IntNotifier& x)
{
    if (newMin <= _min)
        return;
    if (newMin > _max)
        x.empty();
    bool isCompact = (_max - _min + 1) == _sz;
    int nbRemove = isCompact ? newMin - _min : count(_min,newMin - 1);
    _sz = _sz - nbRemove;
    if (!isCompact)
        newMin = findMin(newMin);
    _min = newMin;
    x.changeMin();
    x.change();
    if (_sz==0) x.empty();
    if (_sz==1) x.bind();
}

void BitDomain::removeAbove(int newMax,IntNotifier& x)
{
    if (newMax >= _max)
        return;
    if (newMax < _min)
        x.empty();
    bool isCompact = (_max - _min + 1) == _sz;
    int nbRemove = isCompact ? _max - newMax : count(newMax + 1,_max);
    _sz = _sz - nbRemove;
    if (!isCompact)
        newMax = findMax(newMax);
    _max = newMax;
    x.changeMax();
    x.change();
    if (_sz==0) x.empty();
    if (_sz==1) x.bind();
}

std::ostream& operator<<(std::ostream& os,const BitDomain& x)
{
    if (x.size()==1)
        os << x.min();
    else {
        os << '(' << x.size() << ")[";
        bool first = true;
        bool seq = false;
        int lastIn=x._min;
        for(int k = x._min;k <= x._max;k++) {
            if (x.member(k)) {
                if (first) {
                    os << k;
                    first = seq = false;
                    lastIn = k;
                } else {
                    if (lastIn + 1 == k) {
                        lastIn = k;
                        seq = true;
                    } else {
                        if (seq)
                            os << ".." << lastIn << ',' << k;
                        else
                            os << ',' << k;
                        lastIn = k;
                        seq = false;
                    }
                }
            }
        }
        if (seq)
            os << ".." << lastIn << ']';
        else os << ']';
    }
    return os;
}

// ======================================================================
// SparseSet

SparseSet::SparseSet(Trailer::Ptr eng,int n,int ofs)
    : _values(n),
      _indexes(n),
      _size(eng,n),
      _min(eng,0),
      _max(eng,n-1),
      _ofs(ofs),
      _n(n)
{
    for(int i=0;i < n;i++)
        _values[i] = _indexes[i] = i;
}

void SparseSet::exchangePositions(int val1,int val2)
{
    int i1 = _indexes[val1],i2 = _indexes[val2];
    _values[i1] = val2;
    _values[i2] = val1;
    _indexes[val1] = i2;
    _indexes[val2] = i1;
}

void SparseSet::updateBoundsValRemoved(int val)
{
    updateMaxValRemoved(val);
    updateMinValRemoved(val);
}
void SparseSet::updateMaxValRemoved(int val)
{
    if (!isEmpty() && _max == val) {
        assert(!internalContains(val));
        for(int v=val-1; v >= _min;v--) {
            if (internalContains(v)) {
                _max = v;
                return;
            }
        }
    }
}
void SparseSet::updateMinValRemoved(int val)
{
    if (!isEmpty() && _min == val) {
        assert(!internalContains(val));
        for(int v=val+1;v <= _max;v++) {
            if (internalContains(v)) {
                _min = v;
                return;
            }
        }
    }
}

bool SparseSet::remove(int val)
{
    if (!contains(val))
        return false;
    val -= _ofs;
    assert(checkVal(val));
    int s = _size;
    exchangePositions(val,_values[s - 1]);
    _size = s - 1;
    updateBoundsValRemoved(val);
    return true;
}

void SparseSet::removeAllBut(int v)
{
    assert(contains(v));
    v -= _ofs;
    assert(checkVal(v));
    int val = _values[0];
    int index = _indexes[v];
    _indexes[v] = 0;
    _values[0] = v;
    _indexes[val] = index;
    _values[index] = val;
    _min = v;
    _max = v;
    _size = 1;
}

void SparseSet::removeBelow(int value)
{
    if (max() < value)
        removeAll();
    else
        for(int v= min() ; v < value;v++)
            remove(v);
}

void SparseSet::removeAbove(int value)
{
    if (min() > value)
        removeAll();
    else
        for(int v = value + 1; v <= max();v++)
            remove(v);
}

void SparseSetDomain::assign(int v,IntNotifier& x)
{
    if (_dom.contains(v)) {
        if (_dom.size() != 1) {
            bool maxChanged = max() != v;
            bool minChanged = min() != v;
            _dom.removeAllBut(v);
            if (_dom.size() == 0)
                x.empty();
            x.bind();
            x.change();
            if (maxChanged) x.changeMax();
            if (minChanged) x.changeMin();
        }
    } else {
        _dom.removeAll();
        x.empty();
    }
}

void SparseSetDomain::remove(int v,IntNotifier& x)
{
    if (_dom.contains(v)) {
        bool maxChanged = max() == v;
        bool minChanged = min() == v;
        _dom.remove(v);
        if (_dom.size()==0)
            x.empty();
        x.change();
        if (maxChanged) x.changeMax();
        if (minChanged) x.changeMin();
        if (_dom.size()==1) x.bind();
    }
}

void SparseSetDomain::removeBelow(int newMin,IntNotifier& x)
{
    if (_dom.min() < newMin) {
        _dom.removeBelow(newMin);
        switch(_dom.size()) {
            case 0: x.empty();break;
            case 1: x.bind();
            default:
                x.changeMin();
                x.change();
                break;
        }
    }
}

void SparseSetDomain::removeAbove(int newMax,IntNotifier& x)
{
    if (_dom.max()  > newMax) {
        _dom.removeAbove(newMax);
        switch(_dom.size()) {
            case 0: x.empty();break;
            case 1: x.bind();
            default:
                x.changeMax();
                x.change();
                break;
        }
    }
}

std::ostream& operator<<(std::ostream& os,const SparseSetDomain& x)
{
    os << '{';
    for(int i = x.min();i <= x.max()-1;i++)
        if (x.member(i))
            os << i << ',';
    if (x.size() > 0) os << x.max();
    os << '}';
    return os;
}

