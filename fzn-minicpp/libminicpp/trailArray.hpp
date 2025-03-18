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

#pragma once
#include "trailable.hpp"
#include "store.hpp"

template <class T,typename SizeType = std::size_t> class TrailArray
{
      Trailer::Ptr   _trailer;
      SizeType          _size;
      T*                _data;
      int              _magic; // only to know whether the container changed (_sz up or down).

   public:
      TrailArray():
         _trailer(nullptr),
         _size(0),
         _data(nullptr),
         _magic(0)
      {}

      TrailArray(Trailer::Ptr t, Storage::Ptr storage, SizeType size) :
         _trailer(t),
         _size(size),
         _magic(t->magic()) {
         _data = new (storage) T[_size];
      }

      SizeType size() const { return _size;}

      T const & get(SizeType i) const noexcept { return _data[i];}

      bool changed() const noexcept { return _magic == _trailer->magic();}

      void set(SizeType i, T const & v)
      {
         _trailer->trail(new (_trailer) TrailEntry<T>(_data+i));
         _data[i] = v;
         _magic = _trailer->magic();
      }
};
