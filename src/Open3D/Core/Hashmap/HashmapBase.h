// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Hashmap/Types.h"
#include "Open3D/Core/MemoryManager.h"

namespace open3d {

struct DefaultHash {
    uint64_t OPEN3D_HOST_DEVICE operator()(uint8_t* key_ptr,
                                           uint32_t key_size) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        const int chunks = key_size / sizeof(int);
        int32_t* cast_key_ptr = (int32_t*)(key_ptr);
        for (size_t i = 0; i < chunks; ++i) {
            hash ^= cast_key_ptr[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

/// Base class: shared interface
template <typename Hash>
class Hashmap {
public:
    Hashmap(uint32_t max_keys,
            uint32_t dsize_key,
            uint32_t dsize_value,
            Device device)
        : max_keys_(max_keys),
          dsize_key_(dsize_key),
          dsize_value_(dsize_value),
          device_(device){};

    virtual std::pair<iterator_t*, uint8_t*> Insert(
            uint8_t* input_keys,
            uint8_t* input_values,
            uint32_t input_key_size) = 0;

    virtual std::pair<iterator_t*, uint8_t*> Search(
            uint8_t* input_keys, uint32_t input_key_size) = 0;

    virtual uint8_t* Remove(uint8_t* input_keys, uint32_t input_key_size) = 0;

public:
    uint32_t max_keys_;
    uint32_t dsize_key_;
    uint32_t dsize_value_;

public:
    Device device_;
};

/// Factory
template <typename Hash>
std::shared_ptr<Hashmap<Hash>> CreateHashmap(uint32_t max_keys,
                                             uint32_t dsize_key,
                                             uint32_t dsize_value,
                                             Device device);
}  // namespace open3d
