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
#include "Open3D/Core/Hashmap/Traits.h"
#include "Open3D/Core/MemoryManager.h"

namespace open3d {

struct DefaultHash {
    // Default constructor makes compiler happy. Undefined behavior, must set
    // key_size_ before calling operator().
    DefaultHash() {}
    DefaultHash(size_t key_size) : key_size_(key_size) {}

    uint64_t OPEN3D_HOST_DEVICE operator()(void* key_ptr) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        const int chunks = key_size_ / sizeof(int);
        int32_t* cast_key_ptr = (int32_t*)(key_ptr);
        for (size_t i = 0; i < chunks; ++i) {
            hash ^= cast_key_ptr[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }

    size_t key_size_;
};

struct DefaultKeyEq {
    // Default constructor makes compiler happy. Undefined behavior, must set
    // key_size_ before calling operator().
    DefaultKeyEq() {}
    DefaultKeyEq(size_t key_size) : key_size_(key_size) {}

    bool OPEN3D_HOST_DEVICE operator()(const void* lhs, const void* rhs) const {
        if (lhs == nullptr || rhs == nullptr) {
            return false;
        }
        const int chunks = key_size_ / sizeof(int);
        int* lhs_key_ptr = (int*)(lhs);
        int* rhs_key_ptr = (int*)(rhs);

        bool res = true;
        for (size_t i = 0; i < chunks; ++i) {
            res = res && (lhs_key_ptr[i] == rhs_key_ptr[i]);
        }
        return res;
    }

    size_t key_size_;
};

/// Base class: shared interface
template <typename Hash = DefaultHash, typename KeyEq = DefaultKeyEq>
class Hashmap {
public:
    Hashmap(size_t init_buckets,
            size_t dsize_key,
            size_t dsize_value,
            Device device)
        : bucket_count_(init_buckets),
          dsize_key_(dsize_key),
          dsize_value_(dsize_value),
          device_(device){};

    virtual void Rehash(size_t buckets) = 0;

    /// Essential hashmap operations
    virtual void Insert(void* input_keys,
                        void* input_values,
                        iterator_t* output_iterators,
                        uint8_t* output_masks,
                        size_t count) = 0;

    virtual void Find(void* input_keys,
                      iterator_t* output_iterators,
                      uint8_t* output_masks,
                      size_t count) = 0;

    virtual void Erase(void* input_keys,
                       uint8_t* output_masks,
                       size_t count) = 0;

    virtual size_t GetIterators(iterator_t* output_iterators) = 0;

    /// Parallel iterations
    /// Only write to corresponding entries if they are not nullptr
    /// Only access input_masks if they it is not nullptr
    virtual void UnpackIterators(iterator_t* input_iterators,
                                 uint8_t* input_masks,
                                 void* output_keys,
                                 void* output_values,
                                 size_t count) = 0;

    /// (Optionally) In-place modify iterators returned from Find
    /// Note: key cannot be changed, otherwise the semantic is violated
    virtual void AssignIterators(iterator_t* input_iterators,
                                 uint8_t* input_masks,
                                 void* input_values,
                                 size_t count) = 0;

    /// Bucket-related utilities
    /// Return number of elems per bucket
    virtual std::vector<size_t> BucketSizes() = 0;

    /// Return size / bucket_count
    virtual float LoadFactor() = 0;

public:
    uint32_t bucket_count_;
    uint32_t dsize_key_;
    uint32_t dsize_value_;

public:
    Device device_;
};

/// Factory
template <typename Hash, typename KeyEq>
std::shared_ptr<Hashmap<Hash, KeyEq>> CreateHashmap(size_t init_buckets,
                                                    size_t dsize_key,
                                                    size_t dsize_value,
                                                    Device device);
}  // namespace open3d
