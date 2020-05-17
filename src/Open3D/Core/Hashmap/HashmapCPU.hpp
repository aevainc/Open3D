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

// Implementation for the CPU hashmap. Separated from HashmapCPU.h for brevity.

#include "Open3D/Core/Hashmap/HashmapCPU.h"

namespace open3d {

template <typename Hash, typename KeyEq>
CPUHashmap<Hash, KeyEq>::CPUHashmap(uint32_t init_buckets,
                                    uint32_t dsize_key,
                                    uint32_t dsize_value,
                                    Device device)
    : Hashmap<Hash, KeyEq>(init_buckets, dsize_key, dsize_value, device) {
    cpu_hashmap_impl_ = std::make_shared<
            std::unordered_map<uint8_t*, uint8_t*, Hash, KeyEq>>(
            init_buckets, Hash(dsize_key), KeyEq(dsize_key));
};

template <typename Hash, typename KeyEq>
CPUHashmap<Hash, KeyEq>::~CPUHashmap() {
    for (auto kv_pair : kv_pairs_) {
        MemoryManager::Free(kv_pair.first, this->device_);
        MemoryManager::Free(kv_pair.second, this->device_);
    }
};

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Insert(uint8_t* input_keys,
                                     uint8_t* input_values,
                                     iterator_t* output_iterators,
                                     uint8_t* output_masks,
                                     uint32_t count) {
    for (int i = 0; i < count; ++i) {
        uint8_t* src_key = (uint8_t*)input_keys + this->dsize_key_ * i;
        uint8_t* src_value = (uint8_t*)input_values + this->dsize_value_ * i;

        // Manually copy before insert
        void* dst_key = MemoryManager::Malloc(this->dsize_key_, this->device_);
        void* dst_value =
                MemoryManager::Malloc(this->dsize_value_, this->device_);

        MemoryManager::Memcpy(dst_key, this->device_, src_key, this->device_,
                              this->dsize_key_);
        MemoryManager::Memcpy(dst_value, this->device_, src_value,
                              this->device_, this->dsize_value_);

        // Try insertion
        auto res = cpu_hashmap_impl_->insert(
                {(uint8_t*)dst_key, (uint8_t*)dst_value});

        // Handle memory
        if (res.second) {
            output_iterators[i] =
                    iterator_t((uint8_t*)dst_key, (uint8_t*)dst_value);
            output_masks[i] = 1;
        } else {
            MemoryManager::Free(dst_key, this->device_);
            MemoryManager::Free(dst_value, this->device_);
            output_iterators[i] = iterator_t();
            output_masks[i] = 0;
        }
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Find(uint8_t* input_keys,
                                   iterator_t* output_iterators,
                                   uint8_t* output_masks,
                                   uint32_t count) {
    for (int i = 0; i < count; ++i) {
        uint8_t* key = (uint8_t*)input_keys + this->dsize_key_ * i;

        auto iter = cpu_hashmap_impl_->find(key);
        if (iter == cpu_hashmap_impl_->end()) {
            output_iterators[i] = iterator_t();
            output_masks[i] = 0;
        } else {
            void* key = iter->first;
            output_iterators[i] = iterator_t(iter->first, iter->second);
            output_masks[i] = 1;
        }
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Erase(uint8_t* input_keys,
                                    uint8_t* output_masks,
                                    uint32_t count) {
    for (int i = 0; i < count; ++i) {
        uint8_t* key = (uint8_t*)input_keys + this->dsize_key_ * i;

        size_t erased = cpu_hashmap_impl_->erase(key);
        output_masks[i] = erased > 0;
    }
}

template <typename Hash, typename KeyEq>
uint32_t CPUHashmap<Hash, KeyEq>::GetIterators(iterator_t* output_iterators) {
    uint32_t count = cpu_hashmap_impl_->size();

    int i = 0;
    for (auto iter = cpu_hashmap_impl_->begin();
         iter != cpu_hashmap_impl_->end(); ++iter, ++i) {
        output_iterators[i] = iterator_t(iter->first, iter->second);
    }

    return count;
}

void UnpackIteratorsStep(iterator_t* input_iterators,
                         uint8_t* input_masks,
                         uint8_t* output_keys,
                         uint8_t* output_values,
                         uint32_t dsize_key,
                         uint32_t dsize_value,
                         uint32_t tid) {
    // Valid queries
    if (input_masks == nullptr || input_masks[tid]) {
        if (output_keys != nullptr) {
            uint8_t* dst_key_ptr = output_keys + dsize_key * tid;
            uint8_t* src_key_ptr = input_iterators[tid].first;

            for (int i = 0; i < dsize_key; ++i) {
                dst_key_ptr[i] = src_key_ptr[i];
            }
        }

        if (output_values != nullptr) {
            uint8_t* dst_value_ptr = output_values + dsize_value * tid;
            uint8_t* src_value_ptr = input_iterators[tid].second;

            for (int i = 0; i < dsize_value; ++i) {
                dst_value_ptr[i] = src_value_ptr[i];
            }
        }
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::UnpackIterators(iterator_t* input_iterators,
                                              uint8_t* input_masks,
                                              uint8_t* output_keys,
                                              uint8_t* output_values,
                                              uint32_t iterator_count) {
    for (int i = 0; i < iterator_count; ++i) {
        UnpackIteratorsStep(input_iterators, input_masks, output_keys,
                            output_values, this->dsize_key_, this->dsize_value_,
                            i);
    }
}

void AssignIteratorsStep(iterator_t* input_iterators,
                         uint8_t* input_masks,
                         uint8_t* input_values,
                         uint32_t dsize_value,
                         uint32_t tid) {
    // Valid queries
    if (input_masks == nullptr || input_masks[tid]) {
        uint8_t* src_value_ptr = input_values + dsize_value * tid;
        uint8_t* dst_value_ptr = input_iterators[tid].second;

        // Byte-by-byte copy, can be improved
        for (int i = 0; i < dsize_value; ++i) {
            dst_value_ptr[i] = src_value_ptr[i];
        }
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::AssignIterators(iterator_t* input_iterators,
                                              uint8_t* input_masks,
                                              uint8_t* input_values,
                                              uint32_t iterator_count) {
    for (int i = 0; i < iterator_count; ++i) {
        AssignIteratorsStep(input_iterators, input_masks, input_values,
                            this->dsize_value_, i);
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Rehash(uint32_t buckets) {
    cpu_hashmap_impl_->rehash(buckets);
}

template <typename Hash, typename KeyEq>
std::vector<int> CPUHashmap<Hash, KeyEq>::BucketSizes() {
    size_t bucket_count = cpu_hashmap_impl_->bucket_count();
    std::vector<int> ret;
    for (int i = 0; i < bucket_count; ++i) {
        ret.push_back(cpu_hashmap_impl_->bucket_size(i));
    }
    return std::move(ret);
}

template <typename Hash, typename KeyEq>
std::shared_ptr<CPUHashmap<Hash, KeyEq>> CreateCPUHashmap(
        uint32_t init_buckets,
        uint32_t dsize_key,
        uint32_t dsize_value,
        open3d::Device device) {
    return std::make_shared<CPUHashmap<Hash, KeyEq>>(init_buckets, dsize_key,
                                                     dsize_value, device);
}
}  // namespace open3d
