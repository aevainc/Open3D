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
#include "HashmapCPU.h"

namespace open3d {

template <typename Hash>
CPUHashmap<Hash>::CPUHashmap(uint32_t max_keys,
                             uint32_t dsize_key,
                             uint32_t dsize_value,
                             Device device)
    : Hashmap<Hash>(max_keys, dsize_key, dsize_value, device) {
    cpu_hashmap_impl_ = std::make_shared<
            std::unordered_map<uint8_t*, uint8_t*, Hash, KeyEq>>(
            max_keys, DefaultHash(dsize_key), KeyEq(dsize_key));
};

template <typename Hash>
CPUHashmap<Hash>::~CPUHashmap() {
    for (auto kv_pair : kv_pairs_) {
        MemoryManager::Free(kv_pair, this->device_);
    }
};

template <typename Hash>
std::pair<iterator_t*, uint8_t*> CPUHashmap<Hash>::Insert(
        uint8_t* input_keys, uint8_t* input_values, uint32_t input_key_size) {
    // TODO handle memory release
    auto iterators =
            (iterator_t*)std::malloc(input_key_size * sizeof(iterator_t));
    auto masks = (uint8_t*)std::malloc(input_key_size * sizeof(uint8_t));

    for (int i = 0; i < input_key_size; ++i) {
        uint8_t* src_key = (uint8_t*)input_keys + this->dsize_key_ * i;
        uint8_t* src_value = (uint8_t*)input_values + this->dsize_value_ * i;

        // Manually copy before insert
        void* dst_kvpair = MemoryManager::Malloc(
                this->dsize_key_ + this->dsize_value_, this->device_);
        void* dst_key = dst_kvpair;
        void* dst_value = (void*)((uint8_t*)dst_kvpair + this->dsize_key_);
        MemoryManager::Memcpy(dst_key, this->device_, src_key, this->device_,
                              this->dsize_key_);
        MemoryManager::Memcpy(dst_value, this->device_, src_value,
                              this->device_, this->dsize_value_);

        // Try insertion
        auto res = cpu_hashmap_impl_->insert(
                {(uint8_t*)dst_key, (uint8_t*)dst_value});

        // Handle memory
        if (res.second) {
            iterators[i] = (iterator_t)dst_kvpair;
            masks[i] = 1;
        } else {
            MemoryManager::Free(dst_kvpair, this->device_);
            iterators[i] = nullptr;
            masks[i] = 0;
        }
    }

    return std::make_pair(iterators, masks);
}

template <typename Hash>
std::pair<iterator_t*, uint8_t*> CPUHashmap<Hash>::Search(
        uint8_t* input_keys, uint32_t input_key_size) {
    // TODO: handle memory release
    auto iterators =
            (iterator_t*)std::malloc(input_key_size * sizeof(iterator_t));
    auto masks = (uint8_t*)std::malloc(input_key_size * sizeof(uint8_t));

    for (int i = 0; i < input_key_size; ++i) {
        uint8_t* key = (uint8_t*)input_keys + this->dsize_key_ * i;

        auto iter = cpu_hashmap_impl_->find(key);
        if (iter == cpu_hashmap_impl_->end()) {
            iterators[i] = nullptr;
            masks[i] = 0;
        } else {
            void* key = iter->first;
            iterators[i] = (iterator_t)key;
            masks[i] = 1;
        }
    }

    return std::make_pair(iterators, masks);
}

template <typename Hash>
uint8_t* CPUHashmap<Hash>::Remove(uint8_t* input_keys,
                                  uint32_t input_key_size) {
    utility::LogError("Unimplemented method");
    uint8_t* masks = nullptr;
    return masks;
}

}  // namespace open3d
