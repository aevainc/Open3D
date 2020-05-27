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

// Copyright 2019 Saman Ashkiani
// Rewritten by Wei Dong 2019 - 2020
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing permissions
// and limitations under the License.

#pragma once

#include "Open3D/Core/Hashmap/CUDA/HashmapCUDA.h"
#include "Open3D/Core/Hashmap/CUDA/HashmapCUDAImpl.cuh"

#include <thrust/device_vector.h>

namespace open3d {

/// Interface
template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::CUDAHashmap(size_t initial_buckets,
                                      size_t dsize_key,
                                      size_t dsize_value,
                                      Device device)
    : Hashmap<Hash, KeyEq>(initial_buckets, dsize_key, dsize_value, device) {
    impl_ = std::make_shared<CUDAHashmapImpl<Hash, KeyEq>>(
            this->bucket_count_, this->dsize_key_, this->dsize_value_,
            this->device_);
}

template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::~CUDAHashmap() {}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Insert(const void* input_keys,
                                      const void* input_values,
                                      iterator_t* output_iterators,
                                      uint8_t* output_masks,
                                      size_t count) {
    bool extern_alloc = (output_masks != nullptr);
    if (!extern_alloc) {
        output_masks = (uint8_t*)MemoryManager::Malloc(
                count * sizeof(uint8_t), impl_->device_);
    }
    impl_->Insert((uint8_t*)input_keys, (uint8_t*)input_values,
                               output_iterators, output_masks, count);
    if (!extern_alloc) {
        MemoryManager::Free(output_masks, impl_->device_);
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Find(const void* input_keys,
                                    iterator_t* output_iterators,
                                    uint8_t* output_masks,
                                    size_t count) {
    impl_->Find((uint8_t*)input_keys, output_iterators,
                             output_masks, count);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Erase(const void* input_keys,
                                     uint8_t* output_masks,
                                     size_t count) {
    bool extern_alloc = (output_masks != nullptr);
    if (!extern_alloc) {
        output_masks = (uint8_t*)MemoryManager::Malloc(
                count * sizeof(uint8_t), impl_->device_);
    }
    impl_->Erase((uint8_t*)input_keys, output_masks, count);
    if (!extern_alloc) {
        MemoryManager::Free(output_masks, impl_->device_);
    }
}

template <typename Hash, typename KeyEq>
size_t CUDAHashmap<Hash, KeyEq>::GetIterators(iterator_t* output_iterators) {
    return impl_->GetIterators(output_iterators);
}

__global__ void UnpackIteratorsKernel(const iterator_t* input_iterators,
                                      const uint8_t* input_masks,
                                      void* output_keys,
                                      void* output_values,
                                      size_t dsize_key,
                                      size_t dsize_value,
                                      size_t iterator_count) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Valid queries
    if (tid < iterator_count && (input_masks == nullptr || input_masks[tid])) {
        if (output_keys != nullptr) {
            uint8_t* dst_key_ptr = (uint8_t*)output_keys + dsize_key * tid;
            uint8_t* src_key_ptr = input_iterators[tid].first;

            for (size_t i = 0; i < dsize_key; ++i) {
                dst_key_ptr[i] = src_key_ptr[i];
            }
        }

        if (output_values != nullptr) {
            uint8_t* dst_value_ptr =
                    (uint8_t*)output_values + dsize_value * tid;
            uint8_t* src_value_ptr = input_iterators[tid].second;

            for (size_t i = 0; i < dsize_value; ++i) {
                dst_value_ptr[i] = src_value_ptr[i];
            }
        }
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::UnpackIterators(
        const iterator_t* input_iterators,
        const uint8_t* input_masks,
        void* output_keys,
        void* output_values,
        size_t iterator_count) {
    if (iterator_count == 0) return;

    const size_t num_threads = 32;
    const size_t num_blocks = (iterator_count + num_threads - 1) / num_threads;

    UnpackIteratorsKernel<<<num_blocks, num_threads>>>(
            input_iterators, input_masks, (uint8_t*)output_keys,
            (uint8_t*)output_values, this->dsize_key_, this->dsize_value_,
            iterator_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

__global__ void AssignIteratorsKernel(iterator_t* input_iterators,
                                      const uint8_t* input_masks,
                                      const void* input_values,
                                      size_t dsize_value,
                                      size_t iterator_count) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Valid queries
    if (tid < iterator_count && (input_masks == nullptr || input_masks[tid])) {
        uint8_t* src_value_ptr = (uint8_t*)input_values + dsize_value * tid;
        uint8_t* dst_value_ptr = input_iterators[tid].second;

        // Byte-by-byte copy, can be improved
        for (size_t i = 0; i < dsize_value; ++i) {
            dst_value_ptr[i] = src_value_ptr[i];
        }
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::AssignIterators(iterator_t* input_iterators,
                                               const uint8_t* input_masks,
                                               const void* input_values,
                                               size_t iterator_count) {
    if (iterator_count == 0) return;
    const size_t num_threads = 32;
    const size_t num_blocks = (iterator_count + num_threads - 1) / num_threads;

    AssignIteratorsKernel<<<num_blocks, num_threads>>>(
            input_iterators, input_masks, (uint8_t*)input_values,
            this->dsize_value_, iterator_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Rehash(size_t buckets) {
    // TODO: add a size operator instead of rough estimation
    auto output_iterators = (iterator_t*)MemoryManager::Malloc(
            sizeof(iterator_t) * this->bucket_count_ *
                    impl_->avg_elems_per_bucket_,
            this->device_);
    uint32_t iterator_count = GetIterators(output_iterators);

    auto output_keys = MemoryManager::Malloc(this->dsize_key_ * iterator_count,
                                             this->device_);
    auto output_values = MemoryManager::Malloc(
            this->dsize_value_ * iterator_count, this->device_);

    UnpackIterators(output_iterators, /* masks = */ nullptr, output_keys,
                    output_values, iterator_count);

    this->bucket_count_ = buckets;
    impl_ = std::make_shared<CUDAHashmapImpl<Hash, KeyEq>>(
            this->bucket_count_, this->dsize_key_, this->dsize_value_,
            this->device_);

    /// Insert back
    auto output_masks = (uint8_t*)MemoryManager::Malloc(
            sizeof(uint8_t) * iterator_count, this->device_);
    Insert(output_keys, output_values, output_iterators, output_masks,
           iterator_count);

    MemoryManager::Free(output_iterators, this->device_);
    MemoryManager::Free(output_keys, this->device_);
    MemoryManager::Free(output_values, this->device_);
    MemoryManager::Free(output_masks, this->device_);
}

/// Bucket-related utilitiesx
/// Return number of elems per bucket
template <typename Hash, typename KeyEq>
std::vector<size_t> CUDAHashmap<Hash, KeyEq>::BucketSizes() {
    return impl_->CountElemsPerBucket();
}

/// Return size / bucket_count
template <typename Hash, typename KeyEq>
float CUDAHashmap<Hash, KeyEq>::LoadFactor() {
    return impl_->ComputeLoadFactor();
}

template <typename Hash, typename KeyEq>
std::shared_ptr<CUDAHashmap<Hash, KeyEq>> CreateCUDAHashmap(
        size_t init_buckets,
        size_t dsize_key,
        size_t dsize_value,
        open3d::Device device) {
    return std::make_shared<CUDAHashmap<Hash, KeyEq>>(init_buckets, dsize_key,
                                                      dsize_value, device);
}
}  // namespace open3d
