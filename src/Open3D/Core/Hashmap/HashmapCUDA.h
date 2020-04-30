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

/*
 * Copyright 2019 Saman Ashkiani
 * Rewritten by Wei Dong 2019 - 2020
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>
#include <memory>

#include <thrust/pair.h>

#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/MemoryManager.h"

#include "Open3D/Core/Hashmap/HashmapBase.h"
#include "Open3D/Core/Hashmap/InternalMemoryManager.h"
#include "Open3D/Core/Hashmap/InternalNodeManager.h"

namespace open3d {

template <typename Hash>
class CUDAHashmapImplContext {
public:
    CUDAHashmapImplContext();

    __host__ void Setup(Slab* bucket_list_head,
                        const uint32_t num_buckets,
                        const uint32_t dsize_key,
                        const uint32_t dsize_value,
                        const InternalNodeManagerContext& node_mgr_ctx,
                        const InternalMemoryManagerContext& mem_mgr_ctx);

    __device__ Pair<ptr_t, uint8_t> Insert(uint8_t& lane_active,
                                           const uint32_t lane_id,
                                           const uint32_t bucket_id,
                                           uint8_t* key_ptr,
                                           uint8_t* value_ptr);

    __device__ Pair<ptr_t, uint8_t> Search(uint8_t& lane_active,
                                           const uint32_t lane_id,
                                           const uint32_t bucket_id,
                                           uint8_t* key_ptr);

    __device__ uint8_t Remove(uint8_t& lane_active,
                              const uint32_t lane_id,
                              const uint32_t bucket_id,
                              uint8_t* key_ptr);

    /* Hash function */
    __device__ __host__ uint32_t ComputeBucket(uint8_t* key_ptr) const;
    __device__ __host__ uint32_t bucket_size() const { return num_buckets_; }

    __device__ __forceinline__ ptr_t* get_unit_ptr_from_list_nodes(
            const ptr_t slab_ptr, const uint32_t lane_id) {
        return node_mgr_ctx_.get_unit_ptr_from_slab(slab_ptr, lane_id);
    }
    __device__ __forceinline__ ptr_t* get_unit_ptr_from_list_head(
            const uint32_t bucket_id, const uint32_t lane_id) {
        return reinterpret_cast<uint32_t*>(bucket_list_head_) +
               bucket_id * BASE_UNIT_SIZE + lane_id;
    }

private:
    __device__ __forceinline__ void WarpSyncKey(uint8_t* key_ptr,
                                                const uint32_t lane_id,
                                                uint8_t* ret_key_ptr);
    __device__ __forceinline__ int32_t WarpFindKey(uint8_t* src_key_ptr,
                                                   const uint32_t lane_id,
                                                   const uint32_t ptr);
    __device__ __forceinline__ int32_t WarpFindEmpty(const uint32_t unit_data);

    __device__ __forceinline__ ptr_t AllocateSlab(const uint32_t lane_id);
    __device__ __forceinline__ void FreeSlab(const ptr_t slab_ptr);

public:
    uint32_t num_buckets_;
    uint32_t dsize_key_;
    uint32_t dsize_value_;

    Hash hash_fn_;

    Slab* bucket_list_head_;
    InternalNodeManagerContext node_mgr_ctx_;
    InternalMemoryManagerContext mem_mgr_ctx_;
};

template <typename Hash>
class CUDAHashmapImpl {
public:
    using MemoryManager = open3d::MemoryManager;
    CUDAHashmapImpl(const uint32_t max_bucket_count,
                    const uint32_t max_keyvalue_count,
                    const uint32_t dsize_key,
                    const uint32_t dsize_value,
                    Device device);

    ~CUDAHashmapImpl();

    void Insert(uint8_t* input_keys,
                uint8_t* input_values,
                iterator_t* output_iterators,
                uint8_t* output_masks,
                uint32_t num_keys);
    void Search(uint8_t* input_keys,
                iterator_t* output_iterators,
                uint8_t* output_masks,
                uint32_t num_keys);
    void Remove(uint8_t* input_keys, uint8_t* output_masks, uint32_t num_keys);

    void GetIterators(iterator_t* iterators, uint32_t& num_iterators);

    void ExtractIterators(iterator_t* iterators,
                          uint8_t* keys,
                          uint8_t* values,
                          uint32_t num_iterators);

    std::vector<int> CountElemsPerBucket();
    double ComputeLoadFactor();

private:
    Slab* bucket_list_head_;
    uint32_t num_buckets_;

    CUDAHashmapImplContext<Hash> gpu_context_;

    std::shared_ptr<InternalMemoryManager> mem_mgr_;
    std::shared_ptr<InternalNodeManager> node_mgr_;

    Device device_;
};

template <typename Hash>
class CUDAHashmap : public Hashmap<Hash> {
public:
    ~CUDAHashmap();

    CUDAHashmap(uint32_t max_keys,
                uint32_t dsize_key,
                uint32_t dsize_value,
                Device device);

    std::pair<iterator_t*, uint8_t*> Insert(uint8_t* input_keys,
                                            uint8_t* input_values,
                                            uint32_t input_key_size);

    std::pair<iterator_t*, uint8_t*> Search(uint8_t* input_keys,
                                            uint32_t input_key_size);

    uint8_t* Remove(uint8_t* input_keys, uint32_t input_key_size);

protected:
    uint32_t num_buckets_;

    // Buffer to store temporary results
    uint8_t* output_key_buffer_;
    uint8_t* output_value_buffer_;
    iterator_t* output_iterator_buffer_;
    uint8_t* output_mask_buffer_;

    std::shared_ptr<CUDAHashmapImpl<Hash>> cuda_hashmap_impl_;
};

}  // namespace open3d
