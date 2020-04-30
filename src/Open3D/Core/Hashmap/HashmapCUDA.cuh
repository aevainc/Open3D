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

#include "HashmapCUDA.h"

#include <thrust/device_vector.h>

namespace open3d {

/// Kernels
template <typename Hash>
__global__ void InsertKernel(CUDAHashmapImplContext<Hash> slab_hash_ctx,
                             uint8_t* input_keys,
                             uint8_t* input_values,
                             iterator_t* output_iterators,
                             uint8_t* output_masks,
                             uint32_t num_keys);
template <typename Hash>
__global__ void SearchKernel(CUDAHashmapImplContext<Hash> slab_hash_ctx,
                             uint8_t* input_keys,
                             iterator_t* output_iterators,
                             uint8_t* output_masks,
                             uint32_t num_keys);
template <typename Hash>
__global__ void RemoveKernel(CUDAHashmapImplContext<Hash> slab_hash_ctx,
                             uint8_t* input_keys,
                             uint8_t* output_masks,
                             uint32_t num_keys);
template <typename Hash>
__global__ void GetIteratorsKernel(CUDAHashmapImplContext<Hash> slab_hash_ctx,
                                   iterator_t* output_iterators,
                                   uint32_t* output_iterator_count,
                                   uint32_t num_buckets);
template <typename Hash>
__global__ void CountElemsPerBucketKernel(
        CUDAHashmapImplContext<Hash> slab_hash_ctx,
        uint32_t* bucket_elem_counts);

/// Kernel callers
template <typename Hash>
CUDAHashmapImpl<Hash>::CUDAHashmapImpl(const uint32_t max_bucket_count,
                                       const uint32_t max_keyvalue_count,
                                       const uint32_t dsize_key,
                                       const uint32_t dsize_value,
                                       Device device)
    : num_buckets_(max_bucket_count),
      device_(device),
      bucket_list_head_(nullptr) {
    mem_mgr_ = std::make_shared<InternalMemoryManager>(
            max_keyvalue_count, dsize_key + dsize_value, device_);
    node_mgr_ = std::make_shared<InternalNodeManager>(device_);

    bucket_list_head_ = static_cast<Slab*>(
            MemoryManager::Malloc(num_buckets_ * sizeof(Slab), device_));
    OPEN3D_CUDA_CHECK(
            cudaMemset(bucket_list_head_, 0xFF, sizeof(Slab) * num_buckets_));

    gpu_context_.Setup(bucket_list_head_, num_buckets_, dsize_key, dsize_value,
                       node_mgr_->gpu_context_, mem_mgr_->gpu_context_);
}

template <typename Hash>
CUDAHashmapImpl<Hash>::~CUDAHashmapImpl() {
    MemoryManager::Free(bucket_list_head_, device_);
}

template <typename Hash>
void CUDAHashmapImpl<Hash>::Insert(uint8_t* keys,
                                   uint8_t* values,
                                   iterator_t* iterators,
                                   uint8_t* masks,
                                   uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    InsertKernel<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, values,
                                             iterators, masks, num_keys);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash>
void CUDAHashmapImpl<Hash>::Search(uint8_t* keys,
                                   iterator_t* iterators,
                                   uint8_t* masks,
                                   uint32_t num_keys) {
    OPEN3D_CUDA_CHECK(cudaMemset(masks, 0, sizeof(uint8_t) * num_keys));

    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    SearchKernel<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, iterators,
                                             masks, num_keys);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash>
void CUDAHashmapImpl<Hash>::Remove(uint8_t* keys,
                                   uint8_t* masks,
                                   uint32_t num_keys) {
    OPEN3D_CUDA_CHECK(cudaMemset(masks, 0, sizeof(uint8_t) * num_keys));

    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    RemoveKernel<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, masks,
                                             num_keys);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash>
std::vector<int> CUDAHashmapImpl<Hash>::CountElemsPerBucket() {
    auto elems_per_bucket_buffer = static_cast<uint32_t*>(
            MemoryManager::Malloc(num_buckets_ * sizeof(uint32_t), device_));

    thrust::device_vector<uint32_t> elems_per_bucket(
            elems_per_bucket_buffer, elems_per_bucket_buffer + num_buckets_);
    thrust::fill(elems_per_bucket.begin(), elems_per_bucket.end(), 0);

    const uint32_t blocksize = 128;
    const uint32_t num_blocks = (num_buckets_ * 32 + blocksize - 1) / blocksize;
    CountElemsPerBucketKernel<<<num_blocks, blocksize>>>(
            gpu_context_, thrust::raw_pointer_cast(elems_per_bucket.data()));

    std::vector<int> result(num_buckets_);
    thrust::copy(elems_per_bucket.begin(), elems_per_bucket.end(),
                 result.begin());
    MemoryManager::Free(elems_per_bucket_buffer, device_);
    return std::move(result);
}

template <typename Hash>
double CUDAHashmapImpl<Hash>::ComputeLoadFactor() {
    auto elems_per_bucket = CountElemsPerBucket();
    int total_elems_stored = std::accumulate(elems_per_bucket.begin(),
                                             elems_per_bucket.end(), 0);

    node_mgr_->gpu_context_ = gpu_context_.node_mgr_ctx_;
    auto slabs_per_bucket = node_mgr_->CountSlabsPerSuperblock();
    int total_slabs_stored = std::accumulate(
            slabs_per_bucket.begin(), slabs_per_bucket.end(), num_buckets_);

    double load_factor = double(total_elems_stored) /
                         double(total_slabs_stored * WARP_WIDTH);

    return load_factor;
}

/// Device proxy
template <typename Hash>
CUDAHashmapImplContext<Hash>::CUDAHashmapImplContext()
    : num_buckets_(0), bucket_list_head_(nullptr) {
    static_assert(sizeof(Slab) == (WARP_WIDTH * sizeof(ptr_t)),
                  "Invalid slab size");
}

template <typename Hash>
__host__ void CUDAHashmapImplContext<Hash>::Setup(
        Slab* bucket_list_head,
        const uint32_t num_buckets,
        const uint32_t dsize_key,
        const uint32_t dsize_value,
        const InternalNodeManagerContext& allocator_ctx,
        const InternalMemoryManagerContext& pair_allocator_ctx) {
    bucket_list_head_ = bucket_list_head;

    num_buckets_ = num_buckets;
    dsize_key_ = dsize_key;
    dsize_value_ = dsize_value;

    node_mgr_ctx_ = allocator_ctx;
    mem_mgr_ctx_ = pair_allocator_ctx;
}

/// Device functions
template <typename Hash>
__device__ __host__ __forceinline__ uint32_t
CUDAHashmapImplContext<Hash>::ComputeBucket(uint8_t* key) const {
    return hash_fn_(key, dsize_key_) % num_buckets_;
}

template <typename Hash>
__device__ __forceinline__ void CUDAHashmapImplContext<Hash>::WarpSyncKey(
        uint8_t* key_ptr, const uint32_t lane_id, uint8_t* ret_key_ptr) {
    const int chunks = dsize_key_ / sizeof(int);
#pragma unroll 1
    for (size_t i = 0; i < chunks; ++i) {
        ((int*)(ret_key_ptr))[i] = __shfl_sync(
                ACTIVE_LANES_MASK, ((int*)(key_ptr))[i], lane_id, WARP_WIDTH);
    }
}

__device__ __host__ inline bool cmp(uint8_t* src,
                                    uint8_t* dst,
                                    uint32_t dsize) {
    bool ret = true;
#pragma unroll 1
    for (int i = 0; i < dsize; ++i) {
        ret = ret && (src[i] == dst[i]);
    }
    return ret;
}

template <typename Hash>
__device__ int32_t CUDAHashmapImplContext<Hash>::WarpFindKey(
        uint8_t* key_ptr, const uint32_t lane_id, const ptr_t ptr) {
    uint8_t is_lane_found =
            /* select key lanes */
            ((1 << lane_id) & PAIR_PTR_LANES_MASK)
            /* validate key addrs */
            && (ptr != EMPTY_PAIR_PTR)
            /* find keys in memory heap */
            && cmp(mem_mgr_ctx_.extract_ptr(ptr), key_ptr, dsize_key_);

    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_found)) - 1;
}

template <typename Hash>
__device__ __forceinline__ int32_t
CUDAHashmapImplContext<Hash>::WarpFindEmpty(const ptr_t ptr) {
    uint8_t is_lane_empty = (ptr == EMPTY_PAIR_PTR);

    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_empty)) - 1;
}

template <typename Hash>
__device__ __forceinline__ ptr_t
CUDAHashmapImplContext<Hash>::AllocateSlab(const uint32_t lane_id) {
    return node_mgr_ctx_.WarpAllocate(lane_id);
}

template <typename Hash>
__device__ __forceinline__ void CUDAHashmapImplContext<Hash>::FreeSlab(
        const ptr_t slab_ptr) {
    node_mgr_ctx_.FreeUntouched(slab_ptr);
}

template <typename Hash>
__device__ Pair<ptr_t, uint8_t> CUDAHashmapImplContext<Hash>::Search(
        uint8_t& to_search,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        uint8_t* query_key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = work_queue;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;

    ptr_t iterator = NULL_ITERATOR;
    uint8_t mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_search))) {
        /** 0. Restart from linked list head if the last query is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        uint8_t src_key[MAX_KEY_BYTESIZE];
        WarpSyncKey(query_key, src_lane, src_key);

        /* Each lane in the warp reads a uint in the slab in parallel */
        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** 1. Found in this slab, SUCCEED **/
        if (lane_found >= 0) {
            /* broadcast found value */
            ptr_t found_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANES_MASK, unit_data, lane_found, WARP_WIDTH);

            if (lane_id == src_lane) {
                to_search = false;

                iterator = found_pair_internal_ptr;
                mask = true;
            }
        }

        /** 2. Not found in this slab **/
        else {
            /* broadcast next slab: lane 31 reads 'next' */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** 2.1. Next slab is empty, ABORT **/
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                if (lane_id == src_lane) {
                    to_search = false;
                }
            }
            /** 2.2. Next slab exists, RESTART **/
            else {
                curr_slab_ptr = next_slab_ptr;
            }
        }

        prev_work_queue = work_queue;
    }

    return make_pair(iterator, mask);
}

/*
 * Insert: ABORT if found
 * replacePair: REPLACE if found
 * WE DO NOT ALLOW DUPLICATE KEYS
 */
template <typename Hash>
__device__ Pair<ptr_t, uint8_t> CUDAHashmapImplContext<Hash>::Insert(
        uint8_t& to_be_inserted,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        uint8_t* key,
        uint8_t* value) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;

    ptr_t iterator = NULL_ITERATOR;
    uint8_t mask = false;

    /** WARNING: Allocation should be finished in warp,
     * results are unexpected otherwise **/
    int prealloc_pair_internal_ptr = EMPTY_PAIR_PTR;
    if (to_be_inserted) {
        prealloc_pair_internal_ptr = mem_mgr_ctx_.Allocate();
        // TODO: replace with Assign
        uint8_t* ptr = mem_mgr_ctx_.extract_ptr(prealloc_pair_internal_ptr);
        for (int i = 0; i < dsize_key_; ++i) {
            *ptr++ = key[i];
        }
        for (int i = 0; i < dsize_value_; ++i) {
            *ptr++ = value[i];
        }
    }

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_inserted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        uint8_t src_key[MAX_KEY_BYTESIZE];
        WarpSyncKey(key, src_lane, src_key);

        /* Each lane in the warp reads a uint in the slab */
        uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);
        int32_t lane_empty = WarpFindEmpty(unit_data);

        /** Branch 1: key already existing, ABORT **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                /* free memory heap */
                to_be_inserted = false;
                mem_mgr_ctx_.Free(prealloc_pair_internal_ptr);
            }
        }

        /** Branch 2: empty slot available, try to insert **/
        else if (lane_empty >= 0) {
            if (lane_id == src_lane) {
                // TODO: check why we cannot put malloc here
                const uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_PTR)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_empty)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_empty);
                ptr_t old_pair_internal_ptr =
                        atomicCAS((unsigned int*)unit_data_ptr, EMPTY_PAIR_PTR,
                                  prealloc_pair_internal_ptr);

                /** Branch 2.1: SUCCEED **/
                if (old_pair_internal_ptr == EMPTY_PAIR_PTR) {
                    to_be_inserted = false;

                    iterator = prealloc_pair_internal_ptr;
                    mask = true;
                }
                /** Branch 2.2: failed: RESTART
                 *  In the consequent attempt,
                 *  > if the same key was inserted in this slot,
                 *    we fall back to Branch 1;
                 *  > if a different key was inserted,
                 *    we go to Branch 2 or 3.
                 * **/
            }
        }

        /** Branch 3: nothing found in this slab, goto next slab **/
        else {
            /* broadcast next slab */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** Branch 3.1: next slab existing, RESTART this lane **/
            if (next_slab_ptr != EMPTY_SLAB_PTR) {
                curr_slab_ptr = next_slab_ptr;
            }

            /** Branch 3.2: next slab empty, try to allocate one **/
            else {
                ptr_t new_next_slab_ptr = AllocateSlab(lane_id);

                if (lane_id == NEXT_SLAB_PTR_LANE) {
                    const uint32_t* unit_data_ptr =
                            (curr_slab_ptr == HEAD_SLAB_PTR)
                                    ? get_unit_ptr_from_list_head(
                                              src_bucket, NEXT_SLAB_PTR_LANE)
                                    : get_unit_ptr_from_list_nodes(
                                              curr_slab_ptr,
                                              NEXT_SLAB_PTR_LANE);

                    ptr_t old_next_slab_ptr =
                            atomicCAS((unsigned int*)unit_data_ptr,
                                      EMPTY_SLAB_PTR, new_next_slab_ptr);

                    /** Branch 3.2.1: other thread allocated, RESTART lane
                     *  In the consequent attempt, goto Branch 2' **/
                    if (old_next_slab_ptr != EMPTY_SLAB_PTR) {
                        FreeSlab(new_next_slab_ptr);
                    }
                    /** Branch 3.2.2: this thread allocated, RESTART lane,
                     * 'goto Branch 2' **/
                }
            }
        }

        prev_work_queue = work_queue;
    }

    return make_pair(iterator, mask);
}

template <typename Hash>
__device__ uint8_t
CUDAHashmapImplContext<Hash>::Remove(uint8_t& to_be_deleted,
                                     const uint32_t lane_id,
                                     const uint32_t bucket_id,
                                     uint8_t* key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;

    uint8_t mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_deleted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        uint8_t src_key[MAX_KEY_BYTESIZE];
        WarpSyncKey(key, src_lane, src_key);

        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** Branch 1: key found **/
        if (lane_found >= 0) {
            ptr_t src_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANES_MASK, unit_data, lane_found, WARP_WIDTH);

            if (lane_id == src_lane) {
                uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_PTR)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_found)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_found);
                ptr_t pair_to_delete = *unit_data_ptr;

                // TODO: keep in mind the potential double free problem
                ptr_t old_key_value_pair =
                        atomicCAS((unsigned int*)(unit_data_ptr),
                                  pair_to_delete, EMPTY_PAIR_PTR);
                /** Branch 1.1: this thread reset, free src_addr **/
                if (old_key_value_pair == pair_to_delete) {
                    mem_mgr_ctx_.Free(src_pair_internal_ptr);
                    mask = true;
                }
                /** Branch 1.2: other thread did the job, avoid double free
                 * **/
                to_be_deleted = false;
            }
        } else {  // no matching slot found:
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                // not found:
                to_be_deleted = false;
            } else {
                curr_slab_ptr = next_slab_ptr;
            }
        }
        prev_work_queue = work_queue;
    }

    return mask;
}

template <typename Hash>
__global__ void SearchKernel(CUDAHashmapImplContext<Hash> slab_hash_ctx,
                             uint8_t* keys,
                             iterator_t* iterators,
                             uint8_t* masks,
                             uint32_t num_queries) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    /* This warp is idle */
    if ((tid - lane_id) >= num_queries) {
        return;
    }

    /* Initialize the memory allocator on each warp */
    slab_hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;

    // dummy
    __shared__ uint8_t dummy_key[MAX_KEY_BYTESIZE];
    uint8_t* key = dummy_key;
    Pair<ptr_t, uint8_t> result;

    if (tid < num_queries) {
        lane_active = true;
        key = keys + tid * slab_hash_ctx.dsize_key_;
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    result = slab_hash_ctx.Search(lane_active, lane_id, bucket_id, key);

    if (tid < num_queries) {
        iterators[tid] = slab_hash_ctx.mem_mgr_ctx_.extract_ptr(result.first);
        masks[tid] = result.second;
    }
}

template <typename Hash>
__global__ void InsertKernel(CUDAHashmapImplContext<Hash> slab_hash_ctx,
                             uint8_t* keys,
                             uint8_t* values,
                             iterator_t* iterators,
                             uint8_t* masks,
                             uint32_t num_keys) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;

    // dummy
    __shared__ uint8_t dummy_key[MAX_KEY_BYTESIZE];
    uint8_t* key = dummy_key;
    uint8_t* value;

    if (tid < num_keys) {
        lane_active = true;
        key = keys + tid * slab_hash_ctx.dsize_key_;
        value = values + tid * slab_hash_ctx.dsize_value_;
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    Pair<ptr_t, uint8_t> result =
            slab_hash_ctx.Insert(lane_active, lane_id, bucket_id, key, value);

    if (tid < num_keys) {
        iterators[tid] = slab_hash_ctx.mem_mgr_ctx_.extract_ptr(result.first);
        masks[tid] = result.second;
    }
}

template <typename Hash>
__global__ void RemoveKernel(CUDAHashmapImplContext<Hash> slab_hash_ctx,
                             uint8_t* keys,
                             uint8_t* masks,
                             uint32_t num_keys) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    uint8_t* key;

    if (tid < num_keys) {
        lane_active = true;
        key = keys + tid * slab_hash_ctx.dsize_key_;
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    uint8_t success =
            slab_hash_ctx.Remove(lane_active, lane_id, bucket_id, key);

    if (tid < num_keys) {
        masks[tid] = success;
    }
}

template <typename Hash>
__global__ void GetIteratorsKernel(CUDAHashmapImplContext<Hash> slab_hash_ctx,
                                   iterator_t* iterators,
                                   uint32_t* iterator_count,
                                   uint32_t num_buckets) {
    // global warp ID
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t wid = tid >> 5;
    // assigning a warp per bucket
    if (wid >= num_buckets) {
        return;
    }

    /* uint32_t lane_id = threadIdx.x & 0x1F; */

    /* // initializing the memory allocator on each warp: */
    /* slab_hash_ctx.get_slab_alloc_ctx().Init(tid, lane_id); */

    /* uint32_t src_unit_data = */
    /*         *slab_hash_ctx.get_unit_ptr_from_list_head(wid, lane_id); */
    /* uint32_t active_mask = */
    /*         __ballot_sync(PAIR_PTR_LANES_MASK, src_unit_data !=
     * EMPTY_PAIR_PTR); */
    /* int leader = __ffs(active_mask) - 1; */
    /* uint32_t count = __popc(active_mask); */
    /* uint32_t rank = __popc(active_mask & __lanemask_lt()); */
    /* uint32_t prev_count; */
    /* if (rank == 0) { */
    /*     prev_count = atomicAdd(iterator_count, count); */
    /* } */
    /* prev_count = __shfl_sync(active_mask, prev_count, leader); */

    /* if (src_unit_data != EMPTY_PAIR_PTR) { */
    /*     iterators[prev_count + rank] = src_unit_data; */
    /* } */

    /* uint32_t next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32); */
    /* while (next != EMPTY_SLAB_PTR) { */
    /*     src_unit_data = */
    /*             *slab_hash_ctx.get_unit_ptr_from_list_nodes(next,
     * lane_id);
     */
    /*     count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK, */
    /*                                   src_unit_data != EMPTY_PAIR_PTR));
     */
    /*     next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32); */
    /* } */
    /* // writing back the results: */
    /* if (lane_id == 0) { */
    /* } */
}

/*
 * This kernel can be used to compute total number of elements within each
 * bucket. The final results per bucket is stored in d_count_result array
 */
template <typename Hash>
__global__ void CountElemsPerBucketKernel(
        CUDAHashmapImplContext<Hash> slab_hash_ctx,
        uint32_t* bucket_elem_counts) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // assigning a warp per bucket
    uint32_t wid = tid >> 5;
    if (wid >= slab_hash_ctx.bucket_size()) {
        return;
    }

    slab_hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint32_t count = 0;

    // count head node
    uint32_t src_unit_data =
            *slab_hash_ctx.get_unit_ptr_from_list_head(wid, lane_id);
    count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                  src_unit_data != EMPTY_PAIR_PTR));
    ptr_t next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data,
                             NEXT_SLAB_PTR_LANE, WARP_WIDTH);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
        src_unit_data =
                *slab_hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                      src_unit_data != EMPTY_PAIR_PTR));
        next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data, NEXT_SLAB_PTR_LANE,
                           WARP_WIDTH);
    }

    // write back the results:
    if (lane_id == 0) {
        bucket_elem_counts[wid] = count;
    }
}

template <typename Hash>
CUDAHashmap<Hash>::CUDAHashmap(uint32_t max_keys,
                               uint32_t dsize_key,
                               uint32_t dsize_value,
                               Device device)
    : Hashmap<Hash>(max_keys, dsize_key, dsize_value, device) {
    const uint32_t expected_keys_per_bucket = 10;
    num_buckets_ = (max_keys + expected_keys_per_bucket - 1) /
                   expected_keys_per_bucket;

    output_key_buffer_ = (uint8_t*)MemoryManager::Malloc(
            this->max_keys_ * this->dsize_key_, this->device_);
    output_value_buffer_ = (uint8_t*)MemoryManager::Malloc(
            this->max_keys_ * this->dsize_value_, this->device_);
    output_mask_buffer_ = (uint8_t*)MemoryManager::Malloc(
            this->max_keys_ * sizeof(uint8_t), this->device_);
    output_iterator_buffer_ = (iterator_t*)MemoryManager::Malloc(
            this->max_keys_ * sizeof(iterator_t), this->device_);

    // OPEN3D_CUDA_CHECK(cudaMemcpyFromSymbol(&hash_fn_ptr, default_hash_fn_ptr,
    //                                        sizeof(hash_t)));
    cuda_hashmap_impl_ = std::make_shared<CUDAHashmapImpl<Hash>>(
            this->num_buckets_, this->max_keys_, this->dsize_key_,
            this->dsize_value_, this->device_);
}

template <typename Hash>
CUDAHashmap<Hash>::~CUDAHashmap() {
    MemoryManager::Free(output_key_buffer_, this->device_);
    MemoryManager::Free(output_value_buffer_, this->device_);
    MemoryManager::Free(output_mask_buffer_, this->device_);
    MemoryManager::Free(output_iterator_buffer_, this->device_);
}

template <typename Hash>
std::pair<iterator_t*, uint8_t*> CUDAHashmap<Hash>::Insert(
        uint8_t* input_keys, uint8_t* input_values, uint32_t input_keys_size) {
    // TODO: rehash and increase max_keys_
    if (input_keys_size > this->max_keys_) {
        utility::LogError(
                "CUDAHashmap::Insert: number of input keys {} larger than "
                "reserved number of keys {}",
                input_keys_size, this->max_keys_);
    }

    cuda_hashmap_impl_->Insert(input_keys, input_values,
                               output_iterator_buffer_, output_mask_buffer_,
                               input_keys_size);

    return std::make_pair(output_iterator_buffer_, output_mask_buffer_);
}

template <typename Hash>
std::pair<iterator_t*, uint8_t*> CUDAHashmap<Hash>::Search(
        uint8_t* input_keys, uint32_t input_keys_size) {
    if (input_keys_size > this->max_keys_) {
        utility::LogError(
                "CUDAHashmap::Search: number of input keys {} larger than "
                "reserved number of keys {}",
                input_keys_size, this->max_keys_);
    }

    cuda_hashmap_impl_->Search(input_keys, output_iterator_buffer_,
                               output_mask_buffer_, input_keys_size);

    return std::make_pair(output_iterator_buffer_, output_mask_buffer_);
}

template <typename Hash>
uint8_t* CUDAHashmap<Hash>::Remove(uint8_t* input_keys,
                                   uint32_t input_keys_size) {
    cuda_hashmap_impl_->Remove(input_keys, output_mask_buffer_,
                               input_keys_size);

    return output_mask_buffer_;
}

}  // namespace open3d
