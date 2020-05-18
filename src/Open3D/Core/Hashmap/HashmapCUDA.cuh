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

// Interface for the CUDA hashmap. Separated from HashmapCUDA.h for brevity.

#include "Open3D/Core/Hashmap/HashmapCUDA.h"

#include <thrust/device_vector.h>

namespace open3d {

/// Kernels
template <typename Hash, typename KeyEq>
__global__ void InsertKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                             uint8_t* input_keys,
                             uint8_t* input_values,
                             iterator_t* output_iterators,
                             uint8_t* output_masks,
                             uint32_t input_count);
template <typename Hash, typename KeyEq>
__global__ void FindKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                           uint8_t* input_keys,
                           iterator_t* output_iterators,
                           uint8_t* output_masks,
                           uint32_t input_count);
template <typename Hash, typename KeyEq>
__global__ void EraseKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                            uint8_t* input_keys,
                            uint8_t* output_masks,
                            uint32_t input_count);
template <typename Hash, typename KeyEq>
__global__ void GetIteratorsKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                   iterator_t* output_iterators,
                                   uint32_t* output_iterator_count);
template <typename Hash, typename KeyEq>
__global__ void CountElemsPerBucketKernel(
        CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
        size_t* bucket_elem_counts);

/// Device proxy
template <typename Hash, typename KeyEq>
CUDAHashmapImplContext<Hash, KeyEq>::CUDAHashmapImplContext()
    : bucket_count_(0), bucket_list_head_(nullptr) {
    static_assert(sizeof(Slab) == (WARP_WIDTH * sizeof(ptr_t)),
                  "Invalid slab size");
}

template <typename Hash, typename KeyEq>
__host__ void CUDAHashmapImplContext<Hash, KeyEq>::Setup(
        const uint32_t init_buckets,
        const uint32_t dsize_key,
        const uint32_t dsize_value,
        const InternalNodeManagerContext& allocator_ctx,
        const InternalKvPairManagerContext& pair_allocator_ctx) {
    bucket_count_ = init_buckets;
    dsize_key_ = dsize_key;
    dsize_value_ = dsize_value;

    node_mgr_ctx_ = allocator_ctx;
    mem_mgr_ctx_ = pair_allocator_ctx;

    hash_fn_.key_size_ = dsize_key;
    cmp_fn_.key_size_ = dsize_key;
}

/// Device functions
template <typename Hash, typename KeyEq>
__device__ __host__ __forceinline__ uint32_t
CUDAHashmapImplContext<Hash, KeyEq>::ComputeBucket(uint8_t* key) const {
    return hash_fn_(key) % bucket_count_;
}

template <typename Hash, typename KeyEq>
__device__ __forceinline__ void
CUDAHashmapImplContext<Hash, KeyEq>::WarpSyncKey(uint8_t* key_ptr,
                                                 const uint32_t lane_id,
                                                 uint8_t* ret_key_ptr) {
    const int chunks = dsize_key_ / sizeof(int);
#pragma unroll 1
    for (size_t i = 0; i < chunks; ++i) {
        ((int*)(ret_key_ptr))[i] = __shfl_sync(
                ACTIVE_LANES_MASK, ((int*)(key_ptr))[i], lane_id, WARP_WIDTH);
    }
}

// __device__ __host__ inline bool cmp(uint8_t* src,
//                                     uint8_t* dst,
//                                     uint32_t dsize) {
//     bool ret = true;
// #pragma unroll 1
//     for (int i = 0; i < dsize; ++i) {
//         ret = ret && (src[i] == dst[i]);
//     }
//     return ret;
// }

template <typename Hash, typename KeyEq>
__device__ int32_t CUDAHashmapImplContext<Hash, KeyEq>::WarpFindKey(
        uint8_t* key_ptr, const uint32_t lane_id, const ptr_t ptr) {
    uint8_t is_lane_found =
            /* select key lanes */
            ((1 << lane_id) & PAIR_PTR_LANES_MASK)
            /* validate key addrs */
            && (ptr != EMPTY_PAIR_PTR)
            /* find keys in memory heap */
            && cmp_fn_(mem_mgr_ctx_.extract_iterator(ptr).first, key_ptr);

    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_found)) - 1;
}

template <typename Hash, typename KeyEq>
__device__ __forceinline__ int32_t
CUDAHashmapImplContext<Hash, KeyEq>::WarpFindEmpty(const ptr_t ptr) {
    uint8_t is_lane_empty = (ptr == EMPTY_PAIR_PTR);

    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_empty)) - 1;
}

template <typename Hash, typename KeyEq>
__device__ __forceinline__ ptr_t
CUDAHashmapImplContext<Hash, KeyEq>::AllocateSlab(const uint32_t lane_id) {
    return node_mgr_ctx_.WarpAllocate(lane_id);
}

template <typename Hash, typename KeyEq>
__device__ __forceinline__ void CUDAHashmapImplContext<Hash, KeyEq>::FreeSlab(
        const ptr_t slab_ptr) {
    node_mgr_ctx_.FreeUntouched(slab_ptr);
}

template <typename Hash, typename KeyEq>
__device__ Pair<ptr_t, uint8_t> CUDAHashmapImplContext<Hash, KeyEq>::Find(
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

                /// Actually iterator_ptr
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
template <typename Hash, typename KeyEq>
__device__ Pair<ptr_t, uint8_t> CUDAHashmapImplContext<Hash, KeyEq>::Insert(
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
        iterator_t iter =
                mem_mgr_ctx_.extract_iterator(prealloc_pair_internal_ptr);

        uint8_t* ptr = iter.first;
        for (int i = 0; i < dsize_key_; ++i) {
            *ptr++ = key[i];
        }

        ptr = iter.second;
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

template <typename Hash, typename KeyEq>
__device__ uint8_t
CUDAHashmapImplContext<Hash, KeyEq>::Erase(uint8_t& to_be_deleted,
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

template <typename Hash, typename KeyEq>
__global__ void FindKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                           uint8_t* keys,
                           iterator_t* iterators,
                           uint8_t* masks,
                           uint32_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    /* This warp is idle */
    if ((tid - lane_id) >= input_count) {
        return;
    }

    /* Initialize the memory allocator on each warp */
    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;

    // dummy
    __shared__ uint8_t dummy_key[MAX_KEY_BYTESIZE];
    uint8_t* key = dummy_key;
    Pair<ptr_t, uint8_t> result;

    if (tid < input_count) {
        lane_active = true;
        key = keys + tid * hash_ctx.dsize_key_;
        bucket_id = hash_ctx.ComputeBucket(key);
    }

    result = hash_ctx.Find(lane_active, lane_id, bucket_id, key);

    if (tid < input_count) {
        iterators[tid] = hash_ctx.mem_mgr_ctx_.extract_iterator(result.first);
        masks[tid] = result.second;
    }
}

template <typename Hash, typename KeyEq>
__global__ void InsertKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                             uint8_t* keys,
                             uint8_t* values,
                             iterator_t* iterators,
                             uint8_t* masks,
                             uint32_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= input_count) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;

    // dummy
    __shared__ uint8_t dummy_key[MAX_KEY_BYTESIZE];
    uint8_t* key = dummy_key;
    uint8_t* value;

    if (tid < input_count) {
        lane_active = true;
        key = keys + tid * hash_ctx.dsize_key_;
        value = values + tid * hash_ctx.dsize_value_;
        bucket_id = hash_ctx.ComputeBucket(key);
        // printf("%d -> bucket %d\n", tid, bucket_id); 
    }

    Pair<ptr_t, uint8_t> result =
            hash_ctx.Insert(lane_active, lane_id, bucket_id, key, value);

    if (tid < input_count) {
        iterators[tid] = hash_ctx.mem_mgr_ctx_.extract_iterator(result.first);
        masks[tid] = result.second;
    }
}

template <typename Hash, typename KeyEq>
__global__ void EraseKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                            uint8_t* keys,
                            uint8_t* masks,
                            uint32_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= input_count) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    uint8_t* key;

    if (tid < input_count) {
        lane_active = true;
        key = keys + tid * hash_ctx.dsize_key_;
        bucket_id = hash_ctx.ComputeBucket(key);
    }

    uint8_t success = hash_ctx.Erase(lane_active, lane_id, bucket_id, key);

    if (tid < input_count) {
        masks[tid] = success;
    }
}

__device__ int32_t __lanemask_lt(uint32_t lane_id) {
    return ((int32_t)1 << lane_id) - 1;
}

template <typename Hash, typename KeyEq>
__global__ void GetIteratorsKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                   iterator_t* iterators,
                                   uint32_t* iterator_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // assigning a warp per bucket
    uint32_t wid = tid >> 5;
    if (wid >= hash_ctx.bucket_count_) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint32_t count = 0;
    uint32_t prev_count = 0;

    // TODO simplify code
    // count head node
    uint32_t src_unit_data =
            *hash_ctx.get_unit_ptr_from_list_head(wid, lane_id);
    bool is_active = src_unit_data != EMPTY_PAIR_PTR;
    count = __popc(__ballot_sync(PAIR_PTR_LANES_MASK, is_active));
    if (lane_id == 0) {
        prev_count = atomicAdd(iterator_count, count);
    }
    prev_count = __shfl_sync(ACTIVE_LANES_MASK, prev_count, 0, WARP_WIDTH);

    if (is_active && ((1 << lane_id) & PAIR_PTR_LANES_MASK)) {
        iterators[prev_count + lane_id] =
                hash_ctx.mem_mgr_ctx_.extract_iterator(src_unit_data);
        // printf("head: wid=%d, prev_count=%d, internal_ptr=%d, lane_id=%d, "
        //        "iterators[%d] = %ld, %d\n",
        //        wid, prev_count, src_unit_data, lane_id, prev_count + lane_id,
        //        iterators[prev_count + lane_id].first,
        //        *(int*)(iterators[prev_count + lane_id].first));
    }

    ptr_t next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data,
                             NEXT_SLAB_PTR_LANE, WARP_WIDTH);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
        src_unit_data = *hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        count = __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                     src_unit_data != EMPTY_PAIR_PTR));
        if (lane_id == 0) {
            prev_count = atomicAdd(iterator_count, count);
        }
        printf("list\n");
        prev_count = __shfl_sync(ACTIVE_LANES_MASK, prev_count, 0, WARP_WIDTH);

        uint32_t prev_count = atomicAdd(iterator_count, count);
        if (is_active && ((1 << lane_id) & PAIR_PTR_LANES_MASK)) {
            // printf("list: wid=%d, prev_count=%d, internal_ptr=%d\n", wid,
            //        prev_count, src_unit_data);
            iterators[prev_count + lane_id] =
                    hash_ctx.mem_mgr_ctx_.extract_iterator(src_unit_data);
        }
        next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data, NEXT_SLAB_PTR_LANE,
                           WARP_WIDTH);
    }
}

/*
 * This kernel can be used to compute total number of elements within each
 * bucket. The final results per bucket is stored in d_count_result array
 */
template <typename Hash, typename KeyEq>
__global__ void CountElemsPerBucketKernel(
        CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
        size_t* bucket_elem_counts) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // assigning a warp per bucket
    uint32_t wid = tid >> 5;
    if (wid >= hash_ctx.bucket_count_) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint32_t count = 0;

    // count head node
    uint32_t src_unit_data =
            *hash_ctx.get_unit_ptr_from_list_head(wid, lane_id);
    count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                  src_unit_data != EMPTY_PAIR_PTR));
    ptr_t next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data,
                             NEXT_SLAB_PTR_LANE, WARP_WIDTH);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
        src_unit_data = *hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
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

template <typename Hash, typename KeyEq>
CUDAHashmapImpl<Hash, KeyEq>::CUDAHashmapImpl(const uint32_t init_buckets,
                                              const uint32_t dsize_key,
                                              const uint32_t dsize_value,
                                              Device device)
    : device_(device) {
    const uint32_t est_bucket_factor = 32;
    const uint32_t est_kvpairs = init_buckets * est_bucket_factor;
    mem_mgr_ = std::make_shared<InternalKvPairManager>(est_kvpairs, dsize_key,
                                                       dsize_value, device_);
    node_mgr_ = std::make_shared<InternalNodeManager>(device_);

    gpu_context_.Setup(init_buckets, dsize_key, dsize_value,
                       node_mgr_->gpu_context_, mem_mgr_->gpu_context_);
    gpu_context_.bucket_list_head_ = static_cast<Slab*>(
            MemoryManager::Malloc(sizeof(Slab) * init_buckets, device_));
    OPEN3D_CUDA_CHECK(cudaMemset(gpu_context_.bucket_list_head_, 0xFF,
                                 sizeof(Slab) * init_buckets));
}

template <typename Hash, typename KeyEq>
CUDAHashmapImpl<Hash, KeyEq>::~CUDAHashmapImpl() {
    MemoryManager::Free(gpu_context_.bucket_list_head_, device_);
}

template <typename Hash, typename KeyEq>
void CUDAHashmapImpl<Hash, KeyEq>::Insert(uint8_t* keys,
                                          uint8_t* values,
                                          iterator_t* iterators,
                                          uint8_t* masks,
                                          uint32_t input_count) {
    const uint32_t num_blocks = (input_count + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    InsertKernel<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, values,
                                             iterators, masks, input_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmapImpl<Hash, KeyEq>::Find(uint8_t* keys,
                                        iterator_t* iterators,
                                        uint8_t* masks,
                                        uint32_t input_count) {
    OPEN3D_CUDA_CHECK(cudaMemset(masks, 0, sizeof(uint8_t) * input_count));

    const uint32_t num_blocks = (input_count + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    FindKernel<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, iterators, masks,
                                           input_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmapImpl<Hash, KeyEq>::Erase(uint8_t* keys,
                                         uint8_t* masks,
                                         uint32_t input_count) {
    OPEN3D_CUDA_CHECK(cudaMemset(masks, 0, sizeof(uint8_t) * input_count));

    const uint32_t num_blocks = (input_count + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    EraseKernel<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, masks,
                                            input_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
uint32_t CUDAHashmapImpl<Hash, KeyEq>::GetIterators(iterator_t* iterators) {
    const uint32_t blocksize = 128;
    const uint32_t num_blocks =
            (gpu_context_.bucket_count_ * 32 + blocksize - 1) / blocksize;

    uint32_t* iterator_count_cuda =
            (uint32_t*)MemoryManager::Malloc(sizeof(uint32_t), device_);
    cudaMemset(iterator_count_cuda, 0, sizeof(uint32_t));

    GetIteratorsKernel<<<num_blocks, blocksize>>>(gpu_context_, iterators,
                                                  iterator_count_cuda);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    uint32_t iterator_count;
    MemoryManager::Memcpy(&iterator_count, Device("CPU:0"), iterator_count_cuda,
                          device_, sizeof(uint32_t));
    return iterator_count;
}

template <typename Hash, typename KeyEq>
std::vector<size_t> CUDAHashmapImpl<Hash, KeyEq>::CountElemsPerBucket() {
    auto elems_per_bucket_buffer = static_cast<size_t*>(MemoryManager::Malloc(
            gpu_context_.bucket_count_ * sizeof(size_t), device_));

    thrust::device_vector<size_t> elems_per_bucket(
            elems_per_bucket_buffer,
            elems_per_bucket_buffer + gpu_context_.bucket_count_);
    thrust::fill(elems_per_bucket.begin(), elems_per_bucket.end(), 0);

    const uint32_t blocksize = 128;
    const uint32_t num_blocks =
            (gpu_context_.bucket_count_ * 32 + blocksize - 1) / blocksize;
    CountElemsPerBucketKernel<<<num_blocks, blocksize>>>(
            gpu_context_, thrust::raw_pointer_cast(elems_per_bucket.data()));

    std::vector<size_t> result(gpu_context_.bucket_count_);
    thrust::copy(elems_per_bucket.begin(), elems_per_bucket.end(),
                 result.begin());
    MemoryManager::Free(elems_per_bucket_buffer, device_);
    return std::move(result);
}

template <typename Hash, typename KeyEq>
float CUDAHashmapImpl<Hash, KeyEq>::ComputeLoadFactor() {
    auto elems_per_bucket = CountElemsPerBucket();
    int total_elems_stored = std::accumulate(elems_per_bucket.begin(),
                                             elems_per_bucket.end(), 0);

    node_mgr_->gpu_context_ = gpu_context_.node_mgr_ctx_;

    /// Unrelated factor for now
    // auto slabs_per_bucket = node_mgr_->CountSlabsPerSuperblock();
    // int total_slabs_stored =
    //         std::accumulate(slabs_per_bucket.begin(), slabs_per_bucket.end(),
    //                         gpu_context_.bucket_count_);

    float load_factor =
            float(total_elems_stored) / float(elems_per_bucket.size());

    return load_factor;
}

/// Interface
template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::CUDAHashmap(size_t initial_buckets,
                                      size_t dsize_key,
                                      size_t dsize_value,
                                      Device device)
    : Hashmap<Hash, KeyEq>(initial_buckets, dsize_key, dsize_value, device) {
    cuda_hashmap_impl_ = std::make_shared<CUDAHashmapImpl<Hash, KeyEq>>(
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
    cuda_hashmap_impl_->Insert((uint8_t*)input_keys, (uint8_t*)input_values,
                               output_iterators, output_masks, count);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Find(const void* input_keys,
                                    iterator_t* output_iterators,
                                    uint8_t* output_masks,
                                    size_t count) {
    cuda_hashmap_impl_->Find((uint8_t*)input_keys, output_iterators,
                             output_masks, count);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Erase(const void* input_keys,
                                     uint8_t* output_masks,
                                     size_t count) {
    cuda_hashmap_impl_->Erase((uint8_t*)input_keys, output_masks, count);
}

template <typename Hash, typename KeyEq>
size_t CUDAHashmap<Hash, KeyEq>::GetIterators(iterator_t* output_iterators) {
    return cuda_hashmap_impl_->GetIterators(output_iterators);
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
          uint8_t* dst_value_ptr = (uint8_t*)output_values + dsize_value * tid;
            uint8_t* src_value_ptr = input_iterators[tid].second;

            for (size_t i = 0; i < dsize_value; ++i) {
                dst_value_ptr[i] = src_value_ptr[i];
            }
        }
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::UnpackIterators(const iterator_t* input_iterators,
                                               const uint8_t* input_masks,
                                               void* output_keys,
                                               void* output_values,
                                               size_t iterator_count) {
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
            sizeof(iterator_t) * this->bucket_count_ * 32, this->device_);
    uint32_t iterator_count = GetIterators(output_iterators);

    auto output_keys = MemoryManager::Malloc(this->dsize_key_ * iterator_count,
                                             this->device_);
    auto output_values = MemoryManager::Malloc(
            this->dsize_value_ * iterator_count, this->device_);

    UnpackIterators(output_iterators, /* masks = */ nullptr, output_keys,
                    output_values, iterator_count);

    this->bucket_count_ = buckets;
    cuda_hashmap_impl_ = std::make_shared<CUDAHashmapImpl<Hash, KeyEq>>(
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
    return cuda_hashmap_impl_->CountElemsPerBucket();
}

/// Return size / bucket_count
template <typename Hash, typename KeyEq>
float CUDAHashmap<Hash, KeyEq>::LoadFactor() {
    return cuda_hashmap_impl_->ComputeLoadFactor();
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
