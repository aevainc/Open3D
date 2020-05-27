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

namespace open3d {  /// Kernels
template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* input_keys,
                                  ptr_t* output_iterator_ptrs,
                                  uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* input_keys,
                                  ptr_t* output_iterator_ptrs,
                                  uint8_t* output_masks,
                                  uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass2(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* input_values,
                                  ptr_t* input_iterator_ptrs,
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
__global__ void EraseKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 uint8_t* input_keys,
                                 ptr_t* output_iterator_ptrs,
                                 uint8_t* output_masks,
                                 uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 ptr_t* output_iterator_ptrs,
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

    for (size_t i = 0; i < chunks; ++i) {
        ((int*)(ret_key_ptr))[i] = __shfl_sync(
                ACTIVE_LANES_MASK, ((int*)(key_ptr))[i], lane_id, WARP_WIDTH);
    }
}

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
__device__ uint8_t
CUDAHashmapImplContext<Hash, KeyEq>::Insert(uint8_t& to_be_inserted,
                                            const uint32_t lane_id,
                                            const uint32_t bucket_id,
                                            uint8_t* key,
                                            ptr_t iterator_ptr) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;
    uint8_t src_key[MAX_KEY_BYTESIZE];

    uint8_t mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_inserted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

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

                ptr_t old_iterator_ptr =
                        atomicCAS((unsigned int*)unit_data_ptr, EMPTY_PAIR_PTR,
                                  iterator_ptr);

                // Remember to clean up in another pass
                /** Branch 2.1: SUCCEED **/
                if (old_iterator_ptr == EMPTY_PAIR_PTR) {
                    to_be_inserted = false;
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

    return mask;
}

template <typename Hash, typename KeyEq>
__device__ Pair<ptr_t, uint8_t> CUDAHashmapImplContext<Hash, KeyEq>::Erase(
        uint8_t& to_be_deleted,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        uint8_t* key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;
    uint8_t src_key[MAX_KEY_BYTESIZE];

    ptr_t iterator_ptr = 0;
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

        WarpSyncKey(key, src_lane, src_key);

        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** Branch 1: key found **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_PTR)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_found)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_found);

                uint32_t pair_to_delete = atomicExch(
                        (unsigned int*)unit_data_ptr, EMPTY_PAIR_PTR);
                mask = pair_to_delete != EMPTY_PAIR_PTR;
                iterator_ptr = pair_to_delete;
                // ptr_t old_key_value_pair =
                //         atomicCAS((unsigned int*)(unit_data_ptr),
                //                   pair_to_delete, EMPTY_PAIR_PTR);
                /** Branch 1.1: this thread reset, free src_addr **/
                // if (old_key_value_pair == pair_to_delete) {
                //     iterator_ptr = pair_to_delete;
                //     mask = true;
                // }
                // to_be_deleted = false;

                /** Branch 1.2: other thread did the job, avoid double free
                 * **/
            }
        } else {  // no matching slot found:
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                // not found:
                if (lane_id == src_lane) {
                    to_be_deleted = false;
                }
            } else {
                curr_slab_ptr = next_slab_ptr;
            }
        }
        // printf("src_lane = %d, lane_id = %d, src_key = %d, active = %d\n",
        //         src_lane, lane_id, *((int*)src_key), (int)to_be_deleted);
        prev_work_queue = work_queue;
    }

    return make_pair(iterator_ptr, mask);
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
__global__ void InsertKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* keys,
                                  ptr_t* iterator_ptrs,
                                  uint32_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < input_count) {
        /** First write ALL keys to avoid potential thread conflicts **/
        ptr_t iterator_ptr = hash_ctx.mem_mgr_ctx_.SafeAllocate();
        iterator_t iterator =
                hash_ctx.mem_mgr_ctx_.extract_iterator(iterator_ptr);

        int* dst_key_ptr = (int*)iterator.first;
        int* src_key_ptr = (int*)(keys + tid * hash_ctx.dsize_key_);
        for (int i = 0; i < hash_ctx.dsize_key_ / sizeof(int); ++i) {
            dst_key_ptr[i] = src_key_ptr[i];
        }

        iterator_ptrs[tid] = iterator_ptr;
    }
}

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* keys,
                                  ptr_t* iterator_ptrs,
                                  uint8_t* masks,
                                  uint32_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = tid % 32;

    if ((tid - lane_id) >= input_count) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    ptr_t iterator_ptr = 0;

    // dummy
    uint8_t dummy_key[MAX_KEY_BYTESIZE];
    uint8_t* key = dummy_key;

    if (tid < input_count) {
        lane_active = true;
        key = keys + tid * hash_ctx.dsize_key_;
        iterator_ptr = iterator_ptrs[tid];
        bucket_id = hash_ctx.ComputeBucket(key);
    }

    uint8_t mask =
            hash_ctx.Insert(lane_active, lane_id, bucket_id, key, iterator_ptr);

    if (tid < input_count) {
        masks[tid] = mask;
    }
}

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass2(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* values,
                                  ptr_t* iterator_ptrs,
                                  iterator_t* iterators,
                                  uint8_t* masks,
                                  uint32_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < input_count) {
        ptr_t iterator_ptr = iterator_ptrs[tid];

        if (masks[tid]) {
            iterator_t iterator =
                    hash_ctx.mem_mgr_ctx_.extract_iterator(iterator_ptr);
            // Success: copy remaining values
            int* src_value_ptr = (int*)(values + tid * hash_ctx.dsize_value_);
            int* dst_value_ptr = (int*)iterator.second;
            for (int i = 0; i < hash_ctx.dsize_value_ / sizeof(int); ++i) {
                dst_value_ptr[i] = src_value_ptr[i];
            }

            if (iterators != nullptr) {
                iterators[tid] = iterator;
            }
        } else {
            hash_ctx.mem_mgr_ctx_.Free(iterator_ptr);

            if (iterators != nullptr) {
                iterators[tid] = iterator_t();
            }
        }
    }
}

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 uint8_t* keys,
                                 ptr_t* iterator_ptrs,
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

    __shared__ uint8_t dummy_key[MAX_KEY_BYTESIZE];
    uint8_t* key = dummy_key;

    if (tid < input_count) {
        lane_active = true;
        key = keys + tid * hash_ctx.dsize_key_;
        bucket_id = hash_ctx.ComputeBucket(key);
    }

    auto result = hash_ctx.Erase(lane_active, lane_id, bucket_id, key);

    if (tid < input_count) {
        iterator_ptrs[tid] = result.first;
        masks[tid] = result.second;
    }
}

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 ptr_t* iterator_ptrs,
                                 uint8_t* masks,
                                 uint32_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < input_count && masks[tid]) {
        hash_ctx.mem_mgr_ctx_.Free(iterator_ptrs[tid]);
    }
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
    __syncwarp(0xffffffff);
    prev_count = __shfl_sync(ACTIVE_LANES_MASK, prev_count, 0, WARP_WIDTH);

    if (is_active && ((1 << lane_id) & PAIR_PTR_LANES_MASK)) {
        iterators[prev_count + lane_id] =
                hash_ctx.mem_mgr_ctx_.extract_iterator(src_unit_data);
    }

    ptr_t next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data,
                             NEXT_SLAB_PTR_LANE, WARP_WIDTH);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
        src_unit_data = *hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        is_active = (src_unit_data != EMPTY_PAIR_PTR);
        count = __popc(__ballot_sync(PAIR_PTR_LANES_MASK, is_active));
        if (lane_id == 0) {
            prev_count = atomicAdd(iterator_count, count);
        }
        __syncwarp(0xffffffff);
        prev_count = __shfl_sync(ACTIVE_LANES_MASK, prev_count, 0, WARP_WIDTH);

        if (is_active && ((1 << lane_id) & PAIR_PTR_LANES_MASK)) {
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
    const uint32_t est_kvpairs = init_buckets * avg_elems_per_bucket_;
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

    auto iterator_ptrs =
            (ptr_t*)MemoryManager::Malloc(sizeof(ptr_t) * input_count, device_);

    InsertKernelPass0<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys,
                                                  iterator_ptrs, input_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    InsertKernelPass1<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, iterator_ptrs, masks, input_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    int heap_counter;
    heap_counter =
            *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_);
    std::cout << "Before " << heap_counter << "\n";
    InsertKernelPass2<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, values, iterator_ptrs, iterators, masks, input_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    heap_counter =
            *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_);
    std::cout << "After " << heap_counter << "\n";

    MemoryManager::Free(iterator_ptrs, device_);
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

    auto iterator_ptrs =
            (ptr_t*)MemoryManager::Malloc(sizeof(ptr_t) * input_count, device_);

    EraseKernelPass0<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, iterator_ptrs, masks, input_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    EraseKernelPass1<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, iterator_ptrs,
                                                 masks, input_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    MemoryManager::Free(iterator_ptrs, device_);
}

template <typename Hash, typename KeyEq>
uint32_t CUDAHashmapImpl<Hash, KeyEq>::GetIterators(iterator_t* iterators) {
    const uint32_t blocksize = 128;
    const uint32_t num_blocks =
            (gpu_context_.bucket_count_ * avg_elems_per_bucket_ + blocksize -
             1) /
            blocksize;

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
            (gpu_context_.bucket_count_ * avg_elems_per_bucket_ + blocksize -
             1) /
            blocksize;
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
    //         std::accumulate(slabs_per_bucket.begin(),
    //         slabs_per_bucket.end(),
    //                         gpu_context_.bucket_count_);

    float load_factor =
            float(total_elems_stored) / float(elems_per_bucket.size());

    return load_factor;
}
}  // namespace open3d