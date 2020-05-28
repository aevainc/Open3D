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

#include "Open3D/Core/Hashmap/HashmapBase.h"

#include "Open3D/Core/Hashmap/CUDA/InternalKvPairManager.h"
#include "Open3D/Core/Hashmap/CUDA/InternalNodeManager.h"
#include "Open3D/Core/Hashmap/HashmapBase.h"

namespace open3d {
/// Kernel proxy struct
template <typename Hash, typename KeyEq>
class CUDAHashmapImplContext {
public:
    CUDAHashmapImplContext();

    __host__ void Setup(const uint32_t init_buckets,
                        const uint32_t dsize_key,
                        const uint32_t dsize_value,
                        const InternalNodeManagerContext& node_mgr_ctx,
                        const InternalKvPairManagerContext& mem_mgr_ctx);

    __device__ bool Insert(bool& lane_active,
                           const uint32_t lane_id,
                           const uint32_t bucket_id,
                           uint8_t* key_ptr,
                           ptr_t iterator_ptr);

    __device__ Pair<ptr_t, bool> Find(bool& lane_active,
                                      const uint32_t lane_id,
                                      const uint32_t bucket_id,
                                      uint8_t* key_ptr);

    __device__ Pair<ptr_t, bool> Erase(bool& lane_active,
                                       const uint32_t lane_id,
                                       const uint32_t bucket_id,
                                       uint8_t* key_ptr);

    /* Hash function */
    __device__ __host__ uint32_t ComputeBucket(uint8_t* key_ptr) const;

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
    Hash hash_fn_;
    KeyEq cmp_fn_;

    uint32_t bucket_count_;
    uint32_t dsize_key_;
    uint32_t dsize_value_;

    Slab* bucket_list_head_;
    InternalNodeManagerContext node_mgr_ctx_;
    InternalKvPairManagerContext mem_mgr_ctx_;
};

/// Kernels
template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* input_keys,
                                  ptr_t* output_iterator_ptrs,
                                  uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* input_keys,
                                  ptr_t* output_iterator_ptrs,
                                  bool* output_masks,
                                  uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass2(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  uint8_t* input_values,
                                  ptr_t* input_iterator_ptrs,
                                  iterator_t* output_iterators,
                                  bool* output_masks,
                                  uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void FindKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                           uint8_t* input_keys,
                           iterator_t* output_iterators,
                           bool* output_masks,
                           uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 uint8_t* input_keys,
                                 ptr_t* output_iterator_ptrs,
                                 bool* output_masks,
                                 uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 ptr_t* output_iterator_ptrs,
                                 bool* output_masks,
                                 uint32_t input_count);

template <typename Hash, typename KeyEq>
__global__ void GetIteratorsKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                   iterator_t* output_iterators,
                                   uint32_t* output_iterator_count);

template <typename Hash, typename KeyEq>
__global__ void CountElemsPerBucketKernel(
        CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
        size_t* bucket_elem_counts);

__global__ void UnpackIteratorsKernel(const iterator_t* input_iterators,
                                      const bool* input_masks,
                                      void* output_keys,
                                      void* output_values,
                                      size_t dsize_key,
                                      size_t dsize_value,
                                      size_t iterator_count);

__global__ void AssignIteratorsKernel(iterator_t* input_iterators,
                                      const bool* input_masks,
                                      const void* input_values,
                                      size_t dsize_value,
                                      size_t iterator_count);
}  // namespace open3d
