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

#include <cstddef>
#include <memory>
#include <string>

#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/SparseTensorList.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"

namespace open3d {
namespace core {
/* class DataDescriptor { */
/* public: */
/*     DataDescriptor() = default; */

/*     DataDescriptor(const SizeVector& element_shape, */
/*                    const std::vector<Dtype>& dtypes) { */
/*         ndims_ = static_cast<int64_t>(element_shape.size()); */
/*         for (int64_t i = 0; i < ndims_; ++i) { */
/*             element_shape_[i] = element_shape[i]; */
/*         } */

/*         int64_t stride = 1; */
/*         for (int64_t i = ndims_ - 1; i >= 0; --i) { */
/*             element_strides_[i] = stride; */
/*             // Handles 0-sized dimensions */
/*             stride = */
/*                     element_shape_[i] > 1 ? stride * element_shape_[i] :
 * stride; */
/*         } */
/*         num_elems_ = stride; */
/*         nn num_members_ = static_cast<int64_t>(dtypes.size()); */
/*         for (int64_t i = 0; i < num_members_; ++i) { */
/*             byte_size_[i] = DtypeUtil::ByteSize(dtypes[i]); */
/*         } */

/*         byte_offsets_[0] = 0; */
/*         for (int64_t i = 1; i < num_members_; ++i) { */
/*             byte_offsets_[i] = byte_size_[i - 1] * num_elems_; */
/*         } */
/*     } */

/*     OPEN3D_HOST_DEVICE int64_t GetStride(int64_t i) const { */
/*         return element_strides_[i < 0 ? ndims_ + i : i]; */
/*     } */

/*     OPEN3D_HOST_DEVICE int64_t GetOffset(int64_t tensor_idx, int64_t i) const
 * { */
/*         return byte_offsets_[tensor_idx] + i * byte_size_[tensor_idx]; */
/*     } */

/*     OPEN3D_HOST_DEVICE void* GetInputPtrFrom2D(int64_t tensor_idx, */
/*                                                int64_t u, */
/*                                                int64_t v) const { */
/*         int64_t ndims = inputs_[tensor_idx].ndims_; */
/*         if (u < 0 || v < 0 || v >= inputs_[tensor_idx].shape_[ndims - 2] ||
 */
/*             u >= inputs_[tensor_idx].shape_[ndims - 1]) { */
/*             return nullptr; */
/*         } */
/*         int64_t offset = v * inputs_[tensor_idx].byte_strides_[ndims - 2] +
 */
/*                          u * inputs_[tensor_idx].byte_strides_[ndims - 1]; */
/*         return static_cast<char*>(inputs_[tensor_idx].data_ptr_) + offset; */
/*     } */

/* }; */

static constexpr int64_t MAX_VALUE_TENSOR_DIMS = 10;
static constexpr int64_t MAX_VALUE_TENSORS = 5;

/// SparseIndexer is used for indexing workload of a sequence of sparsely
/// distributed tensors.
/// A workload will be mapped to a key value pair (determined by size_) and the
/// corresponding per-value workload inside a sequence of tensors.
/// Users have to decide how to use the per-value workload index.
/// For example:
/// - VoxelGrid:
/// Per-value workload = (N x N x N) where N is the resolution, while the
/// tensors can be tsdf: (N x N x N) float, weight (N x N x N) byte, color (N x
/// N x N x 3) byte Users have to handle mapping from 3D coordinate to
/// corresponding properties.
/// - Multi-layered Tree:
/// Similar to VoxelGrid, but now we have multiple layers so we need to traverse
/// layer by layer instead of property by property.
/// - MLP:
/// Per-value workload = num of elements
/// Users have to deal with an additional indexing array in order to feed in
/// layers one by one (Not likely to be done on device side though)

class SparseIndexer {
public:
    SparseIndexer(const SparseTensorList& sparse_tl,
                  int64_t workload_per_entry);

    /// Get \key_idx depending on size_, and
    /// \value_idx depending on workload_per_entry
    OPEN3D_HOST_DEVICE void GetSparseWorkloadIdx(
            int64_t workload_idx,
            int64_t* key_idx,
            int64_t* value_workload) const {
        *key_idx = workload_idx / workload_per_entry_;
        *value_workload = workload_idx % workload_per_entry_;
    }

    OPEN3D_HOST_DEVICE void* GetWorkloadKeyPtr(int64_t key_idx) const {
        return static_cast<void*>(
                static_cast<uint8_t*>(ptrs_[key_idx * factor_]));
    }

    OPEN3D_HOST_DEVICE void* GetWorkloadValuePtr(int64_t key_idx,
                                                 int64_t tensor_idx,
                                                 int64_t value_idx) const {
        return static_cast<void*>(
                static_cast<uint8_t*>(ptrs_[key_idx * factor_ + offset_]) +
                GetValueOffset(tensor_idx, value_idx));
    }

    OPEN3D_HOST_DEVICE int64_t GetValueOffset(int64_t tensor_idx,
                                              int64_t value_idx) const {
        return byte_offsets_[tensor_idx] +
               value_idx * dtype_byte_sizes_[tensor_idx];
    }

    OPEN3D_HOST_DEVICE int64_t NumWorkloads() const {
        return size_ * workload_per_entry_;
    }

public:
    void** ptrs_;
    int64_t size_;

    // interleaved_: factor_ = 2, value_offset_ = 1
    // non-interleaved_: factor_ = 1, value_offset_ = size_
    int64_t factor_;
    int64_t offset_;

    int64_t workload_per_entry_;

    int64_t num_tensors_;
    int64_t byte_offsets_[MAX_VALUE_TENSORS];
    int64_t dtype_byte_sizes_[MAX_VALUE_TENSORS];

    // Not used for now, reserved
    // int64_t ndims_[MAX_VALUE_TENSORS];
    // int64_t num_elems_[MAX_VALUE_TENSORS];

    // // Shape
    // int64_t shapes_[MAX_VALUE_TENSORS][MAX_VALUE_TENSOR_DIMS];
};
}  // namespace core
}  // namespace open3d
