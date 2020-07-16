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
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"

namespace open3d {
namespace core {

static constexpr int64_t MAX_VALUE_DIMS = 10;
static constexpr int64_t MAX_MEMBERS = 10;

// TODO: more general desc
class DataDescriptor {
public:
    DataDescriptor() = default;

    DataDescriptor(const SizeVector& element_shape,
                   const std::vector<Dtype>& dtypes) {
        ndims_ = static_cast<int64_t>(element_shape.size());
        for (int64_t i = 0; i < ndims_; ++i) {
            element_shape_[i] = element_shape[i];
        }

        int64_t stride = 1;
        for (int64_t i = ndims_ - 1; i >= 0; --i) {
            element_strides_[i] = stride;
            // Handles 0-sized dimensions
            stride =
                    element_shape_[i] > 1 ? stride * element_shape_[i] : stride;
        }
        num_elems_ = stride;

        num_members_ = static_cast<int64_t>(dtypes.size());
        for (int64_t i = 0; i < num_members_; ++i) {
            byte_size_[i] = DtypeUtil::ByteSize(dtypes[i]);
        }

        byte_offsets_[0] = 0;
        for (int64_t i = 1; i < num_members_; ++i) {
            byte_offsets_[i] = byte_size_[i - 1] * num_elems_;
        }
    }

    OPEN3D_HOST_DEVICE int64_t GetStride(int64_t i) const {
        return element_strides_[i < 0 ? ndims_ + i : i];
    }

    OPEN3D_HOST_DEVICE int64_t GetOffset(int64_t tensor_idx, int64_t i) const {
        return byte_offsets_[tensor_idx] + i * byte_size_[tensor_idx];
    }

    // Shared
    int64_t ndims_;
    int64_t num_elems_;
    int64_t element_shape_[MAX_VALUE_DIMS];
    int64_t element_strides_[MAX_VALUE_DIMS];

    // By default 1
    int64_t num_members_;
    int64_t byte_offsets_[MAX_MEMBERS];
    int64_t byte_size_[MAX_MEMBERS];
};

class SparseTensorList {
public:
    SparseTensorList() = default;

    SparseTensorList(size_t size,
                     void** ptrs,
                     bool interleaved,
                     const SizeVector& element_shape,
                     const std::vector<Dtype>& dtypes,
                     const Device& device = Device("CPU:0"))
        // Interleaved:
        // k0ptr, v0ptr, k1ptr, v1ptr, ...
        // Else:
        // k0ptr, k1ptr, ..., v0ptr, v1ptr, ...
        : size_(size),

          ptrs_(ptrs),
          interleaved_(interleaved),
          device_(device),
          data_desc_(element_shape, dtypes) {}

    /// The shape for each element tensor in the tensorlist.
    int64_t size_;
    Dtype dtype_;
    void** ptrs_;

    bool interleaved_;

    Device device_;

    DataDescriptor data_desc_;
};
}  // namespace core
}  // namespace open3d
