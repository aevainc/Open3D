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
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"

namespace open3d {
namespace core {
static constexpr int64_t MAX_VALUE_DIMS = 10;

class SparseTensorList {
public:
    SparseTensorList() = default;

    SparseTensorList(size_t size,
                     const SizeVector& element_shape,
                     Dtype dtype,
                     void** ptrs,
                     bool interleaved = true,
                     const Device& device = Device("CPU:0"))
        // Interleaved:
        // k0ptr, v0ptr, k1ptr, v1ptr, ...
        // Else:
        // k0ptr, k1ptr, ..., v0ptr, v1ptr, ...
        : size_(size),
          dtype_(dtype),
          ptrs_(ptrs),
          interleaved_(interleaved),
          device_(device) {
        ndims_ = static_cast<int64_t>(element_shape.size());
        for (int64_t i = 0; i < ndims_; ++i) {
            element_shape_[i] = element_shape[i];
        }
        utility::LogInfo("[SparseTensorList] {} {} {} {}", ndims_,
                         element_shape_[0], element_shape_[1],
                         element_shape_[2]);

        // interleaved_ = true;
        // for (int64_t i = 0; i < size_; ++i) {
        //     void* key_ptr = ptrs[i * 2 + 0];
        //     utility::LogInfo("key = {}, {}, {}",
        //                      *static_cast<int64_t*>(key_ptr),
        //                      *(static_cast<int64_t*>(key_ptr) + 1),
        //                      *(static_cast<int64_t*>(key_ptr) + 2));
        //     void* value_ptr = ptrs[i * 2 + 1];

        //     for (int j = 0; j < 4096; ++j) {
        //         *(static_cast<float*>(value_ptr) + j) = 0;
        //         utility::LogInfo("value[{}] = {}", j,
        //                          *(static_cast<float*>(value_ptr) + j));
        //     }
        // }
    }

    /// The shape for each element tensor in the tensorlist.
    int64_t size_;
    Dtype dtype_;
    void** ptrs_;

    bool interleaved_;

    Device device_;

    int64_t ndims_;
    int64_t element_shape_[MAX_VALUE_DIMS];
};
}  // namespace core
}  // namespace open3d
