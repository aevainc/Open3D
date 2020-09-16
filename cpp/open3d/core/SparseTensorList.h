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

class SparseTensorList {
public:
    SparseTensorList() = default;

    SparseTensorList(void** ptrs,
                     size_t size,
                     bool interleaved,
                     const std::vector<SizeVector>& shapes,
                     const std::vector<Dtype>& dtypes,
                     const Device& device = Device("CPU:0"))
        // Interleaved:
        // k0ptr, v0ptr, k1ptr, v1ptr, ...
        // Else:
        // k0ptr, k1ptr, ..., v0ptr, v1ptr, ...
        : ptrs_(ptrs),
          size_(size),
          interleaved_(interleaved),
          device_(device),
          shapes_(shapes),
          dtypes_(dtypes) {}

    /// The shape for each element tensor in the sparse tensor list.
    void** ptrs_;
    int64_t size_;
    bool interleaved_;

    Device device_;

    // Interpret chunk of data with various shapes and dtypes
    std::vector<SizeVector> shapes_;
    std::vector<Dtype> dtypes_;
};
}  // namespace core
}  // namespace open3d
