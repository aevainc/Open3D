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

#include "open3d/core/SparseIndexer.h"

#include <bits/stdint-intn.h>

namespace open3d {
namespace core {
SparseIndexer::SparseIndexer(const SparseTensorList& sparse_tl,
                             int64_t workload_per_entry) {
    ptrs_ = sparse_tl.ptrs_;
    size_ = sparse_tl.size_;

    // interleaved_: factor_ = 2, value's offset_ = 1
    // non-interleaved_: factor_ = 1, value's offset_ = size_
    if (sparse_tl.interleaved_) {
        factor_ = 2;
        offset_ = 1;
    } else {
        factor_ = 1;
        offset_ = size_;
    }

    workload_per_entry_ = workload_per_entry;
    num_tensors_ = static_cast<int64_t>(sparse_tl.shapes_.size());

    for (int64_t i = 0; i < num_tensors_; ++i) {
        dtype_byte_sizes_[i] = sparse_tl.dtypes_[i].ByteSize();
    }

    byte_offsets_[0] = 0;
    for (size_t i = 1; i < sparse_tl.shapes_.size(); ++i) {
        byte_offsets_[i] =
                byte_offsets_[i - 1] + sparse_tl.shapes_[i - 1].NumElements() *
                                               dtype_byte_sizes_[i - 1];
    }
}

}  // namespace core
}  // namespace open3d
