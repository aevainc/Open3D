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
#include "Open3D/Core/Hashmap/HashmapBase.h"
#include "Open3D/Core/Tensor.h"

namespace open3d {

class TensorHash {
public:
    /// <Value, Mask>
    virtual std::pair<Tensor, Tensor> Query(Tensor coords) = 0;
    /// <Key, Mask>
    virtual std::pair<Tensor, Tensor> Insert(Tensor coords, Tensor values) = 0;
    /// Mask
    virtual Tensor Assign(Tensor coords, Tensor values) = 0;

protected:
    std::shared_ptr<Hashmap<DefaultHash, DefaultKeyEq>> hashmap_;
    Dtype key_type_;
    Dtype value_type_;

    int64_t key_dim_;
    int64_t value_dim_;
};

class CPUTensorHash : public TensorHash {
public:
    CPUTensorHash(Tensor coords, Tensor values, bool insert = true);
    std::pair<Tensor, Tensor> Insert(Tensor coords, Tensor values);
    std::pair<Tensor, Tensor> Query(Tensor coords);
    Tensor Assign(Tensor coords, Tensor values);
};

class CUDATensorHash : public TensorHash {
public:
    CUDATensorHash(Tensor coords, Tensor values, bool insert = true);
    std::pair<Tensor, Tensor> Insert(Tensor coords, Tensor values);
    std::pair<Tensor, Tensor> Query(Tensor coords);
    Tensor Assign(Tensor coords, Tensor values);
};

/// Factory
std::shared_ptr<TensorHash> CreateTensorHash(Tensor coords,
                                             Tensor values,
                                             bool insert = true);

/// Temporarily put here to make the module self-contained (for potential branch
/// migration)
std::pair<Tensor, Tensor> Unique(Tensor tensor);

/// Hidden
namespace _factory {
std::shared_ptr<CPUTensorHash> CreateCPUTensorHash(Tensor coords,
                                                   Tensor values,
                                                   bool insert);
std::shared_ptr<CUDATensorHash> CreateCUDATensorHash(Tensor coords,
                                                     Tensor values,
                                                     bool insert);
}  // namespace _factory

}  // namespace open3d
