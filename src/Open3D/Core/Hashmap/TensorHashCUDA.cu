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

#include "Open3D/Core/Hashmap/HashmapCUDA.cuh"
#include "Open3D/Core/Hashmap/TensorHash.h"
#include "Open3D/Core/Tensor.h"

namespace open3d {

CUDATensorHash::CUDATensorHash(Tensor coords,
                               Tensor values,
                               bool insert /* = true */) {
    // Device check
    if (coords.GetDevice().GetType() != Device::DeviceType::CUDA ||
        values.GetDevice().GetType() != Device::DeviceType::CUDA) {
        utility::LogError("CUDATensorHash::Input tensors must be on CUDA.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("CUDATensorHash::Input tensors must be contiguous.");
    }

    // Shape check
    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() != 2) {
        utility::LogError("CUDATensorHash::Input coords shape must be (N, D).");
    }
    if (coords_shape[0] != values_shape[0]) {
        utility::LogError(
                "CUDATensorHash::Input coords and values size mismatch.");
    }

    // Store type and dim info
    key_type_ = coords.GetDtype();
    value_type_ = values.GetDtype();
    key_dim_ = coords_shape[1];
    value_dim_ = values_shape.size() == 1 ? 1 : values_shape[1];

    int64_t N = coords_shape[0];

    size_t key_size = DtypeUtil::ByteSize(key_type_) * key_dim_;
    if (key_size > MAX_KEY_BYTESIZE) {
        utility::LogError(
                "CUDATensorHash::Unsupported key size: at most {} bytes per "
                "key is "
                "supported, received {} bytes per key",
                MAX_KEY_BYTESIZE, key_size);
    }
    size_t value_size = DtypeUtil::ByteSize(value_type_) * value_dim_;

    // Create hashmap and reserve twice input size
    hashmap_ = CreateCUDAHashmap<DefaultHash, DefaultKeyEq>(
            N / 2, key_size, value_size, coords.GetDevice());

    if (insert) {
        auto iterators = MemoryManager::Malloc(sizeof(iterator_t) * N,
                                               coords.GetDevice());
        auto masks =
                MemoryManager::Malloc(sizeof(uint8_t) * N, coords.GetDevice());

        hashmap_->Insert(static_cast<void*>(coords.GetBlob()->GetDataPtr()),
                         static_cast<void*>(values.GetBlob()->GetDataPtr()),
                         static_cast<iterator_t*>(iterators),
                         static_cast<uint8_t*>(masks), N);

        MemoryManager::Free(iterators, coords.GetDevice());
        MemoryManager::Free(masks, coords.GetDevice());
    }
}

std::pair<Tensor, Tensor> CUDATensorHash::Insert(Tensor coords, Tensor values) {
    // Device check
    if (coords.GetDevice().GetType() != Device::DeviceType::CUDA) {
        utility::LogError("CUDATensorHash::Input tensors must be on CUDA.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("CUDATensorHash::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_type_ != coords.GetDtype() || value_type_ != values.GetDtype()) {
        utility::LogError("CUDATensorHash::Input key/value type mismatch.");
    }

    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() == 0 || coords_shape.size() == 0) {
        utility::LogError("CUDATensorHash::Inputs are empty tensors");
    }
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("CUDATensorHash::Input coords shape mismatch.");
    }
    auto value_dim = values_shape.size() == 1 ? 1 : values_shape[1];
    if (value_dim != value_dim_) {
        utility::LogError("CUDATensorHash::Input values shape mismatch.");
    }

    int64_t N = coords.GetShape()[0];

    // Insert
    auto iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * N, coords.GetDevice());

    Tensor output_coord_tensor(SizeVector({N, key_dim_}), key_type_,
                               coords.GetDevice());
    Tensor output_mask_tensor(SizeVector({N}), Dtype::UInt8,
                              coords.GetDevice());
    hashmap_->Insert(
            static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()),
            static_cast<uint8_t*>(values.GetBlob()->GetDataPtr()),
            static_cast<iterator_t*>(iterators),
            static_cast<uint8_t*>(output_mask_tensor.GetBlob()->GetDataPtr()),
            N);

    std::cout << output_mask_tensor.ToString() << "\n";

    hashmap_->UnpackIterators(
            static_cast<iterator_t*>(iterators),
            static_cast<uint8_t*>(output_mask_tensor.GetBlob()->GetDataPtr()),
            static_cast<void*>(output_coord_tensor.GetBlob()->GetDataPtr()),
            /* value = */ nullptr, N);

    MemoryManager::Free(iterators, coords.GetDevice());

    return std::make_pair(output_coord_tensor, output_mask_tensor);
}

std::pair<Tensor, Tensor> CUDATensorHash::Query(Tensor coords) {
    // Device check
    if (coords.GetDevice().GetType() != Device::DeviceType::CUDA) {
        utility::LogError("CUDATensorHash::Input tensors must be on CUDA.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous()) {
        utility::LogError("CUDATensorHash::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_type_ != coords.GetDtype()) {
        utility::LogError("CUDATensorHash::Input coords key type mismatch.");
    }
    auto coords_shape = coords.GetShape();
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("CUDATensorHash::Input coords shape mismatch.");
    }
    int64_t N = coords.GetShape()[0];

    // Search
    auto iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * N, coords.GetDevice());

    Tensor output_value_tensor(SizeVector({N, value_dim_}), value_type_,
                               coords.GetDevice());
    Tensor output_mask_tensor(SizeVector({N}), Dtype::UInt8,
                              coords.GetDevice());

    hashmap_->Find(
            static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()),
            static_cast<iterator_t*>(iterators),
            static_cast<uint8_t*>(output_mask_tensor.GetBlob()->GetDataPtr()),
            N);

    hashmap_->UnpackIterators(
            static_cast<iterator_t*>(iterators),
            static_cast<uint8_t*>(output_mask_tensor.GetBlob()->GetDataPtr()),
            /* coord = */ nullptr,
            static_cast<void*>(output_value_tensor.GetBlob()->GetDataPtr()), N);

    MemoryManager::Free(iterators, coords.GetDevice());

    return std::make_pair(output_value_tensor, output_mask_tensor);
}

Tensor CUDATensorHash::Assign(Tensor coords, Tensor values) {
    // Device check
    if (coords.GetDevice().GetType() != Device::DeviceType::CUDA) {
        utility::LogError("CUDATensorHash::Input tensors must be on CUDA.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("CUDATensorHash::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_type_ != coords.GetDtype() || value_type_ != values.GetDtype()) {
        utility::LogError("CUDATensorHash::Input key/value type mismatch.");
    }

    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() == 0 || coords_shape.size() == 0) {
        utility::LogError("CUDATensorHash::Inputs are empty tensors");
    }
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("CUDATensorHash::Input coords shape mismatch.");
    }
    auto value_dim = values_shape.size() == 1 ? 1 : values_shape[1];
    if (value_dim != value_dim_) {
        utility::LogError("CUDATensorHash::Input values shape mismatch.");
    }

    int64_t N = coords.GetShape()[0];

    // Search
    auto iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * N, coords.GetDevice());

    Tensor output_mask_tensor(SizeVector({N}), Dtype::UInt8,
                              coords.GetDevice());

    hashmap_->Find(
            static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()),
            static_cast<iterator_t*>(iterators),
            static_cast<uint8_t*>(output_mask_tensor.GetBlob()->GetDataPtr()),
            N);

    hashmap_->AssignIterators(
            static_cast<iterator_t*>(iterators),
            static_cast<uint8_t*>(output_mask_tensor.GetBlob()->GetDataPtr()),
            static_cast<uint8_t*>(values.GetBlob()->GetDataPtr()), N);

    MemoryManager::Free(iterators, coords.GetDevice());

    return output_mask_tensor;
}

namespace _factory {
std::shared_ptr<CUDATensorHash> CreateCUDATensorHash(Tensor coords,
                                                     Tensor values,
                                                     bool insert) {
    return std::make_shared<CUDATensorHash>(coords, values, insert);
}
}  // namespace _factory
}  // namespace open3d
