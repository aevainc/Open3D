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
#include "Open3D/Core/Hashmap/HashmapCPU.hpp"
#include "Open3D/Core/Hashmap/TensorHash.h"
#include "Open3D/Core/Tensor.h"

namespace open3d {

CPUTensorHash::CPUTensorHash(Tensor coords,
                             Tensor values,
                             bool insert /* = true */) {
    // Device check
    if (coords.GetDevice().GetType() != Device::DeviceType::CPU ||
        values.GetDevice().GetType() != Device::DeviceType::CPU) {
        utility::LogError("CPUTensorHash::Input tensors must be on CPU.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("CPUTensorHash::Input tensors must be contiguous.");
    }

    // Shape check
    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() != 2) {
        utility::LogError("CPUTensorHash::Input coords shape must be (N, D).");
    }
    if (coords_shape[0] != values_shape[0]) {
        utility::LogError(
                "CPUTensorHash::Input coords and values size mismatch.");
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
                "CPUTensorHash::Unsupported key size: at most {} bytes per "
                "key is "
                "supported, received {} bytes per key",
                MAX_KEY_BYTESIZE, key_size);
    }
    size_t value_size = DtypeUtil::ByteSize(value_type_) * value_dim_;

    // Create hashmap and reserve twice input size
    hashmap_ = CreateCPUHashmap<DefaultHash>(N * 2, key_size, value_size,
                                             coords.GetDevice());

    if (insert) {
        hashmap_->Insert(static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()),
                         static_cast<uint8_t*>(values.GetBlob()->GetDataPtr()),
                         N);
    }
}

/// TODO: move these iterator dispatchers to Hashmap interfaces
void AssignIteratorsIter(iterator_t* iterators,
                         uint8_t* masks,
                         uint8_t* values,
                         size_t key_size,
                         size_t value_size,
                         int tid) {
    // Valid queries
    if (masks[tid]) {
        uint8_t* src_value_ptr = values + value_size * tid;
        uint8_t* dst_value_ptr = iterators[tid].second;

        // Byte-by-byte copy, can be improved
        for (int i = 0; i < value_size; ++i) {
            dst_value_ptr[i] = src_value_ptr[i];
        }
    }
}

void DispatchKeysIter(iterator_t* iterators,
                      uint8_t* masks,
                      uint8_t* keys,
                      size_t key_size,
                      int tid) {
    // Valid queries
    if (masks[tid]) {
        uint8_t* src_key_ptr = iterators[tid].first;
        uint8_t* dst_key_ptr = keys + key_size * tid;

        // Byte-by-byte copy, can be improved
        for (int i = 0; i < key_size; ++i) {
            dst_key_ptr[i] = src_key_ptr[i];
        }
    }
}

std::pair<Tensor, Tensor> CPUTensorHash::Insert(Tensor coords, Tensor values) {
    // Device check
    if (coords.GetDevice().GetType() != Device::DeviceType::CPU) {
        utility::LogError("CPUTensorHash::Input tensors must be on CPU.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("CPUTensorHash::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_type_ != coords.GetDtype() || value_type_ != values.GetDtype()) {
        utility::LogError("CPUTensorHash::Input key/value type mismatch.");
    }

    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() == 0 || coords_shape.size() == 0) {
        utility::LogError("CPUTensorHash::Inputs are empty tensors");
    }
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("CPUTensorHash::Input coords shape mismatch.");
    }
    auto value_dim = values_shape.size() == 1 ? 1 : values_shape[1];
    if (value_dim != value_dim_) {
        utility::LogError("CPUTensorHash::Input values shape mismatch.");
    }

    int64_t N = coords.GetShape()[0];

    // Insert
    auto result = hashmap_->Insert(
            static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()),
            static_cast<uint8_t*>(values.GetBlob()->GetDataPtr()), N);

    // Decode returned iterators
    auto iterators_buf = result.first;
    auto masks_buf = result.second;

    // Assign keys
    auto ret_keys_tensor =
            Tensor(coords.GetShape(), key_type_, hashmap_->device_);

    for (int i = 0; i < N; ++i) {
        DispatchKeysIter(
                iterators_buf, masks_buf,
                static_cast<uint8_t*>(ret_keys_tensor.GetBlob()->GetDataPtr()),
                hashmap_->dsize_key_, i);
    }

    // Dispatch masks
    // Copy mask to avoid duplicate data; dummy deleter avoids double free
    // TODO: more efficient memory reuse
    auto blob = std::make_shared<Blob>(hashmap_->device_,
                                       static_cast<void*>(masks_buf),
                                       [](void* dummy) -> void {});
    auto mask_tensor =
            Tensor(SizeVector({N}), SizeVector({1}),
                   static_cast<void*>(masks_buf), Dtype::UInt8, blob);
    auto ret_mask_tensor = mask_tensor.Copy(hashmap_->device_);

    return std::make_pair(ret_keys_tensor, ret_mask_tensor);
}

void DispatchValuesIter(iterator_t* iterators,
                        uint8_t* masks,
                        uint8_t* values,
                        size_t key_size,
                        size_t value_size,
                        int tid) {
    // Valid queries
    if (masks[tid]) {
        uint8_t* src_value_ptr = iterators[tid].second;
        uint8_t* dst_value_ptr = values + value_size * tid;

        // Byte-by-byte copy, can be improved
        for (int i = 0; i < value_size; ++i) {
            dst_value_ptr[i] = src_value_ptr[i];
        }
    }
}

std::pair<Tensor, Tensor> CPUTensorHash::Query(Tensor coords) {
    // Device check
    if (coords.GetDevice().GetType() != Device::DeviceType::CPU) {
        utility::LogError("CPUTensorHash::Input tensors must be on CPU.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous()) {
        utility::LogError("CPUTensorHash::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_type_ != coords.GetDtype()) {
        utility::LogError("CPUTensorHash::Input coords key type mismatch.");
    }
    auto coords_shape = coords.GetShape();
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("CPUTensorHash::Input coords shape mismatch.");
    }
    int64_t N = coords.GetShape()[0];

    // Search
    auto result = hashmap_->Search(
            static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()), N);
    utility::LogInfo("Searched");

    // Decode returned iterators
    auto iterators_buf = result.first;
    auto masks_buf = result.second;

    // Dispatch values
    auto ret_value_tensor =
            Tensor(SizeVector({N}), value_type_, hashmap_->device_);
    utility::LogInfo("Dispatching");
    for (int i = 0; i < N; ++i) {
        DispatchValuesIter(
                iterators_buf, masks_buf,
                static_cast<uint8_t*>(ret_value_tensor.GetBlob()->GetDataPtr()),
                hashmap_->dsize_key_, hashmap_->dsize_value_, i);
    }

    // Dispatch masks
    // Copy mask to avoid duplicate data; dummy deleter avoids double free
    // TODO: more efficient memory reuse
    utility::LogInfo("Dispatching finished");
    auto blob = std::make_shared<Blob>(hashmap_->device_,
                                       static_cast<void*>(masks_buf),
                                       [](void* dummy) -> void {});
    auto mask_tensor =
            Tensor(SizeVector({N}), SizeVector({1}),
                   static_cast<void*>(masks_buf), Dtype::UInt8, blob);
    auto ret_mask_tensor = mask_tensor.Copy(hashmap_->device_);

    return std::make_pair(ret_value_tensor, ret_mask_tensor);
}

/// TODO: move these iterator dispatchers to Hashmap interfaces
void AssignValuesIter(iterator_t* iterators,
                      uint8_t* masks,
                      uint8_t* values,
                      size_t key_size,
                      size_t value_size,
                      int tid) {
    // Valid queries
    if (masks[tid]) {
        uint8_t* src_value_ptr = values + value_size * tid;
        uint8_t* dst_value_ptr = iterators[tid].second;

        // Byte-by-byte copy, can be improved
        for (int i = 0; i < value_size; ++i) {
            dst_value_ptr[i] = src_value_ptr[i];
        }
    }
}

Tensor CPUTensorHash::Assign(Tensor coords, Tensor values) {
    // Device check
    if (coords.GetDevice().GetType() != Device::DeviceType::CPU) {
        utility::LogError("CPUTensorHash::Input tensors must be on CUDA.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("CPUTensorHash::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_type_ != coords.GetDtype() || value_type_ != values.GetDtype()) {
        utility::LogError("CPUTensorHash::Input key/value type mismatch.");
    }

    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() == 0 || coords_shape.size() == 0) {
        utility::LogError("CPUTensorHash::Inputs are empty tensors");
    }
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("CPUTensorHash::Input coords shape mismatch.");
    }
    auto value_dim = values_shape.size() == 1 ? 1 : values_shape[1];
    if (value_dim != value_dim_) {
        utility::LogError("CPUTensorHash::Input values shape mismatch.");
    }

    int64_t N = coords.GetShape()[0];

    // Search
    auto result = hashmap_->Search(
            static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()), N);

    // Decode returned iterators
    auto iterators_buf = result.first;
    auto masks_buf = result.second;

    // Assign values
    for (int i = 0; i < N; ++i) {
        AssignValuesIter(iterators_buf, masks_buf,
                         static_cast<uint8_t*>(values.GetBlob()->GetDataPtr()),
                         hashmap_->dsize_key_, hashmap_->dsize_value_, i);
    }

    // Dispatch masks
    // Copy mask to avoid duplicate data; dummy deleter avoids double free
    // TODO: more efficient memory reuse
    auto blob = std::make_shared<Blob>(hashmap_->device_,
                                       static_cast<void*>(masks_buf),
                                       [](void* dummy) -> void {});
    auto mask_tensor =
            Tensor(SizeVector({N}), SizeVector({1}),
                   static_cast<void*>(masks_buf), Dtype::UInt8, blob);
    auto ret_mask_tensor = mask_tensor.Copy(hashmap_->device_);

    return ret_mask_tensor;
}

namespace _factory {
std::shared_ptr<CPUTensorHash> CreateCPUTensorHash(Tensor coords,
                                                   Tensor values,
                                                   bool insert) {
    return std::make_shared<CPUTensorHash>(coords, values, insert);
}
}  // namespace _factory
}  // namespace open3d
