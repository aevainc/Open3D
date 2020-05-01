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

#include "TensorHashCPU.cuh"
#include "TensorHashCUDA.cuh"

#include <unordered_map>

namespace open3d {
std::pair<Tensor, Tensor> Unique(Tensor tensor) {
    /// TODO: sanity checks and multiple axises
    std::vector<int64_t> indices_data(tensor.GetShape()[0]);
    std::iota(indices_data.begin(), indices_data.end(), 0);
    Tensor indices(indices_data, {tensor.GetShape()[0]}, Dtype::Int64,
                   tensor.GetDevice());

    auto tensor_hash = CreateTensorHash(tensor, indices, false);
    return tensor_hash->Insert(tensor, indices);
}

std::shared_ptr<TensorHash> CreateTensorHash(Tensor coords,
                                             Tensor values,
                                             bool insert) {
    static std::unordered_map<
            open3d::Device::DeviceType,
            std::function<std::shared_ptr<TensorHash>(Tensor, Tensor, bool)>,
            open3d::utility::hash_enum_class::hash>
            map_device_type_to_tensorhash_constructor = {
                    {Device::DeviceType::CPU, _factory::CreateCPUTensorHash},
#ifdef BUILD_CUDA_MODULE
                    {Device::DeviceType::CUDA, _factory::CreateCUDATensorHash}
#endif
            };

    if (coords.GetDevice() != values.GetDevice()) {
        utility::LogError("Tensor device mismatch between coords and values.");
    }

    auto device = coords.GetDevice();
    if (map_device_type_to_tensorhash_constructor.find(device.GetType()) ==
        map_device_type_to_tensorhash_constructor.end()) {
        utility::LogError("CreateTensorHash: Unimplemented device");
    }

    auto constructor =
            map_device_type_to_tensorhash_constructor.at(device.GetType());
    return constructor(coords, values, insert);
}
}  // namespace open3d
