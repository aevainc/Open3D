// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/utility/MiniVec.h"

#include "open3d/core/Tensor.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"
#include "open3d/utility/TestMiniVec.h"

namespace open3d {
namespace tests {

class MiniVecPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(MiniVec,
                         MiniVecPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(MiniVecPermuteDevices, Constructor) {
    core::Device device = GetParam();

    core::Tensor data = core::Tensor::Ones({3}, core::Float32, device);

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        tests::kernels::TestMiniVecConstructorCPU(data);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(tests::kernels::TestMiniVecConstructorCUDA, data);
    } else {
        utility::LogError("Unimplemented device");
    }

    // core::Tensor output;
    std::cout << data.ToString();
}

}  // namespace tests
}  // namespace open3d
