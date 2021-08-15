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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/TestMiniVec.h"

namespace open3d {
namespace tests {
namespace kernels {

#if defined(__CUDACC__)
void TestMiniVecConstructorCUDA
#else
void TestMiniVecConstructorCPU
#endif
        (core::Tensor& data) {
    // core::Tensor output = core::Tensor::Empty({3}, core::Float32, device);

    auto output_data_ptr = data.GetDataPtr<float>();

    int64_t n = 1;
    core::ParallelFor(data.GetDevice(), n,
                      [=] OPEN3D_DEVICE(int64_t workload_idx) {
                          Eigen::Vector3f A = {1.0, 1.0, 1.0};
                          Eigen::Vector3f B = {2.0, 2.0, 2.0};
                          float C = A.dot(B);

                          Eigen::Matrix3f D;
                          D << 1, 2, 3, 4, 5, 6, 7, 8, 9;
                          auto Dinv = D.inverse();
                        

                          output_data_ptr[0] = C;
                          output_data_ptr[1] = Dinv(2, 0);
                          output_data_ptr[2] = C;
                      });
}

}  // namespace kernels
}  // namespace tests
}  // namespace open3d
