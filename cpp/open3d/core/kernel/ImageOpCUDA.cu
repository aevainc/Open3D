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

#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/core/kernel/ImageOp.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename scalar_t>
void OPEN3D_HOST_DEVICE CUDAUnprojectKernel(int64_t x,
                                            int64_t y,
                                            const void* src,
                                            void* dst0,
                                            void* dst1,
                                            void* dst2,
                                            float inv_fx,
                                            float inv_fy,
                                            float cx,
                                            float cy) {
    scalar_t d = static_cast<scalar_t>(*static_cast<const scalar_t*>(src));
    // printf("%ld, %ld\n", x, y);
    *static_cast<scalar_t*>(dst0) =
            static_cast<scalar_t>(d * (x - cx) * inv_fx);
    *static_cast<scalar_t*>(dst1) =
            static_cast<scalar_t>(d * (y - cy) * inv_fy);
    *static_cast<scalar_t*>(dst2) = d;
}

void ImageUnaryEWCUDA(const Tensor& src,
                      Tensor& dst,
                      const Tensor& intrinsic,
                      ImageOpCode op_code) {
    float fx = intrinsic[0][0].Item<float>();
    float fy = intrinsic[1][1].Item<float>();
    float cx = intrinsic[0][2].Item<float>();
    float cy = intrinsic[1][2].Item<float>();

    switch (op_code) {
        case ImageOpCode::Unproject: {
            Indexer indexer({src}, {dst[0], dst[1], dst[2]},
                            DtypePolicy::ALL_SAME);
            DISPATCH_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
                CUDALauncher::LaunchImageUnaryKernel(
                        indexer, [=] OPEN3D_HOST_DEVICE(
                                         int64_t x, int64_t y, const void* src,
                                         void* dst0, void* dst1, void* dst2) {
                            CUDAUnprojectKernel<scalar_t>(x, y, src, dst0, dst1,
                                                          dst2, 1.0 / fx,
                                                          1.0 / fy, cx, cy);
                        });
            });
            break;
        }
        default: {
            utility::LogError("Unsupported image op");
        }
    }
}
}  // namespace kernel
}  // namespace core
}  // namespace open3d
