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
#include "open3d/core/kernel/SpecialOp.h"

namespace open3d {
namespace core {
namespace kernel {

void OPEN3D_HOST_DEVICE CUDAIntegrateKernel(void* tsdf,
                                            void* weight,
                                            const void* depth,
                                            float zc,
                                            float sdf_trunc) {
    if (depth != nullptr && weight != nullptr && zc > 0) {
        float sdf = (*static_cast<const float*>(depth) - zc);
        if (sdf >= -sdf_trunc) {
            sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
            sdf /= sdf_trunc;

            float tsdf_sum = *static_cast<float*>(tsdf);
            float weight_sum = *static_cast<float*>(weight);
            *static_cast<float*>(tsdf) =
                    (weight_sum * tsdf_sum + sdf) / (weight_sum + 1);
            *static_cast<float*>(weight) = weight_sum + 1;
        }
    }
}

void SpecialOpEWCUDA(const std::vector<Tensor>& input_tensors,
                     const std::vector<SparseTensorList>& input_sparse_tls,
                     SparseTensorList& output_sparse_tl,
                     SpecialOpCode op_code) {
    switch (op_code) {
        case SpecialOpCode::Integrate: {
            // sparse_tls: tsdf grid
            // tensors: depth, intrinsic, extrinsic
            SizeVector grid_shape = output_sparse_tl.shapes_[0];

            SparseIndexer sparse_indexer(output_sparse_tl,
                                         grid_shape.NumElements());
            NDArrayIndexer indexer3d(grid_shape,
                                     DtypeUtil::ByteSize(Dtype::Float32));
            SizeVector chw = input_tensors[0].GetShape();
            NDArrayIndexer indexer2d({chw[1], chw[2]},
                                     DtypeUtil::ByteSize(Dtype::Float32),
                                     input_tensors[0].GetDataPtr());
            Projector projector(input_tensors[1], input_tensors[2]);
            float sdf_trunc = input_tensors[3][0].Item<float>();

            CUDALauncher::LaunchIntegrateKernel(
                    sparse_indexer, indexer3d, indexer2d, projector,
                    [=] OPEN3D_HOST_DEVICE(void* tsdf, void* weight,
                                           const void* depth, float zc) {
                        CUDAIntegrateKernel(tsdf, weight, depth, zc, sdf_trunc);
                    });
            utility::LogInfo("[SpecialOpEWCPU] CUDALauncher finished");
            break;
        };
        default: { utility::LogError("Unsupported special op"); }
    }
}
}  // namespace kernel
}  // namespace core
}  // namespace open3d
