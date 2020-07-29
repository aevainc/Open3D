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

#include "open3d/core/Indexer.h"
#include "open3d/core/SparseIndexer.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/core/kernel/SpecialOp.h"

namespace open3d {
namespace core {
namespace kernel {

void CPUIntegrateKernel(int64_t workload_idx) {}

void SpecialOpEWCPU(const std::vector<Tensor>& input_tensors,
                    const std::vector<SparseTensorList>& input_sparse_tls,
                    SparseTensorList& output_sparse_tl,
                    SpecialOpCode op_code) {
    switch (op_code) {
        case SpecialOpCode::Integrate: {
            // sparse_tls: tsdf grid
            // tensors: depth, intrinsic, extrinsic
            SizeVector grid_shape = output_sparse_tl.shapes_[0];
            float voxel_size = input_tensors[3][0].Item<float>();
            float sdf_trunc = input_tensors[4][0].Item<float>();

            SparseIndexer sparse_indexer(output_sparse_tl,
                                         grid_shape.NumElements());
            NDArrayIndexer indexer3d(grid_shape,
                                     DtypeUtil::ByteSize(Dtype::Float32));
            SizeVector chw = input_tensors[0].GetShape();
            NDArrayIndexer indexer2d({chw[1], chw[2]},
                                     DtypeUtil::ByteSize(Dtype::Float32),
                                     input_tensors[0].GetDataPtr());

            Projector projector(input_tensors[1], input_tensors[2], voxel_size);

            int64_t n = sparse_indexer.NumWorkloads();
            CPULauncher::LaunchGeneralKernel(n, [=](int64_t workload_idx) {
                int64_t key_idx, value_idx;
                sparse_indexer.GetSparseWorkloadIdx(workload_idx, &key_idx,
                                                    &value_idx);

                int64_t xl, yl, zl;
                indexer3d.ConvertOffsetTo3D(value_idx, &xl, &yl, &zl);

                void* key_ptr = sparse_indexer.GetWorkloadKeyPtr(key_idx);
                int64_t xg = *(static_cast<int64_t*>(key_ptr) + 0);
                int64_t yg = *(static_cast<int64_t*>(key_ptr) + 1);
                int64_t zg = *(static_cast<int64_t*>(key_ptr) + 2);

                int64_t resolution = indexer3d.GetShape(0);
                int64_t x = (xg * resolution + xl);
                int64_t y = (yg * resolution + yl);
                int64_t z = (zg * resolution + zl);

                float xc, yc, zc, u, v;
                projector.Transform(static_cast<float>(x),
                                    static_cast<float>(y),
                                    static_cast<float>(z), &xc, &yc, &zc);
                projector.Project(xc, yc, zc, &u, &v);

                if (!indexer2d.InBoundary2D(u, v)) {
                    return;
                }

                int64_t offset;
                indexer2d.Convert2DToOffset(static_cast<int64_t>(u),
                                            static_cast<int64_t>(v), &offset);
                float depth = *static_cast<const float*>(
                        indexer2d.GetPtrFromOffset(offset));

                float sdf = depth - zc;
                if (depth <= 0 || zc <= 0 || sdf < -sdf_trunc) {
                    return;
                }
                sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
                sdf /= sdf_trunc;

                void* tsdf_ptr = sparse_indexer.GetWorkloadValuePtr(key_idx, 0,
                                                                    value_idx);
                void* weight_ptr = sparse_indexer.GetWorkloadValuePtr(
                        key_idx, 1, value_idx);

                float tsdf_sum = *static_cast<float*>(tsdf_ptr);
                float weight_sum = *static_cast<float*>(weight_ptr);
                *static_cast<float*>(tsdf_ptr) =
                        (weight_sum * tsdf_sum + sdf) / (weight_sum + 1);
                *static_cast<float*>(weight_ptr) = weight_sum + 1;
            });
            utility::LogInfo("[SpecialOpEWCPU] CPULauncher finished");
            break;
        };

        case SpecialOpCode::ExtractSurface: {
            // input_sparse_tls: tsdf grid
            // output_sparse_tl: surface grid
            // tensors: voxel_size, sdf_trunc
            SizeVector grid_shape = output_sparse_tl.shapes_[0];
            float voxel_size = input_tensors[0][0].Item<float>();

            NDArrayIndexer indexer3d(grid_shape,
                                     DtypeUtil::ByteSize(Dtype::Int32));

            SparseIndexer tsdf_indexer(input_sparse_tls[0],
                                       grid_shape.NumElements());
            SparseIndexer surf_indexer(output_sparse_tl,
                                       grid_shape.NumElements());
            int* count = static_cast<int*>(MemoryManager::Malloc(
                    sizeof(int), output_sparse_tl.device_));
            *count = 0;

            int64_t n = tsdf_indexer.NumWorkloads();
            CPULauncher::LaunchGeneralKernel(n, [=](int64_t workload_idx) {
                int64_t key_idx, value_idx;
                tsdf_indexer.GetSparseWorkloadIdx(workload_idx, &key_idx,
                                                  &value_idx);

                int64_t xl, yl, zl;
                indexer3d.ConvertOffsetTo3D(value_idx, &xl, &yl, &zl);

                int64_t resolution = indexer3d.GetShape(0);

                /// We don't handle boundaries for now -- come back later!
                if (xl == resolution - 1 || yl == resolution - 1 ||
                    zl == resolution - 1) {
                    return;
                }

                int64_t offset_x, offset_y, offset_z;
                indexer3d.Convert3DToOffset(xl + 1, yl, zl, &offset_x);
                indexer3d.Convert3DToOffset(xl, yl + 1, zl, &offset_y);
                indexer3d.Convert3DToOffset(xl, yl, zl + 1, &offset_z);

                /// Query values from tsdf grid
                float tsdf_o =
                        *static_cast<float*>(tsdf_indexer.GetWorkloadValuePtr(
                                key_idx, 0, value_idx));
                float weight_o =
                        *static_cast<float*>(tsdf_indexer.GetWorkloadValuePtr(
                                key_idx, 1, value_idx));
                if (weight_o == 0) {
                    return;
                }

                void* key_ptr = tsdf_indexer.GetWorkloadKeyPtr(key_idx);
                int64_t xg = *(static_cast<int64_t*>(key_ptr) + 0);
                int64_t yg = *(static_cast<int64_t*>(key_ptr) + 1);
                int64_t zg = *(static_cast<int64_t*>(key_ptr) + 2);

                float x = (xg * resolution + xl) * voxel_size;
                float y = (yg * resolution + yl) * voxel_size;
                float z = (zg * resolution + zl) * voxel_size;

                float tsdf_x = *static_cast<float*>(
                        tsdf_indexer.GetWorkloadValuePtr(key_idx, 0, offset_x));
                float weight_x = *static_cast<float*>(
                        tsdf_indexer.GetWorkloadValuePtr(key_idx, 1, offset_x));

                float tsdf_y = *static_cast<float*>(
                        tsdf_indexer.GetWorkloadValuePtr(key_idx, 0, offset_y));
                float weight_y = *static_cast<float*>(
                        tsdf_indexer.GetWorkloadValuePtr(key_idx, 1, offset_y));

                float tsdf_z = *static_cast<float*>(
                        tsdf_indexer.GetWorkloadValuePtr(key_idx, 0, offset_z));
                float weight_z = *static_cast<float*>(
                        tsdf_indexer.GetWorkloadValuePtr(key_idx, 1, offset_z));

                int* vertex_ind_x = static_cast<int*>(
                        surf_indexer.GetWorkloadValuePtr(key_idx, 0, offset_x));
                int* vertex_ind_y = static_cast<int*>(
                        surf_indexer.GetWorkloadValuePtr(key_idx, 1, offset_y));
                int* vertex_ind_z = static_cast<int*>(
                        surf_indexer.GetWorkloadValuePtr(key_idx, 2, offset_z));

                if (weight_x > 0 && tsdf_x * tsdf_o < 0) {
                    *vertex_ind_x = int(x);
                    printf("x, idx = %d\n", *count);
                    *count = *count + 1;
                }
                if (weight_y > 0 && tsdf_y * tsdf_o < 0) {
                    *vertex_ind_y = int(y);
                    printf("y, idx = %d\n", *count);
                    *count = *count + 1;
                }
                if (weight_z > 0 && tsdf_z * tsdf_o < 0) {
                    *vertex_ind_z = int(z);
                    printf("z, idx = %d\n", *count);
                    *count = *count + 1;
                }
            });

            break;
        };

        default: { utility::LogError("Unsupported special op"); }
    }
    utility::LogInfo("[SpecialOpEWCPU] Exiting SpecialOpEWCPU");
}
}  // namespace kernel
}  // namespace core
}  // namespace open3d
