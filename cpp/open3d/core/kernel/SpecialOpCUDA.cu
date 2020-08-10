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

void SpecialOpEWCUDA(const std::vector<Tensor>& input_tensors,
                     const std::vector<SparseTensorList>& input_sparse_tls,
                     Tensor& output_tensor,
                     SparseTensorList& output_sparse_tl,
                     SpecialOpCode op_code) {
    utility::LogInfo("SpecialOpEWCUDA");
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

            CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_HOST_DEVICE(
                                                         int64_t workload_idx) {
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
            utility::LogInfo("[SpecialOpEWCUDA] CUDALauncher finished");
            break;
        };

        case SpecialOpCode::ExtractSurface: {
            utility::LogInfo("ExtractSurface");
            // input_sparse_tls: tsdf grid
            // output_sparse_tl: surface grid
            // tensors: voxel_size, sdf_trunc
            SizeVector grid_shape = output_sparse_tl.shapes_[0];
            float voxel_size = input_tensors[0][0].Item<float>();

            // res x res x res
            NDArrayIndexer indexer3d(grid_shape,
                                     DtypeUtil::ByteSize(Dtype::Int32));
            // 27 x n
            NDArrayIndexer indexer2d(input_tensors[1].GetShape(),
                                     DtypeUtil::ByteSize(Dtype::Bool),
                                     input_tensors[1].GetDataPtr());
            // n => res x res x res
            SparseIndexer tsdf_indexer(input_sparse_tls[0],
                                       grid_shape.NumElements());
            // 27 x n => res x res x res
            SparseIndexer tsdf_nb_indexer(input_sparse_tls[1],
                                          grid_shape.NumElements());

            SparseIndexer surf_indexer(output_sparse_tl,
                                       grid_shape.NumElements());
            int64_t n = tsdf_indexer.NumWorkloads();
            int64_t m = input_tensors[1].GetShape()[1];

            Device device = output_sparse_tl.device_;
            Tensor count(std::vector<int>{0}, {1}, Dtype::Int32, device);
            int* count_ptr = static_cast<int*>(count.GetDataPtr());

            // TODO: adaptive
            int total_count = 1400000;
            Tensor vertices_x({total_count}, Dtype::Float32, device);
            Tensor vertices_y({total_count}, Dtype::Float32, device);
            Tensor vertices_z({total_count}, Dtype::Float32, device);
            float* vertices_x_ptr =
                    static_cast<float*>(vertices_x.GetDataPtr());
            float* vertices_y_ptr =
                    static_cast<float*>(vertices_y.GetDataPtr());
            float* vertices_z_ptr =
                    static_cast<float*>(vertices_z.GetDataPtr());

            CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                         int64_t workload_idx) {
                int64_t key_idx, value_idx;
                tsdf_indexer.GetSparseWorkloadIdx(workload_idx, &key_idx,
                                                  &value_idx);
                int64_t xl, yl, zl;
                indexer3d.ConvertOffsetTo3D(value_idx, &xl, &yl, &zl);

                int64_t resolution = indexer3d.GetShape(0);

                void* key_ptr = tsdf_indexer.GetWorkloadKeyPtr(key_idx);
                int64_t xg = *(static_cast<int64_t*>(key_ptr) + 0);
                int64_t yg = *(static_cast<int64_t*>(key_ptr) + 1);
                int64_t zg = *(static_cast<int64_t*>(key_ptr) + 2);

                float x = (xg * resolution + xl) * voxel_size;
                float y = (yg * resolution + yl) * voxel_size;
                float z = (zg * resolution + zl) * voxel_size;

                // Check if boundary neighbors exist
                bool flag_x = false, flag_y = false, flag_z = false;
                if (xl == resolution - 1) {
                    flag_x = true;
                    int64_t offset;
                    indexer2d.Convert2DToOffset(key_idx, 14, &offset);
                    void* ptr = indexer2d.GetPtrFromOffset(offset);
                    bool nb_valid = *static_cast<bool*>(ptr);

                    if (!nb_valid) {
                        return;
                    }
                }
                if (yl == resolution - 1) {
                    flag_y = true;
                    int64_t offset;
                    indexer2d.Convert2DToOffset(key_idx, 16, &offset);
                    void* ptr = indexer2d.GetPtrFromOffset(offset);
                    bool nb_valid = *static_cast<bool*>(ptr);

                    if (!nb_valid) {
                        return;
                    }
                }
                if (zl == resolution - 1) {
                    flag_z = true;
                    int64_t offset;
                    indexer2d.Convert2DToOffset(key_idx, 22, &offset);
                    void* ptr = indexer2d.GetPtrFromOffset(offset);
                    bool nb_valid = *static_cast<bool*>(ptr);

                    if (!nb_valid) {
                        return;
                    }
                }

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

                int64_t offset_x, offset_y, offset_z;
                indexer3d.Convert3DToOffset(xl + 1, yl, zl, &offset_x);
                indexer3d.Convert3DToOffset(xl, yl + 1, zl, &offset_y);
                indexer3d.Convert3DToOffset(xl, yl, zl + 1, &offset_z);

                float tsdf_x;
                float weight_x = 0.0;
                if (!flag_x) {
                    tsdf_x = *static_cast<float*>(
                            tsdf_indexer.GetWorkloadValuePtr(key_idx, 0,
                                                             offset_x));
                    weight_x = *static_cast<float*>(
                            tsdf_indexer.GetWorkloadValuePtr(key_idx, 1,
                                                             offset_x));
                } else {
                    // int64_t nb_offset_x;
                    // indexer3d.Convert3DToOffset(0, yl, zl, &nb_offset_x);
                    // void* ptr = tsdf_nb_indexer.GetWorkloadValuePtr(
                    //         2 * m + key_idx, 1, 0);
                    // printf("0, %p / %ld, %ld\n", nb_offset_x, ptr, m, n);
                    // weight_x = *static_cast<float*>(
                    //         tsdf_nb_indexer.GetWorkloadValuePtr(
                    //                 m * 14 + key_idx, 1, 0));
                }

                float tsdf_y;
                float weight_y = 0.0;
                if (!flag_y) {
                    tsdf_y = *static_cast<float*>(
                            tsdf_indexer.GetWorkloadValuePtr(key_idx, 0,
                                                             offset_y));
                    weight_y = *static_cast<float*>(
                            tsdf_indexer.GetWorkloadValuePtr(key_idx, 1,
                                                             offset_y));
                } else {
                    int64_t nb_offset_y;
                    indexer3d.Convert3DToOffset(xl, 0, zl, &nb_offset_y);
                    tsdf_y = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    16 * n + key_idx, 0, nb_offset_y));
                    weight_y = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    16 * n + key_idx, 1, nb_offset_y));
                }

                float tsdf_z;
                float weight_z = 0.0;
                if (!flag_z) {
                    tsdf_z = *static_cast<float*>(
                            tsdf_indexer.GetWorkloadValuePtr(key_idx, 0,
                                                             offset_z));
                    weight_z = *static_cast<float*>(
                            tsdf_indexer.GetWorkloadValuePtr(key_idx, 1,
                                                             offset_z));
                } else {
                    int64_t nb_offset_z;
                    indexer3d.Convert3DToOffset(xl, yl, 0, &nb_offset_z);
                    tsdf_z = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    22 * n + key_idx, 0, nb_offset_z));
                    weight_z = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    22 * n + key_idx, 1, nb_offset_z));
                }

                // int* vertex_ind_x = static_cast<int*>(
                //         surf_indexer.GetWorkloadValuePtr(key_idx, 0,
                //         offset_x));
                // int* vertex_ind_y = static_cast<int*>(
                //         surf_indexer.GetWorkloadValuePtr(key_idx, 1,
                //         offset_y));
                // int* vertex_ind_z = static_cast<int*>(
                //         surf_indexer.GetWorkloadValuePtr(key_idx, 2,
                //         offset_z));

                if (weight_x > 0 && tsdf_x * tsdf_o < 0) {
                    float ratio = tsdf_x / (tsdf_x - tsdf_o);
                    int idx = atomicAdd(count_ptr, 1);
                    // *vertex_ind_x = idx;
                    vertices_x_ptr[idx] = x + ratio * voxel_size;
                    vertices_y_ptr[idx] = y;
                    vertices_z_ptr[idx] = z;
                }
                if (weight_y > 0 && tsdf_y * tsdf_o < 0) {
                    float ratio = tsdf_y / (tsdf_y - tsdf_o);
                    int idx = atomicAdd(count_ptr, 1);
                    // *vertex_ind_y = idx;
                    vertices_x_ptr[idx] = x;
                    vertices_y_ptr[idx] = y + ratio * voxel_size;
                    vertices_z_ptr[idx] = z;
                }
                if (weight_z > 0 && tsdf_z * tsdf_o < 0) {
                    float ratio = tsdf_z / (tsdf_z - tsdf_o);
                    int idx = atomicAdd(count_ptr, 1);
                    // *vertex_ind_z = idx;
                    vertices_x_ptr[idx] = x;
                    vertices_y_ptr[idx] = y;
                    vertices_z_ptr[idx] = z + ratio * voxel_size;
                }
            });

            int actual_count = count[0].Item<int>();

            output_tensor = Tensor({3, actual_count}, Dtype::Float32, device);
            output_tensor[0].Slice(0, 0, actual_count) =
                    vertices_x.Slice(0, 0, actual_count);
            output_tensor[1].Slice(0, 0, actual_count) =
                    vertices_y.Slice(0, 0, actual_count);
            output_tensor[2].Slice(0, 0, actual_count) =
                    vertices_z.Slice(0, 0, actual_count);
            break;
        };

        case SpecialOpCode::Check: {
            utility::LogInfo("Check");
            // 27 x n
            SizeVector grid_shape = input_sparse_tls[0].shapes_[0];

            NDArrayIndexer indexer2d(input_tensors[0].GetShape(),
                                     DtypeUtil::ByteSize(Dtype::Bool),
                                     input_tensors[0].GetDataPtr());
            // n => res x res x res
            SparseIndexer tsdf_indexer(input_sparse_tls[0],
                                       grid_shape.NumElements());
            // 27 x n => res x res x res
            SparseIndexer tsdf_nb_indexer(input_sparse_tls[1],
                                          grid_shape.NumElements());

            int64_t n = tsdf_indexer.NumWorkloads();
            int64_t m = input_tensors[0].GetShape()[1];

            CUDALauncher::LaunchGeneralKernel(
                    n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                        int64_t key_idx, value_idx;
                        tsdf_indexer.GetSparseWorkloadIdx(workload_idx,
                                                          &key_idx, &value_idx);
                        // int64_t key_idx = workload_idx;
                        void* ptr0 =
                                tsdf_indexer.GetWorkloadValuePtr(key_idx, 0, 0);
                        void* ptr1 = tsdf_nb_indexer.GetWorkloadValuePtr(
                                key_idx + 13 * m, 0, 0);
                        // printf("%p %p\n", ptr0, ptr1);
                    });
            break;
        };

        default: { utility::LogError("Unsupported special op"); }
    }
}
}  // namespace kernel
}  // namespace core
}  // namespace open3d
