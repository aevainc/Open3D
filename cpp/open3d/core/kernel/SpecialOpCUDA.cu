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
#include "open3d/tgeometry/MarchingCubesConst.h"

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
                                     Dtype::Float32.ByteSize());
            SizeVector chw = input_tensors[0].GetShape();
            NDArrayIndexer indexer2d({chw[1], chw[2]},
                                     Dtype::Float32.ByteSize(),
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
            utility::LogInfo("Indexer");
            NDArrayIndexer indexer3d(grid_shape,
                                     Dtype::Int32.ByteSize());
            // 27 x n
            SizeVector nshape = input_tensors[1].GetShape();
            NDArrayIndexer indexer2d(input_tensors[1].GetShape(),
                                     Dtype::Bool.ByteSize(),
                                     input_tensors[1].GetDataPtr());
            // n => res x res x res
            SparseIndexer tsdf_indexer(input_sparse_tls[0],
                                       grid_shape.NumElements());
            // 27 x n => res x res x res
            SparseIndexer tsdf_nb_indexer(input_sparse_tls[1],
                                          grid_shape.NumElements());

            utility::LogInfo("Surf indexer");
            SparseIndexer surf_indexer(output_sparse_tl,
                                       grid_shape.NumElements());
            int64_t n = tsdf_indexer.NumWorkloads();
            utility::LogInfo("n = {}", n);
            int64_t m = input_tensors[1].GetShape()[1];

            Device device = output_sparse_tl.device_;
            Tensor count(std::vector<int>{0}, {1}, Dtype::Int32, device);
            int* count_ptr = static_cast<int*>(count.GetDataPtr());

            // TODO: adaptive
            utility::LogInfo("Count");
            int total_count = 18000000;
            Tensor vertices_x({total_count}, Dtype::Float32, device);
            Tensor vertices_y({total_count}, Dtype::Float32, device);
            Tensor vertices_z({total_count}, Dtype::Float32, device);
            float* vertices_x_ptr =
                    static_cast<float*>(vertices_x.GetDataPtr());
            float* vertices_y_ptr =
                    static_cast<float*>(vertices_y.GetDataPtr());
            float* vertices_z_ptr =
                    static_cast<float*>(vertices_z.GetDataPtr());

            utility::LogInfo("Launch");
            CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                         int64_t workload_idx) {
                int64_t key_idx, value_idx;
                tsdf_indexer.GetSparseWorkloadIdx(workload_idx, &key_idx,
                                                  &value_idx);

                int64_t resolution = indexer3d.GetShape(0);

                float tsdf_o =
                        *static_cast<float*>(tsdf_indexer.GetWorkloadValuePtr(
                                key_idx, 0, value_idx));
                float weight_o =
                        *static_cast<float*>(tsdf_indexer.GetWorkloadValuePtr(
                                key_idx, 1, value_idx));
                if (weight_o == 0) return;

                int64_t xl, yl, zl;
                indexer3d.ConvertOffsetTo3D(value_idx, &xl, &yl, &zl);
                for (int i = 0; i < 3; ++i) {
                    int64_t xl_i = xl + int(i == 0);
                    int64_t yl_i = yl + int(i == 1);
                    int64_t zl_i = zl + int(i == 2);
                    // printf("%d, (%ld %ld %ld) => (%ld %ld %ld)\n", i, xl, yl,
                    //        zl, xl_i, yl_i, zl_i);

                    int dx = xl_i >= resolution ? 1 : 0;
                    int dy = yl_i >= resolution ? 1 : 0;
                    int dz = zl_i >= resolution ? 1 : 0;

                    int nb_idx = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;

                    int64_t nb_mask_offset;
                    indexer2d.Convert2DToOffset(key_idx, nb_idx,
                                                &nb_mask_offset);
                    bool nb_valid = *static_cast<bool*>(
                            indexer2d.GetPtrFromOffset(nb_mask_offset));
                    if (!nb_valid) continue;

                    int64_t nb_value_idx;
                    indexer3d.Convert3DToOffset(
                            xl_i - dx * resolution, yl_i - dy * resolution,
                            zl_i - dz * resolution, &nb_value_idx);
                    float tsdf_i = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    nb_idx * m + key_idx, 0, nb_value_idx));
                    float weight_i = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    nb_idx * m + key_idx, 1, nb_value_idx));

                    if (weight_i > 0 && tsdf_i * tsdf_o < 0) {
                        float ratio = tsdf_i / (tsdf_i - tsdf_o);

                        int* vertex_ind = static_cast<int*>(
                                surf_indexer.GetWorkloadValuePtr(key_idx, i,
                                                                 value_idx));

                        int idx = atomicAdd(count_ptr, 1);
                        *vertex_ind = idx;

                        void* key_ptr = tsdf_indexer.GetWorkloadKeyPtr(key_idx);
                        int64_t xg = *(static_cast<int64_t*>(key_ptr) + 0);
                        int64_t yg = *(static_cast<int64_t*>(key_ptr) + 1);
                        int64_t zg = *(static_cast<int64_t*>(key_ptr) + 2);

                        vertices_x_ptr[idx] =
                                voxel_size * (xg * resolution + xl +
                                              (1 - ratio) * int(i == 0));
                        vertices_y_ptr[idx] =
                                voxel_size * (yg * resolution + yl +
                                              (1 - ratio) * int(i == 1));
                        vertices_z_ptr[idx] =
                                voxel_size * (zg * resolution + zl +
                                              (1 - ratio) * int(i == 2));
                    }
                }
            });

            utility::LogInfo("Final");
            int actual_count = count[0].Item<int>();
            std::cout << actual_count << "\n";

            output_tensor = Tensor({3, actual_count}, Dtype::Float32, device);
            output_tensor[0].Slice(0, 0, actual_count) =
                    vertices_x.Slice(0, 0, actual_count);
            output_tensor[1].Slice(0, 0, actual_count) =
                    vertices_y.Slice(0, 0, actual_count);
            output_tensor[2].Slice(0, 0, actual_count) =
                    vertices_z.Slice(0, 0, actual_count);
            utility::LogInfo("end");
            break;
        };

        case SpecialOpCode::MarchingCubesPass0: {
            utility::LogInfo("MC Pass0");
            // input_sparse_tls: tsdf grid
            // output_sparse_tl: surface grid
            // tensors: voxel_size, sdf_trunc
            SizeVector grid_shape = output_sparse_tl.shapes_[0];

            // res x res x res
            NDArrayIndexer indexer3d(grid_shape,
                                     Dtype::Int32.ByteSize());
            // 27 x n
            NDArrayIndexer indexer2d(input_tensors[1].GetShape(),
                                     Dtype::Bool.ByteSize(),
                                     input_tensors[1].GetDataPtr());

            // n => res x res x res x 2
            SparseIndexer tsdf_indexer(input_sparse_tls[0],
                                       grid_shape.NumElements());

            // 27 x n => res x res x res x 2
            SparseIndexer tsdf_nb_indexer(input_sparse_tls[1],
                                          grid_shape.NumElements());

            // 27 x n => res x res x res x 4 [3 x vtx_idx, table_idx]
            SparseIndexer surf_nb_indexer(output_sparse_tl,
                                          grid_shape.NumElements());

            int64_t n = tsdf_indexer.NumWorkloads();
            int64_t m = input_tensors[1].GetShape()[1];

            Device device = output_sparse_tl.device_;

            output_tensor =
                    Tensor(std::vector<int>{0}, {1}, Dtype::Int32, device);
            int* tri_count_ptr = static_cast<int*>(output_tensor.GetDataPtr());

            // Pass 0: allocate points and
            CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                         int64_t workload_idx) {
                int64_t key_idx, value_idx;
                tsdf_indexer.GetSparseWorkloadIdx(workload_idx, &key_idx,
                                                  &value_idx);

                int64_t xl, yl, zl;
                indexer3d.ConvertOffsetTo3D(value_idx, &xl, &yl, &zl);

                int64_t resolution = indexer3d.GetShape(0);

                // Enumerate 8 neighbor corners (including itself)
                int table_index = 0;
                int* table_index_ptr =
                        static_cast<int*>(surf_nb_indexer.GetWorkloadValuePtr(
                                13 * m + key_idx, 3, value_idx));
                *table_index_ptr = 0;
                for (int i = 0; i < 8; ++i) {
                    int64_t xl_i = xl + vtx_shifts[i][0];
                    int64_t yl_i = yl + vtx_shifts[i][1];
                    int64_t zl_i = zl + vtx_shifts[i][2];

                    int dx = xl_i / resolution;
                    int dy = yl_i / resolution;
                    int dz = zl_i / resolution;

                    int nb_idx = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;

                    int64_t nb_mask_offset;
                    indexer2d.Convert2DToOffset(key_idx, nb_idx,
                                                &nb_mask_offset);
                    bool nb_valid = *static_cast<bool*>(
                            indexer2d.GetPtrFromOffset(nb_mask_offset));
                    if (!nb_valid) return;

                    int64_t nb_value_idx;
                    indexer3d.Convert3DToOffset(
                            xl_i - dx * resolution, yl_i - dy * resolution,
                            zl_i - dz * resolution, &nb_value_idx);
                    float weight_i = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    nb_idx * m + key_idx, 1, nb_value_idx));
                    if (weight_i == 0) return;

                    float tsdf_i = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    nb_idx * m + key_idx, 0, nb_value_idx));

                    table_index |= ((tsdf_i < 0) ? (1 << i) : 0);
                }
                *table_index_ptr = table_index;

                if (table_index == 0 || table_index == 255) return;
                atomicAdd(tri_count_ptr, tri_count[table_index]);

                // Enumerate 12 edges
                int edges_w_vertices = edge_table[table_index];
                for (int i = 0; i < 12; ++i) {
                    if (edges_w_vertices & (1 << i)) {
                        int64_t xl_i = xl + edge_shifts[i][0];
                        int64_t yl_i = yl + edge_shifts[i][1];
                        int64_t zl_i = zl + edge_shifts[i][2];
                        int edge_i = edge_shifts[i][3];

                        int dx = xl_i / resolution;
                        int dy = yl_i / resolution;
                        int dz = zl_i / resolution;

                        int nb_idx = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;

                        // no need to check nb_valid now
                        int64_t nb_value_idx;
                        indexer3d.Convert3DToOffset(
                                xl_i - dx * resolution, yl_i - dy * resolution,
                                zl_i - dz * resolution, &nb_value_idx);
                        int* vertex_idx = static_cast<int*>(
                                surf_nb_indexer.GetWorkloadValuePtr(
                                        nb_idx * m + key_idx, edge_i,
                                        nb_value_idx));

                        // Non-atomic write, but we are safe
                        *vertex_idx = -1;
                    }
                }
            });

            break;
        }

        case SpecialOpCode::MarchingCubesPass1: {
            utility::LogInfo("MC Pass1");
            // input_sparse_tls: tsdf grid
            // output_sparse_tl: surface grid
            // tensors: voxel_size, sdf_trunc
            SizeVector grid_shape = output_sparse_tl.shapes_[0];
            float voxel_size = input_tensors[0][0].Item<float>();
            int triangle_count = input_tensors[1][0].Item<int>();
            std::cout << triangle_count << "\n";

            // std::cout << input_tensors[0].ToString() << "\n";
            // std::cout << input_tensors[1].ToString() << "\n";
            // std::cout << input_tensors[2].ToString() << "\n";
            // return;

            // res x res x res
            NDArrayIndexer indexer3d(grid_shape,
                                     Dtype::Int32.ByteSize());
            // 27 x n
            NDArrayIndexer indexer2d(input_tensors[2].GetShape(),
                                     Dtype::Bool.ByteSize(),
                                     input_tensors[2].GetDataPtr());
            // n => res x res x res
            SparseIndexer tsdf_indexer(input_sparse_tls[0],
                                       grid_shape.NumElements());

            // 27 x n => res x res x res
            SparseIndexer tsdf_nb_indexer(input_sparse_tls[1],
                                          grid_shape.NumElements());

            SparseIndexer surf_indexer(output_sparse_tl,
                                       grid_shape.NumElements());

            int64_t n = tsdf_indexer.NumWorkloads();
            int64_t m = input_tensors[2].GetShape()[1];
            std::cout << n << " " << m << "\n";

            Device device = output_sparse_tl.device_;

            int vtx_count_bound = 3 * triangle_count;
            Tensor vertices_x({vtx_count_bound}, Dtype::Float32, device);
            Tensor vertices_y({vtx_count_bound}, Dtype::Float32, device);
            Tensor vertices_z({vtx_count_bound}, Dtype::Float32, device);
            float* vertices_x_ptr =
                    static_cast<float*>(vertices_x.GetDataPtr());
            float* vertices_y_ptr =
                    static_cast<float*>(vertices_y.GetDataPtr());
            float* vertices_z_ptr =
                    static_cast<float*>(vertices_z.GetDataPtr());

            Tensor normals_x({vtx_count_bound}, Dtype::Float32, device);
            Tensor normals_y({vtx_count_bound}, Dtype::Float32, device);
            Tensor normals_z({vtx_count_bound}, Dtype::Float32, device);
            float* normals_x_ptr = static_cast<float*>(normals_x.GetDataPtr());
            float* normals_y_ptr = static_cast<float*>(normals_y.GetDataPtr());
            float* normals_z_ptr = static_cast<float*>(normals_z.GetDataPtr());

            Tensor vtx_count(std::vector<int>{0}, {1}, Dtype::Int32, device);
            int* vtx_count_ptr = static_cast<int*>(vtx_count.GetDataPtr());

            // Pass 1: allocate points and
            CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                         int64_t workload_idx) {
                int64_t key_idx, value_idx;
                tsdf_indexer.GetSparseWorkloadIdx(workload_idx, &key_idx,
                                                  &value_idx);

                int64_t xl, yl, zl;
                indexer3d.ConvertOffsetTo3D(value_idx, &xl, &yl, &zl);

                int64_t resolution = indexer3d.GetShape(0);

                float tsdf_o =
                        *static_cast<float*>(tsdf_indexer.GetWorkloadValuePtr(
                                key_idx, 0, value_idx));
                float n_o[3], n_p[3];

                for (int n = 0; n < 3; ++n) {
                    int64_t xl_p = xl + int(n == 0);
                    int64_t yl_p = yl + int(n == 1);
                    int64_t zl_p = zl + int(n == 2);

                    int64_t xl_n = xl - int(n == 0);
                    int64_t yl_n = yl - int(n == 1);
                    int64_t zl_n = zl - int(n == 2);

                    int dx_p = xl_p / resolution;
                    int dy_p = yl_p / resolution;
                    int dz_p = zl_p / resolution;

                    int dx_n = xl_n >= 0 ? 0 : -1;
                    int dy_n = yl_n >= 0 ? 0 : -1;
                    int dz_n = zl_n >= 0 ? 0 : -1;

                    int nb_idx_p = (dx_p + 1) + (dy_p + 1) * 3 + (dz_p + 1) * 9;
                    int nb_idx_n = (dx_n + 1) + (dy_n + 1) * 3 + (dz_n + 1) * 9;

                    int64_t nb_mask_offset_p;
                    indexer2d.Convert2DToOffset(key_idx, nb_idx_p,
                                                &nb_mask_offset_p);
                    bool nb_valid_p = *static_cast<bool*>(
                            indexer2d.GetPtrFromOffset(nb_mask_offset_p));
                    int64_t nb_value_idx_p;
                    indexer3d.Convert3DToOffset(
                            xl_p - dx_p * resolution, yl_p - dy_p * resolution,
                            zl_p - dz_p * resolution, &nb_value_idx_p);
                    float tsdf_p =
                            nb_valid_p
                                    ? *static_cast<float*>(
                                              tsdf_nb_indexer
                                                      .GetWorkloadValuePtr(
                                                              nb_idx_p * m +
                                                                      key_idx,
                                                              0,
                                                              nb_value_idx_p))
                                    : 0;

                    int64_t nb_mask_offset_n;
                    indexer2d.Convert2DToOffset(key_idx, nb_idx_n,
                                                &nb_mask_offset_n);
                    bool nb_valid_n = *static_cast<bool*>(
                            indexer2d.GetPtrFromOffset(nb_mask_offset_n));
                    int64_t nb_value_idx_n;
                    indexer3d.Convert3DToOffset(
                            xl_n - dx_n * resolution, yl_n - dy_n * resolution,
                            zl_n - dz_n * resolution, &nb_value_idx_n);
                    float tsdf_n =
                            nb_valid_n
                                    ? *static_cast<float*>(
                                              tsdf_nb_indexer
                                                      .GetWorkloadValuePtr(
                                                              nb_idx_n * m +
                                                                      key_idx,
                                                              0,
                                                              nb_value_idx_n))
                                    : 0;

                    n_o[n] = (tsdf_p - tsdf_n) / (2 * voxel_size);
                }

                // Enumerate 8 neighbor corners (including itself)
                for (int i = 0; i < 3; ++i) {
                    int* vertex_idx =
                            static_cast<int*>(surf_indexer.GetWorkloadValuePtr(
                                    key_idx, i, value_idx));

                    if (*vertex_idx != -1) {
                        continue;
                    }

                    int64_t xl_i = xl + int(i == 0);
                    int64_t yl_i = yl + int(i == 1);
                    int64_t zl_i = zl + int(i == 2);

                    int dx = xl_i / resolution;
                    int dy = yl_i / resolution;
                    int dz = zl_i / resolution;

                    int nb_idx = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;

                    int64_t nb_value_idx;
                    indexer3d.Convert3DToOffset(
                            xl_i - dx * resolution, yl_i - dy * resolution,
                            zl_i - dz * resolution, &nb_value_idx);

                    float tsdf_i = *static_cast<float*>(
                            tsdf_nb_indexer.GetWorkloadValuePtr(
                                    nb_idx * m + key_idx, 0, nb_value_idx));

                    for (int n = 0; n < 3; ++n) {
                        int64_t xl_p = xl_i + int(n == 0);
                        int64_t yl_p = yl_i + int(n == 1);
                        int64_t zl_p = zl_i + int(n == 2);

                        int64_t xl_n = xl_i - int(n == 0);
                        int64_t yl_n = yl_i - int(n == 1);
                        int64_t zl_n = zl_i - int(n == 2);

                        int dx_p = xl_p / resolution;
                        int dy_p = yl_p / resolution;
                        int dz_p = zl_p / resolution;

                        int dx_n = xl_n >= 0 ? 0 : -1;
                        int dy_n = yl_n >= 0 ? 0 : -1;
                        int dz_n = zl_n >= 0 ? 0 : -1;

                        int nb_idx_p =
                                (dx_p + 1) + (dy_p + 1) * 3 + (dz_p + 1) * 9;
                        int nb_idx_n =
                                (dx_n + 1) + (dy_n + 1) * 3 + (dz_n + 1) * 9;

                        int64_t nb_mask_offset_p;
                        indexer2d.Convert2DToOffset(key_idx, nb_idx_p,
                                                    &nb_mask_offset_p);
                        bool nb_valid_p = *static_cast<bool*>(
                                indexer2d.GetPtrFromOffset(nb_mask_offset_p));
                        int64_t nb_value_idx_p;
                        indexer3d.Convert3DToOffset(xl_p - dx_p * resolution,
                                                    yl_p - dy_p * resolution,
                                                    zl_p - dz_p * resolution,
                                                    &nb_value_idx_p);
                        float tsdf_p =
                                nb_valid_p
                                        ? *static_cast<float*>(
                                                  tsdf_nb_indexer
                                                          .GetWorkloadValuePtr(
                                                                  nb_idx_p * m +
                                                                          key_idx,
                                                                  0,
                                                                  nb_value_idx_p))
                                        : 0;

                        int64_t nb_mask_offset_n;
                        indexer2d.Convert2DToOffset(key_idx, nb_idx_n,
                                                    &nb_mask_offset_n);
                        bool nb_valid_n = *static_cast<bool*>(
                                indexer2d.GetPtrFromOffset(nb_mask_offset_n));
                        int64_t nb_value_idx_n;
                        indexer3d.Convert3DToOffset(xl_n - dx_n * resolution,
                                                    yl_n - dy_n * resolution,
                                                    zl_n - dz_n * resolution,
                                                    &nb_value_idx_n);
                        float tsdf_n =
                                nb_valid_n
                                        ? *static_cast<float*>(
                                                  tsdf_nb_indexer
                                                          .GetWorkloadValuePtr(
                                                                  nb_idx_n * m +
                                                                          key_idx,
                                                                  0,
                                                                  nb_value_idx_n))
                                        : 0;

                        n_p[n] = (tsdf_p - tsdf_n) / (2 * voxel_size);
                    }

                    float ratio = tsdf_i / (tsdf_i - tsdf_o);

                    void* key_ptr = tsdf_indexer.GetWorkloadKeyPtr(key_idx);
                    int64_t xg = *(static_cast<int64_t*>(key_ptr) + 0);
                    int64_t yg = *(static_cast<int64_t*>(key_ptr) + 1);
                    int64_t zg = *(static_cast<int64_t*>(key_ptr) + 2);

                    // https://stackoverflow.com/questions/4034908/fetch-and-add-using-openmp-atomic-operations
                    int idx = atomicAdd(vtx_count_ptr, 1);

                    *vertex_idx = idx;

                    float ratio_x = (1 - ratio) * int(i == 0);
                    float ratio_y = (1 - ratio) * int(i == 1);
                    float ratio_z = (1 - ratio) * int(i == 2);

                    vertices_x_ptr[idx] =
                            voxel_size * (xg * resolution + xl + (ratio_x));
                    vertices_y_ptr[idx] =
                            voxel_size * (yg * resolution + yl + (ratio_y));
                    vertices_z_ptr[idx] =
                            voxel_size * (zg * resolution + zl + (ratio_z));
                    // printf("%f => %f %f %f => %f %f %f\n", ratio, ratio_x,
                    //        ratio_y, ratio_z, vertices_x_ptr[idx],
                    //        vertices_y_ptr[idx], vertices_z_ptr[idx]);

                    float nx = n_o[0] * (ratio) + n_p[0] * (1 - ratio);
                    float ny = n_o[1] * (ratio) + n_p[1] * (1 - ratio);
                    float nz = n_o[2] * (ratio) + n_p[2] * (1 - ratio);
                    float norm = sqrtf(nx * nx + ny * ny + nz * nz);

                    normals_x_ptr[idx] = nx / norm;
                    normals_y_ptr[idx] = ny / norm;
                    normals_z_ptr[idx] = nz / norm;
                }
            });

            int actual_count = vtx_count[0].Item<int>();
            std::cout << actual_count << "\n";
            output_tensor = Tensor({6, actual_count}, Dtype::Float32, device);
            output_tensor[0].Slice(0, 0, actual_count) =
                    vertices_x.Slice(0, 0, actual_count);
            output_tensor[1].Slice(0, 0, actual_count) =
                    vertices_y.Slice(0, 0, actual_count);
            output_tensor[2].Slice(0, 0, actual_count) =
                    vertices_z.Slice(0, 0, actual_count);
            output_tensor[3].Slice(0, 0, actual_count) =
                    normals_x.Slice(0, 0, actual_count);
            output_tensor[4].Slice(0, 0, actual_count) =
                    normals_y.Slice(0, 0, actual_count);
            output_tensor[5].Slice(0, 0, actual_count) =
                    normals_z.Slice(0, 0, actual_count);

            break;
        }

        case SpecialOpCode::MarchingCubesPass2: {
            utility::LogInfo("MC Pass2");
            // input_sparse_tls: tsdf grid
            // output_sparse_tl: surface grid
            // tensors: voxel_size, sdf_trunc
            SizeVector grid_shape = output_sparse_tl.shapes_[0];
            int triangle_count = input_tensors[1][0].Item<int>();

            // res x res x res
            NDArrayIndexer indexer3d(grid_shape,
                                     Dtype::Int32.ByteSize());
            // 27 x n
            NDArrayIndexer indexer2d(input_tensors[2].GetShape(),
                                     Dtype::Bool.ByteSize(),
                                     input_tensors[2].GetDataPtr());

            SparseIndexer surf_indexer(input_sparse_tls[0],
                                       grid_shape.NumElements());
            SparseIndexer surf_nb_indexer(output_sparse_tl,
                                          grid_shape.NumElements());

            int64_t n = surf_indexer.NumWorkloads();
            int64_t m = input_tensors[2].GetShape()[1];
            std::cout << n << " " << m << "\n";

            Device device = output_sparse_tl.device_;
            output_tensor = Tensor({triangle_count * 3}, Dtype::Int32, device);
            int* tri_ptr = static_cast<int*>(output_tensor.GetDataPtr());

            Tensor count(std::vector<int>{0}, {1}, Dtype::Int32, device);
            int* tri_count_ptr = static_cast<int*>(count.GetDataPtr());

            // Pass 2: connect vertices
            CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                         int64_t workload_idx) {
                int64_t key_idx, value_idx;
                surf_indexer.GetSparseWorkloadIdx(workload_idx, &key_idx,
                                                  &value_idx);

                int64_t xl, yl, zl;
                indexer3d.ConvertOffsetTo3D(value_idx, &xl, &yl, &zl);

                int64_t resolution = indexer3d.GetShape(0);

                int table_index =
                        *static_cast<int*>(surf_indexer.GetWorkloadValuePtr(
                                key_idx, 3, value_idx));
                if (tri_count[table_index] == 0) return;

                for (size_t tri = 0; tri < 16; tri += 3) {
                    if (tri_table[table_index][tri] == -1) return;

                    int tri_idx = atomicAdd(tri_count_ptr, 3);

                    for (size_t vertex = 0; vertex < 3; ++vertex) {
                        int edge = tri_table[table_index][tri + vertex];

                        int64_t xl_i = xl + edge_shifts[edge][0];
                        int64_t yl_i = yl + edge_shifts[edge][1];
                        int64_t zl_i = zl + edge_shifts[edge][2];
                        int64_t edge_i = edge_shifts[edge][3];

                        int dx = xl_i / resolution;
                        int dy = yl_i / resolution;
                        int dz = zl_i / resolution;

                        int nb_idx = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;

                        int64_t nb_value_idx;
                        indexer3d.Convert3DToOffset(
                                xl_i - dx * resolution, yl_i - dy * resolution,
                                zl_i - dz * resolution, &nb_value_idx);

                        int vtx_idx = *static_cast<int*>(
                                surf_nb_indexer.GetWorkloadValuePtr(
                                        nb_idx * m + key_idx, edge_i,
                                        nb_value_idx));
                        // if (vtx_idx < 0) {
                        //     printf("%ld, %ld, (%ld, %ld, %ld), SHOULD NEVER "
                        //            "REACH HERE!\n",
                        //            key_idx, vertex, xl, yl, zl);
                        //     return;
                        // }
                        tri_ptr[tri_idx + 2 - vertex] = vtx_idx;
                    }
                }
            });

            output_tensor = output_tensor.View({triangle_count, 3});
            std::cout << count.ToString() << "\n";
            std::cout << triangle_count << "\n";
            break;
        }

        case SpecialOpCode::Check: {
            utility::LogInfo("Check");
            // 27 x n
            SizeVector grid_shape = input_sparse_tls[0].shapes_[0];

            NDArrayIndexer indexer2d(input_tensors[0].GetShape(),
                                     Dtype::Bool.ByteSize(),
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
}  // namespace kernel
}  // namespace kernel
}  // namespace core
}  // namespace open3d
