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

#include "open3d/tgeometry/VoxelGrid.h"

#include <Eigen/Dense>

#include "open3d/core/EigenAdaptor.h"
#include "open3d/core/SparseTensorList.h"
#include "open3d/core/TensorList.h"
#include "open3d/core/kernel/SpecialOp.h"
#include "open3d/utility/Console.h"
namespace open3d {
namespace tgeometry {
using namespace core;

VoxelGrid::VoxelGrid(float voxel_size,
                     float sdf_trunc,
                     int64_t resolution,
                     int64_t capacity,
                     const Device &device)
    : Geometry3D(Geometry::GeometryType::VoxelGrid),
      voxel_size_(voxel_size),
      sdf_trunc_(sdf_trunc),
      resolution_(resolution),
      capacity_(capacity),
      device_(device) {
    hashmap_ = CreateDefaultHashmap(
            capacity, 3 * sizeof(int64_t),
            2 * resolution * resolution * resolution * sizeof(float), device);
}

void VoxelGrid::Integrate(const tgeometry::Image &depth,
                          const Tensor &intrinsic,
                          const Tensor &extrinsic) {
    /// Unproject
    Tensor vertex_map = depth.Unproject(intrinsic);
    Tensor pcd_map = vertex_map.View({3, 480 * 640});

    /// Inverse is currently not available...
    Eigen::Matrix4f pose_ = ToEigen<float>(extrinsic).inverse();
    Tensor pose = FromEigen(pose_).Copy(device_);
    tgeometry::PointCloud pcd(pcd_map.T());
    pcd.Transform(pose);

    tgeometry::PointCloud pcd_down =
            pcd.VoxelDownSample(voxel_size_ * resolution_);

    Tensor coords = pcd_down.point_dict_["points"].AsTensor().To(Dtype::Int64);
    SizeVector coords_shape = coords.GetShape();
    int64_t N = coords_shape[0];

    void *iterators = MemoryManager::Malloc(sizeof(iterator_t) * N, device_);
    void *masks = MemoryManager::Malloc(sizeof(bool) * N, device_);

    hashmap_->Activate(static_cast<void *>(coords.GetBlob()->GetDataPtr()),
                       static_cast<iterator_t *>(iterators),
                       static_cast<bool *>(masks), N);

    hashmap_->Find(static_cast<void *>(coords.GetBlob()->GetDataPtr()),
                   static_cast<iterator_t *>(iterators),
                   static_cast<bool *>(masks), N);

    utility::LogInfo("Active entries = {}", N);

    SizeVector shape = SizeVector{resolution_, resolution_, resolution_};
    SparseTensorList sparse_tl(static_cast<void **>(iterators), N, true,
                               {shape, shape}, {Dtype::Float32, Dtype::Float32},
                               device_);
    Tensor voxel_size(std::vector<float>{voxel_size_}, {1}, Dtype::Float32);
    Tensor sdf_trunc(std::vector<float>{sdf_trunc_}, {1}, Dtype::Float32);

    Tensor output_dummy;
    kernel::SpecialOpEW(
            {depth.data_, intrinsic, extrinsic, voxel_size, sdf_trunc}, {},
            output_dummy, sparse_tl, kernel::SpecialOpCode::Integrate);
    utility::LogInfo("[VoxelGrid] Kernel launch finished");

    // Then manipulate iterators to integrate!
    MemoryManager::Free(iterators, coords.GetDevice());
    utility::LogInfo("[VoxelGrid] iterators freed");
    MemoryManager::Free(masks, coords.GetDevice());
    utility::LogInfo("[VoxelGrid] masks freed");

    utility::LogInfo("Hashmap size = {}", hashmap_->size());
}

std::pair<SparseTensorList, Tensor> VoxelGrid::ExtractNearestNeighbors() {
    int64_t n = hashmap_->size();
    void *block_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n, device_);

    hashmap_->GetIterators(static_cast<iterator_t *>(block_iterators));

    Tensor keys({n, 3}, Dtype::Int64, device_);
    hashmap_->UnpackIterators(static_cast<iterator_t *>(block_iterators),
                              nullptr, keys.GetDataPtr(), nullptr, n);

    Tensor keys_nb({27, n, 3}, Dtype::Int64, device_);
    void *nb_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n * 27, device_);
    Tensor masks_nb({27, n}, Dtype::Bool, device_);
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor dt = Tensor(std::vector<int64_t>{dx - 1, dy - 1, dz - 1}, {1, 3},
                           Dtype::Int64, device_);
        keys_nb[nb] = keys + dt;
    }

    hashmap_->Find(keys_nb.GetDataPtr(),
                   static_cast<iterator_t *>(nb_iterators),
                   static_cast<bool *>(masks_nb.GetDataPtr()), n * 27);

    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor mask_nb = masks_nb[nb];
        utility::LogInfo("{}: ({}, {}, {}) => {}", nb, dx - 1, dy - 1, dz - 1,
                         mask_nb.To(Dtype::Int32).Sum({0}).Item<int>());
    }

    /// 27 * n nbs (some of them are empty), each pointing to a TSDF and a
    /// weight Tensor
    SizeVector shape = SizeVector{resolution_, resolution_, resolution_};
    SparseTensorList sparse_nb_tsdf_tl(
            static_cast<void **>(nb_iterators), 27 * n, true, {shape, shape},
            {Dtype::Float32, Dtype::Float32}, device_);
    SparseTensorList sparse_tsdf_tl(static_cast<void **>(block_iterators), n,
                                    true, {shape, shape},
                                    {Dtype::Float32, Dtype::Float32}, device_);

    Tensor dummy;
    kernel::SpecialOpEW({masks_nb}, {sparse_tsdf_tl, sparse_nb_tsdf_tl}, dummy,
                        sparse_tsdf_tl, kernel::SpecialOpCode::Check);

    return std::make_pair(sparse_nb_tsdf_tl, masks_nb);
}

tgeometry::PointCloud VoxelGrid::ExtractSurfacePoints() {
    int64_t n = hashmap_->size();
    void *block_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n, device_);
    void *surf_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n, device_);
    void *nb_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n * 27, device_);

    hashmap_->GetIterators(static_cast<iterator_t *>(block_iterators));

    Tensor keys({n, 3}, Dtype::Int64, device_);
    hashmap_->UnpackIterators(static_cast<iterator_t *>(block_iterators),
                              nullptr, keys.GetDataPtr(), nullptr, n);

    // Each voxel corresponds to ptrs to 3 vertices
    auto surface_hashmap = CreateDefaultHashmap(
            hashmap_->size(), 3 * sizeof(int64_t),
            3 * resolution_ * resolution_ * resolution_ * sizeof(int), device_);

    Tensor masks({n}, Dtype::Bool, device_);
    surface_hashmap->Activate(keys.GetDataPtr(),
                              static_cast<iterator_t *>(surf_iterators),
                              static_cast<bool *>(masks.GetDataPtr()), n);

    // Each block corresponds to 27 neighbors (at most)
    Tensor keys_nb({27, n, 3}, Dtype::Int64, device_);
    Tensor masks_nb({27, n}, Dtype::Bool, device_);
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor dt = Tensor(std::vector<int64_t>{dx - 1, dy - 1, dz - 1}, {1, 3},
                           Dtype::Int64, device_);
        keys_nb[nb] = keys + dt;
    }
    hashmap_->Find(keys_nb.GetDataPtr(),
                   static_cast<iterator_t *>(nb_iterators),
                   static_cast<bool *>(masks_nb.GetDataPtr()), n * 27);

    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor mask_nb = masks_nb[nb];
        utility::LogInfo("{}: ({}, {}, {}) => {}", nb, dx - 1, dy - 1, dz - 1,
                         mask_nb.To(Dtype::Int32).Sum({0}).Item<int>());
    }

    SizeVector shape = SizeVector{resolution_, resolution_, resolution_};
    SparseTensorList sparse_tsdf_tl(static_cast<void **>(block_iterators), n,
                                    true, {shape, shape},
                                    {Dtype::Float32, Dtype::Float32}, device_);

    SparseTensorList sparse_surf_tl(static_cast<void **>(surf_iterators), n,
                                    true, {shape, shape, shape},
                                    {Dtype::Int32, Dtype::Int32, Dtype::Int32},
                                    device_);
    SparseTensorList sparse_nb_tsdf_tl(
            static_cast<void **>(nb_iterators), 27 * n, true, {shape, shape},
            {Dtype::Float32, Dtype::Float32}, device_);

    Tensor voxel_size(std::vector<float>{voxel_size_}, {1}, Dtype::Float32);
    Tensor output;
    kernel::SpecialOpEW({voxel_size, masks_nb},
                        {sparse_tsdf_tl, sparse_nb_tsdf_tl}, output,
                        sparse_surf_tl, kernel::SpecialOpCode::ExtractSurface);

    MemoryManager::Free(block_iterators, device_);
    MemoryManager::Free(surf_iterators, device_);
    MemoryManager::Free(nb_iterators, device_);
    MemoryManager::ReleaseCache(device_);

    return tgeometry::PointCloud(output.T());
}

void VoxelGrid::MarchingCubes() {
    // int64_t n = hashmap_->size();
    // void *block_iterators =
    //         MemoryManager::Malloc(sizeof(iterator_t) * n, device_);
    // void *nb_iterators =
    //         MemoryManager::Malloc(sizeof(iterator_t) * n * 27, device_);

    // // Get all TSDF blocks and their keys
    // hashmap_->GetIterators(static_cast<iterator_t *>(block_iterators));

    // Tensor keys({n, 3}, Dtype::Int64, device_);
    // hashmap_->UnpackIterators(static_cast<iterator_t *>(block_iterators),
    //                           nullptr, keys.GetDataPtr(), nullptr, n);

    // // Get neighbors per TSDF block
    // Tensor keys_nb({27, n, 3}, Dtype::Int64, device_);
    // Tensor masks_nb({27, n}, Dtype::Bool, device_);
    // for (int nb = 0; nb < 27; ++nb) {
    //     int dz = nb / 9;
    //     int dy = (nb % 9) / 3;
    //     int dx = nb % 3;
    //     Tensor dt = Tensor(std::vector<int64_t>{dx - 1, dy - 1, dz - 1}, {1,
    //     3},
    //                        Dtype::Int64, device_);
    //     keys_nb[nb] = keys + dt;
    // }
    // hashmap_->Find(keys_nb.GetDataPtr(),
    //                static_cast<iterator_t *>(nb_iterators),
    //                static_cast<bool *>(masks_nb.GetDataPtr()), n * 27);

    // for (int nb = 0; nb < 27; ++nb) {
    //     int dz = nb / 9;
    //     int dy = (nb % 9) / 3;
    //     int dx = nb % 3;
    //     Tensor mask_nb = masks_nb[nb];
    //     utility::LogInfo("{}: ({}, {}, {}) => {}", nb, dx - 1, dy - 1, dz -
    //     1,
    //                      mask_nb.To(Dtype::Int32).Sum({0}).Item<int>());
    // }

    // // Then prepare output tensors:
    // // Smaller hashmap with voxels and triangles
    // // Need: a prefix-sum array for triangles
    // SparseTensorList sparse_tsdf_tl(static_cast<void **>(block_iterators), n,
    //                                 true, {shape, shape},
    //                                 {Dtype::Float32, Dtype::Float32},
    //                                 device_);
    // SparseTensorList sparse_nb_tsdf_tl(
    //         static_cast<void **>(nb_iterators), 27 * n, true, {shape, shape},
    //         {Dtype::Float32, Dtype::Float32}, device_);

    // SizeVector shape = SizeVector{resolution_, resolution_, resolution_};

    // Tensor voxel_size(std::vector<float>{voxel_size_}, {1}, Dtype::Float32);
    // Tensor output;
    // kernel::SpecialOpEW({voxel_size, masks_nb},
    //                     {sparse_tsdf_tl, sparse_nb_tsdf_tl}, output,
    //                     sparse_surf_tl,
    //                     kernel::SpecialOpCode::ExtractSurface);

    // MemoryManager::Free(block_iterators, device_);
    // MemoryManager::Free(surf_iterators, device_);
    // MemoryManager::Free(nb_iterators, device_);
    // MemoryManager::ReleaseCache(device_);
}

}  // namespace tgeometry
}  // namespace open3d
