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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/CUDA/StdGPUHashmap.h"
#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGrid.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGridImpl.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace tsdf {
struct Coord3i {
    OPEN3D_HOST_DEVICE Coord3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}
    bool OPEN3D_HOST_DEVICE operator==(const Coord3i& other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    int64_t x_;
    int64_t y_;
    int64_t z_;
};

void TouchCUDA(std::shared_ptr<core::Hashmap>& hashmap,
               const core::Tensor& points,
               core::Tensor& voxel_block_coords,
               int64_t voxel_grid_resolution,
               float voxel_size,
               float sdf_trunc) {
    int64_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    int64_t n = points.GetLength();
    const float* pcd_ptr = static_cast<const float*>(points.GetDataPtr());

    core::Device device = points.GetDevice();
    core::Tensor block_coordi({8 * n, 3}, core::Dtype::Int32, device);
    int* block_coordi_ptr = static_cast<int*>(block_coordi.GetDataPtr());
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32, device);
    int* count_ptr = static_cast<int*>(count.GetDataPtr());

    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float x = pcd_ptr[3 * workload_idx + 0];
                float y = pcd_ptr[3 * workload_idx + 1];
                float z = pcd_ptr[3 * workload_idx + 2];

                int xb_lo =
                        static_cast<int>(floorf((x - sdf_trunc) / block_size));
                int xb_hi =
                        static_cast<int>(floorf((x + sdf_trunc) / block_size));
                int yb_lo =
                        static_cast<int>(floorf((y - sdf_trunc) / block_size));
                int yb_hi =
                        static_cast<int>(floorf((y + sdf_trunc) / block_size));
                int zb_lo =
                        static_cast<int>(floorf((z - sdf_trunc) / block_size));
                int zb_hi =
                        static_cast<int>(floorf((z + sdf_trunc) / block_size));

                for (int xb = xb_lo; xb <= xb_hi; ++xb) {
                    for (int yb = yb_lo; yb <= yb_hi; ++yb) {
                        for (int zb = zb_lo; zb <= zb_hi; ++zb) {
                            int idx = atomicAdd(count_ptr, 1);
                            block_coordi_ptr[3 * idx + 0] = xb;
                            block_coordi_ptr[3 * idx + 1] = yb;
                            block_coordi_ptr[3 * idx + 2] = zb;
                        }
                    }
                }
            });

    int total_block_count = count.Item<int>();
    if (total_block_count == 0) {
        utility::LogError(
                "[CUDATSDFTouchKernel] No block is touched in TSDF volume, "
                "abort integration. Please check specified parameters, "
                "especially depth_scale and voxel_size");
    }
    block_coordi = block_coordi.Slice(0, 0, total_block_count);
    core::Tensor block_addrs, block_masks;
    hashmap->Activate(block_coordi.Slice(0, 0, count.Item<int>()), block_addrs,
                      block_masks);
    voxel_block_coords = block_coordi.IndexGet({block_masks});
    utility::LogInfo("voxel_block_coords = {}", voxel_block_coords.GetLength());
}

void TouchCUDA(std::shared_ptr<core::Hashmap>& hashmap,
               const core::Tensor& depth,
               const core::Tensor& intrinsics,
               const core::Tensor& extrinsics,
               core::Tensor& voxel_block_coords,
               int64_t voxel_grid_resolution,
               float voxel_size,
               float sdf_trunc,
               float depth_scale,
               float depth_max,
               int stride) {
    core::Device device = depth.GetDevice();
    NDArrayIndexer depth_indexer(depth, 2);
    core::Tensor pose = t::geometry::InverseTransformation(extrinsics);
    TransformIndexer ti(intrinsics, pose, 1.0f);

    // Output
    int64_t rows_strided = depth_indexer.GetShape(0) / stride;
    int64_t cols_strided = depth_indexer.GetShape(1) / stride;

    // Counter
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32, device);
    int* count_ptr = count.GetDataPtr<int>();

    int64_t n = rows_strided * cols_strided;
    core::Tensor block_coordi({8 * n, 3}, core::Dtype::Int32, device);
    int* block_coordi_ptr = static_cast<int*>(block_coordi.GetDataPtr());

    int64_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    core::kernel::CUDALauncher launcher;
    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                int64_t workload_idx) {
            int64_t y = (workload_idx / cols_strided) * stride;
            int64_t x = (workload_idx % cols_strided) * stride;

            float d = *depth_indexer.GetDataPtrFromCoord<scalar_t>(x, y) /
                      depth_scale;
            if (d > 0 && d < depth_max) {
                float x_c = 0, y_c = 0, z_c = 0;
                ti.Unproject(static_cast<float>(x), static_cast<float>(y), d,
                             &x_c, &y_c, &z_c);

                float x_g = 0, y_g = 0, z_g = 0;
                ti.RigidTransform(x_c, y_c, z_c, &x_g, &y_g, &z_g);

                int xb_lo = static_cast<int>(
                        floorf((x_g - sdf_trunc) / block_size));
                int xb_hi = static_cast<int>(
                        floorf((x_g + sdf_trunc) / block_size));
                int yb_lo = static_cast<int>(
                        floorf((y_g - sdf_trunc) / block_size));
                int yb_hi = static_cast<int>(
                        floorf((y_g + sdf_trunc) / block_size));
                int zb_lo = static_cast<int>(
                        floorf((z_g - sdf_trunc) / block_size));
                int zb_hi = static_cast<int>(
                        floorf((z_g + sdf_trunc) / block_size));

                for (int xb = xb_lo; xb <= xb_hi; ++xb) {
                    for (int yb = yb_lo; yb <= yb_hi; ++yb) {
                        for (int zb = zb_lo; zb <= zb_hi; ++zb) {
                            int idx = atomicAdd(count_ptr, 1);
                            block_coordi_ptr[3 * idx + 0] = xb;
                            block_coordi_ptr[3 * idx + 1] = yb;
                            block_coordi_ptr[3 * idx + 2] = zb;
                        }
                    }
                }
            }
        });
    });

    int total_block_count = count.Item<int>();
    if (total_block_count == 0) {
        utility::LogError(
                "[CUDATSDFTouchKernel] No block is touched in TSDF volume, "
                "abort integration. Please check specified parameters, "
                "especially depth_scale and voxel_size");
    }
    block_coordi = block_coordi.Slice(0, 0, total_block_count);
    core::Tensor block_addrs, block_masks;
    hashmap->Activate(block_coordi.Slice(0, 0, count.Item<int>()), block_addrs,
                      block_masks);
    voxel_block_coords = block_coordi.IndexGet({block_masks});
    utility::LogInfo("voxel_block_coords = {}", voxel_block_coords.GetLength());
}

}  // namespace tsdf
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
