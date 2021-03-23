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

#include "open3d/t/pipelines/odometry/RGBDOdometry.h"

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/pipelines/kernel/RGBDOdometry.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/utility/Timer.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

core::Tensor RGBDOdometryMultiScale(const t::geometry::RGBDImage& source,
                                    const t::geometry::RGBDImage& target,
                                    const core::Tensor& intrinsics,
                                    const core::Tensor& init_source_to_target,
                                    float depth_scale,
                                    float depth_diff,
                                    float depth_max,
                                    const std::vector<int>& iterations,
                                    const std::vector<float>& dist_thrs,
                                    const LossType method) {
    core::Device device = source.depth_.GetDevice();
    if (target.depth_.GetDevice() != device) {
        utility::LogError(
                "Device mismatch, got {} for source and {} for target.",
                device.ToString(), target.depth_.GetDevice().ToString());
    }

    core::Tensor intrinsics_d = intrinsics.To(device, true);

    core::Device host("CPU:0");
    core::Tensor trans_d = init_source_to_target.To(host, core::Dtype::Float64);

    if (method == LossType::PointToPlane) {
        int64_t n = int64_t(iterations.size());

        std::vector<core::Tensor> src_vertex_maps(iterations.size());
        std::vector<core::Tensor> src_normal_maps(iterations.size());
        std::vector<core::Tensor> dst_vertex_maps(iterations.size());
        std::vector<core::Tensor> intrinsic_matrices(iterations.size());

        t::geometry::Image src_depth = source.depth_;
        t::geometry::Image dst_depth = target.depth_;

        // Create image pyramid
        utility::Timer timer;
        timer.Start();
        for (int64_t i = 0; i < n; ++i) {
            core::Tensor src_vertex_map =
                    CreateVertexMap(src_depth, intrinsics_d, depth_scale);

            t::geometry::Image src_depth_smooth =
                    src_depth.FilterBilateral(3, 20.0f, 20.0f);
            core::Tensor src_vertex_map_smooth = CreateVertexMap(
                    src_depth_smooth, intrinsics_d, depth_scale);
            core::Tensor src_normal_map =
                    CreateNormalMap(src_vertex_map_smooth);

            // core::Tensor points = src_vertex_map.View({-1, 3});
            // core::Tensor normals = src_normal_map.View({-1, 3});
            // // utility::LogInfo("points = {}", points.ToString());
            // utility::LogInfo("normals = {}", normals.ToString());

            // core::Tensor nxs = normals.Slice(1, 0, 1);
            // core::Tensor mask = nxs.Abs().Le(2);

            // points = points.IndexGet({mask.View({-1})});
            // normals = normals.IndexGet({mask.View({-1})});

            // t::geometry::PointCloud pcd(points);
            // pcd.SetPointNormals(normals);
            // visualization::DrawGeometries(
            //         {std::make_shared<open3d::geometry::PointCloud>(
            //                 pcd.ToLegacyPointCloud())});

            core::Tensor dst_vertex_map =
                    CreateVertexMap(dst_depth, intrinsics_d, depth_scale);

            src_vertex_maps[n - 1 - i] = src_vertex_map;
            src_normal_maps[n - 1 - i] = src_normal_map;
            dst_vertex_maps[n - 1 - i] = dst_vertex_map;

            intrinsic_matrices[n - 1 - i] = intrinsics_d.Clone();

            if (i != n - 1) {
                src_depth = PyrDownDepth(src_depth, depth_scale, depth_diff,
                                         depth_max);
                dst_depth = PyrDownDepth(dst_depth, depth_scale, depth_diff,
                                         depth_max);

                intrinsics_d /= 2;
                intrinsics_d[-1][-1] = 1;
            }
        }
        timer.Stop();
        utility::LogInfo("Preparation {}", timer.GetDuration());

        // Odometry
        for (int64_t i = 0; i < n; ++i) {
            timer.Start();
            for (int iter = 0; iter < iterations[i]; ++iter) {
                core::Tensor delta_src_to_dst = ComputePosePointToPlane(
                        src_vertex_maps[i], dst_vertex_maps[i],
                        src_normal_maps[i], intrinsic_matrices[i], trans_d,
                        dist_thrs[i]);
                trans_d = delta_src_to_dst.Matmul(trans_d);
            }
            timer.Stop();
            utility::LogInfo("Level {}: {} per iter", i,
                             timer.GetDuration() / iterations[i]);
        }
    } else {
        utility::LogError("Odometry method not implemented.");
    }

    return trans_d;
}

t::geometry::Image PyrDownDepth(const t::geometry::Image& depth,
                                float depth_scale,
                                float depth_diff,
                                float depth_max) {
    core::Tensor depth_down;
    kernel::odometry::PyrDownDepth(depth.AsTensor(), depth_down, depth_scale,
                                   depth_diff, depth_max);
    return t::geometry::Image(depth_down);
}

core::Tensor CreateVertexMap(const t::geometry::Image& depth,
                             const core::Tensor& intrinsics,
                             float depth_scale,
                             float depth_max) {
    core::Tensor vertex_map;
    kernel::odometry::CreateVertexMap(depth.AsTensor(), intrinsics, vertex_map,
                                      depth_scale, depth_max);
    return vertex_map;
}

core::Tensor CreateNormalMap(const core::Tensor& vertex_map) {
    core::Tensor normal_map;
    kernel::odometry::CreateNormalMap(vertex_map, normal_map);
    return normal_map;
}

core::Tensor ComputePosePointToPlane(const core::Tensor& source_vtx_map,
                                     const core::Tensor& target_vtx_map,
                                     const core::Tensor& source_normal_map,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     float depth_diff) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    core::Tensor residual;
    kernel::odometry::ComputePosePointToPlane(
            source_vtx_map, target_vtx_map, source_normal_map, intrinsics,
            init_source_to_target, se3_delta, residual, depth_diff);

    core::Tensor T_delta_inv =
            pipelines::kernel::PoseToTransformation(se3_delta);

    // T.inv = [R.T | -R.T @ t]
    core::Tensor R_inv = T_delta_inv.Slice(0, 0, 3).Slice(1, 0, 3);
    core::Tensor t_inv = T_delta_inv.Slice(0, 0, 3).Slice(1, 3, 4);

    core::Tensor T_delta = core::Tensor::Zeros({4, 4}, core::Dtype::Float64);
    T_delta.Slice(0, 0, 3).Slice(1, 0, 3) = R_inv.T();
    T_delta.Slice(0, 0, 3).Slice(1, 3, 4) = R_inv.T().Matmul(t_inv).Neg();
    T_delta[-1][-1] = 1;

    return T_delta;
}

core::Tensor ComputePoseDirectHybrid(const core::Tensor& source_vtx_map,
                                     const core::Tensor& target_vtx_map,
                                     const core::Tensor& source_color,
                                     const core::Tensor& target_color,
                                     const core::Tensor& source_color_dx,
                                     const core::Tensor& target_color_dy,
                                     const core::Tensor& source_depth_dx,
                                     const core::Tensor& target_depth_dy,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     float depth_diff) {
    utility::LogError("Direct hybrid odometry unimplemented.");
}

core::Tensor ComputePoseDirectIntensity(
        const core::Tensor& source_vtx_map,
        const core::Tensor& target_vtx_map,
        const core::Tensor& source_color,
        const core::Tensor& target_color,
        const core::Tensor& source_color_dx,
        const core::Tensor& source_color_dy,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        float depth_diff) {
    utility::LogError("Direct intensity odometry unimplemented.");
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
