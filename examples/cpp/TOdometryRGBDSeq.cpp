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
#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::core;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    VoxelHashing [color_folder] [depth_folder] [options]");
    utility::LogInfo("     Given RGBD images, reconstruct mesh or point cloud from color and depth images");
    utility::LogInfo("     [options]");
    utility::LogInfo("     --voxel_size [=0.0058 (m)]");
    utility::LogInfo("     --intrinsic_path [camera_intrinsic]");
    utility::LogInfo("     --depth_scale [=1000.0]");
    utility::LogInfo("     --max_depth [=3.0]");
    utility::LogInfo("     --sdf_trunc [=0.04]");
    utility::LogInfo("     --device [CPU:0]");
    utility::LogInfo("     --mesh");
    utility::LogInfo("     --pointcloud");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char** argv) {
    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc < 4) {
        PrintHelp();
        return 1;
    }

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    // Color and depth
    std::string color_folder = std::string(argv[1]);
    std::string depth_folder = std::string(argv[2]);

    std::vector<std::string> color_filenames;
    utility::filesystem::ListFilesInDirectory(color_folder, color_filenames);
    std::sort(color_filenames.begin(), color_filenames.end());

    std::vector<std::string> depth_filenames;
    utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    std::sort(depth_filenames.begin(), depth_filenames.end());

    if (color_filenames.size() != depth_filenames.size()) {
        utility::LogError(
                "[VoxelHashing] numbers of color and depth files mismatch. "
                "Please provide folders with same number of images.");
    }

    // Intrinsics
    std::string intrinsic_path = utility::GetProgramOptionAsString(
            argc, argv, "--intrinsic_path", "");
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    if (intrinsic_path.empty()) {
        utility::LogWarning("Using default Primesense intrinsics");
    } else if (!io::ReadIJsonConvertible(intrinsic_path, intrinsic)) {
        utility::LogError("Unable to convert json to intrinsics.");
    }

    // Device
    std::string device_code = "CPU:0";
    if (utility::ProgramOptionExists(argc, argv, "--device")) {
        device_code = utility::GetProgramOptionAsString(argc, argv, "--device");
    }
    core::Device device(device_code);
    utility::LogInfo("Using device: {}", device.ToString());

    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    Tensor intrinsic_t =
            Tensor::Init<float>({{static_cast<float>(focal_length.first), 0,
                                  static_cast<float>(principal_point.first)},
                                 {0, static_cast<float>(focal_length.second),
                                  static_cast<float>(principal_point.second)},
                                 {0, 0, 1}},
                                device);

    int block_count =
            utility::GetProgramOptionAsInt(argc, argv, "--block_count", 1000);

    float voxel_size = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--voxel_size", 3.f / 512.f));
    float depth_scale = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_scale", 1000.f));
    float depth_diff = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_diff", 0.07));
    float sdf_trunc = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--sdf_trunc", 0.04f));

    t::geometry::TSDFVoxelGrid voxel_grid({{"tsdf", core::Dtype::Float32},
                                           {"weight", core::Dtype::UInt16},
                                           {"color", core::Dtype::UInt16}},
                                          voxel_size, sdf_trunc, 16,
                                          block_count, device);

    size_t n = color_filenames.size();
    size_t iterations = static_cast<size_t>(
            utility::GetProgramOptionAsInt(argc, argv, "--iterations", n));

    Tensor T_curr_to_model =
            Tensor::Eye(4, core::Dtype::Float32, core::Device("CPU:0"));

    for (size_t i = 0; i < iterations - 1; ++i) {
        if (i < 83) continue;

        utility::LogInfo("i = {}", i);

        // Load image
        t::geometry::Image src_depth =
                *t::io::CreateImageFromFile(depth_filenames[i]);

        t::geometry::Image dst_depth =
                *t::io::CreateImageFromFile(depth_filenames[i + 1]);

        t::geometry::RGBDImage src, dst;
        src.depth_ = src_depth.To(device).To(core::Dtype::Float32, false, 1.0);
        dst.depth_ = dst_depth.To(device).To(core::Dtype::Float32, false, 1.0);

        core::Tensor trans = core::Tensor::Eye(4, core::Dtype::Float64, device);

        // Visualize before odometry
        auto source_pcd = std::make_shared<open3d::geometry::PointCloud>(
                t::geometry::PointCloud::CreateFromDepthImage(
                        src.depth_, intrinsic_t, trans, depth_scale)
                        .ToLegacyPointCloud());
        source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
        auto target_pcd = std::make_shared<open3d::geometry::PointCloud>(
                t::geometry::PointCloud::CreateFromDepthImage(
                        dst.depth_, intrinsic_t, trans, depth_scale)
                        .ToLegacyPointCloud());
        target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
        visualization::DrawGeometries({source_pcd, target_pcd});

        trans = t::pipelines::odometry::RGBDOdometryMultiScale(
                src, dst, intrinsic_t, trans, depth_scale, depth_diff,
                {10, 5, 3});

        // Visualize after odometry
        source_pcd = std::make_shared<open3d::geometry::PointCloud>(
                t::geometry::PointCloud::CreateFromDepthImage(
                        src.depth_, intrinsic_t, trans.Inverse(), depth_scale)
                        .ToLegacyPointCloud());
        source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
        target_pcd = std::make_shared<open3d::geometry::PointCloud>(
                t::geometry::PointCloud::CreateFromDepthImage(
                        dst.depth_, intrinsic_t,
                        core::Tensor::Eye(4, core::Dtype::Float32, device),
                        depth_scale)
                        .ToLegacyPointCloud());
        target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
        visualization::DrawGeometries({source_pcd, target_pcd});
    }
}
