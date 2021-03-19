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
#include <math.h>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::core;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    TOdometryRGBDSeq [depth_folder] [gt log] [options]");
    utility::LogInfo("     Given depth images, evaluate point-to-plane icp performance");
    utility::LogInfo("     [options]");
    utility::LogInfo("     --intrinsic_path [camera_intrinsic]");
    utility::LogInfo("     --iterations [=10 (5, 3)]");
    utility::LogInfo("     --depth_scale [=1000.0]");
    utility::LogInfo("     --depth_diff [=0.07]");
    utility::LogInfo("     --max_depth [=3.0]");
    utility::LogInfo("     --device [=CPU:0]");
    utility::LogInfo("     --output [=output.npy]");
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

    // Depth
    std::string depth_folder = std::string(argv[1]);

    std::vector<std::string> depth_filenames;
    utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    std::sort(depth_filenames.begin(), depth_filenames.end());

    // Trajectory
    std::string gt_trajectory_path = std::string(argv[2]);
    auto gt_trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(gt_trajectory_path);

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

    // Intrinsics
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    Tensor intrinsic_t =
            Tensor::Init<float>({{static_cast<float>(focal_length.first), 0,
                                  static_cast<float>(principal_point.first)},
                                 {0, static_cast<float>(focal_length.second),
                                  static_cast<float>(principal_point.second)},
                                 {0, 0, 1}},
                                device);

    float depth_scale = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_scale", 1000.f));
    float depth_diff = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_diff", 0.07));
    float depth_max = static_cast<float>(
            utility::GetProgramOptionAsDouble(argc, argv, "--max_depth", 3.0));
    bool debug = utility::ProgramOptionExists(argc, argv, "--debug");

    int iterations =
            utility::GetProgramOptionAsInt(argc, argv, "--iterations", 10);

    int begin = utility::GetProgramOptionAsInt(argc, argv, "--begin", 0);
    int end = utility::GetProgramOptionAsInt(argc, argv, "--end",
                                             depth_filenames.size() - 1);
    end = std::min(end, static_cast<int>(depth_filenames.size() - 1));

    core::Tensor diffs = core::Tensor::Empty(
            {end - begin, 2}, core::Dtype::Float64, core::Device("CPU:0"));
    for (int i = begin; i < end; ++i) {
        utility::LogInfo("i = {}", i);

        // Load image
        t::geometry::Image src_depth =
                *t::io::CreateImageFromFile(depth_filenames[i]);
        t::geometry::Image dst_depth =
                *t::io::CreateImageFromFile(depth_filenames[i + 1]);

        Eigen::Matrix4d src_pose_gt_eigen =
                gt_trajectory->parameters_[i].extrinsic_.inverse().eval();
        Tensor src_pose_gt =
                core::eigen_converter::EigenMatrixToTensor(src_pose_gt_eigen);
        Eigen::Matrix4d dst_pose_gt_eigen =
                gt_trajectory->parameters_[i + 1].extrinsic_.inverse().eval();
        Tensor dst_pose_gt =
                core::eigen_converter::EigenMatrixToTensor(dst_pose_gt_eigen);
        Tensor trans_gt = dst_pose_gt.Inverse().Matmul(src_pose_gt);

        t::geometry::RGBDImage src, dst;
        src.depth_ = src_depth.To(device).To(core::Dtype::Float32, false, 1.0);
        dst.depth_ = dst_depth.To(device).To(core::Dtype::Float32, false, 1.0);

        core::Tensor trans = core::Tensor::Eye(4, core::Dtype::Float64, device);

        // Visualize before odometry
        if (debug) {
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
        }

        utility::Timer timer;
        timer.Start();
        trans = t::pipelines::odometry::RGBDOdometryMultiScale(
                src, dst, intrinsic_t, trans, depth_scale, depth_diff,
                depth_max, {iterations, 0, 0});
        // {iterations, static_cast<int>(std::ceil(iterations * 0.5)),
        // static_cast<int>(std::ceil(iterations * 0.25))});
        timer.Stop();
        utility::LogInfo("est: {}", trans.ToString());
        utility::LogInfo("gt: {}", trans_gt.ToString());

        Tensor diff = trans_gt.Inverse().Matmul(trans);

        double rot_err = std::acos(0.5 * (diff[0][0].Item<double>() +
                                          diff[1][1].Item<double>() +
                                          diff[2][2].Item<double>() - 1));
        double trans_err = std::sqrt(
                diff[0][3].Item<double>() * diff[0][3].Item<double>() +
                diff[1][3].Item<double>() * diff[1][3].Item<double>() +
                diff[2][3].Item<double>() * diff[2][3].Item<double>());
        diffs[i - begin][0] = rot_err;
        diffs[i - begin][1] = trans_err;
        utility::LogInfo("T_diff = {}", diff.ToString());
        utility::LogInfo("rot_err = {}, trans_err = {}", rot_err, trans_err);

        // Visualize after odometry
        if (debug) {
            auto source_pcd = std::make_shared<open3d::geometry::PointCloud>(
                    t::geometry::PointCloud::CreateFromDepthImage(
                            src.depth_, intrinsic_t, trans.Inverse(),
                            depth_scale)
                            .ToLegacyPointCloud());
            source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
            auto target_pcd = std::make_shared<open3d::geometry::PointCloud>(
                    t::geometry::PointCloud::CreateFromDepthImage(
                            dst.depth_, intrinsic_t,
                            core::Tensor::Eye(4, core::Dtype::Float32, device),
                            depth_scale)
                            .ToLegacyPointCloud());
            target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
            visualization::DrawGeometries({source_pcd, target_pcd});
        }
    }

    std::string diffs_name = utility::GetProgramOptionAsString(
            argc, argv, "--output",
            fmt::format("it_{}_diff_{}.npy", iterations, depth_diff));
    diffs.Save(diffs_name);

    return 0;
}
