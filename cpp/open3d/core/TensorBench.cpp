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

#include "open3d/core/TensorBench.h"

#include "open3d/Open3D.h"
#include "open3d/core/Device.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TSDFVoxelGrid.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Timer.h"

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/CUDAState.cuh"
#endif

namespace open3d {
namespace core {

// static std::vector<Device> PermuteDevices() {
// #ifdef BUILD_CUDA_MODULE
//     std::shared_ptr<core::CUDAState> cuda_state =
//             core::CUDAState::GetInstance();
//     if (cuda_state->GetNumDevices() >= 1) {
//         return {core::Device("CPU:0"), core::Device("CUDA:0")};
//     } else {
//         return {core::Device("CPU:0")};
//     }
// #else
//     return {core::Device("CPU:0")};
// #endif
// }

template <typename func_t>
static void RunBenchmark(func_t benchmark_func,
                         const std::string& name,
                         int repeats) {
    if (repeats <= 0) {
        utility::LogError("repeats must be > 0");
    }
    benchmark_func();  // Warm up.

    utility::Timer timer;
    timer.Start();
    for (int i = 0; i < repeats; i++) {
        benchmark_func();
    }
    timer.Stop();
    double avg_time = timer.GetDuration() / static_cast<double>(repeats);
    utility::LogInfo("Name: {}; Avg time: {:.2f}ms; Repeats: {}", name,
                     avg_time, repeats);
}

template <class T, int M, int N, int A>
static Tensor FromEigen(const Eigen::Matrix<T, M, N, A>& matrix) {
    Dtype dtype = Dtype::FromType<T>();
    Eigen::Matrix<T, M, N, Eigen::RowMajor> matrix_row_major = matrix;
    return Tensor(matrix_row_major.data(), {matrix.rows(), matrix.cols()},
                  dtype);
}

void RunTensorBench() {
    // Hard-coded paths
    std::string color_folder =
            "/home/yixing/data/stanford/lounge/lounge_png/color";
    std::string depth_folder =
            "/home/yixing/data/stanford/lounge/lounge_png/depth";
    std::string trajectory_path =
            "/home/yixing/data/stanford/lounge/trajectory.log";
    std::string device_code = "CPU:0";

    // Color and depth
    std::vector<std::string> color_filenames;
    utility::filesystem::ListFilesInDirectory(color_folder, color_filenames);
    std::sort(color_filenames.begin(), color_filenames.end());

    std::vector<std::string> depth_filenames;
    utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    std::sort(depth_filenames.begin(), depth_filenames.end());

    if (color_filenames.size() != depth_filenames.size()) {
        utility::LogError(
                "[TIntegrateRGBD] numbers of color and depth files mismatch. "
                "Please provide folders with same number of images.");
    }

    // Trajectory
    auto trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_path);

    // Intrinsics
    std::string intrinsic_path = "";
    camera::PinholeCameraIntrinsic intrinsic;
    if (intrinsic_path.empty() ||
        !io::ReadIJsonConvertible(intrinsic_path, intrinsic)) {
        utility::LogWarning("Using default value for Primesense camera.");
        intrinsic = camera::PinholeCameraIntrinsic(
                camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    }

    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    Tensor intrinsic_t = Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, Dtype::Float32);

    core::Device device(device_code);
    utility::LogInfo("Using device: {}", device.ToString());

    t::geometry::TSDFVoxelGrid voxel_grid({{"tsdf", core::Dtype::Float32},
                                           {"weight", core::Dtype::UInt16},
                                           {"color", core::Dtype::UInt16}},
                                          3.0f / 512.f, 0.04f, 16, 100, device);

    for (size_t i = 0; i < 10; ++i) {
        // Load image
        std::shared_ptr<geometry::Image> depth_legacy =
                io::CreateImageFromFile(depth_filenames[i]);
        std::shared_ptr<geometry::Image> color_legacy =
                io::CreateImageFromFile(color_filenames[i]);

        t::geometry::Image depth =
                t::geometry::Image::FromLegacyImage(*depth_legacy, device);
        t::geometry::Image color =
                t::geometry::Image::FromLegacyImage(*color_legacy, device);

        Eigen::Matrix4f extrinsic =
                trajectory->parameters_[i].extrinsic_.cast<float>();
        Tensor extrinsic_t = FromEigen(extrinsic).Copy(device);

        utility::Timer timer;
        timer.Start();
        voxel_grid.Integrate(depth, color, intrinsic_t, extrinsic_t);
        timer.Stop();
        utility::LogInfo("{}: Integration takes {}", i, timer.GetDuration());
    }
}

}  // namespace core
}  // namespace open3d
