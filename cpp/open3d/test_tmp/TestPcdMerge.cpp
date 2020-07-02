#include <fmt/format.h>
#include "open3d/Open3D.h"
#include "open3d/tgeometry/Image.h"
#include "open3d/tgeometry/PointCloud.h"

using namespace open3d;
using namespace open3d::core;

int main(int argc, char** argv) {
    std::string root_path = argv[1];

    Tensor intrinsic = Tensor(
            std::vector<float>({525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1}),
            {3, 3}, Dtype::Float32);

    auto trajectory = io::CreatePinholeCameraTrajectoryFromFile(
            fmt::format("{}/trajectory.log", root_path));

    auto pcd_vis = std::make_shared<geometry::PointCloud>();

    for (int i = 0; i < 100; ++i) {
        std::string image_path =
                fmt::format("{}/depth/{:06d}.png", root_path, i + 1);

        std::shared_ptr<geometry::Image> im_legacy =
                io::CreateImageFromFile(image_path);

        auto depth_legacy = im_legacy->ConvertDepthToFloatImage();
        tgeometry::Image im = tgeometry::Image::FromLegacyImage(
                *depth_legacy, Device("CUDA:0"));
        Tensor vertex_map = im.Unproject(intrinsic);
        Tensor pcd_map = vertex_map.View({3, 480 * 640});
        tgeometry::PointCloud pcd(pcd_map.T());

        Eigen::Matrix4d extrinsicd =
                trajectory->parameters_[i].extrinsic_.inverse();
        Eigen::Matrix4f extrinsic = extrinsicd.cast<float>();
        Tensor transform = Tensor(
                std::vector<float>(
                        {extrinsic(0, 0), extrinsic(0, 1), extrinsic(0, 2),
                         extrinsic(0, 3), extrinsic(1, 0), extrinsic(1, 1),
                         extrinsic(1, 2), extrinsic(1, 3), extrinsic(2, 0),
                         extrinsic(2, 1), extrinsic(2, 2), extrinsic(2, 3),
                         extrinsic(3, 0), extrinsic(3, 1), extrinsic(3, 2),
                         extrinsic(3, 3)}),
                {4, 4}, Dtype::Float32, Device("CUDA:0"));
        pcd.Transform(transform);

        tgeometry::PointCloud pcd_down = pcd.VoxelDownSample(0.05);
        *pcd_vis += *std::make_shared<geometry::PointCloud>(
                tgeometry::PointCloud::ToLegacyPointCloud(pcd_down));

        if (i % 30 == 0) {
            visualization::DrawGeometries({pcd_vis});

            // auto pcd_legacy = std::make_shared<geometry::PointCloud>(
            //         tgeometry::PointCloud::ToLegacyPointCloud(global_pcd));
            // visualization::DrawGeometries({pcd_legacy});
        }
    }
}
