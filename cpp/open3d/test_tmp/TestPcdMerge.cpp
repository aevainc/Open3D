#include <fmt/format.h>
#include "open3d/Open3D.h"
#include "open3d/core/EigenAdaptor.h"
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

    std::vector<Device> devices{Device("CPU:0"), Device("CUDA:0")};

    for (auto device : devices) {
        tgeometry::PointCloud pcd_global(Dtype::Float32, device);
        for (int i = 0; i < 300; ++i) {
            /// Load image
            std::string image_path =
                    fmt::format("{}/depth/{:06d}.png", root_path, i + 1);
            std::shared_ptr<geometry::Image> im_legacy =
                    io::CreateImageFromFile(image_path);
            auto depth_legacy = im_legacy->ConvertDepthToFloatImage();
            tgeometry::Image depth =
                    tgeometry::Image::FromLegacyImage(*depth_legacy, device);

            /// Unproject
            Tensor vertex_map = depth.Unproject(intrinsic);
            Tensor pcd_map = vertex_map.View({3, 480 * 640});
            tgeometry::PointCloud pcd(pcd_map.T());

            /// Transform
            Eigen::Matrix4f extrinsic = trajectory->parameters_[i]
                                                .extrinsic_.inverse()
                                                .cast<float>();
            Tensor transform = FromEigen(extrinsic).Copy(device);
            pcd.Transform(transform);

            /// Downsample and append
            tgeometry::PointCloud pcd_down = pcd.VoxelDownSample(0.05);
            pcd_global.point_dict_.at("points") +=
                    pcd_down.point_dict_.at("points");
        }

        auto pcd_vis = std::make_shared<geometry::PointCloud>(
                tgeometry::PointCloud::ToLegacyPointCloud(
                        pcd_global.VoxelDownSample(0.05)));
        visualization::DrawGeometries({pcd_vis});
    }
}
