#include <fmt/format.h>

#include "open3d/Open3D.h"
#include "open3d/core/EigenAdaptor.h"
#include "open3d/tgeometry/Image.h"
#include "open3d/tgeometry/PointCloud.h"
#include "open3d/tgeometry/VoxelGrid.h"

using namespace open3d;
using namespace open3d::core;

int main(int argc, char** argv) {
    std::string root_path = argv[1];

    Tensor intrinsic = Tensor(
            std::vector<float>({525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1}),
            {3, 3}, Dtype::Float32);

    auto trajectory = io::CreatePinholeCameraTrajectoryFromFile(
            fmt::format("{}/trajectory.log", root_path));

    std::vector<Device> devices{Device("CUDA:0")};

    for (auto device : devices) {
        tgeometry::VoxelGrid voxel_grid(0.05, 16, 10000, device);
        for (int i = 0; i < 3000; ++i) {
            /// Load image
            std::string image_path =
                    fmt::format("{}/depth/{:06d}.png", root_path, i + 1);
            std::shared_ptr<geometry::Image> im_legacy =
                    io::CreateImageFromFile(image_path);
            auto depth_legacy = im_legacy->ConvertDepthToFloatImage();
            tgeometry::Image depth =
                    tgeometry::Image::FromLegacyImage(*depth_legacy, device);
            Eigen::Matrix4f pose_ = trajectory->parameters_[i]
                                            .extrinsic_.inverse()
                                            .cast<float>();
            Tensor pose = FromEigen(pose_).Copy(device);

            voxel_grid.Integrate(depth, intrinsic, pose);
        }
    }
}
