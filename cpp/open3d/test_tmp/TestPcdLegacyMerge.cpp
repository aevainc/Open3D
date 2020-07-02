#include <fmt/format.h>
#include "open3d/Open3D.h"
#include "open3d/tgeometry/Image.h"
#include "open3d/tgeometry/PointCloud.h"

using namespace open3d;
using namespace open3d::core;

int main(int argc, char** argv) {
    std::string root_path = argv[1];

    camera::PinholeCameraIntrinsic intrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto trajectory = io::CreatePinholeCameraTrajectoryFromFile(
            fmt::format("{}/trajectory.log", root_path));

    std::shared_ptr<geometry::PointCloud> global_pcd =
            std::make_shared<geometry::PointCloud>();

    for (int i = 0; i < 1; ++i) {
        std::string image_path =
                fmt::format("{}/depth/{:06d}.png", root_path, i + 1);

        std::shared_ptr<geometry::Image> im_legacy =
                io::CreateImageFromFile(image_path);

        auto depth_legacy = im_legacy->ConvertDepthToFloatImage();

        Eigen::Matrix4d extrinsic = trajectory->parameters_[i].extrinsic_;

        auto pcd = geometry::PointCloud::CreateFromDepthImage(*depth_legacy,
                                                              intrinsic);
        std::cout << pcd->points_[0] << "\n";
        pcd->Transform(extrinsic.inverse());
        std::cout << pcd->points_[0] << "\n";
        auto pcd_down = pcd->VoxelDownSample(0.05);

        *global_pcd += *pcd_down;
        // Tensor vertex_map = im.Unproject(intrinsic);
        // Tensor pcd_map = vertex_map.View({3, 480 * 640});
        // tgeometry::PointCloud pcd(pcd_map.T());
        // tgeometry::PointCloud pcd_down = pcd.VoxelDownSample(0.05);

        // Tensor transform = Tensor(
        //         std::vector<float>(
        //                 {extrinsic(0, 0), extrinsic(0, 1), extrinsic(0, 2),
        //                  extrinsic(0, 3), extrinsic(1, 0), extrinsic(1, 1),
        //                  extrinsic(1, 2), extrinsic(1, 3), extrinsic(2, 0),
        //                  extrinsic(2, 1), extrinsic(2, 2), extrinsic(2, 3),
        //                  extrinsic(3, 0), extrinsic(3, 1), extrinsic(3, 2),
        //                  extrinsic(3, 3)}),
        //         {4, 4}, Dtype::Float32, Device("CUDA:0"));
        // pcd_down.Transform(transform);
        // global_pcd.point_dict_.at("points") +=
        //         TensorList(pcd_down.point_dict_.at("points"));

        if (i % 30 == 0) {
            // auto pcd_legacy = std::make_shared<geometry::PointCloud>(
            //         tgeometry::PointCloud::ToLegacyPointCloud(global_pcd));
            visualization::DrawGeometries({global_pcd});
        }
    }
}
