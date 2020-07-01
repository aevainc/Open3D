#include "open3d/Open3D.h"
#include "open3d/tgeometry/Image.h"
#include "open3d/tgeometry/PointCloud.h"

using namespace open3d;
using namespace open3d::core;

int main(int argc, char** argv) {
    std::shared_ptr<geometry::Image> im_legacy =
            io::CreateImageFromFile(argv[1]);
    auto depth_legacy = im_legacy->ConvertDepthToFloatImage();

    utility::LogInfo("From legacy image");
    tgeometry::Image im =
            tgeometry::Image::FromLegacyImage(*depth_legacy, Device("CUDA:0"));

    Tensor intrinsic = Tensor(
            std::vector<float>({525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1}),
            {3, 3}, Dtype::Float32);
    Tensor vertex_map = im.Unproject(intrinsic);

    Tensor pcd_map = vertex_map.View({3, 480 * 640});
    tgeometry::PointCloud pcd(pcd_map.T());

    Tensor transform = Tensor(std::vector<float>({1, 0, 0, 1, 0, -1, 0, 2, 0, 0,
                                                  -1, 3, 0, 0, 0, 1}),
                              {4, 4}, Dtype::Float32, Device("CUDA:0"));
    pcd.Transform(transform);

    auto pcd_legacy = std::make_shared<geometry::PointCloud>(
            tgeometry::PointCloud::ToLegacyPointCloud(pcd));
    visualization::DrawGeometries({pcd_legacy});
}
