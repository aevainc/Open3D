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
    tgeometry::Image im = tgeometry::Image::FromLegacyImage(*depth_legacy);
    // im.data_ = im.AsTensor().To(Dtype::Float32) / 1000.0f;
    // im.data_ = Tensor::Ones({1, 480, 640}, Dtype::Float32);
    // std::cout << im.AsTensor().ToString() << "\n";

    Tensor intrinsic = Tensor(
            std::vector<float>({525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1}),
            {3, 3}, Dtype::Float32);
    Tensor vertex_map = im.Unproject(intrinsic);
    std::cout << vertex_map.ToString() << "\n";

    Tensor pcd_map = vertex_map.View({3, 480 * 640});
    tgeometry::PointCloud pcd(pcd_map.T());

    auto pcd_legacy = std::make_shared<geometry::PointCloud>(
            tgeometry::PointCloud::ToLegacyPointCloud(pcd));
    visualization::DrawGeometries({pcd_legacy});

    // utility::LogInfo("To legacy image");
    // geometry::Image im_legacy_converted =
    // tgeometry::Image::ToLegacyImage(im);

    // utility::LogInfo("Print");
    // std::cout << im.AsTensor().ToString() << "\n";
    // std::cout << im.AsTensor()
    //                      .To(Dtype::Float32)
    //                      .Mean(SizeVector({0, 1, 2}))
    //                      .ToString()
    //           << "\n";
}
