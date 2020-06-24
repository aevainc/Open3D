#include "open3d/Open3D.h"
#include "open3d/tgeometry/Image.h"

using namespace open3d;
using namespace open3d::core;

int main(int argc, char** argv) {
    std::shared_ptr<geometry::Image> im_legacy =
            io::CreateImageFromFile(argv[1]);

    utility::LogInfo("From legacy image");
    tgeometry::Image im = tgeometry::Image::FromLegacyImage(*im_legacy);

    utility::LogInfo("To legacy image");
    geometry::Image im_legacy_converted = tgeometry::Image::ToLegacyImage(im);

    utility::LogInfo("Print");
    std::cout << im.AsTensor().ToString() << "\n";
    std::cout << im.AsTensor()
                         .To(Dtype::Float32)
                         .Mean(SizeVector({0, 1, 2}))
                         .ToString()
              << "\n";
}
