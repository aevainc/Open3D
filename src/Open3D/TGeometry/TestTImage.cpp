#include "Image.h"
#include "Open3D/Open3D.h"

using namespace open3d;

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
