#include "Open3D/Open3D.h"
#include "PointCloud.h"

using namespace open3d;
int main(int argc, char** argv) {
    auto pcd_legacy = io::CreatePointCloudFromFile(argv[1]);
    auto pcd = tgeometry::PointCloud::FromLegacyPointCloud(*pcd_legacy);
    auto pcd_down = pcd.VoxelDownSample(0.05, {"colors"});
    auto pcd_down_legacy = tgeometry::PointCloud::ToLegacyPointCloud(pcd_down);
    visualization::DrawGeometries({pcd_down_legacy});
}
