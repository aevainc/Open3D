#include "Open3D/Open3D.h"
#include "PointCloud.h"

using namespace open3d;
int main(int argc, char** argv) {
    utility::Timer timer;

    auto pcd_legacy = io::CreatePointCloudFromFile(argv[1]);
    auto pcd = tgeometry::PointCloud::FromLegacyPointCloud(
            *pcd_legacy, Dtype::Float32, Device("CUDA:0"));

    timer.Start();
    pcd_legacy->VoxelDownSample(0.01);
    timer.Stop();
    utility::LogInfo("PointCloud time: {}", timer.GetDuration());

    timer.Start();
    auto pcd_down = pcd.VoxelDownSample(0.01);
    timer.Stop();
    utility::LogInfo("TPointCloud time: {}", timer.GetDuration());

    auto pcd_down_legacy = tgeometry::PointCloud::ToLegacyPointCloud(pcd_down);
    visualization::DrawGeometries({pcd_down_legacy});
}
