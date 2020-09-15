#include "open3d/Open3D.h"
#include "open3d/tgeometry/PointCloud.h"

using namespace open3d;
using namespace open3d::core;

int main(int argc, char** argv) {
    utility::Timer timer;

    auto pcd_legacy = io::CreatePointCloudFromFile(argv[1]);

    auto pcd = tgeometry::PointCloud::FromLegacyPointCloud(
            *pcd_legacy, Dtype::Float32, Device("CUDA:0"));
    timer.Start();
    pcd_legacy->VoxelDownSample(0.01);
    timer.Stop();
    utility::LogInfo("[TestTPointCloud] Legacy VoxelDownSample time: {}",
                     timer.GetDuration());

    for (int i = 0; i < 10; ++i) {
        timer.Start();
        auto pcd_down = pcd.VoxelDownSample(0.05);
        timer.Stop();
        utility::LogInfo("[TestTPointCloud] VoxelDownSample: {}",
                         timer.GetDuration());

        auto pcd_down_legacy = std::make_shared<geometry::PointCloud>(
                pcd_down.ToLegacyPointCloud());
        visualization::DrawGeometries({pcd_down_legacy});
    }
}
