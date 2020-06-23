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
    utility::LogInfo("[TestTPointCloud] Legacy VoxelDownSample time: {}",
                     timer.GetDuration());

    for (int i = 0; i < 10; ++i) {
        if (i % 3 == 0) {
            MemoryManager::ReleaseCache(Device("CUDA:0"));
        }
        timer.Start();
        auto pcd_down = pcd.VoxelDownSample(0.01);
        timer.Stop();
        utility::LogInfo("[TestTPointCloud] VoxelDownSample: {}",
                         timer.GetDuration());
    }

    // auto pcd_down_legacy = std::make_shared<geometry::PointCloud>(
    //         tgeometry::PointCloud::ToLegacyPointCloud(pcd_down));

    // utility::LogInfo("pcd size {}", pcd_legacy->points_.size());
    // utility::LogInfo("pcd down size {}", pcd_down_legacy->points_.size());
    // visualization::DrawGeometries({pcd_down_legacy});
}
