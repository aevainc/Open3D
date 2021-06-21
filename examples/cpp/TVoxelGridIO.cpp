// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::core;

int main(int argc, char *argv[]) {
    using namespace open3d;

    std::string filename =
            utility::GetProgramOptionAsString(argc, argv, "--tsdf_input");

    t::geometry::TSDFVoxelGrid voxel_grid;
    t::io::ReadTSDFVoxelGrid(filename, voxel_grid);

    auto voxel_grid_upsampled = voxel_grid.Upsample();

    auto pcd = voxel_grid.ExtractSurfacePoints(-1, 3.0f);
    auto pcd_upsampled = voxel_grid_upsampled.ExtractSurfacePoints(-1, 3.0f);
    visualization::Draw(
            {std::make_shared<geometry::PointCloud>(pcd.ToLegacyPointCloud()),
             std::make_shared<geometry::PointCloud>(
                     pcd_upsampled.ToLegacyPointCloud())});
}
