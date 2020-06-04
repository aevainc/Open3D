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

#pragma once

#include <Eigen/Core>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "Open3D/Core/Tensor.h"
#include "Open3D/Core/TensorList.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/TGeometry/Geometry3D.h"

namespace open3d {
namespace tgeometry {

class VoxelGrid : public Geometry3D {
public:
    /// \brief Default Constructor.
    VoxelGrid() : Geometry3D(Geometry::GeometryType::VoxelGrid) {}

    ~VoxelGrid() override{};

    std::unordered_map<std::string, TensorList> GetVoxels(
            const Tensor coords &);

protected:
    // Map from 3D coords to Int64 indices
    // Usage:
    // Tensor indices = coord_map_.Query(coords);
    // Tensor values_get =
    //         voxel_dict_["desired_property"].AsTensor().IndexGet({indices});
    // voxel_dict_["desired_property"].AsTensor().IndexSet({indices},
    // values_set);

    TensorHash coord_map_;
    std::unordered_map<std::string, TensorList> voxel_dict_;
}
}  // namespace tgeometry
}  // namespace open3d
