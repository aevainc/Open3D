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

#include "open3d/tgeometry/PointCloud.h"

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "open3d/core/EigenConverter.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"
#include "open3d/core/hashmap/TensorHash.h"
#include "open3d/core/linalg/Matmul.h"

namespace open3d {
namespace tgeometry {

PointCloud::PointCloud(core::Dtype dtype, const core::Device &device)
    : Geometry(Geometry::GeometryType::PointCloud, 3),
      device_(device),
      point_attr_(TensorListMap("points")) {
    SetPoints(core::TensorList({3}, dtype, device_));
}

PointCloud::PointCloud(const core::TensorList &points)
    : PointCloud(points.GetDtype(), points.GetDevice()) {
    points.AssertElementShape({3});
    SetPoints(points);
}

PointCloud::PointCloud(const std::unordered_map<std::string, core::TensorList>
                               &map_keys_to_tensorlists)
    : PointCloud(map_keys_to_tensorlists.at("points").GetDtype(),
                 map_keys_to_tensorlists.at("points").GetDevice()) {
    map_keys_to_tensorlists.at("points").AssertElementShape({3});
    point_attr_.Assign(map_keys_to_tensorlists);
}

core::Tensor PointCloud::GetMinBound() const {
    return GetPoints().AsTensor().Min({0});
}

core::Tensor PointCloud::GetMaxBound() const {
    return GetPoints().AsTensor().Max({0});
}

core::Tensor PointCloud::GetCenter() const {
    return GetPoints().AsTensor().Mean({0});
}

PointCloud &PointCloud::Transform(const core::Tensor &transformation) {
    core::Tensor R = transformation.Slice(0, 0, 3).Slice(1, 0, 3);
    core::Tensor t = transformation.Slice(0, 0, 3).Slice(1, 3, 4);
    core::Tensor points_transformed;
    core::Matmul(GetPoints().AsTensor(), R.T(), points_transformed);
    points_transformed += t.T();
    GetPoints().AsTensor() = points_transformed;
    return *this;
}

PointCloud &PointCloud::Translate(const core::Tensor &translation,
                                  bool relative) {
    translation.AssertShape({3});
    core::Tensor transform = translation.Copy();
    if (!relative) {
        transform -= GetCenter();
    }
    GetPoints().AsTensor() += transform;
    return *this;
}

PointCloud &PointCloud::Scale(double scale, const core::Tensor &center) {
    center.AssertShape({3});
    core::Tensor points = GetPoints().AsTensor();
    points.Sub_(center).Mul_(scale).Add_(center);
    return *this;
}

PointCloud &PointCloud::Rotate(const core::Tensor &R,
                               const core::Tensor &center) {
    utility::LogError("Unimplemented");
    return *this;
}

tgeometry::PointCloud PointCloud::FromLegacyPointCloud(
        const geometry::PointCloud &pcd_legacy,
        core::Dtype dtype,
        const core::Device &device) {
    tgeometry::PointCloud pcd(dtype, device);
    if (pcd_legacy.HasPoints()) {
        pcd.SetPoints(core::eigen_converter::EigenVector3dVectorToTensorList(
                pcd_legacy.points_, dtype, device));
    } else {
        utility::LogWarning(
                "Creating from an empty legacy pointcloud, an empty pointcloud "
                "with default dtype and device will be created.");
    }
    if (pcd_legacy.HasColors()) {
        pcd.SetPointColors(
                core::eigen_converter::EigenVector3dVectorToTensorList(
                        pcd_legacy.colors_, dtype, device));
    }
    if (pcd_legacy.HasNormals()) {
        pcd.SetPointNormals(
                core::eigen_converter::EigenVector3dVectorToTensorList(
                        pcd_legacy.normals_, dtype, device));
    }
    return pcd;
}

geometry::PointCloud PointCloud::ToLegacyPointCloud() const {
    geometry::PointCloud pcd_legacy;
    if (HasPoints()) {
        const core::TensorList &points = GetPoints();
        for (int64_t i = 0; i < points.GetSize(); i++) {
            pcd_legacy.points_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(points[i]));
        }
    }
    if (HasPointColors()) {
        const core::TensorList &colors = GetPointColors();
        for (int64_t i = 0; i < colors.GetSize(); i++) {
            pcd_legacy.colors_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(colors[i]));
        }
    }
    if (HasPointNormals()) {
        const core::TensorList &normals = GetPointNormals();
        for (int64_t i = 0; i < normals.GetSize(); i++) {
            pcd_legacy.normals_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(normals[i]));
        }
    }
    return pcd_legacy;
}

geometry::TriangleMesh PointCloud::ToLegacyTriangleMesh() const {
    utility::LogInfo("mesh");
    geometry::TriangleMesh mesh_legacy;

    if (HasPoints()) {
        const core::TensorList &points = GetPoints();
        utility::LogInfo("points = {}", points.AsTensor().GetShape());
        for (int64_t i = 0; i < points.GetSize(); i++) {
            mesh_legacy.vertices_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(points[i]));
        }
    }
    if (HasPointColors()) {
        const core::TensorList &colors = GetPointColors();
        utility::LogInfo("colors = {}", colors.AsTensor().GetShape());
        for (int64_t i = 0; i < colors.GetSize(); i++) {
            mesh_legacy.vertex_colors_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(colors[i]));
        }
    }
    if (HasPointNormals()) {
        const core::TensorList &normals = GetPointNormals();
        utility::LogInfo("normals = {}", normals.AsTensor().GetShape());
        for (int64_t i = 0; i < normals.GetSize(); i++) {
            mesh_legacy.vertex_normals_.push_back(
                    core::eigen_converter::TensorToEigenVector3d(normals[i]));
        }
    }

    const core::TensorList &triangles = GetPointAttr("triangles");
    for (int64_t i = 0; i < triangles.GetSize(); i++) {
        mesh_legacy.triangles_.push_back(
                core::eigen_converter::TensorToEigenVector3i(triangles[i]));
    }

    return mesh_legacy;
}

PointCloud PointCloud::VoxelDownSample(
        float voxel_size,
        const std::unordered_set<std::string> &properties_to_skip) const {
    auto tensor_quantized =

            point_attr_.find("points")->second.AsTensor() / voxel_size;

    auto tensor_quantized_int64 = tensor_quantized.To(core::Dtype::Int64);

    auto result = core::TensorHash::Unique(tensor_quantized_int64);

    core::Tensor coords = result.first;
    core::Tensor masks = result.second;
    auto pcd_down_map = std::unordered_map<std::string, core::TensorList>();
    auto tl_pts = core::TensorList::FromTensor(
            coords.IndexGet({masks}).To(core::Dtype::Float32),
            /* inplace = */ false);

    pcd_down_map.emplace(std::make_pair("points", tl_pts));
    for (auto kv : point_attr_) {
        if (kv.first != "points" &&
            properties_to_skip.find(kv.first) == properties_to_skip.end()) {
            auto tl = core::TensorList::FromTensor(
                    kv.second.AsTensor().IndexGet({masks}), false);
            pcd_down_map.emplace(std::make_pair(kv.first, tl));
        }
    }

    PointCloud pcd_down(pcd_down_map);

    return pcd_down;
}
}  // namespace tgeometry
}  // namespace open3d
