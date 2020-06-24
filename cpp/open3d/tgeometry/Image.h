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

#include <memory>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/tgeometry/Geometry2D.h"
#include "open3d/utility/Console.h"

namespace open3d {
// Legacy class
namespace geometry {
class Image;
}

namespace tgeometry {

/// \class Image
///
/// \brief The Image class stores image with customizable width, height, num of
/// channels and bytes per channel.
class Image : public Geometry2D {
public:
    /// \enum ColorToIntensityConversionType
    ///
    /// \brief Specifies whether R, G, B channels have the same weight when
    /// converting to intensity. Only used for Image with 3 channels.
    ///
    /// When `Weighted` is used R, G, B channels are weighted according to the
    /// Digital ITU BT.601 standard: I = 0.299 * R + 0.587 * G + 0.114 * B.
    enum class ColorToIntensityConversionType {
        /// R, G, B channels have equal weights.
        Equal,
        /// Weighted R, G, B channels: I = 0.299 * R + 0.587 * G + 0.114 * B.
        Weighted,
    };

    /// \enum FilterType
    ///
    /// \brief Specifies the Image filter type.
    enum class FilterType {
        /// Gaussian filter of size 3 x 3.
        Gaussian3,
        /// Gaussian filter of size 5 x 5.
        Gaussian5,
        /// Gaussian filter of size 7 x 7.
        Gaussian7,
        /// Sobel filter along X-axis.
        Sobel3Dx,
        /// Sobel filter along Y-axis.
        Sobel3Dy
    };

public:
    /// \brief Default Constructor.
    Image() : Geometry2D(Geometry::GeometryType::Image) {}
    ~Image() override {}

public:
    Image &Clear() override;
    bool IsEmpty() const override;
    core::Tensor GetMinBound() const override;
    core::Tensor GetMaxBound() const override;

public:
    /// Returns `true` if the Image has valid data.
    virtual bool HasData() const { return data_.GetBlob() != nullptr; }

    /// \brief Prepare Image properties and allocate Image buffer.
    Image &Prepare(int width,
                   int height,
                   int num_of_channels,
                   core::Dtype dtype = core::Dtype::Float32,
                   core::Device device = core::Device("CPU:0")) {
        width_ = width;
        height_ = height;
        num_of_channels_ = num_of_channels;
        dtype_ = dtype;
        device_ = device;

        data_ = core::Tensor(core::SizeVector({height, width, num_of_channels}),
                             dtype, device);
        return *this;
    }

    core::Tensor AsTensor() { return data_; }

    // Usage:
    // std::shared_ptr<geometry::PointCloud> pcd_legacy =
    //         io::CreatePointCloudFromFile(filename);
    // tgeometry::PointCloud pcd =
    //         tgeometry::PointCloud::FromLegacyPointCloud(*pcd_legacy);
    // geometry::PointCloud pcd_legacy_back =
    //         tgeometry::PointCloud::ToLegacyPointCloud(pcd);
    static tgeometry::Image FromLegacyImage(
            const geometry::Image &image_legacy,
            core::Device device = core::Device("CPU:0"));

    static geometry::Image ToLegacyImage(const tgeometry::Image &image);

public:
    int width_ = -1;
    int height_ = -1;
    int num_of_channels_ = -1;
    core::Dtype dtype_;
    core::Device device_;

    core::Tensor data_;
};

}  // namespace tgeometry
}  // namespace open3d
