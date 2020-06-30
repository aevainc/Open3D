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

#include <unordered_map>
#include "open3d/geometry/Image.h"
#include "open3d/tgeometry/Image.h"

namespace open3d {
namespace tgeometry {

using namespace core;

static const std::unordered_map<int, Dtype> kBytesToDtypeMap = {
        {1, Dtype::UInt8}, {2, Dtype::UInt16}, {4, Dtype::Float32}};

tgeometry::Image Image::FromLegacyImage(const geometry::Image &image_legacy,
                                        Device device) {
    tgeometry::Image image;
    if (image_legacy.IsEmpty()) {
        return image;
    }

    auto iter = kBytesToDtypeMap.find(image_legacy.bytes_per_channel_);
    if (iter == kBytesToDtypeMap.end()) {
        utility::LogError("Unsupported image bytes_per_channel ({})",
                          image_legacy.bytes_per_channel_);
    }

    Dtype dtype = iter->second;
    image.Prepare(image_legacy.width_, image_legacy.height_,
                  image_legacy.num_of_channels_, dtype, device);

    size_t num_bytes = image_legacy.height_ * image_legacy.BytesPerLine();
    MemoryManager::MemcpyFromHost(image.data_.GetDataPtr(), device,
                                  image_legacy.data_.data(), num_bytes);
    image.data_ = image.data_.Permute({2, 0, 1}).Contiguous();
    return image;
}

geometry::Image Image::ToLegacyImage(const tgeometry::Image &image) {
    geometry::Image image_legacy;
    if (image.IsEmpty()) {
        return image_legacy;
    }

    utility::LogInfo("{}, {}, {}, {}", image.width_, image.height_,
                     image.num_of_channels_, DtypeUtil::ToString(image.dtype_));

    image_legacy.Prepare(image.width_, image.height_, image.num_of_channels_,
                         DtypeUtil::ByteSize(image.dtype_));
    size_t num_bytes = image_legacy.height_ * image_legacy.BytesPerLine();

    Tensor transposed = image.data_.Permute({1, 2, 0}).Contiguous();
    MemoryManager::MemcpyToHost(image_legacy.data_.data(),
                                transposed.GetDataPtr(), image.device_,
                                num_bytes);

    return image_legacy;
}

}  // namespace tgeometry
}  // namespace open3d
