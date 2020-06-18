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

#include "Open3D/TGeometry/Image.h"

namespace open3d {
namespace tgeometry {

Image &Image::Clear() {
    data_ = Tensor();
    return *this;
}

bool Image::IsEmpty() const { return !HasData(); }

Tensor Image::GetMinBound() const {
    return Tensor(std::vector<float>({0.0, 0.0}), SizeVector({2}),
                  Dtype::Float32, Device("CPU:0"));
}

Tensor Image::GetMaxBound() const {
    return Tensor(std::vector<float>({static_cast<float>(width_),
                                      static_cast<float>(height_)}),
                  SizeVector({2}), Dtype::Float32, Device("CPU:0"));
}

}  // namespace tgeometry
}  // namespace open3d
