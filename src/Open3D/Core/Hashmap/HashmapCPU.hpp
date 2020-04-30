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

#include "HashmapCPU.h"

namespace open3d {

template <typename Hash>
CPUHashmap<Hash>::CPUHashmap(uint32_t max_keys,
                             uint32_t dsize_key,
                             uint32_t dsize_value,
                             Device device)
    : Hashmap<Hash>(max_keys, dsize_key, dsize_value, device) {
    utility::LogError("CPUHashmap is unimplemented!");
};

template <typename Hash>
CPUHashmap<Hash>::~CPUHashmap(){};

template <typename Hash>
std::pair<iterator_t*, uint8_t*> CPUHashmap<Hash>::Insert(
        uint8_t* input_keys, uint8_t* input_values, uint32_t input_key_size) {
    iterator_t* iterators = nullptr;
    uint8_t* masks = nullptr;
    return std::make_pair(iterators, masks);
}

template <typename Hash>
std::pair<iterator_t*, uint8_t*> CPUHashmap<Hash>::Search(
        uint8_t* input_keys, uint32_t input_key_size) {
    iterator_t* iterators = nullptr;
    uint8_t* masks = nullptr;
    return std::make_pair(iterators, masks);
}

template <typename Hash>
uint8_t* CPUHashmap<Hash>::Remove(uint8_t* input_keys,
                                  uint32_t input_key_size) {
    uint8_t* masks = nullptr;
    return masks;
}

}  // namespace open3d
