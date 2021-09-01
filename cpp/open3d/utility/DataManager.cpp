// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/utility/DataManager.h"

#include <string>

namespace open3d {
namespace utility {

DataManager& DataManager::GetInstance() {
    static DataManager instance;
    return instance;
}

void DataManager::SetDataRootCommon(const std::string& data_root) {
    GetInstance().data_root_common_ = data_root;
}

void DataManager::SetDataRootDownload(const std::string& data_root) {
    GetInstance().data_root_download_ = data_root;
}

std::string DataManager::GetDataPathCommon(const std::string& relative_path) {
    if (relative_path.empty()) {
        return GetInstance().data_root_common_;
    } else {
        return GetInstance().data_root_common_ + "/" + relative_path;
    }
}

std::string DataManager::GetDataPathDownload(const std::string& relative_path) {
    if (relative_path.empty()) {
        return GetInstance().data_root_download_;
    } else {
        return GetInstance().data_root_download_ + "/" + relative_path;
    }
}

DataManager::DataManager()
    : data_root_common_("UNDEFINED"), data_root_download_("UNDEFINED") {}

}  // namespace utility
}  // namespace open3d
