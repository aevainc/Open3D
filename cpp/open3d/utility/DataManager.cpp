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

#include "open3d/utility/Logging.h"

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
    const std::string data_root_common = GetInstance().data_root_common_;
    if (data_root_common.empty()) {
        utility::LogError(
                "DataManager::SetDataRootCommon() must be called first to "
                "specify the common data root.");
    }
    if (relative_path.empty()) {
        return data_root_common;
    } else {
        return data_root_common + "/" + relative_path;
    }
}

std::string DataManager::GetDataPathDownload(const std::string& relative_path) {
    const std::string data_root_download = GetInstance().data_root_download_;
    if (data_root_download.empty()) {
        utility::LogError(
                "DataManager::SetDataRootdownload() must be called first to "
                "specify the download data root.");
    }
    if (relative_path.empty()) {
        return data_root_download;
    } else {
        return data_root_download + "/" + relative_path;
    }
}

DataManager::DataManager() : data_root_common_(""), data_root_download_("") {}

}  // namespace utility
}  // namespace open3d
