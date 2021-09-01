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

/// Global singleton class to manage data directories.
/// TODO: if DataManager has more functionality in the future, consider exposing
/// the class in a public interface.
class DataManager {
public:
    /// Computes the full path of a file/directory inside the Open3D common data
    /// root. If \p relative_path is specified, the full path is computed by
    /// appending the \p relative_path to the common data root; otherwise, the
    /// common data root is returned.
    ///
    /// \param relative_path Relative path to Open3D common data root.
    static std::string GetDataPathCommon(
            const std::string& relative_path = "") {
        const std::string data_root_common = GetInstance().data_root_common_;
        if (data_root_common.empty()) {
            utility::LogError(
                    "SetDataRootCommon() must be called first to specify the "
                    "common data root.");
        }
        if (relative_path.empty()) {
            return data_root_common;
        } else {
            return data_root_common + "/" + relative_path;
        }
    }

    /// Computes the full path of a file/directory inside the Open3D
    /// download data root. If \p relative_path is specified, the full
    /// path is computed by appending the \p relative_path to the
    /// download data root; otherwise, the download data root is
    /// returned.
    ///
    /// \param relative_path Relative path to Open3D download data root.
    static std::string GetDataPathDownload(
            const std::string& relative_path = "") {
        const std::string data_root_download =
                GetInstance().data_root_download_;
        if (data_root_download.empty()) {
            utility::LogError(
                    "SetDataRootdownload() must be called first to specify the "
                    "download data root.");
        }
        if (relative_path.empty()) {
            return data_root_download;
        } else {
            return data_root_download + "/" + relative_path;
        }
    }

    /// Set the Open3D common data root. This is typically done once during
    /// the initialization phase of a program.
    ///
    /// \param data_root Full path of the Open3D common data root directory.
    static void SetDataRootCommon(const std::string& data_root) {
        GetInstance().data_root_common_ = data_root;
    }

    /// Set the Open3D download data root. This is typically done once during
    /// the initialization phase of a program.
    ///
    /// \param data_root Full path of the Open3D download data root directory.
    static void SetDataRootDownload(const std::string& data_root) {
        GetInstance().data_root_download_ = data_root;
    }

private:
    static DataManager& GetInstance() {
        static DataManager instance;
        return instance;
    }
    DataManager() : data_root_common_(""), data_root_download_("") {}
    std::string data_root_common_;
    std::string data_root_download_;
};

std::string GetDataPathCommon(const std::string& relative_path) {
    return DataManager::GetDataPathCommon(relative_path);
}

std::string GetDataPathDownload(const std::string& relative_path) {
    return DataManager::GetDataPathDownload(relative_path);
}

void SetDataRootCommon(const std::string& data_root) {
    DataManager::SetDataRootCommon(data_root);
}

void SetDataRootDownload(const std::string& data_root) {
    DataManager::SetDataRootDownload(data_root);
}

}  // namespace utility
}  // namespace open3d
