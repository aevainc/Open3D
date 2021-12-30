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

#include <string>
#include <unordered_map>

#include "open3d/data/Dataset.h"
#include "open3d/data/Download.h"
#include "open3d/data/Extract.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

// Set data meta-information.
namespace {
const std::string url =
        "https://github.com/isl-org/open3d_downloads/releases/"
        "download/data/ICLNUIM_LivingRoomFragments.zip";
const std::string download_cache_sha256 =
        "1f67f1dee630cf42bbb535e45160101e15b1aeebee3032399738d18ac7018db4";
const std::unordered_map<std::string, std::string> file_name_to_file_sha256{
        {"cloud_bin_0.pcd",
         "e1e100802c29ef454c6b523084668ee0e2f365ec52eaeebe79ae804c20447b15"},
        {"cloud_bin_1.pcd",
         "a4c3dc0ad7b1279736491b9b2638991d4c808605997be4f9ab174c24a9fa6e52"},
        {"cloud_bin_2.pcd",
         "1e68e194ebc1941f0f29764e4daf89340e69b224d2b80db5efbc1373a17f8b4a"},
        {"init.log",
         "609896dbdd666b7ae0bb7390c52730ca8aca10c2b5886b895acdb36f4a202156"},
};
}  // namespace

ICLNUIM_LivingRoomFragments::ICLNUIM_LivingRoomFragments(
        const std::string prefix,
        const std::string& data_root,
        const bool cache_download)
    : file_name_to_file_sha256_(file_name_to_file_sha256) {
    // Resolve paths.
    if (data_root.empty()) {
        data_root_ = LocateDataRoot();
    } else {
        data_root_ = data_root;
    }
    const std::string download_dir = data_root_ + "/download_cache";
    download_cache_path_ = download_dir + "/" + prefix;
    data_path_ = data_root_ + "/data/" + prefix;

    // Check if files already present.
    if (!VerifyFiles(data_path_, file_name_to_file_sha256_)) {
        // Check cached download.
        if (!VerifyFiles(download_dir, {{prefix, download_cache_sha256}})) {
            // Download Data.
            DownloadFromURL(url, download_cache_sha256, download_dir,
                            data_root_);
        }
        // Extract data.
        Extract(download_cache_path_, data_path_);
    }
}

void ICLNUIM_LivingRoomFragments::DeleteDownloadCache() const {
    utility::filesystem::DeleteDirectory(download_cache_path_);
}

void ICLNUIM_LivingRoomFragments::DeleteData() const {
    utility::filesystem::DeleteDirectory(data_path_);
}

void ICLNUIM_LivingRoomFragments::DisplayDataTree(const int max_depth) {}

}  // namespace data
}  // namespace open3d
