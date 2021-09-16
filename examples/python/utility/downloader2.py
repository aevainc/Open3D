import zipfile
import os
import io
import sys
import json
import hashlib
import urllib.request
from pathlib import Path

# Typically "Open3D/examples/test_data", the test data dir.
_test_data_dir = Path(__file__).parent.absolute().resolve()

# Typically "Open3D/examples/test_data/open3d_downloads", the download dir.
_download_dir = _test_data_dir / "../dataset_dir"

def _compute_sha256(path):
    """
    Returns sha256 checksum as string.
    """
    # http://stackoverflow.com/a/17782753 with fixed block size
    algo = hashlib.sha256()
    with io.open(str(path), 'br') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            algo.update(chunk)
    return algo.hexdigest()


def _download_file(url, path, sha256, max_retry=3):
    if max_retry == 0:
        raise OSError(f"max_retry reached, cannot download {url}.")

    full_path = _download_dir / Path(path)

    # The saved file must be inside _test_data_dir.
    if _download_dir not in full_path.parents:
        raise AssertionError(f"{full_path} must be inside {_download_dir}.")

    # Supports sub directory inside _test_data_dir, e.g.
    # Open3D/examples/test_data/open3d_downloads/foo/bar/my_file.txt
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if full_path.exists() and _compute_sha256(full_path) == sha256:
        print(f"[download_utils.py] {str(full_path)} already exists, skipped.")
        return

    try:
        urllib.request.urlretrieve(url, full_path)
        print(
            f"[download_utils.py] Downloaded {url}\n        to {str(full_path)}"
        )
        if _compute_sha256(full_path) != sha256:
            raise ValueError(f"{path}'s SHA256 checksum incorrect:\n"
                             f"- Expected: {sha256}\n"
                             f"- Actual  : {_compute_sha256(full_path)}")
    except Exception as e:
        sleep_time = 5
        print(f"[download_utils.py] Failed to download {url}: {str(e)}")
        print(f"[download_utils.py] Retrying in {sleep_time}s")
        time.sleep(sleep_time)
        _download_file(url, path, sha256, max_retry=max_retry - 1)

def get_dataset(dataset_name):

    with open('DatasetList.json') as file:
        dataset_list_json = json.load(file)

    if dataset_name not in dataset_list_json:
        print("dataset " + dataset_name + "is not presesnt in Open3D dataset list.")
    else:
        dataset_files = dataset_list_json[dataset_name]["files"]
        for item in dataset_files:
            url = dataset_files[item]["url"]
            path = dataset_files[item]["path"]
            sha256 = dataset_files[item]["sha256"]
            _download_file(url, path, sha256)

def unzip_data(path_zip, path_extract_to):
    print("Unzipping %s" % path_zip)
    zip_ref = zipfile.ZipFile(path_zip, 'r')
    zip_ref.extractall(path_extract_to)
    zip_ref.close()
    print("Extracted to %s" % path_extract_to)

get_dataset("Redwood")