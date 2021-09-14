import zipfile
import os
import sys
import json
if (sys.version_info > (3, 0)):
    pyver = 3
    from urllib.request import Request, urlopen
else:
    pyver = 2
    from urllib2 import Request, urlopen


def get_dataset(dataset_name):

    with open('DatasetList.json') as file:
        dataset_list_json = json.load(file)

    if dataset_name not in dataset_list_json:
        print("dataset " + dataset_name + "is not presesnt in Open3D dataset list.")
    else:
        dataset = dataset_list_json[dataset_name]
        for url in dataset["url"]:
            print("==================================")
            file_downloader(url)

def file_downloader(url):
    file_name = url.split('/')[-1]
    u = urlopen(url)
    f = open(file_name, "wb")
    if pyver == 2:
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
    elif pyver == 3:
        file_size = int(u.getheader("Content-Length"))
    print("Downloading: %s " % file_name)

    file_size_dl = 0
    block_sz = 8192
    progress = 0
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        if progress + 10 <= (file_size_dl * 100. / file_size):
            progress = progress + 10
            print(" %.1f / %.1f MB (%.0f %%)" % \
                    (file_size_dl/(1024*1024), file_size/(1024*1024), progress))
    f.close()

def unzip_data(path_zip, path_extract_to):
    print("Unzipping %s" % path_zip)
    zip_ref = zipfile.ZipFile(path_zip, 'r')
    zip_ref.extractall(path_extract_to)
    zip_ref.close()
    print("Extracted to %s" % path_extract_to)

get_dataset("Redwood")