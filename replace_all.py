from pathlib import Path
import os
from pprint import pprint

pwd = Path(os.path.dirname(os.path.realpath(__file__)))


def get_source_files(root_dir):
    extensions = [".cpp", ".cc", ".h", ".cu", ".cuh"]
    source_files = []
    for extension in extensions:
        source_files.extend(Path(root_dir).glob("**/*" + extension))
    return source_files


def replace_string_in_file(file_path, src, dst):
    with open(file_path) as f:
        lines = f.readlines()
        lines = [line.replace(src, dst) for line in lines]

    with open(file_path, "w") as f:
        f.writelines(lines)


if __name__ == '__main__':
    root_dir = pwd / "cpp"
    file_paths = get_source_files(root_dir)
    file_paths = sorted(file_paths)

    srcs_dsts = [
        ("core::Dtype::Undefined", "core::Undefined"),
        ("core::Dtype::Float32", "core::Float32"),
        ("core::Dtype::Float64", "core::Float64"),
        ("core::Dtype::Int8", "core::Int8"),
        ("core::Dtype::Int16", "core::Int16"),
        ("core::Dtype::Int32", "core::Int32"),
        ("core::Dtype::Int64", "core::Int64"),
        ("core::Dtype::UInt8", "core::UInt8"),
        ("core::Dtype::UInt16", "core::UInt16"),
        ("core::Dtype::UInt32", "core::UInt32"),
        ("core::Dtype::UInt64", "core::UInt64"),
        ("core::Dtype::Bool", "core::Bool"),
    ]

    for src, dst in srcs_dsts:
        for file_path in file_paths:
            replace_string_in_file(file_path, src, dst)
