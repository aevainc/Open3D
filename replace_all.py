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


if __name__ == '__main__':
    root_dir = pwd / "cpp"
    source_files = get_source_files(root_dir)
    source_files = sorted(source_files)
    pprint(source_files)
