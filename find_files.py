from pathlib import Path
import os

mit_license = """# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
"""


def starts_with_license(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        if "".join(lines).startswith(mit_license):
            return True
        else:
            return False


def file_not_empty(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        if len(lines) > 0:
            return True
        else:
            return False


pwd = Path(os.path.dirname(os.path.realpath(__file__)))

python_dirs = [
    "examples",
    "docs",
    "python",
    "util",
]

python_files = []
for python_dir in python_dirs:
    python_files.extend(list(Path(python_dir).rglob('*.py')))

for python_file in python_files:
    if file_not_empty(python_file) and not starts_with_license(python_file):
        print("#############################################################")
        print("#", str(python_file))
        print("#############################################################")
        with open(python_file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for i in range(min(10, len(lines))):
                print(lines[i])
        print("#############################################################")
        print("")
