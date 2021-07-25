import numpy as np
from pathlib import Path
import os

pwd = Path(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    npz_file = pwd / "build" / "out.npz"
    with np.load(npz_file) as data:
        for key in data:
            print(key, data[key])

    npz_file = pwd / "build" / "tensor.npz"
    if npz_file.exists():
        with np.load(npz_file) as data:
            for key in data:
                print(key, data[key])
    else:
        print(f"{npz_file} not found")
