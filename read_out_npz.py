import numpy as np
from pathlib import Path
import os

pwd = Path(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    npz_file = pwd / "build" / "out.npz"
    with np.load(npz_file) as data:
        for key in data:
            print(key, data[key])
        np.testing.assert_array_equal(data["t0"], [100, 200])
        assert data["t0"].dtype == np.int32
        np.testing.assert_array_equal(data["t1"], [[0, 1, 2], [3, 4, 5]])
        assert data["t1"].dtype == np.float64

    npz_file = pwd / "build" / "tensors.npz"
    if npz_file.exists():
        with np.load(npz_file) as data:
            for key in data:
                print(key, data[key])
    else:
        print(f"{npz_file} not found")
