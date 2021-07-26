import numpy as np
from pathlib import Path
import os

pwd = Path(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    npz_file = pwd / "build" / "out.npz"
    with np.load(npz_file) as data:
        for key in data:
            print(key, data[key])
        np.testing.assert_equal(data["t0"], [100, 200])
        assert data["t0"].dtype == np.int32
        np.testing.assert_equal(data["t1"], [[0, 1, 2], [3, 4, 5]])
        assert data["t1"].dtype == np.float64

    npz_file = pwd / "build" / "tensors.npz"
    if npz_file.exists():
        with np.load(npz_file) as data:
            for key in data:
                print(key, data[key])

            np.testing.assert_equal(data["t0"], [[1, 2], [3, 4]])
            assert data["t0"].dtype == np.int32

            np.testing.assert_equal(
                data["t1"],
                np.array([0, 2, 8, 10, 12, 14, 20, 22]).reshape((2, 2, 2)))
            assert data["t1"].dtype == np.float32

            np.testing.assert_allclose(data["t2"],
                                       np.array(3.14, dtype=np.float32))
            assert data["t2"].dtype == np.float32

            np.testing.assert_equal(data["t3"], np.ones((0,), dtype=np.float32))
            assert data["t3"].dtype == np.float32

            np.testing.assert_equal(data["t4"], np.ones((0, 0),
                                                        dtype=np.float32))
            assert data["t4"].dtype == np.float32

            np.testing.assert_equal(data["t5"],
                                    np.ones((0, 1, 0), dtype=np.float32))
            assert data["t5"].dtype == np.float32
    else:
        # print(f"{npz_file} not found")
        pass
