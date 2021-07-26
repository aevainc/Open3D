import numpy as np
from pathlib import Path
import os

pwd = Path(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    npz_file = pwd / "examples" / "test_data" / "tensors_compressed.npz"
    np.savez_compressed(
        npz_file,
        t0=np.array([[1, 2], [3, 4]], dtype=np.int32),
        t1=np.array([0, 2, 8, 10, 12, 14, 20, 22], dtype=np.float32).reshape(
            (2, 2, 2)),
        t2=np.array(3.14, dtype=np.float32),
        t3=np.ones((0,), dtype=np.float32),
        t4=np.ones((0, 0), dtype=np.float32),
        t5=np.ones((0, 1, 0), dtype=np.float32),
    )
