import numpy as np
import open3d.core as o3c

size = int(10e8)
repeat = 10

print("Numpy compute")
a = np.ones((size))
for i in range(repeat):
    print(i)
    a * a

print("Open3D compute")
a = o3c.Tensor.ones((int(10e8)))
for i in range(repeat):
    print(i)
    a * a
