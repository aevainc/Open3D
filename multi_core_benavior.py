import numpy as np
import open3d.core as o3c

print(np)
print(np.__version__)

size = int(10e8)
repeat = 3

print("Numpy compute binary")
a = np.ones((size), dtype=np.float64)
b = np.ones((size), dtype=np.float64)
for i in range(repeat):
    print(i)
    c = a + b

print("Numpy compute unary")
a = np.ones((size), dtype=np.float64)
for i in range(repeat):
    print(i)
    b = np.sin(a)

print("Open3D compute binary")
a = o3c.Tensor.ones(size, dtype=o3c.float64)
b = o3c.Tensor.ones(size, dtype=o3c.float64)
for i in range(repeat):
    print(i)
    c = a + b

print("Open3D compute unary")
a = o3c.Tensor.ones(size, dtype=o3c.float64)
for i in range(repeat):
    print(i)
    b = a.sin()
