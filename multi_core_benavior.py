import numpy as np
import open3d.core as o3c
import time

print(np)
print(np.__version__)

size = int(10e8)
repeat = 10

# a = np.ones((2, 3), dtype=np.int32)
# b = np.ones((2, 3), dtype=np.int32)
# c = a + b

# a = np.ones((200, 300), dtype=np.int32)
# b = np.ones((200, 300), dtype=np.int32)
# c = a + b

###########################################

start_time = time.time()
print("Numpy compute binary")
a = np.ones((size), dtype=np.int32)
b = np.ones((size), dtype=np.int32)
for i in range(repeat):
    print(i)
    c = a + b
print(f"Numpy time: {time.time() - start_time}")

start_time = time.time()
print("Open3D compute binary")
a = o3c.Tensor.ones(size, dtype=o3c.int32)
b = o3c.Tensor.ones(size, dtype=o3c.int32)
for i in range(repeat):
    print(i)
    c = a + b
print(f"Open3D time: {time.time() - start_time}")
