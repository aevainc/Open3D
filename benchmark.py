import numpy as np
import torch
import open3d.core as o3c
import time
import os

print("##############################")

# print(np)
# print(np.__version__)

size = int(10e8)
repeat = 5

# a = np.ones((2, 3), dtype=np.int32)
# b = np.ones((2, 3), dtype=np.int32)
# c = a + b

# a = np.ones((200, 300), dtype=np.int32)
# b = np.ones((200, 300), dtype=np.int32)
# c = a + b

###########################################

omp = os.getenv("OMP_NUM_THREADS")

# print("Numpy compute binary op")
a = np.ones(size, dtype=np.int32)
b = np.ones(size, dtype=np.int32)
start_time = time.time()
for i in range(repeat):
    c = a + b
print(f"OMP={omp}, contiguous, Numpy: {time.time() - start_time} sec")

# print("Torch compute binary op")
a = torch.ones(size, dtype=torch.int32)
b = torch.ones(size, dtype=torch.int32)
start_time = time.time()
for i in range(repeat):
    c = a + b
print(f"OMP={omp}, contiguous, Torch: {time.time() - start_time} sec")

# print("Open3D compute binary op")
a = o3c.Tensor.ones(size, dtype=o3c.int32)
b = o3c.Tensor.ones(size, dtype=o3c.int32)
start_time = time.time()
for i in range(repeat):
    c = a + b
print(f"OMP={omp}, contiguous, Open3D: {time.time() - start_time} sec")

# print("Numpy compute binary op")
a_ = np.ones(size * 2, dtype=np.int32)
b_ = np.ones(size * 2, dtype=np.int32)
a = a_[0:-1:2]
b = b_[0:-1:2]
start_time = time.time()
for i in range(repeat):
    c = a + b
print(f"OMP={omp}, non-contiguous, Numpy: {time.time() - start_time} sec")

# print("Torch compute binary op")
a_ = torch.ones(size * 2, dtype=torch.int32)
b_ = torch.ones(size * 2, dtype=torch.int32)
a = a_[0:-1:2]
b = b_[0:-1:2]
start_time = time.time()
for i in range(repeat):
    c = a + b
print(f"OMP={omp}, non-contiguous, Torch: {time.time() - start_time} sec")

# print("Open3D compute binary op")
a_ = o3c.Tensor.ones(size * 2, dtype=o3c.int32)
b_ = o3c.Tensor.ones(size * 2, dtype=o3c.int32)
a = a_[0:-1:2]
b = b_[0:-1:2]
start_time = time.time()
for i in range(repeat):
    c = a + b
print(f"OMP={omp}, non-contiguous, Open3D: {time.time() - start_time} sec")

print("##############################")
