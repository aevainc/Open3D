import numpy as np
import open3d.core as o3c

print(np)
print(np.__version__)

size = int(10e8)
repeat = 3

a = np.ones((2, 3), dtype=np.int32)
b = np.ones((2, 3), dtype=np.int32)
c = a + b

a = np.ones((200, 300), dtype=np.int32)
b = np.ones((200, 300), dtype=np.int32)
c = a + b

###########################################

# print("Numpy compute binary")
# a = np.ones((size), dtype=np.int32)
# b = np.ones((size), dtype=np.int32)
# for i in range(repeat):
#     print(i)
#     c = a + b

# print("Numpy compute unary")
# a = np.ones((size), dtype=np.int32)
# for i in range(repeat):
#     print(i)
#     b = np.sin(a)

# print("Open3D compute binary")
# a = o3c.Tensor.ones(size, dtype=o3c.int32)
# b = o3c.Tensor.ones(size, dtype=o3c.int32)
# for i in range(repeat):
#     print(i)
#     c = a + b

# print("Open3D compute unary")
# a = o3c.Tensor.ones(size, dtype=o3c.int32)
# for i in range(repeat):
#     print(i)
#     b = a.sin()
