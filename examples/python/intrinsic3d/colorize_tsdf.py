import open3d as o3d
import numpy as np

import torch
import torch.utils.dlpack

def to_torch(tensor):
    return torch.utils.dlpack.from_dlpack(tensor.to_dlpack())

volume = o3d.t.io.read_tsdf_voxelgrid('tsdf.json')
hashmap = volume.get_block_hashmap()

value = hashmap.get_value_tensor()
key = hashmap.get_key_tensor()

# Load TSDF
value_np = value.cpu().numpy().view(np.float32)

tsdf = value_np[:, :, :, :, 0]
tsdf_c = np.ascontiguousarray(np.expand_dims(tsdf, axis=-1))

weight = value_np[:, :, :, :, 1].astype(np.uint16)
weight_c = np.ascontiguousarray(np.expand_dims(weight, axis=-1))

mask_weight = weight_c > 0
mask_tsdf = np.abs(tsdf_c) < (0.004 * 2 / 0.015)

mask_general = (mask_weight & mask_tsdf).squeeze()

tsdf_c = tsdf_c[mask_general]
weight_c = weight_c[mask_general]

# Organized in the (z, y, x) order
x, y = np.meshgrid(np.arange(8), np.arange(8))
x = np.tile(x, (8, 1)).reshape(8, 8, 8)
y = np.tile(y, (8, 1)).reshape(8, 8, 8)
z = np.expand_dims(np.arange(8), axis=-1)
z = np.tile(z, (1, 64)).reshape(8, 8, 8)

# print(x[3, 4, 5])
# print(y[3, 4, 5])
# print(z[3, 4, 5])

key_np = key.cpu().numpy()
n_blocks = len(key_np)

block_x = key_np[:, 0:1]
block_y = key_np[:, 1:2]
block_z = key_np[:, 2:3]

block_x = np.tile(block_x, (1, 512)).reshape(-1, 8, 8, 8)
block_y = np.tile(block_y, (1, 512)).reshape(-1, 8, 8, 8)
block_z = np.tile(block_z, (1, 512)).reshape(-1, 8, 8, 8)

voxel_x = np.tile(x.reshape(-1, 8, 8, 8), (n_blocks, 1, 1, 1))
voxel_y = np.tile(y.reshape(-1, 8, 8, 8), (n_blocks, 1, 1, 1))
voxel_z = np.tile(z.reshape(-1, 8, 8, 8), (n_blocks, 1, 1, 1))

global_x = block_x * 8 + voxel_x
global_y = block_y * 8 + voxel_y
global_z = block_z * 8 + voxel_z

x = global_x[mask_general].flatten()
y = global_y[mask_general].flatten()
z = global_z[mask_general].flatten()
xyz = np.stack((x, y, z)).astype(np.float64).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz * 0.004)
o3d.visualization.draw_geometries([pcd])
