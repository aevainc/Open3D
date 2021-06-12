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

point_hashmap = o3d.core.Hashmap(len(xyz), o3d.core.Dtype.Int32,
                                 o3d.core.Dtype.Float32, (3), (1),
                                 o3d.core.Device('CUDA:0'))

point_xyz = o3d.core.Tensor(np.stack((x, y, z)).astype(np.int32).T,
                            device=o3d.core.Device('CUDA:0'))
point_tsdf = o3d.core.Tensor(tsdf_c, device=o3d.core.Device('CUDA:0'))

addrs, masks = point_hashmap.insert(point_xyz, point_tsdf)

xyz_xp = point_xyz + o3d.core.Tensor(
    [[1, 0, 0]], dtype=o3d.core.Dtype.Int32, device=o3d.core.Device('CUDA:0'))
xyz_xm = point_xyz - o3d.core.Tensor(
    [[1, 0, 0]], dtype=o3d.core.Dtype.Int32, device=o3d.core.Device('CUDA:0'))
xyz_yp = point_xyz + o3d.core.Tensor(
    [[0, 1, 0]], dtype=o3d.core.Dtype.Int32, device=o3d.core.Device('CUDA:0'))
xyz_ym = point_xyz - o3d.core.Tensor(
    [[0, 1, 0]], dtype=o3d.core.Dtype.Int32, device=o3d.core.Device('CUDA:0'))
xyz_zp = point_xyz + o3d.core.Tensor(
    [[0, 0, 1]], dtype=o3d.core.Dtype.Int32, device=o3d.core.Device('CUDA:0'))
xyz_zm = point_xyz - o3d.core.Tensor(
    [[0, 0, 1]], dtype=o3d.core.Dtype.Int32, device=o3d.core.Device('CUDA:0'))

addr_xp, mask_xp = point_hashmap.find(xyz_xp)
addr_xm, mask_xm = point_hashmap.find(xyz_xm)

addr_yp, mask_yp = point_hashmap.find(xyz_yp)
addr_ym, mask_ym = point_hashmap.find(xyz_ym)

addr_zp, mask_zp = point_hashmap.find(xyz_zp)
addr_zm, mask_zm = point_hashmap.find(xyz_zm)

mask_nb_laplacian = mask_xp & mask_xm & mask_yp & mask_ym & mask_zp & mask_zm

index_o = to_torch(addrs[mask_nb_laplacian].to(o3d.core.Dtype.Int64))

index_xp = to_torch(addr_xp[mask_nb_laplacian].to(o3d.core.Dtype.Int64))
index_yp = to_torch(addr_yp[mask_nb_laplacian].to(o3d.core.Dtype.Int64))
index_zp = to_torch(addr_zp[mask_nb_laplacian].to(o3d.core.Dtype.Int64))

index_xm = to_torch(addr_xm[mask_nb_laplacian].to(o3d.core.Dtype.Int64))
index_ym = to_torch(addr_ym[mask_nb_laplacian].to(o3d.core.Dtype.Int64))
index_zm = to_torch(addr_zm[mask_nb_laplacian].to(o3d.core.Dtype.Int64))

mapped_keys = to_torch(point_hashmap.get_key_tensor())
mapped_tsdf = to_torch(point_hashmap.get_value_tensor())

nx = mapped_tsdf[index_xp] - mapped_tsdf[index_xm]
ny = mapped_tsdf[index_yp] - mapped_tsdf[index_ym]
nz = mapped_tsdf[index_zp] - mapped_tsdf[index_zm]
nnorm = ((nx * nx + ny * ny + nz * nz).sqrt())

# Visualize
nx = (nx / nnorm)
ny = (ny / nnorm)
nz = (nz / nnorm)

normal = torch.cat((nx, ny, nz), axis=1)
xyz = (mapped_keys[index_o].float() * 0.004)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
pcd.normals = o3d.utility.Vector3dVector(normal.cpu().numpy())
o3d.visualization.draw_geometries([pcd])
