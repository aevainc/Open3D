import open3d as o3d
import numpy as np
import os
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

pcd_xyz = np.stack((x, y, z)).astype(np.float64) * 0.004
pcd_color = np.zeros_like(pcd_xyz)

# Then load images and cameras
path = '/home/wei/Workspace/data/intrinsic3d/lion-rgbd/'


def read_poses_from_log(traj_log):
    import numpy as np

    trans_arr = []
    with open(traj_log) as f:
        content = f.readlines()

        # Load .log file.
        for i in range(0, len(content), 5):
            # format %d (src) %d (tgt) %f (fitness)
            data = list(map(float, content[i].strip().split(' ')))
            ids = (int(data[0]), int(data[1]))
            fitness = data[2]

            # format %f x 16
            T_gt = np.array(
                list(map(float, (''.join(
                    content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

            trans_arr.append(T_gt)

    return trans_arr


K_depth = np.array([577.871, 0, 319.623, 0, 580.258, 239.624, 0, 0,
                    1]).reshape(3, 3)
K_color = np.array([1170.19, 0, 647.75, 0, 1170.19, 483.75, 0, 0,
                    1]).reshape(3, 3)

color_fnames = sorted(os.listdir(path + 'color'))
depth_fnames = sorted(os.listdir(path + 'depth'))
trajectory = read_poses_from_log(path + 'trajectory.log')

with open('keyframes.txt') as f:
    lines = f.readlines()

color_kfnames = []
depth_kfnames = []
trajectory_kf = []
for i, line in enumerate(lines):
    weight, mask = line.strip().split(' ')
    if int(mask):
        color_kfnames.append(color_fnames[i])
        depth_kfnames.append(depth_fnames[i])
        trajectory_kf.append(trajectory[i])

for color_kfname in color_kfnames:
    color = o3d.io.read_image(f'{path}/color/{color_kfname}')
    h, w, _ = np.asarray(color).shape

pcd_map = o3d.geometry.PointCloud()
for i, (color_kfname,
        depth_kfname) in enumerate(zip(color_kfnames, depth_kfnames)):
    depth = o3d.io.read_image(f'{path}/depth/{depth_kfname}')
    depth_np = np.asarray(depth)
    dh, dw = depth_np.shape

    depth_np_resized = np.zeros((h, w), dtype=np.uint16)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()

    # Unproject and project for resizing
    depth_xyz_resized = np.stack((u, v, np.ones_like(u)))
    projection = (K_depth @ np.linalg.inv(K_color)) @ depth_xyz_resized

    uu = projection[0].round().astype(int)
    vv = projection[1].round().astype(int)

    mask = (uu >= 0) & (uu < dw) & (vv >= 0) & (vv < dh)

    depth_np_resized[v[mask], u[mask]] = depth_np[vv[mask], uu[mask]]
    depth_resized = o3d.geometry.Image(depth_np_resized)
    color = o3d.io.read_image(f'{path}/color/{color_kfname}')
    color_np = np.asarray(color)

    T = np.linalg.inv(trajectory_kf[i])
    R = T[:3, :3]
    t = T[:3, 3:]

    projection = K_color @ (R @ pcd_xyz + t)
    uu = (projection[0] / projection[2]).round().astype(int)
    vv = (projection[1] / projection[2]).round().astype(int)

    mask = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)

    # Naive assignment. TODO: k-selection and depth culling
    pcd_color.T[mask, :] = color_np[vv[mask], uu[mask]]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_xyz.T)
pcd.colors = o3d.utility.Vector3dVector(pcd_color.T / 255.0)
o3d.visualization.draw_geometries([pcd])

tsdf = value_np[:, :, :, :, 0]
tsdf_c = np.ascontiguousarray(np.expand_dims(tsdf, axis=-1))
tsdf_c = tsdf_c.view(np.uint8)

weight = value_np[:, :, :, :, 1].astype(np.uint16)
weight_c = np.ascontiguousarray(np.expand_dims(weight, axis=-1))
weight_c = weight_c.view(np.uint8)

color_c = np.zeros((*tsdf.shape, 3), dtype=np.uint16)
color_c[mask_general] = pcd_color.T.astype(np.uint16) * 255
color_c = color_c.view(np.uint8)

value_np = np.concatenate((tsdf_c, weight_c, color_c), axis=-1)

colored_volume = o3d.t.geometry.TSDFVoxelGrid(
    {
        'tsdf': o3d.core.Dtype.Float32,
        'weight': o3d.core.Dtype.UInt16,
        'color': o3d.core.Dtype.UInt16
    },
    voxel_size=0.004,
    sdf_trunc=0.015,
    block_resolution=8,
    block_count=hashmap.capacity())

hashmap = colored_volume.get_block_hashmap()
hashmap.insert(key.cpu(), o3d.core.Tensor(value_np))

mesh = colored_volume.extract_surface_mesh().to_legacy_triangle_mesh()
o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh('mesh_colored.ply', mesh)
