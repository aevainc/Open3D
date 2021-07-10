import open3d as o3d
import numpy as np

data = np.load('tsdf.npz')

print('loading')
xyz = data['xyz']
tsdf = data['tsdf']
weight = data['weight']


def touch():
    print('touching')
    device = o3d.core.Device('CPU:0')
    point_hashmap = o3d.core.Hashmap(len(xyz), o3d.core.Dtype.Int32,
                                     o3d.core.Dtype.Int32, (3), (1), device)
    block_xyz = o3d.core.Tensor(np.floor(xyz / 16).astype(np.int32))

    addrs, masks = point_hashmap.activate(block_xyz)
    block_active = block_xyz[masks]

    return block_xyz, block_active


block_xyz, block_active = touch()

volume = o3d.t.geometry.TSDFVoxelGrid(
    {
        'tsdf': o3d.core.Dtype.Float32,
        'weight': o3d.core.Dtype.Float32
    },
    voxel_size=0.0058,
    sdf_trunc=0.04,
    block_resolution=16,
    block_count=len(block_active))
block_hashmap = volume.get_block_hashmap()

print('activating')
block_hashmap.activate(block_active)

print('finding')
addrs, masks = block_hashmap.find(block_xyz)
assert (masks.numpy().sum() == len(block_xyz))

print('calculating')
voxel_coord = xyz.astype(np.int32) - block_xyz.numpy() * 16

indices = addrs.to(o3d.core.Dtype.Int64).numpy()
voxel_coord = voxel_coord.astype(np.int64)

print('assigning in numpy')
buf = np.zeros((len(block_active), 16, 16, 16, 2), dtype=np.float32)
buf[indices, voxel_coord[:, 2], voxel_coord[:, 1], voxel_coord[:, 0], 0] = tsdf
buf[indices, voxel_coord[:, 2], voxel_coord[:, 1], voxel_coord[:, 0], 1] = weight

print('assigning in tsdf')
value_tensor = block_hashmap.get_value_tensor()
value_tensor[:] = o3d.core.Tensor(buf.view(np.uint8))

mesh = volume.extract_surface_mesh()
o3d.io.write_triangle_mesh('world.ply', mesh.to_legacy_triangle_mesh())
