import open3d as o3d
import numpy as np

import torch
import torch.utils.dlpack

volume = o3d.t.io.read_tsdf_voxelgrid('tsdf.json')
hashmap = volume.get_block_hashmap()

value = hashmap.get_value_tensor()
key = hashmap.get_key_tensor()

# Load TSDF
value_np = value.cpu().numpy().view(np.float32)

# Process data
tsdf = value_np[:, :, :, :, 0]
weight = value_np[:, :, :, :, 1].astype(np.uint16)
color = np.ones((*tsdf.shape, 3), dtype=np.uint16) * 65535
color[:, :, :, :, ::2] = 0

# Put back value and generate mesh
tsdf_c = np.ascontiguousarray(np.expand_dims(tsdf, axis=-1))
tsdf_c = tsdf_c.view(np.uint8)

weight_c = np.ascontiguousarray(np.expand_dims(weight, axis=-1))
weight_c = weight_c.view(np.uint8)

color_c = color.view(np.uint8)

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

mesh = colored_volume.extract_surface_mesh()
o3d.io.write_triangle_mesh('mesh_colored.ply', mesh.to_legacy_triangle_mesh())

