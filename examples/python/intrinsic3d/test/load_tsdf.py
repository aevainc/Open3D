import open3d as o3d
import numpy as np

import torch
import torch.utils.dlpack

tsdf_grid = o3d.t.io.read_tsdf_voxelgrid('tsdf.json')
hashmap = tsdf_grid.get_block_hashmap()

value = hashmap.get_value_tensor()
key = hashmap.get_key_tensor()

# Share buffer in torch
value_torch_lhs = torch.utils.dlpack.from_dlpack(value.to_dlpack())

# We have to use numpy for reinterpreting datatype
# TODO: write Float temporarily or implement view in Open3D Tensor
value_np = value_torch_lhs.cpu().numpy().view(np.float32)

# Process
tsdf = value_np[:, :, :, :, 0]
weight = value_np[:, :, :, :, 1]

# Cross out some of the volumes for sanity check
tsdf[:, :, :, 0] = 0

# Put back value
value_np = np.stack((tsdf, weight), axis=-1)
value_np = value_np.view(np.uint8)
value_torch_rhs = torch.from_numpy(value_np).cuda()
value_torch_lhs[:] = value_torch_rhs[:]

mesh = tsdf_grid.extract_surface_mesh()
o3d.io.write_triangle_mesh('mesh.ply', mesh.to_legacy_triangle_mesh())
