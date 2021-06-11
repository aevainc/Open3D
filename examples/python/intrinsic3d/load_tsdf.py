import open3d as o3d
import numpy as np

import torch
import torch.utils.dlpack

tsdf_grid = o3d.t.io.read_tsdf_voxelgrid('tsdf.json')
hashmap = tsdf_grid.get_block_hashmap()

