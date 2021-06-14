import open3d as o3d
import numpy as np

import argparse
from pcd_util import make_o3d_pcd

voxel_size = 0.004
sdf_trunc = 0.015


def load_tsdf_kv(path_tsdf_json):
    # Don't load json directly, as it may change the order of key-values
    prefix = ''.join(path_tsdf_json.split('.')[:-1])
    path_key = prefix + '.hash.key.npy'
    path_value = prefix + '.hash.value.npy'

    keys = np.load(path_key)

    # [:,:,:,:,0]: tsdf, [:,:,:,:,1]: weight, as float
    values = np.load(path_value).view(np.float32)

    return keys, values


def generate_voxel_coords(keys, mask):
    n_blocks = len(keys)

    # Generate spatially-hashed block-wise coordinates
    block_x = keys[:, 0:1]
    block_y = keys[:, 1:2]
    block_z = keys[:, 2:3]

    block_x = np.tile(block_x, (1, 512)).reshape(-1, 8, 8, 8)
    block_y = np.tile(block_y, (1, 512)).reshape(-1, 8, 8, 8)
    block_z = np.tile(block_z, (1, 512)).reshape(-1, 8, 8, 8)

    # Generate local volume in zyx order
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    x = np.tile(x, (8, 1)).reshape(8, 8, 8)
    y = np.tile(y, (8, 1)).reshape(8, 8, 8)
    z = np.expand_dims(np.arange(8), axis=-1)
    z = np.tile(z, (1, 64)).reshape(8, 8, 8)

    voxel_x = np.tile(x.reshape(-1, 8, 8, 8), (n_blocks, 1, 1, 1))
    voxel_y = np.tile(y.reshape(-1, 8, 8, 8), (n_blocks, 1, 1, 1))
    voxel_z = np.tile(z.reshape(-1, 8, 8, 8), (n_blocks, 1, 1, 1))

    # Compute voxel coordinates
    global_x = block_x * 8 + voxel_x
    global_y = block_y * 8 + voxel_y
    global_z = block_z * 8 + voxel_z

    x = global_x[mask].flatten()
    y = global_y[mask].flatten()
    z = global_z[mask].flatten()

    return np.stack((x, y, z)).T


def select_voxels(tsdf, weight, sdf_thr, weight_thr=0):
    mask_tsdf = np.abs(tsdf) < sdf_thr
    mask_weight = weight > weight_thr
    return mask_tsdf & mask_weight


def tsdf_value_merge_color(tsdf, weight, color):
    # TSDF: (N, 8, 8, 8, dtype=float) -> (N, 8, 8, 8, 1, dtype=float),
    # Weight: (N, 8, 8, 8, dtype=float) -> (N, 8, 8, 8, 1, dtype=uint16)
    # Color: (N, 8, 8, 8, 3, dtype=float) -> (N, 8, 8, 8, 3, dtype=uint16)
    # Note: assuming color is in range(0, 1) per channel
    # Every array should be contiguous and viewed in uint8
    tsdf_c = np.ascontiguousarray(np.expand_dims(tsdf, axis=-1)).view(np.uint8)
    weight_c = np.ascontiguousarray(np.expand_dims(weight, -1)).astype(
        np.uint16).view(np.uint8)
    color = ((color * 65535).astype(np.uint16)).view(np.uint8)
    return np.concatenate((tsdf_c, weight_c, color), axis=-1)


def construct_colored_tsdf_volume(keys, values_with_color):
    colored_volume = o3d.t.geometry.TSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        block_resolution=8,
        block_count=len(keys))

    hashmap = colored_volume.get_block_hashmap()
    hashmap.insert(o3d.core.Tensor(keys), o3d.core.Tensor(values_with_color))

    return colored_volume
