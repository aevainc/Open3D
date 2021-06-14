import numpy as np
import argparse

from tsdf_util import *
from voxel_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_tsdf')
    parser.add_argument('--spatial', default='voxels_spatial.npz')
    parser.add_argument('--output', default='voxels_init.npz')
    args = parser.parse_args()

    # Load
    keys, values = load_tsdf_kv(args.path_tsdf)

    # Separate
    tsdf = values[:, :, :, :, 0]
    weight = values[:, :, :, :, 1]

    # Select
    volume_mask = select_voxels(tsdf,
                                weight,
                                sdf_thr=(3 * voxel_size / sdf_trunc))

    # Generate coordinate for projection and hashing
    voxel_coords = generate_voxel_coords(keys, volume_mask)
    voxel_tsdf = tsdf[volume_mask]

    voxel_nbs = find_neighbors(voxel_coords)

    # Constant, wont' be changed during optimization
    np.savez(args.spatial,
             volume_mask=volume_mask,
             voxel_coords=voxel_coords * voxel_size,
             **voxel_nbs)

    # To be refined
    np.savez(args.output, voxel_tsdf=voxel_tsdf)
