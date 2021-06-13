import numpy as np
import argparse

from tsdf_util import *
from voxel_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_tsdf')
    parser.add_argument('--path_voxel', default='voxels.npz')
    args = parser.parse_args()

    # Load
    keys, values = load_tsdf_kv(args.path_tsdf)

    # Separate
    tsdf = values[:, :, :, :, 0]
    weight = values[:, :, :, :, 1]

    # Select
    mask = select_voxels(tsdf, weight, sdf_thr=(2 * voxel_size / sdf_trunc))

    # Generate coordinate for projection and hashing
    tsdf = tsdf[mask]
    voxel_coords = generate_voxel_coords(keys, mask)

    nbs = find_neighbors(voxel_coords)

    np.savez(args.path_voxel,
             mask=mask,
             voxel_coords=voxel_coords,
             tsdf=tsdf,
             **nbs)
