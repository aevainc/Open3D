import numpy as np
import argparse

from tsdf_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_voxel', default='voxels.npz')
    parser.add_argument('--path_voxel_refined', default='voxels_refined.npz')
    args = parser.parse_args()

    # Load data
    data = np.load(args.path_voxel)
    mask = data['mask']
    tsdf = data['tsdf']
    voxel_coords = data['voxel_coords']

    # TODO: optimize tsdf with intrinsic3d
    tsdf_refined = tsdf * 0.5

    # TODO: optimize albedo with intrinsic3d
    albedo_refined = np.ones(len(tsdf)) * 0.6

    # TODO: optimize/assign color with projection model
    color_refined = np.zeros((len(tsdf), 3))
    color_refined[:, 1] = 1

    np.savez(args.path_voxel_refined,
             mask=mask,
             tsdf=tsdf,
             albedo=albedo_refined,
             color=color_refined)
