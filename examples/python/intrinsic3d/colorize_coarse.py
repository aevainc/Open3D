import numpy as np
import argparse

from tsdf_util import *
from voxel_util import *
from rgbd_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset')
    parser.add_argument('--spatial', default='voxels_spatial.npz')

    parser.add_argument('--input', default='voxels_init.npz')
    parser.add_argument('--output', default='colored_voxels_coarse.npz')

    args = parser.parse_args()

    # Load data
    spatial = np.load(args.spatial)
    voxel_coords = spatial['voxel_coords']

    input_data = np.load(args.input)
    voxel_tsdf = input_data['voxel_tsdf']

    colors, depths, poses = load_keyframes(args.path_dataset, check=False)

    n_kf = len(poses)
    n_voxel = len(voxel_coords)
    corres_mask = np.zeros((n_kf, n_voxel), dtype=bool)
    corres_weight = np.zeros((n_kf, n_voxel))
    corres_color = np.zeros((n_kf, n_voxel, 3))

    for i, (color, depth, pose) in enumerate(zip(colors, depths, poses)):
        mask, weight, color = project(voxel_coords, color, depth, pose)

        corres_mask[i] = mask
        corres_weight[i] = weight
        corres_color[i] = color / 255.0

    corres_valid_mask = corres_weight.sum(axis=0) >= 1

    weighted_color = corres_color * np.expand_dims(corres_weight, axis=-1)
    sum_weighted_color = corres_color.sum(axis=0)
    sum_weight = corres_weight.sum(axis=0)
    voxel_color = sum_weighted_color / np.expand_dims(sum_weight, axis=-1)

    voxel_color[~corres_valid_mask] = 0

    pcd = make_o3d_pcd(voxel_coords[corres_valid_mask],
                       normals=None,
                       colors=voxel_color[corres_valid_mask])
    o3d.visualization.draw([pcd])

    np.savez(args.output, voxel_tsdf=voxel_tsdf, voxel_color=voxel_color)
