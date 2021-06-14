import numpy as np
import argparse

from tsdf_util import *
from voxel_util import *
from rgbd_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset')
    parser.add_argument('--path_voxel', default='voxels.npz')
    parser.add_argument('--path_voxel_init', default='voxels_init.npz')
    args = parser.parse_args()

    # Load data
    data = np.load(args.path_voxel)
    tsdf = data['tsdf']

    voxel_coords = data['voxel_coords'] * voxel_size

    colors, depths, poses = load_keyframes(args.path_dataset, check=False)

    corres_mask = np.zeros((len(poses), len(voxel_coords)), dtype=bool)
    corres_weight = np.zeros((len(poses), len(voxel_coords)))
    corres_color = np.zeros((len(poses), len(voxel_coords), 3))

    for i, (color, depth, pose) in enumerate(zip(colors, depths, poses)):
        mask, weight, color = project(voxel_coords, color, depth, pose)

        corres_mask[i] = mask
        corres_weight[i] = weight
        corres_color[i] = color / 255.0

    t_max = 5
    corres_mask_valid = (corres_mask.sum(axis=0) >=
                         1) & (corres_weight.sum(axis=0) > 0)
    corres_indices_sorted = np.argsort(-corres_mask.astype(int), axis=0)[:t_max]

    # pick up max weights
    pcd_indices = np.arange(len(voxel_coords),
                            dtype=np.int64)[corres_mask_valid]

    c_sum = np.zeros((len(pcd_indices), 3))
    w_sum = np.zeros((len(pcd_indices), 1))
    for i in range(t_max):
        w = corres_weight[corres_indices_sorted[i, corres_mask_valid],
                          pcd_indices]
        c = corres_color[corres_indices_sorted[i, corres_mask_valid],
                         pcd_indices, :]
        c_sum += np.expand_dims(w, axis=-1) * c
        w_sum += np.expand_dims(w, axis=-1)

    pcd_colors = c_sum / w_sum
    pcd = make_o3d_pcd(voxel_coords[pcd_indices],
                       normals=None,
                       colors=pcd_colors)
    o3d.visualization.draw([pcd])

    colors = np.zeros((len(tsdf), 3))
    colors[pcd_indices] = pcd_colors

    np.savez('voxels_init.npz', tsdf=tsdf, color=colors)
