import numpy as np
import argparse

from tsdf_util import *
from voxel_util import *
from rgbd_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset')
    parser.add_argument('--spatial', default='voxels_spatial.npz')

    parser.add_argument('--input', default='colored_voxels_coarse.npz')
    parser.add_argument('--output', default='colored_voxels_fine.npz')
    args = parser.parse_args()

    # Load data
    spatial = np.load(args.spatial)
    voxel_coords = spatial['voxel_coords']
    voxel_nbs = get_nb_dict(spatial)

    input_data = np.load(args.input)
    voxel_tsdf = input_data['voxel_tsdf']
    voxel_color = input_data['voxel_color']

    n_voxel = len(voxel_tsdf)

    # Mask 1: voxel with a valid normal from neighbors
    voxel_normals, voxel_normal_mask = compute_normals(voxel_tsdf, voxel_nbs)
    valid_voxel_indices = np.arange(n_voxel)[voxel_normal_mask]

    voxel_surface_pts = compute_nearest_surface(
        voxel_coords[voxel_normal_mask],
        voxel_tsdf[voxel_normal_mask] * sdf_trunc,
        voxel_normals[voxel_normal_mask])
    voxel_surfaces = np.zeros((n_voxel, 3))
    voxel_surfaces[valid_voxel_indices] = voxel_surface_pts

    print('Loading keyframes ...')
    colors, depths, poses = load_keyframes(args.path_dataset, check=False)
    print('Loading finished.')

    n_kf = len(poses)
    corres_mask = np.zeros((n_kf, n_voxel), dtype=bool)
    corres_weight = np.zeros((n_kf, n_voxel))
    corres_color = np.zeros((n_kf, n_voxel, 3))

    for i, (color, depth, pose) in enumerate(zip(colors, depths, poses)):
        mask, weight, color = project(voxel_surfaces[voxel_normal_mask],
                                      color,
                                      depth,
                                      pose,
                                      normal=voxel_normals[voxel_normal_mask])
        corres_mask[i, valid_voxel_indices] = mask
        corres_weight[i, valid_voxel_indices] = weight
        corres_color[i, valid_voxel_indices] = color / 255.0

    # At least one; at most 5 correspondences.
    # Corres mask: (n_voxels)
    t_max = 5
    corres_mask_saved = corres_mask

    corres_mask = np.logical_and(
        corres_mask_saved.sum(axis=0) >= 1,
        corres_weight.sum(axis=0) > 0)
    valid_voxel_indices = np.arange(n_voxel)[corres_mask]

    # (t_max, num of valid voxels (both for normal estimation and with correspondences))
    corres_kf_sorted = np.argsort(-corres_weight, axis=0)

    sum_color = np.zeros((n_voxel, 3))
    sum_weight = np.zeros((n_voxel, 1))

    for k in range(t_max):
        kf_sel = corres_kf_sorted[k, valid_voxel_indices]
        w = corres_weight[kf_sel, valid_voxel_indices]
        c = corres_color[kf_sel, valid_voxel_indices]
        sum_color[valid_voxel_indices] += np.expand_dims(w, axis=-1) * c
        sum_weight[valid_voxel_indices] += np.expand_dims(w, axis=-1)

    voxel_color[valid_voxel_indices] = sum_color[
        valid_voxel_indices] / sum_weight[valid_voxel_indices]

    pcd = make_o3d_pcd(voxel_surfaces[corres_mask],
                       normals=None,
                       colors=voxel_color[corres_mask])
    o3d.visualization.draw_geometries([pcd])

    np.savez(args.output, voxel_tsdf=voxel_tsdf, voxel_color=voxel_color)
