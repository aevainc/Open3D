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
    voxel_nbs = get_nb_dict(data)

    # mask 0: nb_mask, voxels with valid normals
    voxel_normals, nb_mask = compute_normals(tsdf, voxel_nbs)

    # Keep track of the indices for indexing, reserved for mesh assignment
    nb_indices = np.arange(len(tsdf))[nb_mask]

    voxel_surface_pts = compute_nearest_surface(voxel_coords[nb_mask],
                                                tsdf[nb_mask] * sdf_trunc,
                                                voxel_normals[nb_mask])
    pcd = make_o3d_pcd(voxel_surface_pts, voxel_normals[nb_mask])

    # For sanity check
    # mesh = o3d.io.read_triangle_mesh('mesh_CUDA:0.ply')
    # o3d.visualization.draw_geometries([pcd, mesh])

    colors, depths, poses = load_keyframes(args.path_dataset, check=False)

    # mask 1: correspondence_mask, voxels with sufficient k>=5 correspondences
    corres_mask = np.zeros((len(poses), len(voxel_surface_pts)), dtype=bool)
    corres_weight = np.zeros((len(poses), len(voxel_surface_pts)))
    corres_color = np.zeros((len(poses), len(voxel_surface_pts), 3))

    for i, (color, depth, pose) in enumerate(zip(colors, depths, poses)):
        mask, weight, color = project(voxel_surface_pts, voxel_normals[nb_mask],
                                      color, depth, pose)

        # pcd = make_o3d_pcd(voxel_surface_pts[mask],
        #                    normals=None,
        #                    colors=color[mask] / 255.0)
        # o3d.visualization.draw([pcd])

        corres_mask[i] = mask
        corres_weight[i] = weight
        corres_color[i] = color / 255.0

    t_max = 5
    corres_mask_valid = (corres_mask.sum(axis=0) >=
                         1) & (corres_weight.sum(axis=0) > 0)

    corres_nb_indices = nb_indices[corres_mask_valid]
    print(len(corres_nb_indices))

    corres_indices_sorted = np.argsort(-corres_weight, axis=0)[:t_max]

    # pick up max weights
    pcd_indices = np.arange(len(voxel_surface_pts),
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

    # corres_indices = np.argsort(corres_weight[:, corres_mask_valid], axis=0)[:5]
    # print(corres_indices.shape)
    # print(corres_indices)
    # print(corres_weight[corres_indices])

    pcd = make_o3d_pcd(voxel_surface_pts[pcd_indices],
                       normals=None,
                       colors=pcd_colors)
    o3d.visualization.draw([pcd])

    colors = np.zeros((len(tsdf), 3))
    colors[corres_nb_indices] = pcd_colors
    print(colors[corres_nb_indices])
    print(corres_nb_indices)
    print(colors.sum(axis=0))

    np.savez('voxels_init.npz', tsdf=tsdf, color=colors)
