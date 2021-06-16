import open3d as o3d
import numpy as np
import argparse

from tsdf_util import *
from lighting_util import *
from voxel_util import *
from rgbd_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset')
    parser.add_argument('--spatial', default='voxels_spatial.npz')

    # After color is initialized
    parser.add_argument('--selection', default='tsdf_selection.npz')
    parser.add_argument('--t_max', type=int, default=5)

    # Refined association with t-nb selection
    parser.add_argument('--input', default='colored_voxels_fine.npz')
    parser.add_argument('--output', default='tsdf_association.npz')

    args = parser.parse_args()

    # Load data
    spatial = np.load(args.spatial)
    selection = dict(np.load(args.selection))

    voxel_coords = spatial['voxel_coords']
    voxel_nbs = get_nb_dict(spatial)

    input_data = np.load(args.input)
    voxel_tsdf = input_data['voxel_tsdf']
    voxel_color = input_data['voxel_color']

    index_data = selection['index_data_c']
    nx = voxel_tsdf[voxel_nbs['index_xp'][index_data]] - voxel_tsdf[
        voxel_nbs['index_xm'][index_data]]
    ny = voxel_tsdf[voxel_nbs['index_yp'][index_data]] - voxel_tsdf[
        voxel_nbs['index_ym'][index_data]]
    nz = voxel_tsdf[voxel_nbs['index_zp'][index_data]] - voxel_tsdf[
        voxel_nbs['index_zm'][index_data]]
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    normals = np.stack((nx, ny, nz)).T / np.expand_dims(norm, axis=-1)

    voxel_surfaces = voxel_coords[index_data] - np.expand_dims(
        voxel_tsdf[index_data], axis=-1) * normals * sdf_trunc
    pcd = make_o3d_pcd(voxel_surfaces,
                       normals=normals,
                       colors=voxel_color[index_data])
    o3d.visualization.draw([pcd])

    print('Loading keyframes ...')
    colors, depths, poses = load_keyframes(args.path_dataset, check=False)
    print('Loading finished.')

    # Directly select from index_data according to projection
    n_kf = len(poses)
    n_voxel = len(index_data)

    corres_mask = np.zeros((n_kf, n_voxel), dtype=bool)
    corres_weight = np.zeros((n_kf, n_voxel))
    for i, (color, depth, pose) in enumerate(zip(colors, depths, poses)):
        mask, weight, _ = project(voxel_surfaces,
                                  color,
                                  depth,
                                  pose,
                                  normal=normals)
        corres_mask[i] = mask
        corres_weight[i] = weight
        # pcd = make_o3d_pcd(voxel_surfaces[mask],
        #                    normals=normals[mask],
        #                    colors=voxel_color[index_data][mask])
        # o3d.visualization.draw([pcd])

    sel = np.arange(n_voxel, dtype=np.int64)
    corres_kf_sorted = np.argsort(-corres_weight, axis=0)

    corres_final = np.zeros((n_kf, n_voxel), dtype=bool)
    for k in range(args.t_max):
        kf_sel = corres_kf_sorted[k]
        corres_final[kf_sel, sel] = corres_mask[kf_sel, sel]

    for i in range(n_kf):
        sel = corres_final[i]
        pcd = make_o3d_pcd(voxel_surfaces[sel],
                           normals=normals[sel],
                           colors=voxel_color[index_data][sel])
        o3d.visualization.draw([pcd])

    np.savez(args.output, corres_final=corres_final)
