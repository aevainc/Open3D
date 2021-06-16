import open3d as o3d
import numpy as np
import argparse
import torch
import torch.nn

from tsdf_util import *
from lighting_util import *
from voxel_util import *
from rgbd_util import *


def to_torch(np_dict):
    torch_dict = {}
    for key in np_dict:
        torch_dict[key] = torch.from_numpy(np_dict[key]).cuda()

    return torch_dict


def compute_surface_and_normal(voxel_coords, param_tsdf, key, selection,
                               voxel_nbs):
    index_data = selection[key]
    nx = param_tsdf[voxel_nbs['index_xp'][index_data]] - param_tsdf[
        voxel_nbs['index_xm'][index_data]]
    ny = param_tsdf[voxel_nbs['index_yp'][index_data]] - param_tsdf[
        voxel_nbs['index_ym'][index_data]]
    nz = param_tsdf[voxel_nbs['index_zp'][index_data]] - param_tsdf[
        voxel_nbs['index_zm'][index_data]]
    norm = (nx**2 + ny**2 + nz**2).sqrt()

    normals = torch.stack((nx, ny, nz)).T / norm.unsqueeze(-1)
    surfaces = voxel_coords[index_data] - param_tsdf[index_data].unsqueeze(
        -1) * normals * sdf_trunc

    return surfaces, normals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset')

    # Selections
    parser.add_argument('--spatial', default='voxels_spatial.npz')
    parser.add_argument('--selection', default='tsdf_selection.npz')
    parser.add_argument('--association', default='tsdf_association.npz')

    parser.add_argument('--input', default='colored_voxels_fine.npz')
    parser.add_argument('--output', default='tsdf_association.npz')

    args = parser.parse_args()

    # Load data
    spatial = np.load(args.spatial)
    selection = to_torch(dict(np.load(args.selection)))
    voxel_nbs = to_torch(get_nb_dict(spatial))

    input_data = np.load(args.input)
    voxel_coords = torch.from_numpy(spatial['voxel_coords']).cuda()

    voxel_color = torch.from_numpy(input_data['voxel_tsdf']).cuda()
    voxel_tsdf = torch.from_numpy(input_data['voxel_color']).cuda()
    param_tsdf = torch.nn.Parameter(
        torch.from_numpy(input_data['voxel_tsdf']).cuda())

    # TODO: if it eats too much memory,
    # map index_data_xp to index_data_c in another hashmap in preprocessing
    surfaces_c, normals_c = compute_surface_and_normal(voxel_coords, param_tsdf,
                                                       'index_data_c',
                                                       selection, voxel_nbs)
    surfaces_xp, normals_xp = compute_surface_and_normal(
        voxel_coords, param_tsdf, 'index_data_xp', selection, voxel_nbs)
    surfaces_yp, normals_yp = compute_surface_and_normal(
        voxel_coords, param_tsdf, 'index_data_yp', selection, voxel_nbs)
    surfaces_zp, normals_zp = compute_surface_and_normal(
        voxel_coords, param_tsdf, 'index_data_zp', selection, voxel_nbs)

    pcd_c = make_o3d_pcd(surfaces_c.detach().cpu().numpy(),
                         normals=normals_c.detach().cpu().numpy())
    pcd_xp = make_o3d_pcd(surfaces_xp.detach().cpu().numpy(),
                          normals=normals_xp.detach().cpu().numpy())
    pcd_yp = make_o3d_pcd(surfaces_yp.detach().cpu().numpy(),
                          normals=normals_yp.detach().cpu().numpy())
    pcd_zp = make_o3d_pcd(surfaces_zp.detach().cpu().numpy(),
                          normals=normals_zp.detach().cpu().numpy())
    o3d.visualization.draw([pcd_c, pcd_xp, pcd_yp, pcd_zp])

    print('Loading keyframes ...')
    colors, depths, poses = load_keyframes(args.path_dataset, check=False)
    print('Loading finished.')
