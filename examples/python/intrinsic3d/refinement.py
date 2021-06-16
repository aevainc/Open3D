import open3d as o3d
import torch
import torch.nn
import numpy as np

from rgbd_util import *
from lighting_util import *
from voxel_util import *
from pcd_util import *

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset')
    parser.add_argument('--spatial', default='voxels_spatial.npz')

    parser.add_argument('--input', default='colored_voxels_fine.npz')
    parser.add_argument('--output', default='colored_voxels_refined.npz')
    args = parser.parse_args()

    # Load data
    spatial = np.load(args.spatial)
    voxel_coords = spatial['voxel_coords']
    voxel_nbs = get_nb_dict(spatial)

    input_data = np.load(args.input)
    voxel_tsdf = input_data['voxel_tsdf']
    voxel_color = input_data['voxel_color']
    kf_association = input_data['kf_association']

    # Intensity estimation
    voxel_intensity = color_to_intensity(voxel_color)

    n_voxel = len(voxel_tsdf)
    voxel_normals, voxel_normal_mask = compute_normals(voxel_tsdf, voxel_nbs)
    valid_voxel_indices = np.arange(n_voxel)[voxel_normal_mask]
    voxel_albedo = np.ones_like(voxel_tsdf) * 0.6

    # Solve for lighting coeffs
    l = backward_sh(voxel_normals[voxel_normal_mask],
                    voxel_intensity[voxel_normal_mask],
                    voxel_albedo[voxel_normal_mask])
    print(f'illumination params = {l}')

    # Initialize with a uniform lighting (for now)
    voxel_illumination = forward_sh(l, voxel_normals[voxel_normal_mask])

    # Update albedo
    voxel_albedo[voxel_normal_mask] = voxel_intensity[
        voxel_normal_mask] / voxel_illumination

    voxel_chromaticity = voxel_color / np.expand_dims(voxel_intensity, axis=-1)
    voxel_chromaticity[np.isnan(voxel_chromaticity)] = 0
    mask_voxel_chromaticity = voxel_chromaticity > 0
    # Note: it is this in the original implementaion
    # weight_voxel_chromaticity = np.maximum(1 - voxel_chromaticity, 0.01)
    # weight_voxel_chromaticity[np.isnan(weight_voxel_chromaticity)] = 0
    # weight_voxel_chromaticity[np.isinf(weight_voxel_chromaticity)] = 0

    ## Setup optimization
    param_tsdf = torch.nn.Parameter(torch.from_numpy(voxel_tsdf).cuda())
    param_albedo = torch.nn.Parameter(torch.from_numpy(voxel_albedo).cuda())

    init_tsdf_const = torch.from_numpy(voxel_tsdf).cuda()
    lighting_const = torch.from_numpy(l).cuda()

    # Neighbor
    mask_plus = voxel_nbs['mask_xp'] & voxel_nbs['mask_yp'] & voxel_nbs[
        'mask_zp']
    mask_minus = voxel_nbs['mask_xm'] & voxel_nbs['mask_ym'] & voxel_nbs[
        'mask_zm']
    mask = mask_plus & mask_minus

    index_xp = torch.from_numpy(voxel_nbs['index_xp'][mask]).cuda()
    index_yp = torch.from_numpy(voxel_nbs['index_yp'][mask]).cuda()
    index_zp = torch.from_numpy(voxel_nbs['index_zp'][mask]).cuda()
    index_xm = torch.from_numpy(voxel_nbs['index_xm'][mask]).cuda()
    index_ym = torch.from_numpy(voxel_nbs['index_ym'][mask]).cuda()
    index_zm = torch.from_numpy(voxel_nbs['index_zm'][mask]).cuda()

    mask = torch.from_numpy(mask).cuda()

    # For data term: selection from all the parameters
    valid_normal_indices = torch.from_numpy(valid_voxel_indices).cuda()

    optimizer = torch.optim.Adam([param_tsdf, param_albedo], lr=1e-4)

    # Regularizer: all parameters
    voxel_chromaticity = torch.from_numpy(voxel_chromaticity).cuda()

    robust_kernel = lambda x: 1 / (1 + x)**3
    for i in range(100):
        # TODO: revisit colorize_util and get a better association for data term
        # TODO: figure out the mask for valid neighbors, in a chain or a brute-force search, or another hashmap?

        # TSDF stablizer
        loss_stable = (param_tsdf - init_tsdf_const)
        loss_stable = (loss_stable**2).sum()

        # TSDF laplacian regularizer
        dxx = param_tsdf[index_xp] + param_tsdf[index_xm] - 2 * param_tsdf[mask]
        dyy = param_tsdf[index_yp] + param_tsdf[index_ym] - 2 * param_tsdf[mask]
        dzz = param_tsdf[index_zp] + param_tsdf[index_zm] - 2 * param_tsdf[mask]
        loss_laplacian = dxx + dyy + dzz
        loss_laplacian = (loss_laplacian**2).sum()

        # Albedo regularizer
        # TODO: case-by-case, not a shared mask
        w_xp = (voxel_chromaticity[mask] -
                voxel_chromaticity[index_xp]).norm(dim=1)
        w_xm = (voxel_chromaticity[mask] -
                voxel_chromaticity[index_xm]).norm(dim=1)
        w_yp = (voxel_chromaticity[mask] -
                voxel_chromaticity[index_yp]).norm(dim=1)
        w_ym = (voxel_chromaticity[mask] -
                voxel_chromaticity[index_ym]).norm(dim=1)
        w_zp = (voxel_chromaticity[mask] -
                voxel_chromaticity[index_zp]).norm(dim=1)
        w_zm = (voxel_chromaticity[mask] -
                voxel_chromaticity[index_zm]).norm(dim=1)

        diff_albedo_xp = robust_kernel(w_xp) * (param_albedo[index_xp] -
                                                param_albedo[mask])**2
        diff_albedo_xm = robust_kernel(w_xm) * (param_albedo[index_xm] -
                                                param_albedo[mask])**2
        diff_albedo_yp = robust_kernel(w_yp) * (param_albedo[index_yp] -
                                                param_albedo[mask])**2
        diff_albedo_ym = robust_kernel(w_ym) * (param_albedo[index_ym] -
                                                param_albedo[mask])**2
        diff_albedo_zp = robust_kernel(w_zp) * (param_albedo[index_zp] -
                                                param_albedo[mask])**2
        diff_albedo_zm = robust_kernel(w_zm) * (param_albedo[index_zm] -
                                                param_albedo[mask])**2
        loss_albedo = diff_albedo_xp + diff_albedo_xm \
                    + diff_albedo_yp + diff_albedo_ym \
                    + diff_albedo_zp + diff_albedo_zm
        loss_albedo = loss_albedo.sum()

        # dummy
        loss = loss_stable + loss_laplacian + loss_albedo
        loss.backward()
        print(loss.item())
        optimizer.step()
