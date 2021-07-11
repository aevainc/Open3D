import numpy as np
import argparse

from tsdf_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--spatial', default='voxels_spatial.npz')
    parser.add_argument('--input', default='colored_voxels_fine.npz')
    args = parser.parse_args()

    # Copy and simple modification
    spatial = np.load(args.spatial)
    volume_mask = spatial['volume_mask']

    input_data = np.load(args.input)

    voxel_tsdf_refined = input_data['voxel_tsdf']
    voxel_color = input_data['voxel_color']

    keys, values = load_tsdf_kv(args.path + '/tsdf.json')
    tsdf = values[:, :, :, :, 0]
    weight = values[:, :, :, :, 1]
    color = np.zeros((*tsdf.shape, 3))

    # Update geometry
    tsdf[volume_mask] = voxel_tsdf_refined
    color[volume_mask] = voxel_color

    # Build new volume and extract mesh
    values_with_color = tsdf_value_merge_color(tsdf, weight, color)
    colored_volume = construct_colored_tsdf_volume(keys, values_with_color)
    mesh = colored_volume.extract_surface_mesh()
    mesh = mesh.to_legacy_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])

    o3d.io.write_triangle_mesh(''.join(args.input.split('.')[:-1]) + '.ply',
                               mesh)
