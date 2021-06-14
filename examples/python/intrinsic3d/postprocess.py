import numpy as np
import argparse

from tsdf_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_tsdf')
    parser.add_argument('--path_voxel', default='voxels.npz')
    parser.add_argument('--path_voxel_refined', default='voxels_init.npz')
    args = parser.parse_args()

    # Copy and simple modification
    data = np.load(args.path_voxel)
    data_refined = np.load(args.path_voxel_refined)

    mask = data['mask']
    tsdf_refined = data_refined['tsdf']
    color_refined = data_refined['color']

    keys, values = load_tsdf_kv(args.path_tsdf)
    tsdf = values[:, :, :, :, 0]
    weight = values[:, :, :, :, 1]
    color = np.zeros((*tsdf.shape, 3))

    # Update geometry
    tsdf[mask] = tsdf_refined
    color[mask] = color_refined
    print(color_refined.sum(axis=0))

    # Build new volume and extract mesh
    values_with_color = tsdf_value_merge_color(tsdf, weight, color)
    colored_volume = construct_colored_tsdf_volume(keys, values_with_color)
    mesh = colored_volume.extract_surface_mesh()
    o3d.visualization.draw_geometries([mesh.to_legacy_triangle_mesh()])
