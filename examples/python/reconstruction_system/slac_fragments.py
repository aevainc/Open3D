# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import math
import os, sys
import time
import open3d as o3d
import argparse

sys.path.append("../utility")
from file import *
sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_path',
        type=str,
        help='path to the dataset.'
        'It should contain 16bit depth images in a folder named depth/'
        'and rgb images in a folder named color/ or rgb/')
    parser.add_argument('trajectory_path',
                        type=str,
                        help='path to the trajectory in open3d\'s .log format')
    parser.add_argument('--intrinsic_path',
                        type=str,
                        help='path to the intrinsic.json config file.'
                        'By default PrimeSense intrinsics is used.')
    parser.add_argument('--fragment_size', type=int, default=50)
    parser.add_argument(
        '--block_count',
        type=int,
        default=2000,
        help='estimated number of 16x16x16 voxel blocks to represent a scene.'
        'Typically with a 6mm resolution,'
        'a lounge scene requires around 30K blocks,'
        'while a large apartment requires 80K blocks.'
        'Open3D will dynamically increase the block count on demand,'
        'but a rough upper bound will be useful especially when memory is limited.'
    )
    parser.add_argument(
        '--voxel_size',
        type=float,
        default=3.0 / 512,
        help='voxel resolution.'
        'For small scenes, 6mm preserves fine details.'
        'For large indoor scenes, 1cm or larger will be reasonable for limited memory.'
    )
    parser.add_argument(
        '--depth_scale',
        type=float,
        default=1000.0,
        help='depth factor. Converting from a uint16 depth image to meter.')
    parser.add_argument('--max_depth',
                        type=float,
                        default=3.0,
                        help='max range in the scene to integrate.')
    parser.add_argument('--sdf_trunc',
                        type=float,
                        default=0.04,
                        help='SDF truncation threshold.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='.')
    args = parser.parse_args()

    device = o3d.core.Device(args.device)

    # Load RGBD
    [color_files, depth_files] = get_rgbd_file_lists(args.dataset_path)

    # Load intrinsics
    if args.intrinsic_path is None:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    else:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(args.intrinsic_path)

    intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix,
                                o3d.core.Dtype.Float32, device)

    # Load extrinsics
    trajectory = read_poses_from_log(args.trajectory_path)

    n_files = len(color_files)
    n_frags = n_files // args.fragment_size

    os.makedirs(args.output, exist_ok=True)

    # Setup volume
    for f in range(n_frags):
        volume = o3d.t.geometry.TSDFVoxelGrid(
            {
                'tsdf': o3d.core.Dtype.Float32,
                'weight': o3d.core.Dtype.UInt16,
                'color': o3d.core.Dtype.UInt16
            },
            voxel_size=args.voxel_size,
            sdf_trunc=args.sdf_trunc,
            block_resolution=16,
            block_count=args.block_count,
            device=device)

        pose_graph = o3d.pipelines.registration.PoseGraph()
        extrinsics = [np.linalg.inv(trajectory[i]) for i in range(n_files)]

        start = time.time()
        fragment_extrinsic = extrinsics[f * args.fragment_size]

        f_start = f * args.fragment_size
        f_end = (f + 1) * args.fragment_size
        if f == n_frags - 1:
            f_end = n_files

        for i in range(f_start, f_end):
            rgb = o3d.io.read_image(color_files[i])
            rgb = o3d.t.geometry.Image.from_legacy_image(rgb, device=device)

            depth = o3d.io.read_image(depth_files[i])
            depth = o3d.t.geometry.Image.from_legacy_image(depth, device=device)

            extrinsic = o3d.core.Tensor(extrinsics[i], o3d.core.Dtype.Float32,
                                        device)


            volume.integrate(depth, rgb, intrinsic, extrinsic, args.depth_scale,
                             args.max_depth)
            pose = fragment_extrinsic @ np.linalg.inv(extrinsics[i])
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))

        end = time.time()

        print('Integration ({:03d}-{:03d}) {:03d}/{:03d} takes {:.3f} ms'.format(
            f_start, f_end, f, n_frags, (end - start)))
        pcd = volume.extract_surface_points().to_legacy_pointcloud()
        pcd.transform(fragment_extrinsic)
        o3d.io.write_point_cloud('{}/fragment_{:03d}.ply'.format(args.output, f), pcd)
        o3d.io.write_pose_graph('{}/fragment_optimized_{:03d}.json'.format(args.output, f), pose_graph)
