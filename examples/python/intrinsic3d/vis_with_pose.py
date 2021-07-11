import open3d as o3d
import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import rgbd_util
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_file')
    parser.add_argument('log_file')
    parser.add_argument('key_file')
    parser.add_argument('output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    poses = rgbd_util.load_poses(args.log_file)

    poses_kf = []
    indices_kf = []

    with open(args.key_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        weight, mask = line.strip().split(' ')
        if int(mask):
            poses_kf.append(poses[i])
            indices_kf.append(i)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1296, height=968)
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()

    for pose, idx in zip(poses_kf, indices_kf):
        print(idx)
        camera = ctr.convert_to_pinhole_camera_parameters()
        camera.extrinsic = np.linalg.inv(pose)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.width = 1296
        intrinsic.height = 968
        # optimized
        # intrinsic.intrinsic_matrix = np.array([[1174.89, 0, 648.608],
        #                                        [0, 1167.31, 483.56], [0, 0, 1]])

        # init
        intrinsic.intrinsic_matrix = np.array([[1170.19, 0, 647.75],
                                               [0, 1170.19, 483.75], [0, 0, 1]])

        camera.intrinsic = intrinsic
        ctr.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("{}/{:03d}.png".format(args.output, idx))

    vis.destroy_window()
