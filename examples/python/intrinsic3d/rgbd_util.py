import os
import numpy as np
import open3d as o3d
import argparse

K_depth = np.array([577.871, 0, 319.623, 0, 580.258, 239.624, 0, 0,
                    1]).reshape(3, 3)
K_color = np.array([1170.19, 0, 647.75, 0, 1170.19, 483.75, 0, 0,
                    1]).reshape(3, 3)


def load_poses(traj_log):
    trans_arr = []
    with open(traj_log) as f:
        content = f.readlines()

        # Load .log file.
        for i in range(0, len(content), 5):
            # format %d (src) %d (tgt) %f (fitness)
            data = list(map(float, content[i].strip().split(' ')))
            ids = (int(data[0]), int(data[1]))
            fitness = data[2]

            # format %f x 16
            T_gt = np.array(
                list(map(float, (''.join(
                    content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

            trans_arr.append(T_gt)

    return trans_arr


def load_keyframes(path_dataset, check=True):
    # Load all filenames
    color_fnames = sorted(os.listdir(path_dataset + '/color'))
    depth_fnames = sorted(os.listdir(path_dataset + '/depth'))
    poses = load_poses(path_dataset + '/trajectory.log')
    keyframes_fname = path_dataset + '/keyframes.txt'

    # Load keyframes description txt
    with open(keyframes_fname) as f:
        lines = f.readlines()

    color_kfnames = []
    depth_kfnames = []
    poses_kf = []
    for i, line in enumerate(lines):
        weight, mask = line.strip().split(' ')
        if int(mask):
            color_kfnames.append(color_fnames[i])
            depth_kfnames.append(depth_fnames[i])
            poses_kf.append(poses[i])

    # Determine h, w
    color = o3d.io.read_image(f'{path_dataset}/color/{color_kfnames[0]}')
    h, w, _ = np.asarray(color).shape

    colors = []
    depths = []
    for i, (color_kfname,
            depth_kfname) in enumerate(zip(color_kfnames, depth_kfnames)):
        depth = o3d.io.read_image(f'{path_dataset}/depth/{depth_kfname}')
        depth_np = np.asarray(depth)
        dh, dw = depth_np.shape

        depth_np_resized = np.zeros((h, w), dtype=np.uint16)
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()

        # Unproject and project for resizing
        depth_xyz_resized = np.stack((u, v, np.ones_like(u)))
        projection = (K_depth @ np.linalg.inv(K_color)) @ depth_xyz_resized

        uu = projection[0].round().astype(int)
        vv = projection[1].round().astype(int)

        mask = (uu >= 0) & (uu < dw) & (vv >= 0) & (vv < dh)

        depth_np_resized[v[mask], u[mask]] = depth_np[vv[mask], uu[mask]]
        depth_resized = o3d.geometry.Image(depth_np_resized)

        depths.append(depth_resized)
        colors.append(o3d.io.read_image(f'{path_dataset}/color/{color_kfname}'))

    if check:
        pcd_map = o3d.geometry.PointCloud()

        for depth, color, pose in zip(depths, colors, poses_kf):
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, 1000.0, 2.0, False)

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                w, h, K_color[0, 0], K_color[1, 1], K_color[0, 2], K_color[1,
                                                                           2])
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic, np.linalg.inv(pose))
            pcd_map += pcd

        mesh = o3d.io.read_triangle_mesh('mesh_CUDA:0.ply')
        o3d.visualization.draw_geometries([pcd_map, mesh])

    return colors, depths, poses_kf


def project(pcd, color, depth, pose, normal=None):
    T = np.linalg.inv(pose)
    R = T[:3, :3]
    t = T[:3, 3:]

    projection = K_color @ (R @ pcd.T + t)

    u = (projection[0] / projection[2]).round().astype(int)
    v = (projection[1] / projection[2]).round().astype(int)

    color_np = np.asarray(color)
    depth_np = np.asarray(depth).astype(np.float32) / 1000.0
    h, w, _ = color_np.shape

    # mask at point's shape
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    corres_depth = np.zeros_like(mask, dtype=np.float64)
    corres_depth[mask] = depth_np[v[mask], u[mask]]

    mask[corres_depth == 0] = False
    color = np.zeros((len(u), 3), dtype=np.float64)
    color[mask] = color_np[v[mask], u[mask]]

    if normal is not None:
        view_angle = np.linalg.inv(K_color) @ np.stack((u, v, np.ones_like(u)))
        view_angle = view_angle / np.expand_dims(np.linalg.norm(view_angle), axis=0)
        normal = R @ normal.T

        dot = np.sum(-normal * view_angle, axis=0)

        weight = np.zeros_like(mask, dtype=np.float64)
        weight[mask] = dot[mask] / (corres_depth[mask]**2)
    else:
        weight = np.ones_like(mask, dtype=np.float64)

    return mask, weight, color
