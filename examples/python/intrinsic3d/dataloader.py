import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

path = '/home/wei/Workspace/data/intrinsic3d/lion-rgbd/'

K_depth = np.array([577.871, 0, 319.623, 0, 580.258, 239.624, 0, 0,
                    1]).reshape(3, 3)
K_color = np.array([1170.19, 0, 647.75, 0, 1170.19, 483.75, 0, 0,
                    1]).reshape(3, 3)

color_fnames = sorted(os.listdir(path + 'color'))
depth_fnames = sorted(os.listdir(path + 'depth'))

with open('keyframes.txt') as f:
    lines = f.readlines()

color_kfnames = []
depth_kfnames = []

for i, line in enumerate(lines):
    weight, mask = line.strip().split(' ')
    if int(mask):
        color_kfnames.append(color_fnames[i])
        depth_kfnames.append(depth_fnames[i])

for color_kfname in color_kfnames:
    color = o3d.io.read_image(f'{path}/color/{color_kfname}')
    h, w, _ = np.asarray(color).shape

for color_kfname, depth_kfname in zip(color_kfnames, depth_kfnames):
    depth = o3d.io.read_image(f'{path}/depth/{depth_kfname}')
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
    color = o3d.io.read_image(f'{path}/color/{color_kfname}')

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth_resized, 1000.0, 2.0, False)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K_color[0, 0],
                                                  K_color[1, 1], K_color[0, 2],
                                                  K_color[1, 2])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    o3d.visualization.draw_geometries([pcd])
