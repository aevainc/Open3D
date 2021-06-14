import open3d as o3d
import numpy as np

def make_o3d_pcd(xyz, normals=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        if len(colors.shape) == 1 or colors.shape[1] == 1:
            colors = np.tile(colors.squeeze(), (1, 3)).reshape(3, -1).T

        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
