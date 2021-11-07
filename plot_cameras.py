import numpy as np
from pathlib import Path
import open3d as o3d


def cameracenter_from_translation(R, t):
    # - R.T @ t
    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()


def cameracenter_from_T(T):
    R, t = T[:3, :3], T[:3, 3]
    C = cameracenter_from_translation(R, t)
    return C


def get_camera_frame(T, size, color):

    R, t = T[:3, :3], T[:3, 3]

    C0 = cameracenter_from_translation(R, t).ravel()
    C1 = (C0 + R.T.dot(
        np.array([[-size], [-size], [3 * size]], dtype=np.float32)).ravel())
    C2 = (C0 + R.T.dot(
        np.array([[-size], [+size], [3 * size]], dtype=np.float32)).ravel())
    C3 = (C0 + R.T.dot(
        np.array([[+size], [+size], [3 * size]], dtype=np.float32)).ravel())
    C4 = (C0 + R.T.dot(
        np.array([[+size], [-size], [3 * size]], dtype=np.float32)).ravel())

    ls = o3d.geometry.LineSet()
    points = np.array([C0, C1, C2, C3, C4])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls


def get_camera_frames(Ts, size=0.1, color=np.array([0, 0, 1])):
    camera_frames = o3d.geometry.LineSet()
    for T in Ts:
        camera_frame = get_camera_frame(T, size=size, color=color)
        camera_frames += camera_frame
    return camera_frames


def get_camera_centers_lineset(Ts, color=np.array([1, 0, 0])):
    num_nodes = len(Ts)
    camera_centers = [cameracenter_from_T(T) for T in Ts]

    ls = o3d.geometry.LineSet()
    lines = [[x, x + 1] for x in range(num_nodes - 1)]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(camera_centers)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls


if __name__ == "__main__":

    test_data_dir = Path.home() / "repo/Open3D/examples/test_data"
    camera_trajectory_path = test_data_dir / "camera_trajectory.json"
    pcd_path = test_data_dir / "fragment.ply"

    trajectory = o3d.io.read_pinhole_camera_trajectory(
        str(camera_trajectory_path))
    Ts = [param.extrinsic for param in trajectory.parameters]

    camera_centers_ls = get_camera_centers_lineset(Ts)
    camera_frames = get_camera_frames(Ts, size=0.02)
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    o3d.visualization.draw_geometries([pcd, camera_centers_ls, camera_frames])
