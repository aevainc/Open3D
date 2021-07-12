import open3d as o3d
import os
import numpy as np
from pathlib import Path

pwd = Path(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    pcd_file = str(pwd / "examples/test_data/fragment.ply")
    pcd = o3d.io.read_point_cloud(pcd_file)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()

    camera = ctr.convert_to_pinhole_camera_parameters()
    print("intrinsic width:\n", camera.intrinsic.width)
    print("intrinsic height:\n", camera.intrinsic.height)
    print("intrinsic matrix:\n", camera.intrinsic.intrinsic_matrix)
    print("extrinsic:\n", camera.extrinsic)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(pwd / "before.png"))

    # Set new camera parameters, you may modify from the original camera params.
    # Intrinsic parameters has to match window size typically, see:
    # ViewControl::ConvertFromPinholeCameraParameters.
    camera.extrinsic = np.array([
        [1., 0., 0., -2],
        [-0., 1., -0., -1],
        [-0., -0., 1., 2],
        [0., 0., 0., 1.],
    ])
    print("new intrinsic width:\n", camera.intrinsic.width)
    print("new intrinsic height:\n", camera.intrinsic.height)
    print("new intrinsic matrix:\n", camera.intrinsic.intrinsic_matrix)
    print("new extrinsic:\n", camera.extrinsic)
    ctr.convert_from_pinhole_camera_parameters(camera)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(pwd / "after.png"))

    vis.destroy_window()
