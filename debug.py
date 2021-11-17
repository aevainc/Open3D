import open3d as o3d
import numpy as np
import re
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/python/test")
from open3d_test import download_fountain_dataset

assert o3d.__version__.startswith("0.13.0")
print(f"Testing with Open3D version {o3d.__version__}")


def load_fountain_dataset():

    def get_file_list(path, extension=None):

        def sorted_alphanum(file_list_ordered):
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [
                convert(c) for c in re.split('([0-9]+)', key)
            ]
            return sorted(file_list_ordered, key=alphanum_key)

        if extension is None:
            file_list = [
                path + f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
        else:
            file_list = [
                path + f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and
                os.path.splitext(f)[1] == extension
            ]
        file_list = sorted_alphanum(file_list)
        return file_list

    path = download_fountain_dataset()
    depth_image_path = get_file_list(os.path.join(path, "depth/"),
                                     extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image/"),
                                     extension=".jpg")
    assert (len(depth_image_path) == len(color_image_path))

    rgbd_images = []
    for i in range(len(depth_image_path)):
        depth = o3d.io.read_image(os.path.join(depth_image_path[i]))
        color = o3d.io.read_image(os.path.join(color_image_path[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
        os.path.join(path, "scene/key.log"))
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(path, "scene", "integrated.ply"))

    return mesh, rgbd_images, camera_trajectory


if __name__ == "__main__":
    # Load dataset
    mesh, rgbd_images, camera_trajectory = load_fountain_dataset()
    print(camera_trajectory.parameters[0].extrinsic)

    # Rigid Optimization
    start = time.time()
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, camera_trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory,
            o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=5))
    print(f"Rigid optimization takes {time.time() - start}")
    print(camera_trajectory.parameters[0].extrinsic)

    # Non-rigid Optimization
    start = time.time()
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory,
            o3d.pipelines.color_map.NonRigidOptimizerOption(
                maximum_iteration=5))
    print(f"Non-rigid optimization takes {time.time() - start}")
    print(camera_trajectory.parameters[0].extrinsic)
