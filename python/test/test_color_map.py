import open3d as o3d
import numpy as np
import re
import os
import sys
from open3d_test import download_fountain_dataset


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


def test_color_map():
    """
    Hard-coded values are from the 0.12 release. We expect the values to match
    exactly when OMP_NUM_THREADS=1. If more threads are used, there could be
    some small numerical differences.
    """

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

    camera = o3d.io.read_pinhole_camera_trajectory(
        os.path.join(path, "scene/key.log"))
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(path, "scene", "integrated.ply"))

    # Computes averaged color without optimization
    option = o3d.pipelines.color_map.ColorMapOptimizationOption()
    option.maximum_iteration = 0
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images,
                                                       camera, option)
    vertex_mean = np.mean(np.asarray(mesh.vertex_colors), axis=0)
    extrinsic_mean = np.array([c.extrinsic for c in camera.parameters
                              ]).mean(axis=0)
    print(f"vertex_mean: \n{vertex_mean}")
    print(f"extrinsic_mean: \n{extrinsic_mean}")

    # Rigid Optimization
    option.maximum_iteration = 10
    option.non_rigid_camera_coordinate = False
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images,
                                                       camera, option)
    vertex_mean = np.mean(np.asarray(mesh.vertex_colors), axis=0)
    extrinsic_mean = np.array([c.extrinsic for c in camera.parameters
                              ]).mean(axis=0)
    print(f"vertex_mean: \n{vertex_mean}")
    print(f"extrinsic_mean: \n{extrinsic_mean}")

    # Non-rigid Optimization
    option.maximum_iteration = 10
    option.non_rigid_camera_coordinate = True
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images,
                                                       camera, option)
    vertex_mean = np.mean(np.asarray(mesh.vertex_colors), axis=0)
    extrinsic_mean = np.array([c.extrinsic for c in camera.parameters
                              ]).mean(axis=0)
    print(f"vertex_mean: \n{vertex_mean}")
    print(f"extrinsic_mean: \n{extrinsic_mean}")
