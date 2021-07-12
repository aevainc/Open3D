import open3d as o3d
import open3d.visualization.rendering as rendering
import os
import numpy as np
from pathlib import Path

pwd = Path(os.path.dirname(os.path.realpath(__file__)))


def main():
    render = rendering.OffscreenRenderer(640, 480)

    pcd_file = str(pwd / "examples/test_data/fragment.ply")
    pcd = o3d.io.read_point_cloud(pcd_file)

    material = rendering.Material()
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.shader = "defaultLit"

    render.scene.add_geometry("pcd", pcd, material)
    render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])

    img = render.render_to_image()
    o3d.io.write_image("test.png", img, 9)

    render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
    img = render.render_to_image()
    o3d.io.write_image("test2.png", img, 9)


if __name__ == "__main__":
    main()
