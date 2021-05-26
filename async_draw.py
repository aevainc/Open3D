import open3d as o3d
import time

if __name__ == "__main__":
    o3d.visualization.webrtc_server.enable_webrtc()
    cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    o3d.visualization.draw(cube_red, non_blocking_and_return_uid=True)
    print("Cube red created")

    # cube_blue = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    # cube_blue.compute_vertex_normals()
    # cube_blue.paint_uniform_color((0.0, 0.0, 1.0))
    # o3d.visualization.draw(cube_blue, non_blocking_and_return_uid=True)
    # print("Cube blue created")

    while True:
        time.sleep(0.001)
