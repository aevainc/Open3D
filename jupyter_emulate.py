import open3d as o3d
from open3d.web_visualizer import _async_event_loop
import functools


def my_draw(geometry=None,
            title="Open3D",
            width=640,
            height=480,
            actions=None,
            lookat=None,
            eye=None,
            up=None,
            field_of_view=60.0,
            bg_color=(1.0, 1.0, 1.0, 1.0),
            bg_image=None,
            show_ui=None,
            point_size=None,
            animation_time_step=1.0,
            animation_duration=None,
            rpc_interface=False,
            on_init=None,
            on_animation_frame=None,
            on_animation_tick=None):
    """Draw in Jupyter Cell"""

    _async_event_loop.run_sync(
        functools.partial(o3d.visualization.draw,
                          geometry=geometry,
                          title=title,
                          width=width,
                          height=height,
                          actions=actions,
                          lookat=lookat,
                          eye=eye,
                          up=up,
                          field_of_view=field_of_view,
                          bg_color=bg_color,
                          bg_image=bg_image,
                          show_ui=show_ui,
                          point_size=point_size,
                          animation_time_step=animation_time_step,
                          animation_duration=animation_duration,
                          rpc_interface=rpc_interface,
                          on_init=on_init,
                          on_animation_frame=on_animation_frame,
                          on_animation_tick=on_animation_tick,
                          non_blocking_and_return_uid=True))

if __name__ == "__main__":
    o3d.visualization.webrtc_server.enable_webrtc()

    cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    my_draw(cube_red)
    print("Cube red created")

    cube_blue = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_blue.compute_vertex_normals()
    cube_blue.paint_uniform_color((0.0, 0.0, 1.0))
    my_draw(cube_blue)
    print("Cube blue created")
