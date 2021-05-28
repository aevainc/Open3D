import open3d as o3d
import functools
import threading
import functools
import time


class _AsyncEventLoop:

    class _Task:
        _g_next_id = 0

        def __init__(self, f):
            self.task_id = self._g_next_id
            self.func = f
            _AsyncEventLoop._Task._g_next_id += 1

    def __init__(self):
        # TODO (yixing): find a better solution. Currently py::print acquires
        # GIL which causes deadlock when AsyncEventLoop is used. By calling
        # reset_print_function(), all C++ prints will be directed to the
        # terminal while python print will still remain in the cell.
        o3d.utility.reset_print_function()
        self._lock = threading.Lock()
        self._run_queue = []
        self._return_vals = {}

        self._thread = threading.Thread(target=self._thread_main)
        self._thread.start()
        self._started = True

    def run_sync(self, f):
        with self._lock:
            task = _AsyncEventLoop._Task(f)
            self._run_queue.append(task)

        while True:
            with self._lock:
                if task.task_id in self._return_vals:
                    return self._return_vals[task.task_id]
            # Give up our timeslice (Windows requires non-zero to actually work),
            # so that we don't use 100% CPU here
            time.sleep(0.0001)

    #def _start_async(self, main_func):
    #    self._thread = threading.Thread(target=main_func)
    #    self._thread.start()
    #    self._started = True
    #    self._thread_main()

    def _thread_main(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        done = False
        while not done:
            with self._lock:
                for task in self._run_queue:
                    retval = task.func()
                    self._return_vals[task.task_id] = retval
                self._run_queue.clear()

            done = not app.run_one_tick()


# The _AsyncEventLoop class shall only be used to create a singleton instance.
# There are different ways to achieve this, here we use the module as a holder
# for singleton variables, see: https://stackoverflow.com/a/31887/1255535.
#
# Note: the _AsyncEventLoop is started whenever web_visualizer module is imported.
_async_event_loop = _AsyncEventLoop()


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


def main():
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

if __name__ == "__main__":
    o3d.visualization.webrtc_server.enable_webrtc()
    main()
