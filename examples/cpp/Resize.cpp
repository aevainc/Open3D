// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <iostream>
#include <thread>

#include "open3d/Open3D.h"
#include "open3d/visualization/gui/BitmapWindowSystem.h"

using namespace open3d;

namespace open3d {
namespace visualization {

static void DrawAndResize() {
    auto bitmap_ws = std::make_shared<gui::BitmapWindowSystem>();
    gui::Application::GetInstance().SetWindowSystem(bitmap_ws);
    auto on_draw_callback = [](const gui::Window *window,
                               std::shared_ptr<geometry::Image> image) -> void {
        static int frame_id = 0;
        std::string file_name = fmt::format("im_{:04d}.png", frame_id++);
        io::WriteImage(file_name, *image);
        utility::LogInfo("Frame saved to {}", file_name);
    };
    bitmap_ws->SetOnWindowDraw(on_draw_callback);

    auto cube = geometry::TriangleMesh::CreateBox(1, 2, 4);
    visualization::gui::Application::GetInstance().Initialize();
    auto visualizer =
            std::make_shared<visualization::visualizer::O3DVisualizer>(
                    "Open3D", 640, 480);
    visualizer->AddGeometry("Cube", cube);
    visualizer->ResetCameraToDefault();
    visualizer->ShowSettings(true);
    visualization::gui::Application::GetInstance().AddWindow(visualizer);

    auto emulate_events = [bitmap_ws, visualizer]() -> void {
        auto emulate_mouse_events = [bitmap_ws, visualizer]() -> void {
            gui::MouseEvent me;
            me = gui::MouseEvent{gui::MouseEvent::Type::BUTTON_DOWN, 139, 366,
                                 0};
            me.button.button = gui::MouseButton::LEFT;
            bitmap_ws->PostMouseEvent(visualizer->GetOSWindow(), me);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            me = gui::MouseEvent{gui::MouseEvent::Type::DRAG, 209, 338, 0};
            me.move.buttons = 1;
            bitmap_ws->PostMouseEvent(visualizer->GetOSWindow(), me);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            me = gui::MouseEvent{gui::MouseEvent::Type::BUTTON_UP, 263, 318, 0};
            me.button.button = gui::MouseButton::LEFT;
            bitmap_ws->PostMouseEvent(visualizer->GetOSWindow(), me);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        };

        // Mouse drag
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        emulate_mouse_events();

        // Resize
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        bitmap_ws->SetWindowSize(visualizer->GetOSWindow(), 720, 480);

        // Mouse drag
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        emulate_mouse_events();
    };
    std::thread thead(emulate_events);

    visualizer.reset();
    visualization::gui::Application::GetInstance().Run();
}

}  // namespace visualization
}  // namespace open3d

int main(int argc, char **argv) {
    open3d::visualization::DrawAndResize();
    return 0;
}
