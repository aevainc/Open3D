add_library(Open3D)

target_sources(Open3D PRIVATE
    Timer.cpp
)
target_compile_options(Open3D PRIVATE -fsycl -fsycl-unnamed-lambda)
target_link_libraries(Open3D PRIVATE sycl -fsycl)

open3d_show_and_abort_on_warning(Open3D)
open3d_set_global_properties(Open3D)
open3d_link_3rdparty_libraries(Open3D)

include(CMakePackageConfigHelpers)




