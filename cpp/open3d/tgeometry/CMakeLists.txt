# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(tgeometry OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(tgeometry)
open3d_set_global_properties(tgeometry)
open3d_link_3rdparty_libraries(tgeometry)

target_include_directories(tgeometry PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
