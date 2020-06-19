# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(TGeometry OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(TGeometry)
open3d_set_global_properties(TGeometry)
open3d_link_3rdparty_libraries(TGeometry)

target_include_directories(TGeometry PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(TestTPointCloud TestTPointCloud.cpp)
target_link_libraries(TestTPointCloud ${CMAKE_PROJECT_NAME})

add_executable(TestTImage TestTImage.cpp)
target_link_libraries(TestTImage ${CMAKE_PROJECT_NAME})

add_executable(TestMemoryManager TestMemoryManager.cpp)
target_link_libraries(TestMemoryManager ${CMAKE_PROJECT_NAME})
