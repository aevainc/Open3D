open3d_ispc_add_executable(tests)

target_sources(tests PRIVATE
    Main.cpp
)

target_compile_definitions(tests PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/test_data")

target_link_libraries(tests PRIVATE
    Open3D::Open3D
    Open3D::3rdparty_jsoncpp
    Open3D::3rdparty_googletest
    Open3D::3rdparty_threads
)

if (TARGET Open3D::3rdparty_openmp)
    target_link_libraries(tests PRIVATE
        Open3D::3rdparty_openmp
    )
endif()

open3d_show_and_abort_on_warning(tests)
open3d_set_global_properties(tests)

