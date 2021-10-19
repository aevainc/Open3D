# open3d_set_open3d_lib_properties() sets properties for the Open3D lib itself.
# This should be used for all object libraries that make up the Open3D lib.
#
# In comparison, open3d_set_global_properties() sets properties for the Open3D
# lib, and targets that links to the Open3D lib, e.g pybind, unit tests, etc.
function(open3d_set_open3d_lib_properties target)
    cmake_parse_arguments(arg "HIDDEN" "" "" ${ARGN})
    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(${target} PUBLIC OPEN3D_STATIC)
    endif()
    if (arg_HIDDEN)
        set_target_properties(${target} PROPERTIES
                        CXX_VISIBILITY_PRESET hidden
                        VISIBILITY_INLINES_HIDDEN ON
                        )
    else ()
        target_compile_definitions(${target} PRIVATE OPEN3D_ENABLE_DLL_EXPORTS)
    endif()
endfunction()


add_library(Open3D)

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
target_sources(Open3D PRIVATE
    Timer.cpp
)

# Source group for Visual Studio
open3d_show_and_abort_on_warning(Open3D)
open3d_set_global_properties(Open3D)
open3d_set_open3d_lib_properties(Open3D)
open3d_link_3rdparty_libraries(Open3D)

add_library(Open3D::Open3D ALIAS Open3D)

include(CMakePackageConfigHelpers)




