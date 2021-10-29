#
# Open3D 3rd party library integration
#
set(Open3D_3RDPARTY_DIR "${CMAKE_CURRENT_LIST_DIR}")

# EXTERNAL_MODULES
# CMake modules we depend on in our public interface. These are modules we
# need to find_package() in our CMake config script, because we will use their
# targets.
set(Open3D_3RDPARTY_EXTERNAL_MODULES)

# PUBLIC_TARGETS
# CMake targets we link against in our public interface. They are
# either locally defined and installed, or imported from an external module
# (see above).
set(Open3D_3RDPARTY_PUBLIC_TARGETS)

# PRIVATE_TARGETS
# CMake targets for dependencies which are not exposed in the public API. This
# will include anything else we use internally.
set(Open3D_3RDPARTY_PRIVATE_TARGETS)

find_package(PkgConfig QUIET)

function(open3d_link_3rdparty_libraries target)
    # Directly pass public and private dependencies to the target.
    target_link_libraries(${target} PRIVATE ${Open3D_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${Open3D_3RDPARTY_PUBLIC_TARGETS})
endfunction()

function(open3d_set_global_properties target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
        $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}>
    )

    if (GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    else()
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    endif()
endfunction()


# DPC++
add_library(SYCL INTERFACE)
target_compile_options(SYCL INTERFACE
    $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<COMPILE_LANGUAGE:ISPC>>>:-fsycl -fsycl-unnamed-lambda>)
target_link_libraries(SYCL INTERFACE
    $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<LINK_LANGUAGE:ISPC>>>:sycl -fsycl>)
add_library(Open3D::SYCL ALIAS SYCL)
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS Open3D::SYCL)

# Compactify list of external modules.
# This must be called after all dependencies are processed.
list(REMOVE_DUPLICATES Open3D_3RDPARTY_EXTERNAL_MODULES)
