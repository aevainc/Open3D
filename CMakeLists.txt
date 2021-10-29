cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS          "Build shared libraries"                   ON )
option(GLIBCXX_USE_CXX11_ABI      "Set -D_GLIBCXX_USE_CXX11_ABI=1 or O"      OFF)

project(Open3D VERSION 0.13 LANGUAGES C CXX)


set(Open3D_3RDPARTY_PRIVATE_TARGETS)

function(open3d_link_3rdparty_libraries target)
    target_link_libraries(${target} PRIVATE ${Open3D_3RDPARTY_PRIVATE_TARGETS})
endfunction()

function(open3d_set_global_properties target)
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

add_library(Open3D)
target_sources(Open3D PRIVATE
    Open3D.cpp
)
open3d_set_global_properties(Open3D)
open3d_link_3rdparty_libraries(Open3D)

add_executable(main)
target_sources(main PRIVATE
    Main.cpp
)
target_link_libraries(main PRIVATE
    Open3D
)
open3d_set_global_properties(main)
