cmake_minimum_required(VERSION 3.20)

option(GLIBCXX_USE_CXX11_ABI "Set -D_GLIBCXX_USE_CXX11_ABI=1 or O" OFF)

set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Open3D)

add_executable(main Main.cpp)
target_compile_options(main PRIVATE -fsycl -fsycl-unnamed-lambda)
target_link_libraries(main PRIVATE sycl -fsycl)

if (GLIBCXX_USE_CXX11_ABI)
    target_compile_definitions(main PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
else()
    target_compile_definitions(main PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
endif()
