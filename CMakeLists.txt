cmake_minimum_required(VERSION 3.19.2)
# If you're using Ubuntu 18.04, we suggest you install the latest CMake from the
# official repository https://apt.kitware.com/.  CMake 3.19+ is required to:
# - allow linking with OBJECT libraries
# - prevent erroneous -gencode option deduplication with CUDA
# - simplify generator expressions for selecting compile flags and setting
#   global hardened link flags
# - parse JSON files for downloading files from the open3d_downloads repository
# - use first-class language support for ISPC (3.19.2 patch release required)

option(USE_ONE_API                "Use Intel OneAPI BaseKit"                 ON )
if(UNIX)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
endif(UNIX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Open3D build options
option(BUILD_SHARED_LIBS          "Build shared libraries"                   OFF)
option(BUILD_EXAMPLES             "Build Open3D examples programs"           OFF)
option(BUILD_UNIT_TESTS           "Build Open3D unit tests"                  OFF)
option(BUILD_BENCHMARKS           "Build the micro benchmarks"               OFF)
option(WITH_OPENMP                "Use OpenMP multi-threading"               ON )
option(GLIBCXX_USE_CXX11_ABI      "Set -D_GLIBCXX_USE_CXX11_ABI=1"           OFF)

# 3rd-party build options
option(USE_BLAS                   "Use BLAS/LAPACK instead of MKL"           ON )
option(USE_SYSTEM_EIGEN3          "Use system pre-installed eigen3"          OFF)
option(USE_SYSTEM_FMT             "Use system pre-installed fmt"             OFF)
option(USE_SYSTEM_GLEW            "Use system pre-installed glew"            OFF)
option(USE_SYSTEM_GLFW            "Use system pre-installed glfw"            OFF)
option(USE_SYSTEM_IMGUI           "Use system pre-installed imgui"           OFF)
option(USE_SYSTEM_JPEG            "Use system pre-installed jpeg"            OFF)
option(USE_SYSTEM_LIBLZF          "Use system pre-installed liblzf"          OFF)
option(USE_SYSTEM_PNG             "Use system pre-installed png"             OFF)
option(USE_SYSTEM_PYBIND11        "Use system pre-installed pybind11"        OFF)
option(USE_SYSTEM_QHULLCPP        "Use system pre-installed qhullcpp"        OFF)
option(USE_SYSTEM_TINYGLTF        "Use system pre-installed tinygltf"        OFF)
option(USE_SYSTEM_TINYOBJLOADER   "Use system pre-installed tinyobjloader"   OFF)
option(BUILD_FILAMENT_FROM_SOURCE "Build filament from source"               OFF)
option(PREFER_OSX_HOMEBREW        "Prefer Homebrew libs over frameworks"     ON )
option(WITH_FAISS                 "Enable Faiss"                             ON )

# Sensor options
option(BUILD_LIBREALSENSE         "Build support for Intel RealSense camera" OFF)
option(USE_SYSTEM_LIBREALSENSE    "Use system pre-installed librealsense"    OFF)
option(BUILD_AZURE_KINECT         "Build support for Azure Kinect sensor"    OFF)

# ML library options
option(BUILD_TENSORFLOW_OPS       "Build ops for TensorFlow"                 OFF)
option(BUILD_PYTORCH_OPS          "Build ops for PyTorch"                    OFF)
option(BUNDLE_OPEN3D_ML           "Includes the Open3D-ML repo in the wheel" OFF)

# Default build type on single-config generators.
# For multi-config generators (e.g. Visual Studio), CMAKE_CONFIGURATION_TYPES
# will be set, and we don't specify a default CMAKE_BUILD_TYPE.
# https://blog.kitware.com/cmake-and-the-default-build-type/
if(NOT CMAKE_CONFIGURATION_TYPES)
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "Setting build type to RelWithDebInfo as none was specified.")
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
        # Set the possible values of build type for cmake-gui.
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
    message(STATUS "CMAKE_BUILD_TYPE is set to ${CMAKE_BUILD_TYPE}.")
endif()

find_program(CCACHE "ccache")
if (CCACHE)
    message(STATUS "ccache found at ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif()

# In ExternalProject_Add, if OPEN3D_THIRD_PARTY_DOWNLOAD_DIR is specified, CMake will
# use this directory to cache downloaded 3rd party dependencies and automatically skip
# downloading from the Internet if the files are available. This is only supported by
# a limited number of 3rd party libraries.
set(OPEN3D_THIRD_PARTY_DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty_downloads")
message(STATUS "Downloading 3rdparty dependencies to ${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}")

set(FILAMENT_PRECOMPILED_ROOT "" CACHE PATH "Path to precompiled Filament library (used if BUILD_FILAMENT_FROM_SOURCE=OFF)")

# Set OpenGL policy
if(NOT USE_SYSTEM_GLFW)
    cmake_policy(SET CMP0072 OLD)
endif()
cmake_policy(GET CMP0072 CMP0072_VALUE)

project(Open3D VERSION 0.13 LANGUAGES C CXX)

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/CMake")

string(CONCAT PROJECT_VERSION_THREE_NUMBER "${OPEN3D_VERSION_MAJOR}"
                                           ".${OPEN3D_VERSION_MINOR}"
                                           ".${OPEN3D_VERSION_PATCH}")

# Set installation paths
if(UNIX)
    include(GNUInstallDirs)
    set(Open3D_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(Open3D_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(Open3D_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(Open3D_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
endif()

# Put build results in some predictable places
# The $<CONFIG> generator expression makes sure that XCode or Visual Studio do not
# append additional path components, as we need to know *exactly* where the build results
# end up.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Global Security options (including 3rd party code)
# Add -fPIC for library and -fPIE for executable to compiler and linker. Does not add -pie !
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(Open3DMakeHardeningFlags)
open3d_make_hardening_flags(HARDENING_CFLAGS HARDENING_LDFLAGS)
open3d_make_hardening_definitions(HARDENING_DEFINITIONS)

message(STATUS "Using security hardening compiler flags: ${HARDENING_CFLAGS}")
message(STATUS "Using security hardening linker flags: ${HARDENING_LDFLAGS}")
message(STATUS "Using security hardening compiler definitions: ${HARDENING_DEFINITIONS}")

add_compile_options("${HARDENING_CFLAGS}")
add_link_options("${HARDENING_LDFLAGS}")
add_compile_definitions("${HARDENING_DEFINITIONS}")

# Explicitly specifiy the preference of using -pthread over -lpthread.
# This must be defined here since CUDA calls find_package(Threads) internally.
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Folder view for project files
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Check if the compiler defines the _GLIBCXX_USE_CXX11_ABI macro
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#include <cxxabi.h>
int main() { return _GLIBCXX_USE_CXX11_ABI; }" HAS_GLIBCXX_USE_CXX11_ABI)

# Include convenience functions
include(Open3DLink3rdpartyLibraries)
include(Open3DSetGlobalProperties)
include(Open3DShowAndAbortOnWarning)

# Enumerate all third-party libraries which we need later
# This creates the necessary targets and sets the
# Open3D_3RDPARTY_*_TARGETS variables we use in open3d_link_3rdparty_libraries
include(3rdparty/find_dependencies.cmake)

# Open3D library
add_subdirectory(cpp)
