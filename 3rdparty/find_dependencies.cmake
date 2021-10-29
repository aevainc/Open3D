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

# HEADER_TARGETS
# CMake targets we use in our public interface, but as a special case we only
# need to link privately against the library. This simplifies dependencies
# where we merely expose declared data types from other libraries in our
# public headers, so it would be overkill to require all library users to link
# against that dependency.
set(Open3D_3RDPARTY_HEADER_TARGETS)

# PRIVATE_TARGETS
# CMake targets for dependencies which are not exposed in the public API. This
# will include anything else we use internally.
set(Open3D_3RDPARTY_PRIVATE_TARGETS)

find_package(PkgConfig QUIET)


# open3d_find_package_3rdparty_library(name ...)
#
# Creates an interface library for a find_package dependency.
#
# The function will set ${name}_FOUND to TRUE or FALSE
# indicating whether or not the library could be found.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    REQUIRED
#        finding the package is required
#    QUIET
#        finding the package is quiet
#    PACKAGE <pkg>
#        the name of the queried package <pkg> forwarded to find_package()
#    PACKAGE_VERSION_VAR <pkg_version>
#        the variable <pkg_version> where to find the version of the queried package <pkg> find_package().
#        If not provided, PACKAGE_VERSION_VAR will default to <pkg>_VERSION.
#    TARGETS <target> [<target> ...]
#        the expected targets to be found in <pkg>
#    INCLUDE_DIRS
#        the expected include directory variable names to be found in <pkg>.
#        If <pkg> also defines targets, use them instead and pass them via TARGETS option.
#    LIBRARIES
#        the expected library variable names to be found in <pkg>.
#        If <pkg> also defines targets, use them instead and pass them via TARGETS option.
#
function(open3d_find_package_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;REQUIRED;QUIET" "PACKAGE;PACKAGE_VERSION_VAR" "TARGETS;INCLUDE_DIRS;LIBRARIES" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: open3d_find_package_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_PACKAGE)
        message(FATAL_ERROR "open3d_find_package_3rdparty_library: Expected value for argument PACKAGE")
    endif()
    if(NOT arg_PACKAGE_VERSION_VAR)
        set(arg_PACKAGE_VERSION_VAR "${arg_PACKAGE}_VERSION")
    endif()
    set(find_package_args "")
    if(arg_REQUIRED)
        list(APPEND find_package_args "REQUIRED")
    endif()
    if(arg_QUIET)
        list(APPEND find_package_args "QUIET")
    endif()
    find_package(${arg_PACKAGE} ${find_package_args})
    if(${arg_PACKAGE}_FOUND)
        message(STATUS "Using installed third-party library ${name} ${${arg_PACKAGE}_VERSION}")
        add_library(${name} INTERFACE)
        if(arg_TARGETS)
            foreach(target IN LISTS arg_TARGETS)
                if (TARGET ${target})
                    target_link_libraries(${name} INTERFACE ${target})
                else()
                    message(WARNING "Skipping undefined target ${target}")
                endif()
            endforeach()
        endif()
        if(arg_INCLUDE_DIRS)
            foreach(incl IN LISTS arg_INCLUDE_DIRS)
                target_include_directories(${name} INTERFACE ${${incl}})
            endforeach()
        endif()
        if(arg_LIBRARIES)
            foreach(lib IN LISTS arg_LIBRARIES)
                target_link_libraries(${name} INTERFACE ${${lib}})
            endforeach()
        endif()
        if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
            install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
            # Ensure that imported targets will be found again.
            if(arg_TARGETS)
                list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES ${arg_PACKAGE})
                set(Open3D_3RDPARTY_EXTERNAL_MODULES ${Open3D_3RDPARTY_EXTERNAL_MODULES} PARENT_SCOPE)
            endif()
        endif()
        set(${name}_FOUND TRUE PARENT_SCOPE)
        set(${name}_VERSION ${${arg_PACKAGE_VERSION_VAR}} PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# List of linker options for libOpen3D client binaries (eg: pybind) to hide Open3D 3rd
# party dependencies. Only needed with GCC, not AppleClang.
set(OPEN3D_HIDDEN_3RDPARTY_LINK_OPTIONS)

# # Threads
# open3d_find_package_3rdparty_library(3rdparty_threads
#     REQUIRED
#     PACKAGE Threads
#     TARGETS Threads::Threads
# )

# # OpenMP
# if(WITH_OPENMP)
#     open3d_find_package_3rdparty_library(3rdparty_openmp
#         PACKAGE OpenMP
#         PACKAGE_VERSION_VAR OpenMP_CXX_VERSION
#         TARGETS OpenMP::OpenMP_CXX
#     )
#     if(3rdparty_openmp_FOUND)
#         message(STATUS "Building with OpenMP")
#         list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS Open3D::3rdparty_openmp)
#     endif()
# endif()

if (USE_ONE_API)
    # # DPC++ compiler
    # list(APPEND CMAKE_MODULE_PATH /opt/intel/oneapi/compiler/latest/linux/cmake/SYCL)
    # find_package(IntelDPCPP REQUIRED)
    # if(IntelDPCPP_FOUND)
    #     add_library(SYCL INTERFACE)
    #     message(STATUS "SYCL_INCLUDE_DIR: ${SYCL_INCLUDE_DIR}")
    #     message(STATUS "SYCL_FLAGS: ${SYCL_FLAGS}")
    #     # target_compile_options(SYCL INTERFACE -fsycl)
    #     # target_include_directories(SYCL INTERFACE ${SYCL_INCLUDE_DIR})
    #     # # target_link_options(SYCL INTERFACE ${SYCL_FLAGS})
    #     # add_library(${PROJECT_NAME}::SYCL ALIAS SYCL)
    #     # list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS Open3D::SYCL)
    # else()
    #     message(FATAL_ERROR "IntelDPCPP_FOUND cannot be found.")
    # endif()

    # DPC++
    add_library(SYCL INTERFACE)
    target_compile_options(SYCL INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<COMPILE_LANGUAGE:ISPC>>>:-fsycl -fsycl-unnamed-lambda>)
    target_link_libraries(SYCL INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<NOT:$<LINK_LANGUAGE:ISPC>>>:sycl -fsycl>)
    add_library(Open3D::SYCL ALIAS SYCL)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS Open3D::SYCL)

    # oneTBB
    list(APPEND CMAKE_MODULE_PATH /opt/intel/oneapi/tbb/latest/lib/cmake/tbb)
    find_package(TBB REQUIRED)
    message(STATUS "TBB_FOUND: ${TBB_FOUND}")
    message(STATUS "TBB_IMPORTED_TARGETS: ${TBB_IMPORTED_TARGETS}")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS ${TBB_IMPORTED_TARGETS})

    # oneDPL
    list(APPEND CMAKE_MODULE_PATH /opt/intel/oneapi/dpl/latest/lib/cmake/oneDPL)
    find_package(oneDPL REQUIRED)
    target_compile_definitions(oneDPL INTERFACE _GLIBCXX_USE_TBB_PAR_BACKEND=0)
    target_compile_definitions(oneDPL INTERFACE PSTL_USE_PARALLEL_POLICIES=0)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS oneDPL)

    # oneMKL
    set(MKL_THREADING tbb_thread)
    list(APPEND CMAKE_MODULE_PATH /opt/intel/oneapi/mkl/latest/lib/cmake/mkl)
    find_package(MKL CONFIG REQUIRED)
    add_library(3rdparty_mkl INTERFACE)
    target_include_directories(3rdparty_mkl INTERFACE ${MKL_INCLUDE})
    target_link_libraries(3rdparty_mkl INTERFACE
        MKL::mkl_intel_ilp64
        MKL::mkl_core
        MKL::mkl_tbb_thread
    )
    add_library(Open3D::3rdparty_mkl ALIAS 3rdparty_mkl)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS Open3D::3rdparty_mkl)
else()
    message(FATAL_ERROR "Must use OneAPI")
endif()

# Compactify list of external modules.
# This must be called after all dependencies are processed.
list(REMOVE_DUPLICATES Open3D_3RDPARTY_EXTERNAL_MODULES)
