add_library(utility OBJECT)

target_sources(utility PRIVATE
    Console.cpp
    CPUInfo.cpp
    Eigen.cpp
    FileSystem.cpp
    Helper.cpp
    IJsonConvertible.cpp
    Logging.cpp
    Parallel.cpp
    Timer.cpp
    TestMiniVecCPU.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(utility  PRIVATE
        TestMiniVecCUDA.cu
    )
endif()

open3d_show_and_abort_on_warning(utility)
open3d_set_global_properties(utility)
open3d_set_open3d_lib_properties(utility)
open3d_link_3rdparty_libraries(utility)

if(BUILD_CUDA_MODULE)
    target_include_directories(utility SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
