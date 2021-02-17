# Exports: ${CUDPP_INCLUDE_DIRS}
# Exports: ${CUDPP_LIB_DIR}
# Exports: ${CUDPP_LIBRARIES}

include(ExternalProject)

ExternalProject_Add(
    ext_cudpp
    PREFIX cudpp
    GIT_REPOSITORY https://github.com/yxlao/cudpp.git
    GIT_TAG cmake-fix
    GIT_SHALLOW ON  # Do not download the history.
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
)

ExternalProject_Get_Property(ext_cudpp INSTALL_DIR)
set(CUDPP_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CUDPP_LIB_DIR ${INSTALL_DIR}/lib)
set(CUDPP_LIBRARIES cudpp_hash cudpp) # The order is important.
