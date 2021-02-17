# Exports: ${SLABHASH_INCLUDE_DIRS}
# Exports: ${SLABHASH_LIB_DIR}
# Exports: ${SLABHASH_LIBRARIES}

include(ExternalProject)

ExternalProject_Add(
    ext_slabhash
    PREFIX slabhash
    GIT_REPOSITORY https://github.com/yxlao/SlabHash.git
    GIT_TAG cmake-refactor
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
        -DBUILD_BENCHMARKS=OFF
        -DBUILD_TESTS=OFF
)

ExternalProject_Get_Property(ext_slabhash INSTALL_DIR)
set(SLABHASH_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(SLABHASH_LIB_DIR "")                           # Header-only
set(SLABHASH_LIBRARIES "")                         # Header-only
