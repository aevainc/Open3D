set (KERNEL_SRC
    Kernel/IndexGetSet.cpp
    Kernel/IndexGetSetCPU.cpp
    Kernel/UnaryEW.cpp
    Kernel/UnaryEWCPU.cpp
    Kernel/BinaryEW.cpp
    Kernel/BinaryEWCPU.cpp
    Kernel/Reduction.cpp
    Kernel/ReductionCPU.cpp
)

set (KERNEL_CUDA_SRC
    Kernel/IndexGetSetCUDA.cu
    Kernel/UnaryEWCUDA.cu
    Kernel/BinaryEWCUDA.cu
    Kernel/ReductionCUDA.cu
)

set (CORE_SRC
    DLPack/DLPackConverter.cpp
    AdvancedIndexing.cpp
    ShapeUtil.cpp
    CUDAUtils.cpp
    Indexer.cpp
    MemoryManager.cpp
    MemoryManagerCPU.cpp
    MemoryManagerCUDA.cu
    Tensor.cpp
    TensorKey.cpp
    TensorList.cpp
)

set (CORE_CUDA_SRC
    MemoryManagerCUDA.cu
)

set (HASHMAP_SRC
    # Hashmap/Hashmap.cpp
    # Hashmap/HashmapCPU.cpp
    Hashmap/TensorHash.cpp
    Hashmap/TensorHashCPU.cpp
)

set (HASHMAP_CUDA_SRC
    # Hashmap/Hashmap.cu
    # Hashmap/HashmapCUDA.cu
    # TODO: conditional compile and put corresponding parts in cpp
    Hashmap/TensorHashCUDA.cu
)

if (BUILD_CUDA_MODULE)
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${CORE_CUDA_SRC}
        ${KERNEL_SRC}
        ${KERNEL_CUDA_SRC}
        ${HASHMAP_SRC}
        ${HASHMAP_CUDA_SRC}
    )
else()
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${HASHMAP_SRC}
        ${KERNEL_SRC}
    )
endif()

# Create object library
add_library(Core OBJECT ${ALL_CORE_SRC})
set_target_properties(Core PROPERTIES POSITION_INDEPENDENT_CODE ON)


#####################
## Temporary solution
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    find_package( CUDA REQUIRED ) # required for cuda_select_nvcc_arch_flags
    message(STATUS "Building CUDA enabled")
    add_definitions(-DBUILD_CUDA_MODULE)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    # get gencode flags
    if( "${CUDA_ARCH}" STREQUAL "User" )
        cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH_USER}")
    elseif( "${CUDA_ARCH}" STREQUAL "BasicPTX" )
        # include oldest and most recent PTX and rely on JIT compilation
        set(CUDA_GENCODES "-gencode arch=compute_30,code=compute_30;-gencode arch=compute_75,code=compute_75")
    else()
        cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH}")
    endif()
    string( REPLACE ";" " " CUDA_GENCODES "${CUDA_GENCODES}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODES}")
    target_include_directories(Core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set(CUDA_NVCC_FLAGS -std=c++11)
    set(CMAKE_CXX_STANDARD 11)

    cuda_add_executable(test_hashmap_cuda Hashmap/test_hashmap.cu)
    target_link_libraries(test_hashmap_cuda ${CMAKE_PROJECT_NAME})

    add_executable(test_hashmap_cpu Hashmap/test_hashmap.cpp)
    target_link_libraries(test_hashmap_cpu ${CMAKE_PROJECT_NAME})
endif()
###

if (BUILD_CUDA_MODULE)
    add_executable(test_tensorhash Hashmap/test_tensorhash.cpp)
    target_link_libraries(test_tensorhash ${CMAKE_PROJECT_NAME})
endif()
