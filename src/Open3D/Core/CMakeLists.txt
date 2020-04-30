set (KERNEL_SRC
    Kernel/IndexGetSet.cpp
    Kernel/IndexGetSetCPU.cpp
    Kernel/UnaryEW.cpp
    Kernel/UnaryEWCPU.cpp
    Kernel/BinaryEW.cpp
    Kernel/BinaryEWCPU.cpp
)

set (KERNEL_CUDA_SRC
    Kernel/IndexGetSetCUDA.cu
    Kernel/UnaryEWCUDA.cu
    Kernel/BinaryEWCUDA.cu
)

set (CORE_SRC
    DLPack/DLPackConverter.cpp
    AdvancedIndexing.cpp
    Broadcast.cpp
    CUDAUtils.cpp
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

if (BUILD_CUDA_MODULE)
    target_include_directories(Core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set(CUDA_NVCC_FLAGS -std=c++11)
    set(CMAKE_CXX_STANDARD 11)

    cuda_add_executable(test_hashmap Hashmap/test_hashmap.cu)
    target_link_libraries(test_hashmap ${CMAKE_PROJECT_NAME})

    add_executable(test_tensorhash Hashmap/test_tensorhash.cpp)
    target_link_libraries(test_tensorhash ${CMAKE_PROJECT_NAME})

endif()
