set (KERNEL_SRC
    Kernel/IndexGetSet.cpp
    Kernel/IndexGetSetCPU.cpp
    Kernel/NonZero.cpp
    Kernel/NonZeroCPU.cpp
    Kernel/UnaryEW.cpp
    Kernel/UnaryEWCPU.cpp
    Kernel/BinaryEW.cpp
    Kernel/BinaryEWCPU.cpp
    Kernel/Reduction.cpp
    Kernel/ReductionCPU.cpp
)

set (KERNEL_CUDA_SRC
    Kernel/IndexGetSetCUDA.cu
    Kernel/NonZeroCUDA.cu
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
    Hashmap/TensorHash.cpp
    Hashmap/CPU/DefaultHashmapCPU.cpp
    Hashmap/DefaultHashmap.cpp
)

set (HASHMAP_CUDA_SRC
    Hashmap/CUDA/DefaultHashmapCUDA.cu
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
open3d_set_global_properties(Core)
open3d_link_3rdparty_libraries(Core)

if (BUILD_CUDA_MODULE)
   target_include_directories(Core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # cuda_add_executable(test_hashmap_cuda Hashmap/test_hashmap.cu)
    # target_link_libraries(test_hashmap_cuda ${CMAKE_PROJECT_NAME})

    cuda_add_executable(TestHashmap Hashmap/TestHashmap.cu)
    target_link_libraries(TestHashmap ${CMAKE_PROJECT_NAME})

    # add_executable(test_hashmap_cpu Hashmap/test_hashmap.cpp)
    # target_link_libraries(test_hashmap_cpu ${CMAKE_PROJECT_NAME})

    add_executable(test_tensorhash Hashmap/test_tensorhash.cpp)
    target_link_libraries(test_tensorhash ${CMAKE_PROJECT_NAME})
endif()
