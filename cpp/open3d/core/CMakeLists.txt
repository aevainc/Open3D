set (KERNEL_SRC
    kernel/IndexGetSet.cpp
    kernel/IndexGetSetCPU.cpp
    kernel/NonZero.cpp
    kernel/NonZeroCPU.cpp
    kernel/UnaryEW.cpp
    kernel/UnaryEWCPU.cpp
    kernel/BinaryEW.cpp
    kernel/BinaryEWCPU.cpp
    kernel/Reduction.cpp
    kernel/ReductionCPU.cpp
    kernel/ImageOp.cpp
    kernel/ImageOpCPU.cpp
    kernel/SpecialOp.cpp
    kernel/SpecialOpCPU.cpp
)

set (KERNEL_CUDA_SRC
    kernel/IndexGetSetCUDA.cu
    kernel/NonZeroCUDA.cu
    kernel/UnaryEWCUDA.cu
    kernel/BinaryEWCUDA.cu
    kernel/ReductionCUDA.cu
    kernel/ImageOpCUDA.cu
    kernel/SpecialOpCUDA.cu
)

set (CORE_SRC
    AdvancedIndexing.cpp
    SparseIndexer.cpp
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
    hashmap/TensorHash.cpp
    hashmap/CPU/DefaultHashmapCPU.cpp
    hashmap/DefaultHashmap.cpp
)

set (HASHMAP_CUDA_SRC
    hashmap/CUDA/DefaultHashmapCUDA.cu
)

set(ALGEBRA_SRC
  algebra/Matmul.cpp
  algebra/MatmulCPU.cpp
  algebra/Solver.cpp
  algebra/SolverCPU.cpp
)

set(ALGEBRA_CUDA_SRC
  algebra/MatmulCUDA.cu
  algebra/SolverCUDA.cu
)

if (BUILD_CUDA_MODULE)
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${CORE_CUDA_SRC}
        ${KERNEL_SRC}
        ${KERNEL_CUDA_SRC}
        ${HASHMAP_SRC}
        ${HASHMAP_CUDA_SRC}
        ${ALGEBRA_SRC}
        ${ALGEBRA_CUDA_SRC}
    )
else()
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${KERNEL_SRC}
        ${HASHMAP_SRC}
        ${ALGEBRA_SRC}
    )
endif()

# Create object library
add_library(core OBJECT ${ALL_CORE_SRC})
open3d_set_global_properties(core)
open3d_link_3rdparty_libraries(core)
open3d_show_and_abort_on_warning(core)

find_package(MKL REQUIRED)
find_package(MAGMA REQUIRED)
if (BUILD_CUDA_MODULE)
    target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${MAGMA_INCLUDE_DIR} ${MKL_INCLUDE_DIR})
else()
    target_include_directories(core SYSTEM PRIVATE ${MKL_INCLUDE_DIR})
endif()
