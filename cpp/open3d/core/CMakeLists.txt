# Private symbols and functions (i.e. OPEN3D_LOCAL)
add_library(core_internal OBJECT)
# Public symbols (i.e. OPEN3D_API)
add_library(core OBJECT $<TARGET_OBJECTS:core_internal>)

target_sources(core_internal PRIVATE
    hashmap/DeviceHashmap.cpp
    hashmap/CPU/CreateCPUHashmap.cpp
)
target_sources(core PRIVATE
    hashmap/Hashmap.cpp
)

target_sources(core_internal PRIVATE
    kernel/Arange.cpp
    kernel/ArangeCPU.cpp
    kernel/BinaryEW.cpp
    kernel/BinaryEWCPU.cpp
    kernel/IndexGetSet.cpp
    kernel/IndexGetSetCPU.cpp
    kernel/Kernel.cpp
    kernel/NonZero.cpp
    kernel/NonZeroCPU.cpp
    kernel/Reduction.cpp
    kernel/ReductionCPU.cpp
    kernel/UnaryEW.cpp
    kernel/UnaryEWCPU.cpp
)

target_sources(core_internal PRIVATE
    linalg/InverseCPU.cpp
    linalg/LeastSquaresCPU.cpp
    linalg/LUCPU.cpp
    linalg/MatmulCPU.cpp
    linalg/SolveCPU.cpp
    linalg/SVDCPU.cpp
    linalg/TriCPU.cpp
)
target_sources(core PRIVATE
    linalg/Det.cpp
    linalg/Inverse.cpp
    linalg/LeastSquares.cpp
    linalg/LU.cpp
    linalg/Matmul.cpp
    linalg/Solve.cpp
    linalg/SVD.cpp
    linalg/Tri.cpp
)

target_sources(core_internal PRIVATE
    nns/FixedRadiusIndex.cpp
    nns/NanoFlannIndex.cpp
    nns/NNSIndex.cpp
)
target_sources(core PRIVATE
    nns/NearestNeighborSearch.cpp
    )

if (BUILD_CUDA_MODULE)
    target_sources(core_internal PRIVATE
        MemoryManagerCUDACached.cu
        MemoryManagerCUDASimple.cu
    )

    target_sources(core_internal PRIVATE
        hashmap/CUDA/CreateCUDAHashmap.cu
        hashmap/CUDA/CUDAHashmapBufferAccessor.cu
        hashmap/CUDA/SlabNodeManager.cu
    )

    target_sources(core_internal PRIVATE
        kernel/ArangeCUDA.cu
        kernel/BinaryEWCUDA.cu
        kernel/IndexGetSetCUDA.cu
        kernel/NonZeroCUDA.cu
        kernel/ReductionCUDA.cu
        kernel/UnaryEWCUDA.cu
    )

    target_sources(core_internal PRIVATE
        linalg/InverseCUDA.cpp
        linalg/LeastSquaresCUDA.cpp
        linalg/LinalgUtils.cpp
        linalg/LUCUDA.cpp
        linalg/MatmulCUDA.cpp
        linalg/SolveCUDA.cpp
        linalg/SVDCUDA.cpp
        linalg/TriCUDA.cu
    )

    target_sources(core_internal PRIVATE
        nns/FixedRadiusSearch.cu
    )
endif()

if (WITH_FAISS)
    target_sources(core_internal PRIVATE
        nns/FaissIndex.cpp
    )
endif()

target_sources(core PRIVATE
    AdvancedIndexing.cpp
    CUDAUtils.cpp
    Dtype.cpp
    EigenConverter.cpp
    Indexer.cpp
    MemoryManager.cpp
    MemoryManagerCPU.cpp
    NumpyIO.cpp
    ShapeUtil.cpp
    Tensor.cpp
    TensorKey.cpp
    TensorList.cpp
)

open3d_show_and_abort_on_warning(core_internal)
open3d_set_global_properties(core_internal)
open3d_set_open3d_lib_properties(core_internal HIDDEN)
open3d_link_3rdparty_libraries(core_internal)

open3d_show_and_abort_on_warning(core)
open3d_set_global_properties(core)
open3d_set_open3d_lib_properties(core)
open3d_link_3rdparty_libraries(core)

if(BUILD_CUDA_MODULE)
    target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    find_package(CUB REQUIRED)
    target_include_directories(core SYSTEM PRIVATE ${CUB_INCLUDE_DIR})
endif()
