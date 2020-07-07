find_package(MKL REQUIRED)
find_package(MAGMA REQUIRED)

if (BUILD_CUDA_MODULE)
    # core
    cuda_add_executable(TestHashmap TestHashmap.cu)
    open3d_set_global_properties(TestHashmap)
    target_link_libraries(TestHashmap ${CMAKE_PROJECT_NAME})

    add_executable(TestTensorHash TestTensorHash.cpp)
    target_link_libraries(TestTensorHash ${CMAKE_PROJECT_NAME})

    add_executable(TestMemoryManager TestMemoryManager.cpp)
    target_link_libraries(TestMemoryManager ${CMAKE_PROJECT_NAME})

    # tgeometry
    add_executable(TestTPointCloud TestTPointCloud.cpp)
    target_link_libraries(TestTPointCloud ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})

    add_executable(TestTImage TestTImage.cpp)
    target_link_libraries(TestTImage ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})

    add_executable(TestPcdMergeLegacy TestPcdLegacyMerge.cpp)
    target_link_libraries(TestPcdMergeLegacy ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})

    add_executable(TestPcdMerge TestPcdMerge.cpp)
    target_link_libraries(TestPcdMerge ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})

    add_executable(TestVoxelGrid TestVoxelGrid.cpp)
    target_link_libraries(TestVoxelGrid ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})

    # algebra
    add_executable(TestMatmul TestMatmul.cpp)
    target_link_libraries(TestMatmul ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})

    add_executable(TestSolver TestSolve.cpp)
    target_link_libraries(TestSolver ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})
endif()
