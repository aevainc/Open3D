
function(link_everything target)
    # Link Open3D
    target_link_libraries(${target} PRIVATE Open3D)

    # Link 3rd-prty libraries
    open3d_link_3rdparty_libraries(${target})

    # Other properties
    open3d_show_and_abort_on_warning(${target})
    open3d_set_global_properties(${target})
    if (BUILD_CUDA_MODULE)
        target_include_directories(${target} SYSTEM PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
    endif()

    # Binary dir: build/examples/hashmap
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/hashmap"
    )
endfunction()

add_executable(demo_stdgpu demo_stdgpu.cu)
link_everything(demo_stdgpu)

add_executable(TestHashmap TestHashmap.cu)
link_everything(TestHashmap)
