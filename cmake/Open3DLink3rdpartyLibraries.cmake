# open3d_link_3rdparty_libraries(target)
#
# Links <target> against all 3rdparty libraries.
# We need this because we create a lot of object libraries to assemble the main Open3D library.
function(open3d_link_3rdparty_libraries target)
    # Directly pass public and private dependencies to the target.
    target_link_libraries(${target} PRIVATE ${Open3D_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${Open3D_3RDPARTY_PUBLIC_TARGETS})
endfunction()
