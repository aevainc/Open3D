# Open3D 0.14 Release Note

## Summary

We are excited to present the new Open3D version 0.14.0! In this release, you will find Tensorboard visualization support along with an upgraded GUI, accelerated Tensor and IO performance, new state-of-the-art 3D learning models in Open3D-ML, and many more. 
**[TODO: any other highlights that we want to put here?]**



## Installation and Build system

- `Open3D` now works with Python 3.9. 
- `Open3D-ML` is now recommended to be used along with [PyTorch](https://pytorch.org/) 1.8.1 and/or [Tensorflow](https://www.tensorflow.org/) 2.5.0.

- `Open3D` 0.14 is the last version that supports conda installation. Starting from version 0.15, users will need to install Open3D with `pip install open3d`. A recommended practice is to create a conda virtual environment and `pip install` the package.

- The CMake build system of Open3D is refactored. Git submodules are no longer required in Open3D, i.e., the `--recursive` flag is no longer used; `git clone https://github.com/isl-org/Open3D.git` should suffice.



## New Features

### Visualization

#### Tensorboard

Now you can use Open3D within [Tensorboard](https://www.tensorflow.org/tensorboard) for interactive 3D visualization! At a glance, you can:

- *Sequentially* save and visualize geometry along with their properties. This enables interactive visualization and debugging of training 3D models.
- Visualize 3D semantic segmentation and object detection with input data, ground truth, and predictions. In addition, any *customized* properties for a `PointCloud`, from scalar to vector, can be easily visualized.
- *Synchronize* time steps and viewpoints during different runs. This helps debugging and monitoring the effect of parameter tuning.

For more details on how to use TensorBoard with Open3D, check out this [tutorial](link). **[TODO:@Sameer is there a doc link, or is this gif self-included?]**![img](https://lh4.googleusercontent.com/UN0_Yzb-9PintyBA5o2HmZUAUpHbR0Bp5jEVnfGOQuoQYIffgMJmzke0gMujUr8kQkLaPL9C6SHeH2YdovCTZl886bG6Kh-vuwYMWvkanTuYTgGp9XUbXeH4NN400ywUKg3sQKZo)



#### GUI visualizer

Further enhancements have been added to the GUI viewer. Now you can:

- Direct visualize tensor-based geometry classes including `PointCloud`, `TriangleMesh`, and `LineSet`.

- Use physically based rendering (PBR) materials that deliver appealing appearance.

- Use all the functionality in Tensorboard!

  ![img](https://lh3.googleusercontent.com/MRYlCK2LFxZaZ7GlgKfSvZg47K_Hj94Xhad3jTzomAf4z4vDBixbYBy2_QABhu3XiwMCpriShG30gdBZp7jTs0gwa9TOunigLM_FDiQ6WAJfeHFz5va4d9gHv7UnoSJdRFUTZ1QW)



### Core

- Open3D now supports [Intel ISPC compiler](https://ispc.github.io/). It automatically generates vectorized code to accelerate tensor operations. 

- Linear Algebra performance have been optimized for small matrices, especially on 4x4 transformations.

- A major upgrade of Parallel HashMap is done. Now you can choose from multi-valued HashMap and HashSet depending your value types. A comprehensive [tutorial](http://www.open3d.org/docs/release/tutorial/core/hashmap.html) is also available.

- Semantics for tensor and tensor-based geometry have been improved, especially on device selection.

- Functions expecting a Tensor now accept numpy arrays and Python lists.

  ```python
  mesh = o3d.t.geometry.TriangleMesh()
  mesh.vertex['positions'] = np.array([[0,0,0], [0,0,1], [0,1,1]], dtype=np.float32)
  mesh.triangle['indices'] = [[0,1,2]]
  ```



### I/O

- We now support I/O from/to numpy compatible `.npz` `.npy` formats for Open3D tensors and tensor maps. It is now easier to convert between Open3D geometry classes and numpy properties.
- We improved I/O performance for tensor-based point cloud and triangle-mesh file-formats, including `.ply`, `.pcd`, `.pts`. Geometry loading time is hence improved for the stand-alone visualizer app.



### Geometry

- We introduce a new class `RaycastingScene` with basic ray intersections functions and distance transform for meshes, utilizing the award winning [Intel Embree library](https://www.embree.org/). 
  ![](http://www.open3d.org/docs/latest/_images/distance_field_animation.gif)
- Normal estimation for tensor `PointCloud` is supported with the tensor-compatible nearest neighbor search modules.
- Customizable tensor based `TriangleMesh` and `VoxelBlockGrid` are implemented that allows user-defined properties.



### Pipelines

- We enhanced point cloud registration (ICP) with a tensor interface:

  - Float64 (double) precision point cloud is supported for a higher numerical stability
  - Robust Kernels, including Huber, Tuckey, and GM losses are supported for robust registration.
  - Colored-ICP is now supported in the unified tensor geometry API.
  - **[TODO: link once the tutorial is finished]**

- We also provide with an initial tensor-based reconstruction system in Python, including

  - Customizable volumetric RGB-D integration;
  - Dense RGB-D SLAM with a GUI;
  - Upgraded [tutorial](http://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/index.html ).

  

### Open3D-ML

The Open3D-ML library welcomes more state-of-the-art models and operators that are ready to use for advanced 3D perception, especially semantic segmentation, including

- New state of the art [Point Transformer](https://arxiv.org/abs/2012.09164) for Semantic Segmentation. **[TODO: @Sanskar put image here]**
- Highly Efficient Point-Voxel Convolution for Semantic Segmentation **[TODO: @Sanskar put image and reference here]**
- RaggedTensor integration that enables batch SparseConvolution and SparseConvolutionTranspose along with PyTorch.
- Batched voxelization for fast point-voxel conversions.
- Update requirements for PyTorch and Tensorflow **[TODO: @Sanskar put instructions here]**



## Changelog

- Tensorboard plugin materials support (#4078) 
- Set point size if material is provided to draw (#4175)
- Add simplified path to update geometry in O3DVisualizer (#4202)
- Fix node certificate (#4194)
- Msgpack serialization for Materials (#3985) 
- Tutorial update (#4125)
- fix critical issues in slac python (#4181)
- Add support for Material Properties to Tensor Geometries (#3858) 
- Concatenate op for Tensor (#4131)
- Add missing flag on headless rendering tutorial. (#4166)
- Fix minor typo (#4168)
- typo in common.py of new reconstruction system example, python (#4155)
- Asserts in NNS (#4139) 
- Support per-triangle normals for flat shading (#4118)
- Refactor unittests for NearestNeighborSearch (#4082) 
- Docs requirements in a file. (#4151)
- fix memleak in implicit conversion to Tensor in python (#4148)
- filter files in style checks (#4150)
- Asserts in core::Tensor (#4141)
- update functions using Transform, and docstrings. (#4136)
- fix docs (#4143)
- Matmul Performance Improvement. (#4147) 
- Default AABB and OBB color to white instead of black (#4034)
- Fix issues with IBL Combobox behavior (#4114) 
- wrench add Qhull::qhull_r when USE_SYSTEM_QHULLCPP is ON (#4144)
- asserts added in linear algebra module (#4140)
- Append Op for Tensor (#4086)
- Join Identical Vertices [Fixing #3799] (#4119) 
- removed Assign, and removed TensorList (#4134) 
- Asserts in t (#4105)
- add temporary fix for nodejs certificate (#4121)
- Add Python benchmarks for Tensor's Binary and Unary EW ops (#4117)
- Add devoxelize Op (#4003) 
- Tensor iterator (#4099)
- Add template overloads of Tensor indexing functions (#4110)
- Revert name change of Gradient points attribute (#4070) 
- Vectorize BinaryEW and UnaryEW ops (#4107)
- Add nthreads argument to RaycastingScene (#4100)
- Fix Jupyter WebRTC (#4091)
- Add fast path for indexing contiguous tensors (#4084)
- Tensorboard plugin (#3956) 
- New voxel grid (#4067)
- explicitly set max ccache size in docker build (#4101)
- chown inside docker (#4102)
- check multiple dtypes with AssertTensorDtypes (#4098)
- Add function for testing ray occlusions to RaycastingScene (#4095) 
- Add option to build with detected ISPC ISAs (#4081)
- uninstall before install-pip-package (#4089)
- silent wget in docker build (#4092)
- Rename tensor-registration functions (#4072) 
- fix debug lib name for libpng (#4055) 
- Simplify ISPC language emulation module (#4080)
- Avoid -pthread flag with ISPC compiler (#4076)
- install ml ops shared libraries (#4069) 
- Remove redundant neighbour search (#4050) 
- Style check for license header (#4002)
- compile MessageProcessor only with BUILD_GUI=ON (#4062)
- Add vectorized ParallelFor function and helper macros (#4039)
- SizeVector implicit conversion in Pybind (#4045)
- add support for more types to InvertNeighborsListOp (#4048)
- Upgrade Eigen dependency (#4046)
- throw exception when slicing a 0-dim tensor (#4043)
- Introduce initial ISPC language support (#3996)
- Unify KnnSearch & Radius Search in ml module (#3984)
- Tensor Legacy PTS Read IO (#3981)
- Tensor Colored-ICP (#3940)
- TIO PLY Read Performance Improvement (#4017)
- Build Docker with scripts (#4009)
- Fix wheel name inside Docker (#3989)
- set -euo pipefail (#4010)
- Wrap TEST_DATA_DIR in helper functions (#3929)
- Allow RSBagreader test to fail for now. (#4008)
- Consistently specify pthread preference (#4004)
- Refactor HashMap and backends (#3986)
- Fix Tensor API check in PCD IO (#4001)
- Use new tensor check api (#3998)
- t-IO PCD Support (#3978)
- Fixed wrong maximum count character in strncpy. (#3991)
- Fix inconsistent kernel object library handling (#3997)  
- Fix crash when visualizing single point Point Cloud (#3979) 
- Close window on ESC keypress (#3992)
- Helper functions for checking shape, dtype, device (#3953)
- Tensor Registration - GetInformationMatrixFromPointClouds (#3987)
- update documentation for nns chaing Int64 to Int32 (#3933)
- Update cpp file license (#3980)
- Wait for docs to be ready (#3966) 
- relax data sanity checks to allow sending partial data (#3963)
- Fix typo of online-processing.py (#3969)
- Make Normal Estimate optional to speed up loading (#3887)
- Build release wheel in CUDA Docker (#3893)
- Update Python file license (#3944)
- Refactor RPC serialization (#3821) 
- Minimize cached CUDA state (#3946)
- AssertShape() accepts both shape and dynamic shape (#3954)
- Hashmap upgrade (#3909)
- Remove SmallArray<T, N> class (#3949)
- Automatically handle 3rdparty external modules (#3939)
- removed num_neighbors checks, faster performance, less memory (#3945)
- remove console-specific prints (#3942)
- Simplify header target usage (#3937)
- EstimateNormals for TPointCloud (#3691)
- fix #3818 (#3927)
- Rename pipelines and add new reconstruction system basic config (#3892)
- CUDAState() static instance (#3925)
- Assert CUDA Device-ID Available (#3922) 
- Add warnings to help debug CUDA / pybind issues (#3918)
- Make docs archive available after wheels are ready. (#3919)
- Revert "Throw proper error for invalid CUDA device id. (#3890)" (#3907) 
- refactor unit test for FixedRadiusIndex (#3895)
- Default Constructor for Tensor PointCloud and TriangleMesh Pybind (#3880 
- Throw proper error for invalid CUDA device id. (#3890) 
- New KnnIndex for GPU knn search (#3784)
- revert ccache -M 1G (#3904)
- Parallelize FastGlobalRegistration (#3849) 
- Bypass update_geometry on Windows (#3877)
- Tensor LineSet (#3867)
- fix property of trianglemesh pybind (#3897)
- changed read_point_cloud defaults (#3879)
- set ccache size to 1G (#3885) 
- Geometry rename (#3855)
- Downgrade sphinx to fix doc search. (#3875)
- Randomize C++ and Python unit test order (#3871) 
- Write PLY Tensor PointCloud Performance Improvement. (#3835)
- Removed CPU() and CUDA() wrappers for device transfer in cpp. (#3854)
- NPZ support for PointCloud IO (#3851)
- TSDF IO (#3750)
- fix master style (#3861)
- update readme about zeromq fork (#3860)
- Refactoring: FromLegacy and ToLegacy ops types (#3853)
- use 2gpus on n1-standard machine (#3847)
- RANSACConvergenceCriteria confidence python docstring (#3790) 
- Fix undefined names: docstr and VisibleDeprecationWarning (#3844) 
- Full CUDA static linking on Linux (#3817)
- Add tensor I/O support for .npz (#3796)
- Deterministic registration (#3737)
- Factor out common functions into cmake subdirectory (#3813)
- Add batch support in voxelize op (#3694)
- Refactor FixedRadiusSearch (#3768)
- Add StdAllocator and apply to stdgpu backend (#3804)
- pybind write io default param to write binary (#3816)
- PointToPlane ICP Bug Fix (#3803)
- Update repository name (#3814)
- Build 3rdparty libraries in Release mode by default (#3801)
- Remove USE_SYSTEM_GOOGLETEST option (#3808)
- Helper functions for CUDA device synchronization (#3795)
- ParallelFor takes device as argument (#3749)
- Nacho/generalized icp (#3181) 
- Tensor Geometry::Transform kernels. (#3648)
- torch RaggedTensor class and batch support for SparseConv (#3572)
- F64 and Robust Kernel Support in RegistrationICP (#3690) 
- Fix inconsistencies in 3rd party dependency handling (#3772)
- Cleanup CUDA utilities (#3767)
- Shortcuts for Dtypes (#3741)
- Move Numpy I/O to t/io/NumpyIO.h (#3752)
- remove BUILD_RPC_INTERFACE from build scripts and code (#3771)
- Replies for DataChannelCallback (#3764) 
- Cleanup CUDAScopedStream and add stream support to NPPImage (#3748)
- Add missing include in Logging.h (#3763)
- Print on every Malloc and Free for debugging (#3747) 
- WebRTC data channel callback registration (#3759) 
- Python 3.9 (#3611)
- changed Assimp flags (#3743)
- Ensure CUDAState is initialized before Cacher (#3745)
- Introduce CUDAScopedStream class for Multi-Stream Support (#3732)
- fix 0-D index set and get (#3389) 
- fix args to tensor.index() for torch 1.8 and later (#3740)
- Introduce generic CachedMemoryManager (#3678)
- Replace *_TARGET variable usage by Open3D::3rdparty_* targets (#3731)
- expose BufferConnection to python (#3729)
- Tutorials for RaycastingScene (#3680) 
- Add DEPENDS option to import and build 3rdparty functions (#3702)
- Linear Algebra Device Functions for Kernels. (#3695) 
- Port remaining git submodules to URL downloading (#3703)
- Simplify HTML copy command (#3696)
- Port build submodules to URL downloading (#3689)
- Release CUDA cache in faiss index initialization (#3677)
- Exposed NeighbourCounts as return from HybridSearch (#3588) 
- RaycastingScene and implicit conversion to Tensor in python (#3637)
- Refactor cpu_launcher and cuda_launcher as namespaces (#3638)
- Return failure exit code on leaks at program end (#3666)
- Port header-only submodules to URL downloading (#3643) 
- Remove boost dependency (#3656) 
- libusb not required for running Open3D in docker. (#3657) 
- Use CUDA 10.1 to test Open3D builds with min supported version (#3662)
- Remove shebang from python examples (#3658) 
- Transpose rotation before applying on the right. (#3661)
- Add octree IO in python bind (#3646) 
- Registration Reduction6x6Impl template data type support. (#3636) 
- Reduce noise of k4a archive extraction (#3644)
- Console->Logger in pybind and unit test (#3642)
- Introduce memory manager statistics (#3639) 
- Allow building without Python (#3560)
- Last batch of fixes from the Klocwork scan (#3631) 
- Migrate console functions to Console.h (#3626) 
- Remove Flann dependency (#3620) 
- Respect sort flag in NanoFlannIndex (#3624)
- Rename Console to Logging (#3622) 
- Cleanup git submodules (#3621)
- Fix many small issues detected by klocwork analyzer (#3591)
- Fix CUDA compiler detection (#3607)
- Update issue template (#3602) 
- Reduce common CUDA archs. (#3604)
- Use ccache 4.3 for CI (#3603) 
- License update 2018-2021 (#3600)
- Migrate assert to OPEN3D_ASSERT (#3594)
- Cleanup ssh key individually for each GCE run when deleting VM (#3586) 
- Add general sparse conv operators (#3565) 
- Minimize download size of 3rd party dependencies (#3569)
- Workaround Visual Studio 16.10 compiler bug with CUDA and C++14 (#3574)
- fix windows ci pts test error (#3573)
- Test for -load_hidden flag (#3576) 
- Clarify minimum required CUDA version (#3575)
- Add batch support for fixed_radius_search and SparseConvolution (#3553)
- Hide 3rd party library symbols from client code (#3542)
- Add shader and material build functions (#3547) 
- Use same warning level in python bindings (#3551) 
- Propagate CUDA architectures to librealsense (#3549)
- Improve performance of Tensorflow and PyTorch modules (#3550)
- Fix and enable deprecation warnings (#3527) 
- Modernize CMake build scripts (#3456)
- disable the -S link flag for macros (#3536)
