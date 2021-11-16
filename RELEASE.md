# Open3D 0.14 Release Note

## Summary

We are excited to present the new Open3D version 0.14.0! In this release, you will find TensorBoard visualization, upgraded GUI, accelerated Tensor and I/O performance, new state-of-the-art 3D learning models in Open3D-ML, and many more.

**[TODO: any other highlights that we want to put here?]**

## Installation and Build system

- Open3D now works with Python 3.9. We release Open3D pre-compiled Python packages in Python 3.6, 3.7 3.8 and 3.9.
- Open3D 0.14 is the last version that supports conda installation. Starting from version 0.15, users will need to install Open3D with `pip install open3d`. We recommend installing Open3D with `pip` inside a conda virtual environment.
- Git submodules are no longer required in Open3D. You can now clone Open3D with `git clone https://github.com/isl-org/Open3D.git` without the `--recursive` flag.
- Open3D-ML is now recommended to be used along with [PyTorch](https://github.com/isl-org/Open3D-ML/blob/master/requirements-tensorflow.txt) 1.8.2 and/or [Tensorflow](https://github.com/isl-org/Open3D-ML/blob/master/requirements-tensorflow.txt) 2.5.2. Checkout [Open3D-ML](https://github.com/isl-org/Open3D-ML/) for more information.

## Tensorboard Integration

Now you can use Open3D within [Tensorboard](https://www.tensorflow.org/tensorboard) for interactive 3D visualization! At a glance, you can:

- *Sequentially* save and visualize geometry along with their properties. This enables interactive visualization and debugging of training 3D models.
- Visualize 3D semantic segmentation and object detection with input data, ground truth, and predictions. In addition, any *customized* properties for a `PointCloud`, from scalar to vector, can be easily visualized.
- *Synchronize* time steps and viewpoints during different runs. This helps debugging and monitoring the effect of parameter tuning.

For more details on how to use TensorBoard with Open3D, check out this [tutorial](link). **[TODO:@Sameer is there a doc link, or is this gif self-included?]**![img](https://lh4.googleusercontent.com/UN0_Yzb-9PintyBA5o2HmZUAUpHbR0Bp5jEVnfGOQuoQYIffgMJmzke0gMujUr8kQkLaPL9C6SHeH2YdovCTZl886bG6Kh-vuwYMWvkanTuYTgGp9XUbXeH4NN400ywUKg3sQKZo)

## GUI Visualizer

Further enhancements have been added to the GUI viewer. Now you can:

- Directly visualize tensor-based geometry classes including `PointCloud`, `TriangleMesh`, and `LineSet`.
- Use physically based rendering (PBR) materials that deliver appealing appearance.
- Use all the functionality in Tensorboard!
  ![img](https://lh3.googleusercontent.com/MRYlCK2LFxZaZ7GlgKfSvZg47K_Hj94Xhad3jTzomAf4z4vDBixbYBy2_QABhu3XiwMCpriShG30gdBZp7jTs0gwa9TOunigLM_FDiQ6WAJfeHFz5va4d9gHv7UnoSJdRFUTZ1QW)

## Core

- The Open3D Tensor class received a major performance boost with the help of [Intel ISPC compiler](https://ispc.github.io/) and optimization for the contiguous code path.
  ![img](https://raw.githubusercontent.com/isl-org/Open3D/wei/doc-014/0.13_vs_0.14.png)
  (See `python/benchmarks/core` for the benchmark scripts. For each operation, the geometric mean of run times with different data types is reported. The time is measured with Intel i9-10980XE CPU.)
- A major upgrade of Parallel `HashMap` is done. Now you can choose from multi-valued `HashMap` and `HashSet` depending your value types. A comprehensive [tutorial](http://www.open3d.org/docs/release/tutorial/core/hashmap.html) is also available.
- Linear Algebra performance have been optimized for small matrices, especially on 4x4 transformations.
- Semantics for tensor and tensor-based geometry have been improved, especially on device selection.
- Functions expecting a Tensor now accept Numpy arrays and Python lists. For example:
  ```python
  import open3d as o3d
  import numpy as np

  mesh = o3d.t.geometry.TriangleMesh()
  mesh.vertex['positions'] = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
  mesh.vertex['colors'] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
  mesh.triangle['indices'] = [[0, 1, 2]]
  o3d.visualization.draw(mesh)
  ```

## I/O

- We now support I/O from/to Numpy compatible `.npz` `.npy` formats for Open3D tensors and tensor maps. It is now easier to convert between Open3D geometry classes and Numpy properties.
- We have improved I/O performance for tensor-based point cloud and triangle-mesh file-formats, including `.ply`, `.pcd`, `.pts`. Geometry loading time is hence improved for the stand-alone visualizer app.

## Geometry

- We introduce a new class `RaycastingScene` with basic ray intersections functions and distance transform for meshes, utilizing the award winning [Intel Embree library](https://www.embree.org/).
  ![](http://www.open3d.org/docs/latest/_images/distance_field_animation.gif)
- Normal estimation for tensor `PointCloud` is supported with the tensor-compatible nearest neighbor search modules.
- Customizable tensor based `TriangleMesh` and `VoxelBlockGrid` are implemented that allows user-defined properties.

## Pipelines

- We have enhanced point cloud registration (ICP) with a tensor interface:
  - Float64 (double) precision point cloud is supported for a higher numerical stability
  - Robust Kernels, including Huber, Tuckey, and GM losses are supported for robust registration.
  - Colored-ICP is now supported in the unified tensor geometry API.
  - **[TODO: link once the tutorial is finished]**
- We also provide with an initial tensor-based reconstruction system in Python, including
  - Customizable volumetric RGB-D integration;
  - Dense RGB-D SLAM with a GUI;
  - Upgraded [tutorial](http://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/index.html ).


## Open3D-ML

The Open3D-ML library welcomes more state-of-the-art models and operators that are ready to use for advanced 3D perception, especially semantic segmentation, including

- New state of the art [Point Transformer](https://arxiv.org/abs/2012.09164) for Semantic Segmentation. **[TODO: @Sanskar put image here]**
- Highly Efficient Point-Voxel Convolution for Semantic Segmentation **[TODO: @Sanskar put image and reference here]**
- RaggedTensor integration that enables batch SparseConvolution and SparseConvolutionTranspose along with PyTorch.
- Batched voxelization for fast point-voxel conversions.


## Acknowledgment

We thank all contributors for this release. **[TODO: @yixing gather all community contributors' github id]**.
