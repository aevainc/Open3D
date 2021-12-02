# Open3D 0.14 Release Note

## Summary

We are excited to present the new Open3D version 0.14! In this release, you will find TensorBoard visualization, upgraded GUI, improved Tensor and I/O performance, new state-of-the-art 3D learning models in Open3D-ML, improved interoperability with Numpy and many more.

**[TODO: any other highlights that we want to put here?]**

## Installation and Build system

- Open3D now works with Python 3.9. We release Open3D pre-compiled Python packages in Python 3.6, 3.7 3.8 and 3.9.
- Open3D 0.14 is the last version that supports conda installation. Starting from version 0.15, users will need to install Open3D with `pip install open3d`. We recommend installing Open3D with `pip` inside a conda virtual environment.
- Git submodules are no longer required in Open3D. You can now clone Open3D with `git clone https://github.com/isl-org/Open3D.git` without the `--recursive` flag. Also please note the updated Github URL.
- Open3D will now build in `Release` mode by default if `CMAKE_BUILD_TYPE` is not specified. `Python` is no longer required for building Open3D for C++ users.
- Open3D-ML is now recommended to be used along with [PyTorch](https://github.com/isl-org/Open3D-ML/blob/master/requirements-tensorflow.txt) 1.8.2 and/or [Tensorflow](https://github.com/isl-org/Open3D-ML/blob/master/requirements-tensorflow.txt) 2.5.2. Checkout [Open3D-ML](https://github.com/isl-org/Open3D-ML/) for more information.

## Tensorboard Integration

Now you can use Open3D within [Tensorboard](https://www.tensorflow.org/tensorboard) for interactive 3D visualization! At a glance, you can:

- Save and visualize geometry sequences and their properties. This enables interactive visualization and debugging of 3D data and 3DML model training.
- Visualize 3D semantic segmentation and object detection with input data, ground truth, and predictions. In addition, any *custom* properties for a `PointCloud`, from scalar to vector, can be easily visualized.
- *Synchronize* time steps and viewpoints during different runs. This helps debug and monitor the effect of parameter tuning.

#### Rich PBR materials
![tensorboard_demo_scene](https://user-images.githubusercontent.com/41028320/142651184-03e9fd59-e821-47c8-b470-f99a09a80757.png)
#### Object detection
![tensorboard_objdet_full_2_vp9 webm](https://user-images.githubusercontent.com/41028320/142651663-db1cbfcf-f0f6-4089-bf32-11696f408d5b.jpg)
#### Semantic segmentation
![tensorboard_sync_view_vp9](https://user-images.githubusercontent.com/41028320/142651665-a4c155c1-a4a1-4f9a-80e6-e13cdc1e5569.jpg)

To get started, write some sample geometry data to a TensorBoard summary with this snippet:

```python
from torch.utils.tensorboard import SummaryWriter  # TensorFlow also works, see docs.
import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
writer = SummaryWriter("demo_logs/")
cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
cube.compute_vertex_normals()
colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
for step in range(3):
    cube.paint_uniform_color(colors[step])
    writer.add_3d('cube', to_dict_batch([cube]), step=step)
```

Now you can visualize this in TensorBoard with `tensorboard --logdir demo_logs`. For more details on how to use TensorBoard with Open3D, check out this [tutorial](http://www.open3d.org/docs/release/tutorial/visualization/tensorboard_plugin.html).

## Visualizer

Further enhancements have been added to the GUI viewer. Now you can:

- Directly visualize tensor-based geometry classes including `PointCloud`, `TriangleMesh`, and `LineSet`.
- Use physically based rendering (PBR) materials that deliver appealing appearance.
- New default lighting environment and skybox improves visual appeal
- Use all the functionality in Tensorboard!

  ![img](https://user-images.githubusercontent.com/3722407/143294455-46800d35-3cab-4124-9df9-cb6fdc3d9e7b.png)

``` python
import open3d as o3d
import open3d.visualization as vis
a_sphere = o3d.geometry.TriangleMesh.create_sphere(2.5, create_uv_map=True)
a_sphere.compute_vertex_normals()
a_sphere = o3d.t.geometry.TriangleMesh.from_legacy(a_sphere)
# Compare this...
vis.draw(a_sphere)
a_sphere.material = vis.Material('defaultLit')
a_sphere.material.texture_maps['albedo'] =
    o3d.t.io.read_image('examples/test_data/demo_scene_assets/Tiles074_Color.jpg')
a_sphere.material.texture_maps['roughness'] =
    o3d.t.io.read_image('examples/test_data/demo_scene_assets/Tiles074_Roughness.jpg')
a_sphere.material.texture_maps['normal'] =
    o3d.t.io.read_image('examples/test_data/demo_scene_assets/Tiles074_NormalDX.jpg')
# With this!
vis.draw(a_sphere)
```

A complete, complex demo scene can be found at `examples/python/gui/demo-scene.py` 

## Core

- The Open3D `Tensor` class received a major performance boost with the help of [Intel ISPC compiler](https://ispc.github.io/) and optimization for the contiguous code path.
  ![img](https://raw.githubusercontent.com/isl-org/Open3D/wei/doc-014/0.13_vs_0.14.png)
  (See `python/benchmarks/core` for the benchmark scripts. For each operation, the geometric mean of run times with different data types is reported. The time is measured with an Intel i9-10980XE CPU.)
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
- We added support for material data to the MessagePack serialization format of the RPC module.


## Geometry

- We introduce a new class `RaycastingScene` with basic ray intersections functions and distance transform for meshes, utilizing the award winning [Intel Embree library](https://www.embree.org/).

  Example code for rendering a depth map:
  ```python
  import open3d as o3d
  import matplotlib.pyplot as plt

  # Create scene and add a cube
  cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(cube)

  # Use a helper function to create rays for a pinhole camera.
  rays = scene.create_rays_pinhole(fov_deg=60, center=[0.5,0.5,0.5], eye=[-1,-1,-1], up=[0,0,1],
                                   width_px=320, height_px=240)

  # Compute the ray intersections and visualize the hit distance (depth)
  ans = scene.cast_rays(rays)
  plt.imshow(ans['t_hit'].numpy())
  ```
  Distance transform generated with `RaycastingScene`:

  ![](http://www.open3d.org/docs/release/_images/distance_field_animation.gif)

  See the tutorials for more information ([Ray casting](http://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html), [Distance queries](http://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html)).

- Normal estimation for tensor `PointCloud` is supported with the tensor-compatible nearest neighbor search modules.
- Customizable tensor based `TriangleMesh`, `VoxelBlockGrid`, and `LineSet` are implemented that allow user-defined properties.
  (@yixing: add code snippets on how to set properties of geometries.)

## Pipelines

- We have enhanced point cloud registration (ICP) with a tensor interface:
  - Float64 (double) precision point cloud is supported for a higher numerical stability
  - Robust Kernels, including Huber, Tukey, and GM losses are supported for robust registration.
  - Colored-ICP is now supported in the unified tensor-based API.

Detailed explanations are available at the upgraded [registration tutorials](http://www.open3d.org/docs/0.14.1/tutorial/t_pipelines/t_icp_registration.html).
```python
import open3d as o3d
if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg
import numpy as np

# 1. Import pointclouds. [Locate the sample data in Open3D/examples/test_data]
source = o3d.t.io.read_point_cloud("../../test_data/ColoredICP/frag_115.ply")
target = o3d.t.io.read_point_cloud("../../test_data/ColoredICP/frag_116.ply")
# For Colored-ICP `colors` attribute must be of the same dtype as `positions`
# and `normals` attribute.
source.point["colors"] = source.point["colors"].to(o3d.core.float32) / 255.0
target.point["colors"] = target.point["colors"].to(o3d.core.float32) / 255.0
# To use the GPU, transfer the pointclouds to the GPU device.
source = source.cuda(0)
target = target.cuda(0)

# 2. Setup the parameters.
# Initial alignment.
init_source_to_target = np.identity(4)
# Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
estimation = treg.TransformationEstimationForColoredICP(
    lambda_geometric=0.968,
    kernel=treg.robust_kernel.RobustKernel(
        treg.robust_kernel.RobustKernelMethod.HuberLoss, 0.01))
# Convergence criteria.
criteria_list = [
    treg.ICPConvergenceCriteria(relative_fitness=0.0001,
                                relative_rmse=0.0001,
                                max_iteration=50),
    treg.ICPConvergenceCriteria(0.00001, 0.00001, 30),
    treg.ICPConvergenceCriteria(0.000001, 0.000001, 14)
]
# Search radius.
max_correspondence_distances = o3d.utility.DoubleVector([0.08, 0.04, 0.02])
# Downsampling paramter (voxel-size).
voxel_sizes = o3d.utility.DoubleVector([0.04, 0.02, 0.01])
# Save loss logs.
save_loss_log = True

# 3. Get `registration_result`.
registration_result = treg.multi_scale_icp(source, target, voxel_sizes,
                                           criteria_list,
                                           max_correspondence_distances,
                                           init_source_to_target, estimation,
                                           save_loss_log)

# 4. Visualize the registration result.
o3d.visualization.draw(
    [source.clone().transform(registration_result.transformation), target])

# 5. Visualize the loss-logs
from matplotlib import pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
axes.set_title("Inlier RMSE vs Iteration")
axes.plot(registration_result.loss_log["index"].numpy(),
          registration_result.loss_log["inlier_rmse"].numpy())
```

- We also provide with an initial tensor-based reconstruction system in Python, including
  - Customizable volumetric RGB-D integration;
  - Dense RGB-D SLAM with a GUI;
  - Upgraded [tutorial](http://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/index.html).


## Open3D-ML

The Open3D-ML library welcomes more state-of-the-art models and operators that are ready to use for advanced 3D perception, especially semantic segmentation, including

- New state-of-the-art [Point Transformer](https://arxiv.org/abs/2012.09164) for Semantic Segmentation.
  ![img](https://raw.githubusercontent.com/isl-org/Open3D/wei/doc-014/PointTransformer_S3DIS.png)
- Highly Efficient [Point-Voxel Convolution](https://arxiv.org/abs/1907.03739) for Semantic Segmentation.
  ![img](https://raw.githubusercontent.com/isl-org/Open3D/wei/doc-014/PVCNN_S3DIS.png)
- RaggedTensor integration that enables batch `SparseConvolution` and `SparseConvolutionTranspose` along with PyTorch.
- Batched voxelization for fast point-voxel conversions.

Refer to the tutorial for training and inference on new models. ([PyTorch](https://github.com/isl-org/Open3D-ML/blob/master/docs/tutorial/notebook/train_ss_model_using_pytorch.rst) [TensorFlow](https://github.com/isl-org/Open3D-ML/blob/master/docs/tutorial/notebook/train_ss_model_using_tensorflow.rst)).

## Acknowledgment

We thank all the community contributors for this release (please let us know if we omitted your name)!

@benjaminum
@cclauss
@chrockey
@chunibyo-wly
@cosama
@errissa
@gsakkis
@junha-l
@leomariga
@li6in9muyou
@marcov868
@michaelbeale-IL
@muskie82
@nachovizzo
@NobuoTsukamoto
@plusk01
@reyanshsolis
@sanskar107
@ShubhamAgarwal12
@SoftwareApe
@ssheorey
@stanleyshly
@stotko
@theNded
@yxlao

- [Chunibyo](https://github.com/chunibyo-wly)
- [Christian Clauss](https://github.com/cclauss)
- [Parker Lusk](https://github.com/plusk01)
- [Carl Mueller-Roemer](https://github.com/SoftwareApe)
- [stanleyshly](https://github.com/stanleyshly)
- [Marco Venturelli](https://github.com/marcov868)
- [Nobuo Tsukamoto](https://github.com/NobuoTsukamoto)
- [muskie](https://github.com/muskie82)
- [li6in9muyou](https://github.com/li6in9muyou)
- [George Sakkis](https://github.com/gsakkis)
- [Marco Salathe](https://github.com/cosama)
- [Ignacio Vizzo](https://github.com/nachovizzo)
- [Leonardo Mariga](https://github.com/leomariga)
- [Michael Beale](https://github.com/michaelbeale-IL)
- [applesauce49](https://github.com/applesauce49)
- [Ajinkya Khoche](https://github.com/ajinkyakhoche)
- [Kyle Vedder](https://github.com/kylevedder)
- [Nicholas Mitchell](https://github.com/Nicholas-Mitchell)
- [kukuruza](https://github.com/kukuruza)
- [bhaskar-anand-iith](https://github.com/bhaskar-anand-iith)
- [Zdenko Pikula](https://github.com/RZetko)
- [lc-guy](https://github.com/lc-guy)
- [Josiah Coad](https://github.com/josiahcoad)
- [Felix Rothe](https://github.com/jokokojote)
