# Hashmap benchmark experiments

## analyze_synthetic.py

```bash
python run_synthetic.py ../../build/bin/hashmap/TestHashmapInt3 output
python analyze_synthetic.py output
```

## analyze_ablation.py

```bash

```

## analyze_voxelization.py

- X: voxel size
- Y: voxelization time
- Compare: ours/MinkowskiEngine/Open3D

```bash
# Hard-coded values
python analyze_voxelization.py
```

## analyze_voxelhashing.py

- X: integration/raycasting/meshing
- Y: time for each step
- Compare: ours/InfiniTAM/VoxelHashing/GPU-robust

```bash
# Hard-coded values
python analyze_voxelhashing.py
```
