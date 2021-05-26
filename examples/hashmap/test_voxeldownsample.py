import time
import numpy as np
import open3d as o3d
import torch
import MinkowskiEngine as ME
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pcd')
    parser.add_argument('--voxel_size', default=0.05)

    args = parser.parse_args()

    # warm up
    pcd_o3d = o3d.t.io.read_point_cloud(args.pcd)
    for i in range(100):
        pcd_o3d_cuda = pcd_o3d.cuda()
        pcd_o3d_cuda.voxel_down_sample(args.voxel_size)
        torch.cuda.synchronize()

    pcd_legacy = pcd_o3d.to_legacy_pointcloud()
    start = time.time()
    for i in range(100):
        pcd_legacy.voxel_down_sample(args.voxel_size)
    end = time.time()
    print(end - start)
    o3d.visualization.draw_geometries(
        [pcd_legacy.voxel_down_sample(args.voxel_size)])

    # o3d
    start = time.time()
    for i in range(100):
        pcd_o3d_cuda = pcd_o3d.cuda()
        pcd_o3d_cuda.voxel_down_sample(args.voxel_size)
        torch.cuda.synchronize()
    end = time.time()
    print(end - start)
    o3d.visualization.draw_geometries([
        pcd_o3d.cuda().voxel_down_sample(
            args.voxel_size).to_legacy_pointcloud()
    ])

    # mink
    pcd_mink = torch.from_numpy(pcd_o3d.point['points'].cpu().numpy())
    color_mink = torch.from_numpy(pcd_o3d.point['colors'].cpu().numpy()) / 255.0
    start = time.time()
    for i in range(100):
        ME.utils.sparse_quantize(pcd_mink,
                                 color_mink,
                                 quantization_size=args.voxel_size,
                                 device='cuda')
        torch.cuda.synchronize()
    end = time.time()
    print(end - start)

    pcd_down, color_down = ME.utils.sparse_quantize(
        pcd_mink,
        color_mink,
        quantization_size=args.voxel_size,
        device='cuda')

    pcd_down_vis = o3d.geometry.PointCloud()
    pcd_down_vis.points = o3d.utility.Vector3dVector(pcd_down.cpu().numpy())
    pcd_down_vis.colors = o3d.utility.Vector3dVector(color_down.cpu().numpy())
    o3d.visualization.draw_geometries([pcd_down_vis])
