import open3d as o3d
import numpy as np
import argparse
import torch
import torch.nn
import pytorch3d.transforms as transform3d

from tsdf_util import *
from lighting_util import *
from voxel_util import *
from rgbd_util import *

svsh_block_size = 0.2


def to_torch(nparray):
    if not isinstance(nparray, np.ndarray):
        nparray = nparray.numpy()
    return torch.from_numpy(nparray).cuda()


def dict_to_torch(np_dict):
    torch_dict = {}
    for key in np_dict:
        torch_dict[key] = torch.from_numpy(np_dict[key]).cuda()

    return torch_dict


def compute_surface_and_normal(voxel_coords, voxel_tsdf, key, selection,
                               voxel_nbs):
    index_data = selection[key]
    nx = voxel_tsdf[voxel_nbs['index_xp'][index_data]] - voxel_tsdf[
        voxel_nbs['index_xm'][index_data]]
    ny = voxel_tsdf[voxel_nbs['index_yp'][index_data]] - voxel_tsdf[
        voxel_nbs['index_ym'][index_data]]
    nz = voxel_tsdf[voxel_nbs['index_zp'][index_data]] - voxel_tsdf[
        voxel_nbs['index_zm'][index_data]]
    norm = (nx**2 + ny**2 + nz**2).sqrt()

    normals = torch.stack((nx, ny, nz)).T / norm.unsqueeze(-1)
    surfaces = voxel_coords[index_data] - voxel_tsdf[index_data].unsqueeze(
        -1) * normals * sdf_trunc

    return surfaces, normals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset')

    # Selections
    parser.add_argument('--spatial', default='voxels_spatial.npz')
    parser.add_argument('--selection', default='tsdf_selection.npz')
    parser.add_argument('--association', default='tsdf_association.npz')

    parser.add_argument('--input', default='colored_voxels_fine.npz')
    parser.add_argument('--output', default='colored_voxels_refined.npz')

    args = parser.parse_args()

    # Load data
    spatial = np.load(args.spatial)
    selection = dict_to_torch(dict(np.load(args.selection)))
    input_data = np.load(args.input)
    association = dict_to_torch(np.load(args.association))

    voxel_nbs = dict_to_torch(get_nb_dict(spatial))
    voxel_coords = torch.from_numpy(spatial['voxel_coords']).float()
    voxel_tsdf = torch.from_numpy(input_data['voxel_tsdf']).float()

    voxel_color = torch.from_numpy(input_data['voxel_color']).float().cuda()
    voxel_intensity = color_to_intensity(voxel_color)
    voxel_albedo = torch.ones_like(voxel_intensity) * 0.6

    index_data = selection['index_data_c']
    index_data_xp = selection['index_data_xp']
    index_data_yp = selection['index_data_yp']
    index_data_zp = selection['index_data_zp']

    surfaces, normals = compute_surface_and_normal(voxel_coords, voxel_tsdf,
                                                   'index_data_c', selection,
                                                   voxel_nbs)
    intensity = voxel_intensity[index_data]
    albedo = voxel_albedo[index_data]

    l_init = backward_sh(normals.cpu().numpy(),
                         intensity.cpu().numpy(),
                         albedo.cpu().numpy())

    active_coords = voxel_coords[index_data]

    # First allocate coordinates
    # Ensure every one has a valid neighbor so that lighting computation can work
    unique_hashmap = o3d.core.Hashmap(8 * len(active_coords),
                                      o3d.core.Dtype.Int32,
                                      o3d.core.Dtype.Int32, (3), (1))
    svsh_coord = (active_coords / svsh_block_size).floor().int().numpy()
    svsh_coord_000 = svsh_coord
    svsh_coord_001 = svsh_coord + np.array([[0, 0, 1]], dtype=np.int32)
    svsh_coord_010 = svsh_coord + np.array([[0, 1, 0]], dtype=np.int32)
    svsh_coord_011 = svsh_coord + np.array([[0, 1, 1]], dtype=np.int32)
    svsh_coord_100 = svsh_coord + np.array([[1, 0, 0]], dtype=np.int32)
    svsh_coord_101 = svsh_coord + np.array([[1, 0, 1]], dtype=np.int32)
    svsh_coord_110 = svsh_coord + np.array([[1, 1, 0]], dtype=np.int32)
    svsh_coord_111 = svsh_coord + np.array([[1, 1, 1]], dtype=np.int32)

    unique_hashmap.activate(o3d.core.Tensor(svsh_coord_000))
    unique_hashmap.activate(o3d.core.Tensor(svsh_coord_001))
    unique_hashmap.activate(o3d.core.Tensor(svsh_coord_010))
    unique_hashmap.activate(o3d.core.Tensor(svsh_coord_011))
    unique_hashmap.activate(o3d.core.Tensor(svsh_coord_100))
    unique_hashmap.activate(o3d.core.Tensor(svsh_coord_101))
    unique_hashmap.activate(o3d.core.Tensor(svsh_coord_110))
    unique_hashmap.activate(o3d.core.Tensor(svsh_coord_111))

    active_addrs = unique_hashmap.get_active_addrs()
    active_svsh_coord = unique_hashmap.get_key_tensor()[active_addrs.to(
        o3d.core.Dtype.Int64)]

    # Compactify and redo hashing
    n_svsh = active_svsh_coord.shape[0]
    svsh_hashmap = o3d.core.Hashmap(n_svsh, o3d.core.Dtype.Int32,
                                    o3d.core.Dtype.Int32, (3), (1))
    svsh_hashmap.activate(active_svsh_coord)

    # Now query
    # yapf: disable
    addr_000, mask_000 = unique_hashmap.find(o3d.core.Tensor(svsh_coord_000))
    addr_001, mask_001 = unique_hashmap.find(o3d.core.Tensor(svsh_coord_001))
    addr_010, mask_010 = unique_hashmap.find(o3d.core.Tensor(svsh_coord_010))
    addr_011, mask_011 = unique_hashmap.find(o3d.core.Tensor(svsh_coord_011))
    addr_100, mask_100 = unique_hashmap.find(o3d.core.Tensor(svsh_coord_100))
    addr_101, mask_101 = unique_hashmap.find(o3d.core.Tensor(svsh_coord_101))
    addr_110, mask_110 = unique_hashmap.find(o3d.core.Tensor(svsh_coord_110))
    addr_111, mask_111 = unique_hashmap.find(o3d.core.Tensor(svsh_coord_111))
    # yapf: enable

    # Also query for 1-ring neighbor in laplacian
    svsh_center = active_svsh_coord.numpy()
    lap_center, _ = unique_hashmap.find(o3d.core.Tensor(svsh_center))
    lap_xp, mask_xp = unique_hashmap.find(
        o3d.core.Tensor(svsh_center + np.array([[1, 0, 0]], dtype=np.int32)))
    lap_yp, mask_yp = unique_hashmap.find(
        o3d.core.Tensor(svsh_center + np.array([[0, 1, 0]], dtype=np.int32)))
    lap_zp, mask_zp = unique_hashmap.find(
        o3d.core.Tensor(svsh_center + np.array([[0, 0, 1]], dtype=np.int32)))
    lap_xm, mask_xm = unique_hashmap.find(
        o3d.core.Tensor(svsh_center + np.array([[-1, 0, 0]], dtype=np.int32)))
    lap_ym, mask_ym = unique_hashmap.find(
        o3d.core.Tensor(svsh_center + np.array([[0, -1, 0]], dtype=np.int32)))
    lap_zm, mask_zm = unique_hashmap.find(
        o3d.core.Tensor(svsh_center + np.array([[0, 0, -1]], dtype=np.int32)))
    mask = mask_xp & mask_yp & mask_zp & mask_xm & mask_ym & mask_zm

    lap_center = to_torch(lap_center[mask]).long()
    lap_xp = to_torch(lap_xp[mask]).long()
    lap_yp = to_torch(lap_yp[mask]).long()
    lap_zp = to_torch(lap_zp[mask]).long()
    lap_xm = to_torch(lap_zm[mask]).long()
    lap_ym = to_torch(lap_ym[mask]).long()
    lap_zm = to_torch(lap_zm[mask]).long()

    # Sanity check: all valid
    assert mask_000.numpy().sum() == len(mask_000.numpy())
    assert mask_001.numpy().sum() == len(mask_001.numpy())
    assert mask_010.numpy().sum() == len(mask_010.numpy())
    assert mask_011.numpy().sum() == len(mask_011.numpy())
    assert mask_100.numpy().sum() == len(mask_100.numpy())
    assert mask_101.numpy().sum() == len(mask_101.numpy())
    assert mask_110.numpy().sum() == len(mask_110.numpy())
    assert mask_111.numpy().sum() == len(mask_111.numpy())

    # Now compute interpolation ratio
    svsh_coordf = (active_coords / svsh_block_size).numpy()
    diff = svsh_coordf - svsh_coord_000
    ratio_000 = (1 - diff[:, 0]) * (1 - diff[:, 1]) * (1 - diff[:, 2])
    ratio_001 = (1 - diff[:, 0]) * (1 - diff[:, 1]) * (diff[:, 2])
    ratio_010 = (1 - diff[:, 0]) * (diff[:, 1]) * (1 - diff[:, 2])
    ratio_011 = (1 - diff[:, 0]) * (diff[:, 1]) * (diff[:, 2])
    ratio_100 = (diff[:, 0]) * (1 - diff[:, 1]) * (1 - diff[:, 2])
    ratio_101 = (diff[:, 0]) * (1 - diff[:, 1]) * (diff[:, 2])
    ratio_110 = (diff[:, 0]) * (diff[:, 1]) * (1 - diff[:, 2])
    ratio_111 = (diff[:, 0]) * (diff[:, 1]) * (diff[:, 2])

    # Convert to torch
    index_000 = to_torch(addr_000).long()
    index_001 = to_torch(addr_001).long()
    index_010 = to_torch(addr_010).long()
    index_011 = to_torch(addr_011).long()
    index_100 = to_torch(addr_100).long()
    index_101 = to_torch(addr_101).long()
    index_110 = to_torch(addr_110).long()
    index_111 = to_torch(addr_111).long()

    ratio_000 = to_torch(ratio_000).unsqueeze(-1)
    ratio_001 = to_torch(ratio_001).unsqueeze(-1)
    ratio_010 = to_torch(ratio_010).unsqueeze(-1)
    ratio_011 = to_torch(ratio_011).unsqueeze(-1)
    ratio_100 = to_torch(ratio_100).unsqueeze(-1)
    ratio_101 = to_torch(ratio_101).unsqueeze(-1)
    ratio_110 = to_torch(ratio_110).unsqueeze(-1)
    ratio_111 = to_torch(ratio_111).unsqueeze(-1)

    # print(ratio_000 + ratio_001 + ratio_010 + ratio_011 + ratio_100 + ratio_101 + ratio_110 + ratio_111)

    # Now setup the svsh coordinates and derive the computation model
    param_svsh = torch.nn.Parameter(
        torch.Tensor.repeat(torch.from_numpy(l_init), (n_svsh, 1)).cuda())
    optimizer = torch.optim.Adam([param_svsh], lr=1e-3)

    lam_diffuse = 0.01
    for i in range(100):
        # Setup the computation model
        svsh_interp = ratio_000 * param_svsh[index_000] \
                    + ratio_001 * param_svsh[index_001] \
                    + ratio_010 * param_svsh[index_010] \
                    + ratio_011 * param_svsh[index_011] \
                    + ratio_100 * param_svsh[index_100] \
                    + ratio_101 * param_svsh[index_101] \
                    + ratio_110 * param_svsh[index_110] \
                    + ratio_111 * param_svsh[index_111]
        illum = forward_svsh(svsh_interp, normals.cuda())
        loss_data = (albedo * illum - intensity)**2
        loss_data = loss_data.sum()

        loss_diffuse = (param_svsh[lap_center] - param_svsh[lap_xp])**2 \
                 + (param_svsh[lap_center] - param_svsh[lap_yp])**2 \
                 + (param_svsh[lap_center] - param_svsh[lap_zp])**2 \
                 + (param_svsh[lap_center] - param_svsh[lap_xm])**2 \
                 + (param_svsh[lap_center] - param_svsh[lap_ym])**2 \
                 + (param_svsh[lap_center] - param_svsh[lap_zm])**2
        loss_diffuse = loss_diffuse.sum()
        loss = loss_data + lam_diffuse * loss_diffuse
        loss.backward()
        print(
            '{:2d} loss: {:.2f}, loss_data: {:.2f}, loss_diffuse: {:.2f}'.format(
                i, loss.item(), loss_data.item(),
                lam_diffuse * loss_diffuse.item()))
        optimizer.step()

    # After optimization
    svsh_interp = ratio_000 * param_svsh[index_000] \
                + ratio_001 * param_svsh[index_001] \
                + ratio_010 * param_svsh[index_010] \
                + ratio_011 * param_svsh[index_011] \
                + ratio_100 * param_svsh[index_100] \
                + ratio_101 * param_svsh[index_101] \
                + ratio_110 * param_svsh[index_110] \
                + ratio_111 * param_svsh[index_111]

    svsh = np.zeros((len(voxel_coords), 9))
    svsh[index_data.cpu().numpy()] = svsh_interp.detach().cpu().numpy()
    np.save('svsh.npy', svsh)
