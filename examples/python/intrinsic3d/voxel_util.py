import open3d as o3d

import numpy as np
import torch

def find_neighbors(xyz, check=True):
    device = o3d.core.Device('CUDA:0')
    dtype = o3d.core.Dtype.Int32

    xyz_int = xyz.astype(np.int32)
    indices = np.arange(len(xyz_int)).astype(np.int64)

    point_hashmap = o3d.core.Hashmap(len(xyz_int), o3d.core.Dtype.Int32,
                                     o3d.core.Dtype.Int64, (3), (1), device)

    t_xyz = o3d.core.Tensor(xyz_int, device=device)
    t_indices = o3d.core.Tensor(indices, device=device)

    # Construction
    addr_self, mask_self = point_hashmap.insert(t_xyz, t_indices)
    key = point_hashmap.get_key_tensor()
    value_indices = point_hashmap.get_value_tensor()

    # Query
    xyz_xp = t_xyz + o3d.core.Tensor([[1, 0, 0]], dtype=dtype, device=device)
    xyz_xm = t_xyz - o3d.core.Tensor([[1, 0, 0]], dtype=dtype, device=device)

    xyz_yp = t_xyz + o3d.core.Tensor([[0, 1, 0]], dtype=dtype, device=device)
    xyz_ym = t_xyz - o3d.core.Tensor([[0, 1, 0]], dtype=dtype, device=device)

    xyz_zp = t_xyz + o3d.core.Tensor([[0, 0, 1]], dtype=dtype, device=device)
    xyz_zm = t_xyz - o3d.core.Tensor([[0, 0, 1]], dtype=dtype, device=device)

    addr_xp, mask_xp = point_hashmap.find(xyz_xp)
    addr_xm, mask_xm = point_hashmap.find(xyz_xm)

    addr_yp, mask_yp = point_hashmap.find(xyz_yp)
    addr_ym, mask_ym = point_hashmap.find(xyz_ym)

    addr_zp, mask_zp = point_hashmap.find(xyz_zp)
    addr_zm, mask_zm = point_hashmap.find(xyz_zm)

    index_dtype = o3d.core.Dtype.Int64
    index_xp = value_indices[addr_xp.to(index_dtype)].cpu().numpy().squeeze()
    index_yp = value_indices[addr_yp.to(index_dtype)].cpu().numpy().squeeze()
    index_zp = value_indices[addr_zp.to(index_dtype)].cpu().numpy().squeeze()

    index_xm = value_indices[addr_xm.to(index_dtype)].cpu().numpy().squeeze()
    index_ym = value_indices[addr_ym.to(index_dtype)].cpu().numpy().squeeze()
    index_zm = value_indices[addr_zm.to(index_dtype)].cpu().numpy().squeeze()

    mask_xp = mask_xp.cpu().numpy().squeeze()
    mask_yp = mask_yp.cpu().numpy().squeeze()
    mask_zp = mask_zp.cpu().numpy().squeeze()

    mask_xm = mask_xm.cpu().numpy().squeeze()
    mask_ym = mask_ym.cpu().numpy().squeeze()
    mask_zm = mask_zm.cpu().numpy().squeeze()

    # Sanity check
    if check:
        # Identity map
        assert mask_self.cpu().numpy().sum() == len(xyz_int)
        key_self = key[addr_self.to(index_dtype)].cpu().numpy().squeeze()
        value_self = value_indices[addr_self.to(
            index_dtype)].cpu().numpy().squeeze()
        assert np.allclose(value_self, indices)
        assert np.allclose(key_self, xyz_int)

        # Neighbor mapping is agnostic of the hashmap, only depending on xyz
        lhs = xyz[index_xp[mask_xp]]
        rhs = xyz[mask_xp] + np.array([1, 0, 0])
        assert np.allclose(lhs, rhs), 'xp assertion failed'

        lhs = xyz[index_yp[mask_yp]]
        rhs = xyz[mask_yp] + np.array([0, 1, 0])
        assert np.allclose(lhs, rhs), 'yp assertion failed'

        lhs = xyz[index_zp[mask_zp]]
        rhs = xyz[mask_zp] + np.array([0, 0, 1])
        assert np.allclose(lhs, rhs), 'zp assertion failed'

        lhs = xyz[index_xm[mask_xm]]
        rhs = xyz[mask_xm] + np.array([-1, 0, 0])
        assert np.allclose(lhs, rhs), 'xm assertion failed'

        lhs = xyz[index_ym[mask_ym]]
        rhs = xyz[mask_ym] + np.array([0, -1, 0])
        assert np.allclose(lhs, rhs), 'ym assertion failed'

        lhs = xyz[index_zm[mask_zm]]
        rhs = xyz[mask_zm] + np.array([0, 0, -1])
        assert np.allclose(lhs, rhs), 'zm assertion failed'

    return {
        # Indices
        'index_xp': index_xp,
        'index_yp': index_yp,
        'index_zp': index_zp,
        'index_xm': index_xm,
        'index_ym': index_ym,
        'index_zm': index_zm,

        # Masks
        'mask_xp': mask_xp,
        'mask_yp': mask_yp,
        'mask_zp': mask_zp,
        'mask_xm': mask_xm,
        'mask_ym': mask_ym,
        'mask_zm': mask_zm
    }


def get_nb_dict(data):
    nb_dict = {}
    keys = [
        'index_xp',
        'index_yp',
        'index_zp',
        'index_xm',
        'index_ym',
        'index_zm',
        'mask_xp',
        'mask_yp',
        'mask_zp',
        'mask_xm',
        'mask_ym',
        'mask_zm',
    ]

    for key in list(data.keys()):
        if key in keys:
            nb_dict[key] = data[key]

    return nb_dict


def compute_normals(tsdf, nb_dict):
    # TODO: use pytorch to make it differentiable
    mask_plus = nb_dict['mask_xp'] & nb_dict['mask_yp'] & nb_dict['mask_zp']
    mask_minus = nb_dict['mask_xm'] & nb_dict['mask_ym'] & nb_dict['mask_zm']
    mask = mask_plus & mask_minus

    voxel_normal = np.zeros((len(tsdf), 3))
    nx = tsdf[nb_dict['index_xp'][mask]] - tsdf[nb_dict['index_xm'][mask]]
    ny = tsdf[nb_dict['index_yp'][mask]] - tsdf[nb_dict['index_ym'][mask]]
    nz = tsdf[nb_dict['index_zp'][mask]] - tsdf[nb_dict['index_zm'][mask]]
    norm = np.sqrt((nx**2 + ny**2 + nz**2))

    voxel_normal[mask, 0] = nx / norm
    voxel_normal[mask, 1] = ny / norm
    voxel_normal[mask, 2] = nz / norm

    return voxel_normal, mask


def compute_normals_torch(param_tsdf, nb_dict):
    mask_plus = nb_dict['mask_xp'] & nb_dict['mask_yp'] & nb_dict['mask_zp']
    mask_minus = nb_dict['mask_xm'] & nb_dict['mask_ym'] & nb_dict['mask_zm']
    mask = mask_plus & mask_minus

    index_xp = torch.from_numpy(nb_dict['index_xp'][mask]).cuda()
    index_yp = torch.from_numpy(nb_dict['index_yp'][mask]).cuda()
    index_zp = torch.from_numpy(nb_dict['index_zp'][mask]).cuda()
    index_xm = torch.from_numpy(nb_dict['index_xm'][mask]).cuda()
    index_ym = torch.from_numpy(nb_dict['index_ym'][mask]).cuda()
    index_zm = torch.from_numpy(nb_dict['index_zm'][mask]).cuda()


    nx = param_tsdf[index_xp] - param_tsdf[index_xm]
    ny = param_tsdf[index_yp] - param_tsdf[index_ym]
    nz = param_tsdf[index_zp] - param_tsdf[index_zm]
    norm = (nx **2 + ny **2 + nz **2).sqrt()

    voxel_normal = torch.stack((nx / norm, ny / norm, nz / norm))

    return voxel_normal


def compute_nearest_surface(points, sdf, normals):
    return points - np.expand_dims(sdf, axis=-1) * normals
