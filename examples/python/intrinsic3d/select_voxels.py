import open3d as o3d
import numpy as np
import argparse

from tsdf_util import *
from lighting_util import *
from voxel_util import *
from rgbd_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset')
    parser.add_argument('--spatial', default='voxels_spatial.npz')
    parser.add_argument('--input', default='colored_voxels_fine.npz')

    parser.add_argument('--output', default='tsdf_selection.npz')
    args = parser.parse_args()

    dict_voxel_selection = {}

    # Load data
    spatial = np.load(args.spatial)
    voxel_coords = spatial['voxel_coords']
    voxel_nbs = get_nb_dict(spatial)
    n_voxel = len(voxel_coords)

    input_data = np.load(args.input)
    voxel_tsdf = input_data['voxel_tsdf']
    voxel_color = input_data['voxel_color']
    voxel_intensity = color_to_intensity(voxel_color)

    # Rule out un-colorized points first
    mask_intensity = voxel_intensity > 0
    index = np.arange(n_voxel, dtype=np.int64)[mask_intensity]

    # Mask 1: voxel with valid normals
    mask_plus = voxel_nbs['mask_xp'][index] & \
        voxel_nbs['mask_yp'][index] & \
        voxel_nbs['mask_zp'][index]
    mask_minus = voxel_nbs['mask_xm'][index] & \
        voxel_nbs['mask_ym'][index] & \
        voxel_nbs['mask_zm'][index]

    # TODO: if it is insufficient, switch to positive part only
    mask_1ring = mask_plus & mask_minus
    index_1ring = index[mask_1ring]

    ## For TSDF laplacian
    dict_voxel_selection['index_lap_c'] = index_1ring
    dict_voxel_selection['index_lap_xp'] = voxel_nbs['index_xp'][index_1ring]
    dict_voxel_selection['index_lap_yp'] = voxel_nbs['index_yp'][index_1ring]
    dict_voxel_selection['index_lap_zp'] = voxel_nbs['index_zp'][index_1ring]
    dict_voxel_selection['index_lap_xm'] = voxel_nbs['index_xm'][index_1ring]
    dict_voxel_selection['index_lap_ym'] = voxel_nbs['index_ym'][index_1ring]
    dict_voxel_selection['index_lap_zm'] = voxel_nbs['index_zm'][index_1ring]
    print(
        f'voxels with 1-ring nbs {len(index_1ring)}/{n_voxel}: {len(index_1ring) / float(n_voxel)}'
    )

    onering_hashmap = o3d.core.Hashmap(len(index_1ring), o3d.core.Dtype.Int64,
                                       o3d.core.Dtype.Int64, (1), (1))
    onering_hashmap.activate(o3d.core.Tensor(index_1ring))

    # Next: check if positive neighbors are in index_1ring
    index_1ring_xp = voxel_nbs['index_xp'][index_1ring]
    index_1ring_yp = voxel_nbs['index_yp'][index_1ring]
    index_1ring_zp = voxel_nbs['index_zp'][index_1ring]
    addr_xp, mask_xp = onering_hashmap.find(o3d.core.Tensor(index_1ring_xp))
    addr_yp, mask_yp = onering_hashmap.find(o3d.core.Tensor(index_1ring_yp))
    addr_zp, mask_zp = onering_hashmap.find(o3d.core.Tensor(index_1ring_zp))
    mask_xp = mask_xp.numpy() & voxel_nbs['mask_xp'][index_1ring]
    mask_yp = mask_yp.numpy() & voxel_nbs['mask_yp'][index_1ring]
    mask_zp = mask_zp.numpy() & voxel_nbs['mask_zp'][index_1ring]
    mask_1ring_grad = mask_xp & mask_yp & mask_zp
    index_1ring_grad = index_1ring[mask_1ring_grad]

    ## For TSDF-albedo joint data term
    dict_voxel_selection['index_data_c'] = index_1ring_grad
    dict_voxel_selection['index_data_xp'] = voxel_nbs['index_xp'][
        index_1ring_grad]
    dict_voxel_selection['index_data_yp'] = voxel_nbs['index_yp'][
        index_1ring_grad]
    dict_voxel_selection['index_data_zp'] = voxel_nbs['index_zp'][
        index_1ring_grad]
    print(
        f'voxels with 1-ring nbs and gradient {len(index_1ring_grad)}/{n_voxel}: {len(index_1ring_grad) / float(n_voxel)}'
    )

    ## For albedo regularizer
    intensity_hashmap = o3d.core.Hashmap(len(index), o3d.core.Dtype.Int64,
                                         o3d.core.Dtype.Int64, (1), (1))
    intensity_hashmap.activate(o3d.core.Tensor(index))
    _, mask_xp = intensity_hashmap.find(
        o3d.core.Tensor(voxel_nbs['index_xp'][index]))
    _, mask_yp = intensity_hashmap.find(
        o3d.core.Tensor(voxel_nbs['index_yp'][index]))
    _, mask_zp = intensity_hashmap.find(
        o3d.core.Tensor(voxel_nbs['index_zp'][index]))
    _, mask_xm = intensity_hashmap.find(
        o3d.core.Tensor(voxel_nbs['index_xm'][index]))
    _, mask_ym = intensity_hashmap.find(
        o3d.core.Tensor(voxel_nbs['index_ym'][index]))
    _, mask_zm = intensity_hashmap.find(
        o3d.core.Tensor(voxel_nbs['index_zm'][index]))

    mask_xp = mask_xp.numpy() & voxel_nbs['mask_xp'][index]
    mask_yp = mask_yp.numpy() & voxel_nbs['mask_yp'][index]
    mask_zp = mask_zp.numpy() & voxel_nbs['mask_zp'][index]
    mask_xm = mask_xm.numpy() & voxel_nbs['mask_xm'][index]
    mask_ym = mask_ym.numpy() & voxel_nbs['mask_ym'][index]
    mask_zm = mask_zm.numpy() & voxel_nbs['mask_zm'][index]

    index_xp_self = index[mask_xp]
    index_yp_self = index[mask_yp]
    index_zp_self = index[mask_zp]
    index_xm_self = index[mask_xm]
    index_ym_self = index[mask_ym]
    index_zm_self = index[mask_zm]

    dict_voxel_selection['index_xp_self'] = index_xp_self
    dict_voxel_selection['index_yp_self'] = index_yp_self
    dict_voxel_selection['index_zp_self'] = index_zp_self
    dict_voxel_selection['index_xm_self'] = index_xm_self
    dict_voxel_selection['index_ym_self'] = index_ym_self
    dict_voxel_selection['index_zm_self'] = index_zm_self

    dict_voxel_selection['index_xp_nb'] = voxel_nbs['index_xp'][index_xp_self]
    dict_voxel_selection['index_yp_nb'] = voxel_nbs['index_yp'][index_yp_self]
    dict_voxel_selection['index_zp_nb'] = voxel_nbs['index_zp'][index_zp_self]
    dict_voxel_selection['index_xm_nb'] = voxel_nbs['index_xm'][index_xm_self]
    dict_voxel_selection['index_ym_nb'] = voxel_nbs['index_ym'][index_ym_self]
    dict_voxel_selection['index_zm_nb'] = voxel_nbs['index_zm'][index_zm_self]

    np.savez(args.output, **dict_voxel_selection)

    # Sanity check for data term
    assert np.all(voxel_intensity[dict_voxel_selection['index_data_c']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_data_xp']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_data_yp']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_data_zp']] > 0)

    c2xp = voxel_coords[dict_voxel_selection['index_data_c']] - voxel_coords[
        dict_voxel_selection['index_data_xp']]
    assert np.allclose(c2xp, np.array([-voxel_size, 0, 0]))

    c2yp = voxel_coords[dict_voxel_selection['index_data_c']] - voxel_coords[
        dict_voxel_selection['index_data_yp']]
    assert np.allclose(c2yp, np.array([0, -voxel_size, 0]))

    c2zp = voxel_coords[dict_voxel_selection['index_data_c']] - voxel_coords[
        dict_voxel_selection['index_data_zp']]
    assert np.allclose(c2zp, np.array([0, 0, -voxel_size]))

    # Sanity check for laplacian
    c2xp = voxel_coords[dict_voxel_selection['index_lap_c']] - voxel_coords[
        dict_voxel_selection['index_lap_xp']]
    assert np.allclose(c2xp, np.array([-voxel_size, 0, 0]))

    c2yp = voxel_coords[dict_voxel_selection['index_lap_c']] - voxel_coords[
        dict_voxel_selection['index_lap_yp']]
    assert np.allclose(c2yp, np.array([0, -voxel_size, 0]))

    c2zp = voxel_coords[dict_voxel_selection['index_lap_c']] - voxel_coords[
        dict_voxel_selection['index_lap_zp']]
    assert np.allclose(c2zp, np.array([0, 0, -voxel_size]))

    c2xm = voxel_coords[dict_voxel_selection['index_lap_c']] - voxel_coords[
        dict_voxel_selection['index_lap_xm']]
    assert np.allclose(c2xm, np.array([+voxel_size, 0, 0]))

    c2ym = voxel_coords[dict_voxel_selection['index_lap_c']] - voxel_coords[
        dict_voxel_selection['index_lap_ym']]
    assert np.allclose(c2ym, np.array([0, +voxel_size, 0]))

    c2zm = voxel_coords[dict_voxel_selection['index_lap_c']] - voxel_coords[
        dict_voxel_selection['index_lap_zm']]
    assert np.allclose(c2zm, np.array([0, 0, +voxel_size]))

    # Sanity check for albedo
    assert np.all(voxel_intensity[dict_voxel_selection['index_xp_self']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_xm_self']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_yp_self']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_ym_self']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_zp_self']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_zm_self']] > 0)

    assert np.all(voxel_intensity[dict_voxel_selection['index_xp_nb']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_xm_nb']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_yp_nb']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_ym_nb']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_zp_nb']] > 0)
    assert np.all(voxel_intensity[dict_voxel_selection['index_zm_nb']] > 0)

    c2xp = voxel_coords[dict_voxel_selection['index_xp_self']] - voxel_coords[
        dict_voxel_selection['index_xp_nb']]
    assert np.allclose(c2xp, np.array([-voxel_size, 0, 0]))

    c2yp = voxel_coords[dict_voxel_selection['index_yp_self']] - voxel_coords[
        dict_voxel_selection['index_yp_nb']]
    assert np.allclose(c2yp, np.array([0, -voxel_size, 0]))

    c2zp = voxel_coords[dict_voxel_selection['index_zp_self']] - voxel_coords[
        dict_voxel_selection['index_zp_nb']]
    assert np.allclose(c2zp, np.array([0, 0, -voxel_size]))

    c2xm = voxel_coords[dict_voxel_selection['index_xm_self']] - voxel_coords[
        dict_voxel_selection['index_xm_nb']]
    assert np.allclose(c2xm, np.array([+voxel_size, 0, 0]))

    c2ym = voxel_coords[dict_voxel_selection['index_ym_self']] - voxel_coords[
        dict_voxel_selection['index_ym_nb']]
    assert np.allclose(c2ym, np.array([0, +voxel_size, 0]))

    c2zm = voxel_coords[dict_voxel_selection['index_zm_self']] - voxel_coords[
        dict_voxel_selection['index_zm_nb']]
    assert np.allclose(c2zm, np.array([0, 0, +voxel_size]))

    np.savez(args.output, **dict_voxel_selection)
