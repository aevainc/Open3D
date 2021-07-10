import open3d as o3d
import argparse


def read_edges_from_log(traj_log):
    import numpy as np

    ids = []
    trans_arr = []
    with open(traj_log) as f:
        content = f.readlines()

        # Load .log file.
        for i in range(0, len(content), 5):
            # format %d (src) %d (tgt) %total
            data = list(map(float, content[i].strip().split('\t')))
            indices = (int(data[0]), int(data[1]))

            # format %f x 16
            T_gt = np.array(
                list(map(float, (''.join(
                    content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

            ids.append(indices)
            trans_arr.append(T_gt)

    return ids, trans_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('traj_log')
    parser.add_argument('edge_log')
    args = parser.parse_args()

    n_frags = 50
    traj_ids, traj_poses = read_edges_from_log(args.traj_log)
    loop_ids, loop_transforms = read_edges_from_log(args.edge_log)

    import numpy as np
    base = np.eye(4)
    base[:3, 3] = np.array([2, 2, -0.3])
    base_inv = np.linalg.inv(base)

    pose_graph = o3d.pipelines.registration.PoseGraph()
    for i in range(len(traj_poses) // n_frags):
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(traj_poses[i * n_frags]))

    for loop_id, loop_transform in zip(loop_ids, loop_transforms):
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                loop_id[0], loop_id[1],
                np.linalg.inv(base_inv @ loop_transform @ base)))

    o3d.io.write_pose_graph('pose_graph.json', pose_graph)
