import argparse
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import open3d as o3d

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # does not affect results

pwd = Path(os.path.dirname(os.path.realpath(__file__)))
open3d_root = pwd.parent.parent.parent

from benchmark_utils import measure_time, print_system_info, print_table, sample_points


# Define NNS methods
class O3DKnnGPU:

    def __init__(self):
        pass

    def setup(self, points, queries):
        index = o3d.core.nns.KnnIndex()
        index.set_tensor_data(points)
        return index, queries

    def search(self, index, queries, knn):
        ans = index.knn_search(queries, knn)
        return ans


class O3DFaiss(O3DKnnGPU):

    def __init__(self):
        pass

    def setup(self, points, queries):
        index = o3d.core.nns.FaissIndex()
        index.set_tensor_data(points)
        return index, queries


class O3DKnnGPUNew(O3DKnnGPU):

    def search(self, index, queries, knn):
        ans = index.knn_search_new(queries, knn)
        return ans


class O3DKnnCPU(O3DKnnGPU):

    def setup(self, points, queries):
        points_cpu = points.cpu()
        queries_cpu = queries.cpu()
        index = o3d.core.nns.NearestNeighborSearch(points_cpu)
        index.knn_index()
        return index, queries_cpu

    def search(self, index, queries, knn):
        indices, distances = index.knn_search(queries, knn)
        indices_gpu = indices.cuda()
        distances_gpu = distances.cuda()
        return indices_gpu, distances_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--file",
                        action="append",
                        default=[str(open3d_root / "small_tower.ply")])
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    # fix seed
    np.random.seed(777)

    # cuda device
    o3d_cuda_dev = o3d.core.Device(o3d.core.Device.CUDA, 0)
    # collects runtimes for all examples
    results = OrderedDict()

    # setup dataset examples
    datasets = OrderedDict()

    # random data
    num_points = 100000
    # num_points = 10000
    for i, dim in enumerate([3, 4, 8, 16, 32]):
        points = o3d.core.Tensor.from_numpy(
            np.random.rand(num_points, dim).astype(np.float32))
        queries = o3d.core.Tensor.from_numpy(
            np.random.rand(num_points, dim).astype(np.float32))
        datasets[f'random_{dim}'] = {'points': points, 'queries': queries}

    # prepare methods
    # methods = [O3DKnnGPU(), O3DKnnGPUNew(), O3DKnnCPU()]
    methods = [O3DKnnGPU(), O3DKnnCPU(), O3DKnnGPUNew()]
    method_names = [m.__class__.__name__ for m in methods]

    if args.sanity:
        for example in datasets.values():
            points = example['points']
            queries = example['queries']
            points = points.contiguous().to(o3d_cuda_dev)
            queries = queries.contiguous().to(o3d_cuda_dev)

            index1, queries1 = methods[0].setup(points, queries)
            indices1, dists1 = methods[0].search(index1, queries1, 6)

            index2, queries2 = methods[1].setup(points, queries)
            indices2, dists2 = methods[1].search(index2, queries2, 6)

            index3, queries3 = methods[2].setup(points, queries)
            indices3, dists3 = methods[2].search(index3, queries3, 6)

            indices1 = indices1.cpu().numpy()
            indices2 = indices2.cpu().numpy()
            indices3 = indices3.cpu().numpy()
            dists1 = dists1.cpu().numpy()
            dists2 = dists2.cpu().numpy()
            dists3 = dists3.cpu().numpy()
            print("indices: ", indices1.shape, indices2.shape, indices3.shape)
            np.testing.assert_allclose(indices3, indices1)
            np.testing.assert_allclose(indices3, indices2)
            np.testing.assert_allclose(dists3, dists1, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(dists3, dists2, rtol=1e-6, atol=1e-6)

    # run benchmark
    for method_name, method in zip(method_names, methods):

        print(method_name)
        if not args.overwrite and os.path.exists(f"{method_name}.pkl"):
            print(f"skip {method_name}")
            continue

        for example_name, example in datasets.items():
            print(example_name)
            points = example['points']
            queries = example['queries']

            for knn in (1, 8, 32, 64):
                print(knn)
                points = example['points']
                queries = example['queries']

                # points = sample_points(points, num_sample=num_points)
                # queries = sample_points(queries, num_sample=num_points)
                points = points.contiguous().to(o3d_cuda_dev)
                queries = queries.contiguous().to(o3d_cuda_dev)

                example_results = {'k': knn, 'num_points': points.shape[0]}

                if hasattr(method, "prepare_data"):
                    points, queries = method.prepare_data(points, queries)

                ans = measure_time(lambda: method.setup(points, queries))
                example_results['knn_setup'] = ans

                index, queries = method.setup(points, queries)

                ans = measure_time(lambda: method.search(index, queries, knn),
                                   min_samples=2,
                                   max_samples=2)
                example_results['knn_search'] = ans

                del index
                o3d.core.cuda.release_cache()

                results[
                    f'{example_name} n={points.shape[0]} k={knn}'] = example_results

                del points
                del queries
                o3d.core.cuda.release_cache()

        with open(f'{method_name}.pkl', 'wb') as f:
            pickle.dump(results, f)

    results = []
    for method_name, method in zip(method_names, methods):
        with open(f"{method_name}.pkl", "rb") as f:
            print(f"{method_name}.pkl")
            data = pickle.load(f)
            results.append(data)

    print_system_info()
    print_table(method_names, results)
