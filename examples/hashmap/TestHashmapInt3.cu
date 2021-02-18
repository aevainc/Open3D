#include <stdgpu/iterator.h>  // device_begin, device_end
#include <stdgpu/memory.h>    // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>  // STDGPU_HOST_DEVICE
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <iostream>
#include <stdgpu/unordered_map.cuh>  // stdgpu::unordered_map

#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Timer.h"

class hash_int3 {
public:
    STDGPU_HOST_DEVICE stdgpu::index_t operator()(const int3& key) const {
        const int p0 = 73856093;
        const int p1 = 19349669;
        const int p2 = 83492791;

        return (key.x * p0) ^ (key.y * p1) ^ (key.z * p2);
    }
};

STDGPU_HOST_DEVICE bool operator==(const int3& a, const int3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__global__ void insert_int3(const int* d_keys,
                            const stdgpu::index_t n,
                            stdgpu::unordered_map<int3, int, hash_int3> map) {
    stdgpu::index_t i =
            static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n) return;
    map.emplace(
            make_int3(d_keys[3 * i + 0], d_keys[3 * i + 1], d_keys[3 * i + 2]),
            1);
}

int main(int argc, char** argv) {
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::unordered_map is used to compute a
    // duplicate-free set of numbers.
    //
    using namespace open3d;

    core::Device device("CUDA:0");
    std::string filename =
            utility::GetProgramOptionAsString(argc, argv, "--npy", "");
    double voxel_size =
            utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", 0.05);
    core::Tensor pcd = core::Tensor::Load(filename).To(device);

    core::Tensor pcd_int3 = (pcd / voxel_size).Floor().To(core::Dtype::Int32);

    utility::Timer timer;
    stdgpu::index_t n = pcd_int3.GetLength();
    for (int i = 0; i < 1000; ++i) {
        utility::LogInfo("i = {}", i);
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{3}, core::SizeVector{1}, device);
        core::Tensor t_addrs({n}, core::Dtype::Int32, device);
        core::Tensor t_masks({n}, core::Dtype::Bool, device);

        timer.Start();
        hashmap.Activate(pcd_int3, t_addrs, t_masks);
        cudaDeviceSynchronize();
        timer.Stop();
        utility::LogInfo("slabhash takes {}", timer.GetDuration());

        stdgpu::unordered_map<int3, int, hash_int3> map =
                stdgpu::unordered_map<int3, int, hash_int3>::createDeviceObject(
                        n);
        stdgpu::index_t threads = 128;
        stdgpu::index_t blocks = (n + threads - 1) / threads;

        timer.Start();
        insert_int3<<<static_cast<unsigned int>(blocks),
                      static_cast<unsigned int>(threads)>>>(
                static_cast<const int*>(pcd_int3.GetDataPtr()), n, map);
        cudaDeviceSynchronize();
        timer.Stop();
        utility::LogInfo("stdgpu takes {}", timer.GetDuration());

        if (map.size() != hashmap.Size()) {
            utility::LogError("Failed at iteration {}, {} vs {}", i, map.size(),
                              hashmap.Size());
        }

        stdgpu::unordered_map<int3, int, hash_int3>::destroyDeviceObject(map);
    }

    // // Warm up
    // {

    //     hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
    //     hashmap.Find(t_keys, t_addrs, t_masks);
    //     cudaDeviceSynchronize();
    // }

    // utility::Timer timer;

    // double insert_time = 0;
    // double find_time = 0;
    // for (int i = 0; i < runs; ++i) {
    //     core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
    //                           core::SizeVector{1}, core::SizeVector{1},
    //                           device);
    //     core::Tensor t_addrs({n}, core::Dtype::Int32, device);
    //     core::Tensor t_masks({n}, core::Dtype::Bool, device);

    //     timer.Start();
    //     hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
    //     cudaDeviceSynchronize();
    //     timer.Stop();
    //     insert_time += timer.GetDuration();

    //     timer.Start();
    //     hashmap.Find(t_keys, t_addrs, t_masks);
    //     cudaDeviceSynchronize();
    //     timer.Stop();
    //     find_time += timer.GetDuration();

    //     if (hashmap.Size() != cycle) {
    //         utility::LogError("ours: incorrect insertion");
    //     }
    // }
    // utility::LogInfo("ours takes {} on average for insertion",
    //                  insert_time / runs);
    // utility::LogInfo("ours takes {} on average for query", find_time / runs);

    // // stdgpu
    // int* d_keys = createDeviceArray<int>(n);
    // copyHost2DeviceArray<int>(kv.first.data(), n, d_keys,
    // MemoryCopy::NO_CHECK); int* d_values = createDeviceArray<int>(n);
    // copyHost2DeviceArray<int>(kv.second.data(), n, d_values,
    //                           MemoryCopy::NO_CHECK);

    // insert_time = 0;
    // find_time = 0;
    // for (int i = 0; i < runs; ++i) {
    //     stdgpu::unordered_map<int, int> map =
    //             stdgpu::unordered_map<int, int>::createDeviceObject(n);
    //     timer.Stop();
    //     insert_time += timer.GetDuration();

    //     timer.Start();
    //     find_numbers<<<static_cast<unsigned int>(blocks),
    //                    static_cast<unsigned int>(threads)>>>(d_keys,
    //                    d_values,
    //                                                          n, map);
    //     cudaDeviceSynchronize();
    //     timer.Stop();
    //     find_time += timer.GetDuration();

    //     if (map.size() != cycle) {
    //         utility::LogError("stdgpu: incorrect insertion");
    //     }

    //     stdgpu::unordered_map<int, int>::destroyDeviceObject(map);
    // }
    // utility::LogInfo("stdgpu takes {} on average for insertion",
    //                  insert_time / runs);
    // utility::LogInfo("stdgpu takes {} on average for query", find_time /
    // runs);
}
