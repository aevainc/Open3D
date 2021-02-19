#include <stdgpu/iterator.h>  // device_begin, device_end
#include <stdgpu/memory.h>    // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>  // STDGPU_HOST_DEVICE
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <iostream>
#include <random>
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
    utility::LogInfo("n = {}", n);

    double slabhash_time = 0;
    double stdgpu_time = 0;
    int count = 100;
    for (int i = 0; i < count; ++i) {
        utility::LogInfo("i = {}", i);
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{3}, core::SizeVector{1}, device);
        core::Tensor t_addrs({n}, core::Dtype::Int32, device);
        core::Tensor t_masks({n}, core::Dtype::Bool, device);

        timer.Start();
        hashmap.Activate(pcd_int3, t_addrs, t_masks);
        cudaDeviceSynchronize();
        timer.Stop();
        slabhash_time += timer.GetDuration();

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
        stdgpu_time += timer.GetDuration();

        if (map.size() != hashmap.Size()) {
            utility::LogError("Failed at iteration {}, {} vs {}", i, map.size(),
                              hashmap.Size());
        }

        stdgpu::unordered_map<int3, int, hash_int3>::destroyDeviceObject(map);
    }

    utility::LogInfo("Slabhash speed = {}, stdgpu speed = {}",
                     slabhash_time / count, stdgpu_time / count);
}
