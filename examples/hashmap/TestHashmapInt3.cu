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
                            const int* d_values,
                            const stdgpu::index_t n,
                            stdgpu::unordered_map<int3, int, hash_int3> map) {
    stdgpu::index_t i =
            static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n) return;
    map.emplace(
            make_int3(d_keys[3 * i + 0], d_keys[3 * i + 1], d_keys[3 * i + 2]),
            d_values[i]);
}

__global__ void find_int3(const int* d_keys,
                          int* d_values,
                          const stdgpu::index_t n,
                          stdgpu::unordered_map<int3, int, hash_int3> map) {
    stdgpu::index_t i =
            static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n) return;
    d_values[i] = map.find(make_int3(d_keys[3 * i + 0], d_keys[3 * i + 1],
                                     d_keys[3 * i + 2]))
                          ->second;
}

std::pair<std::vector<int>, std::vector<int>> GenerateKVVector(int n) {
    std::vector<int> k(n * 3), v(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-100, 100);

    for (int i = 0; i < n; ++i) {
        k[i * 3 + 1] = dis(gen);
        k[i * 3 + 2] = dis(gen);
        k[i * 3 + 3] = dis(gen);

        v[i] = i;
    }
    return std::make_pair(k, v);
}

int main(int argc, char** argv) {
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::unordered_map is used to compute a
    // duplicate-free set of numbers.
    //
    using namespace open3d;

    stdgpu::index_t n =
            utility::GetProgramOptionAsInt(argc, argv, "--n", 10000);
    int runs = utility::GetProgramOptionAsInt(argc, argv, "--runs", 1000);

    auto kv = GenerateKVVector(n);

    utility::LogInfo("n = {}", n);

    // Ours
    core::Tensor t_keys = core::Tensor(kv.first, {n, 3}, core::Dtype::Int32,
                                       core::Device("CUDA:0"));
    core::Tensor t_values = core::Tensor(kv.second, {n}, core::Dtype::Int32,
                                         core::Device("CUDA:0"));

    // Warm up
    core::Device device("CUDA:0");
    {
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{3, 1}, core::SizeVector{1},
                              device);
        core::Tensor t_addrs({n}, core::Dtype::Int32, device);
        core::Tensor t_masks({n}, core::Dtype::Bool, device);

        hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
        hashmap.Find(t_keys, t_addrs, t_masks);
        cudaDeviceSynchronize();
    }

    utility::Timer timer;

    double insert_time = 0;
    double find_time = 0;
    for (int i = 0; i < runs; ++i) {
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{3, 1}, core::SizeVector{1},
                              device);
        core::Tensor t_addrs({n}, core::Dtype::Int32, device);
        core::Tensor t_masks({n}, core::Dtype::Bool, device);

        timer.Start();
        hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
        cudaDeviceSynchronize();
        timer.Stop();
        insert_time += timer.GetDuration();

        timer.Start();
        hashmap.Find(t_keys, t_addrs, t_masks);
        cudaDeviceSynchronize();
        timer.Stop();
        find_time += timer.GetDuration();
    }

    utility::LogInfo("slabhash insertion rate: {}",
                     float(n) / (insert_time / runs));
    utility::LogInfo("slabhash query rate: {}", float(n) / (find_time / runs));

    // stdgpu
    insert_time = 0;
    find_time = 0;
    int* d_keys = static_cast<int*>(t_keys.GetDataPtr());
    int* d_values = static_cast<int*>(t_values.GetDataPtr());
    for (int i = 0; i < runs; ++i) {
        stdgpu::unordered_map<int3, int, hash_int3> map =
                stdgpu::unordered_map<int3, int, hash_int3>::createDeviceObject(
                        n);
        stdgpu::index_t threads = 128;
        stdgpu::index_t blocks = (n + threads - 1) / threads;

        timer.Start();
        insert_int3<<<static_cast<unsigned int>(blocks),
                      static_cast<unsigned int>(threads)>>>(d_keys, d_values, n,
                                                            map);
        cudaDeviceSynchronize();
        timer.Stop();
        insert_time += timer.GetDuration();

        timer.Start();
        find_int3<<<static_cast<unsigned int>(blocks),
                    static_cast<unsigned int>(threads)>>>(d_keys, d_values, n,
                                                          map);
        cudaDeviceSynchronize();
        timer.Stop();
        find_time += timer.GetDuration();

        stdgpu::unordered_map<int3, int, hash_int3>::destroyDeviceObject(map);
    }
    utility::LogInfo("stdgpu insertion rate: {}",
                     float(n) / (insert_time / runs));
    utility::LogInfo("stdgpu query rate: {}", float(n) / (find_time / runs));
}
