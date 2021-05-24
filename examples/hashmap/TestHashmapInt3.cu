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

using iterator_t =
        typename stdgpu::unordered_map<int3, int, hash_int3>::iterator;

__global__ void insert_int3(const int* d_keys,
                            const int* d_values,
                            bool* d_masks,
                            iterator_t* d_output,
                            const stdgpu::index_t n,
                            stdgpu::unordered_map<int3, int, hash_int3> map) {
    stdgpu::index_t i =
            static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n) return;
    auto res = map.emplace(
            make_int3(d_keys[3 * i + 0], d_keys[3 * i + 1], d_keys[3 * i + 2]),
            d_values[i]);
    d_masks[i] = res.second;
    if (res.first) {
        d_output[i] = res.first;
    }
}

__global__ void find_int3(const int* d_keys,
                          bool* d_masks,
                          iterator_t* d_output,
                          const stdgpu::index_t n,
                          stdgpu::unordered_map<int3, int, hash_int3> map) {
    stdgpu::index_t i =
            static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n) return;
    auto iter = map.find(
            make_int3(d_keys[3 * i + 0], d_keys[3 * i + 1], d_keys[3 * i + 2]));
    d_masks[i] = (iter == map.end());
    d_output[i] = iter;
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
    stdgpu::index_t threads = 128;
    stdgpu::index_t blocks = (n + threads - 1) / threads;

    // Insert experiments
    {
        double insert_time = 0;
        for (int i = 0; i < runs; ++i) {
            timer.Start();
            core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                                  core::SizeVector{3, 1}, core::SizeVector{1},
                                  device);
            core::Tensor t_addrs({n}, core::Dtype::Int32, device);
            core::Tensor t_masks({n}, core::Dtype::Bool, device);
            hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
            timer.Stop();
            insert_time += timer.GetDuration();
        }

        utility::LogInfo("ours insertion time: {}", insert_time / runs);

        // stdgpu
        insert_time = 0;
        int* d_keys = static_cast<int*>(t_keys.GetDataPtr());
        int* d_values = static_cast<int*>(t_values.GetDataPtr());
        for (int i = 0; i < runs; ++i) {
            timer.Start();
            stdgpu::unordered_map<int3, int, hash_int3> map =
                    stdgpu::unordered_map<int3, int,
                                          hash_int3>::createDeviceObject(n);
            bool* d_masks = createDeviceArray<bool>(n);
            iterator_t* d_output = createDeviceArray<iterator_t>(n);

            insert_int3<<<static_cast<unsigned int>(blocks),
                          static_cast<unsigned int>(threads)>>>(
                    d_keys, d_values, d_masks, d_output, n, map);

            destroyDeviceArray<bool>(d_masks);
            destroyDeviceArray<iterator_t>(d_output);
            stdgpu::unordered_map<int3, int, hash_int3>::destroyDeviceObject(
                    map);
            cudaDeviceSynchronize();
            timer.Stop();
            insert_time += timer.GetDuration();
        }

        utility::LogInfo("stdgpu insert time: {}", insert_time / runs);
    }

    // Find experiments
    {
        double find_time = 0;
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{3, 1}, core::SizeVector{1},
                              device);
        core::Tensor t_addrs({n}, core::Dtype::Int32, device);
        core::Tensor t_masks({n}, core::Dtype::Bool, device);
        hashmap.Insert(t_keys, t_values, t_addrs, t_masks);

        for (int i = 0; i < runs; ++i) {
            timer.Start();
            t_addrs = core::Tensor({n}, core::Dtype::Int32, device);
            t_masks = core::Tensor({n}, core::Dtype::Bool, device);
            hashmap.Find(t_keys, t_addrs, t_masks);
            timer.Stop();
            find_time += timer.GetDuration();
        }
        utility::LogInfo("ours find time: {}", find_time / runs);

        // stdgpu
        find_time = 0;
        int* d_keys = static_cast<int*>(t_keys.GetDataPtr());
        int* d_values = static_cast<int*>(t_values.GetDataPtr());
        stdgpu::unordered_map<int3, int, hash_int3> map =
                stdgpu::unordered_map<int3, int, hash_int3>::createDeviceObject(
                        n);
        bool* d_masks = createDeviceArray<bool>(n);
        iterator_t* d_output = createDeviceArray<iterator_t>(n);
        insert_int3<<<static_cast<unsigned int>(blocks),
                      static_cast<unsigned int>(threads)>>>(
                d_keys, d_values, d_masks, d_output, n, map);
        destroyDeviceArray<bool>(d_masks);
        destroyDeviceArray<iterator_t>(d_output);
        cudaDeviceSynchronize();

        for (int i = 0; i < runs; ++i) {
            timer.Start();
            bool* d_masks = createDeviceArray<bool>(n);
            iterator_t* d_output = createDeviceArray<iterator_t>(n);

            find_int3<<<static_cast<unsigned int>(blocks),
                        static_cast<unsigned int>(threads)>>>(d_keys, d_masks,
                                                              d_output, n, map);
            destroyDeviceArray<bool>(d_masks);
            destroyDeviceArray<iterator_t>(d_output);
            cudaDeviceSynchronize();
            timer.Stop();
            find_time += timer.GetDuration();
        }

        utility::LogInfo("stdgpu find time: {}", find_time / runs);
    }
}
