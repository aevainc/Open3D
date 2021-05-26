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

/// Int3 type
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

// Value type
template <size_t N>
struct int_blob {
    STDGPU_HOST_DEVICE int_blob() {}
    STDGPU_HOST_DEVICE int_blob(int v) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] = v;
        }
    }

    int data_[N];
};

// Data generation for int3 keys
template <size_t N>
std::pair<std::vector<int>, std::vector<int>> GenerateKVVector(int n) {
    std::vector<int> k(n * 3);
    std::vector<int> v(n * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-100, 100);

    for (int i = 0; i < n; ++i) {
        int x = dis(gen);
        int y = dis(gen);
        int z = dis(gen);

        k[i * 3 + 0] = x;
        k[i * 3 + 1] = y;
        k[i * 3 + 2] = z;
        v[i * N + 0] = x + y + z;
    }
    return std::make_pair(k, v);
}

template <typename T>
__global__ void insert_int3(
        const int* d_keys,
        const T* d_values,
        bool* d_masks,
        typename stdgpu::unordered_map<int3, T, hash_int3>::iterator* d_output,
        const stdgpu::index_t n,
        stdgpu::unordered_map<int3, T, hash_int3> map) {
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

template <typename T>
__global__ void find_int3(
        const int* d_keys,
        bool* d_masks,
        typename stdgpu::unordered_map<int3, T, hash_int3>::iterator* d_output,
        const stdgpu::index_t n,
        stdgpu::unordered_map<int3, T, hash_int3> map) {
    stdgpu::index_t i =
            static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n) return;
    auto iter = map.find(
            make_int3(d_keys[3 * i + 0], d_keys[3 * i + 1], d_keys[3 * i + 2]));
    d_masks[i] = (iter == map.end());
    d_output[i] = iter;
}

int main(int argc, char** argv) {
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::unordered_map is used to compute a
    // duplicate-free set of numbers.
    //
    using namespace open3d;
    constexpr stdgpu::index_t C = 128;
    using T = int_blob<C>;
    using iterator_t =
            typename stdgpu::unordered_map<int3, T, hash_int3>::iterator;

    stdgpu::index_t n =
            utility::GetProgramOptionAsInt(argc, argv, "--n", 10000);
    int runs = utility::GetProgramOptionAsInt(argc, argv, "--runs", 1000);

    auto kv = GenerateKVVector<C>(n);
    utility::LogInfo("n = {}, c = {}", n, C);

    // Ours
    core::Tensor t_keys = core::Tensor(kv.first, {n, 3}, core::Dtype::Int32,
                                       core::Device("CUDA:0"));
    core::Tensor t_values = core::Tensor(kv.second, {n, C}, core::Dtype::Int32,
                                         core::Device("CUDA:0"));

    // Warm up
    core::Device device("CUDA:0");
    {
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{3, 1}, core::SizeVector{C},
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
                                  core::SizeVector{3, 1}, core::SizeVector{C},
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
        T* d_values = static_cast<T*>(t_values.GetDataPtr());
        for (int i = 0; i < runs; ++i) {
            timer.Start();
            auto map = stdgpu::unordered_map<int3, T,
                                             hash_int3>::createDeviceObject(n);
            bool* d_masks = createDeviceArray<bool>(n);
            iterator_t* d_output = createDeviceArray<iterator_t>(n);

            insert_int3<T><<<static_cast<unsigned int>(blocks),
                             static_cast<unsigned int>(threads)>>>(
                    d_keys, d_values, d_masks, d_output, n, map);

            destroyDeviceArray<bool>(d_masks);
            destroyDeviceArray<iterator_t>(d_output);
            stdgpu::unordered_map<int3, T, hash_int3>::destroyDeviceObject(map);
            cudaDeviceSynchronize();
            timer.Stop();
            insert_time += timer.GetDuration();
        }

        utility::LogInfo("stdgpu insert time: {}", insert_time / runs);
    }

    // Find experiments
    // bool saved = false;
    {
        double find_time = 0;
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{3, 1}, core::SizeVector{C},
                              device);
        core::Tensor t_addrs({n}, core::Dtype::Int32, device);
        core::Tensor t_masks({n}, core::Dtype::Bool, device);
        hashmap.Insert(t_keys, t_values, t_addrs, t_masks);

        // if (!saved) {
        //     auto query_keys = t_keys.IndexGet({t_masks});
        //     auto query_values = hashmap.GetValueTensor().IndexGet(
        //             {t_addrs.To(core::Dtype::Int64).IndexGet({t_masks})});
        //     query_keys.Save("keys.npy");
        //     query_values.Save("values.npy");
        //     saved = true;
        // }

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
        T* d_values = static_cast<T*>(t_values.GetDataPtr());
        auto map =
                stdgpu::unordered_map<int3, T, hash_int3>::createDeviceObject(
                        n);
        bool* d_masks = createDeviceArray<bool>(n);
        iterator_t* d_output = createDeviceArray<iterator_t>(n);
        insert_int3<T><<<static_cast<unsigned int>(blocks),
                         static_cast<unsigned int>(threads)>>>(
                d_keys, d_values, d_masks, d_output, n, map);
        destroyDeviceArray<bool>(d_masks);
        destroyDeviceArray<iterator_t>(d_output);
        cudaDeviceSynchronize();

        for (int i = 0; i < runs; ++i) {
            timer.Start();
            bool* d_masks = createDeviceArray<bool>(n);
            iterator_t* d_output = createDeviceArray<iterator_t>(n);

            find_int3<T><<<static_cast<unsigned int>(blocks),
                           static_cast<unsigned int>(threads)>>>(
                    d_keys, d_masks, d_output, n, map);
            destroyDeviceArray<bool>(d_masks);
            destroyDeviceArray<iterator_t>(d_output);
            cudaDeviceSynchronize();
            timer.Stop();
            find_time += timer.GetDuration();
        }

        utility::LogInfo("stdgpu find time: {}", find_time / runs);
    }
}
