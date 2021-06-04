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

using namespace open3d;

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
std::pair<std::vector<int>, std::vector<int>> GenerateKVVector(
        int n, double density = 0.9) {
    std::vector<int> k(n * 3);
    std::vector<int> v(n * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-10 * n, 10 * n);

    int valid_entries = n * density;
    std::vector<int> x_pool(n);
    std::vector<int> y_pool(n);
    std::vector<int> z_pool(n);

    // Generate random keys
    for (int i = 0; i < valid_entries; ++i) {
        x_pool[i] = dist(gen);
        y_pool[i] = dist(gen);
        z_pool[i] = dist(gen);
    }

    // Reuse generated keys to ensure density
    std::uniform_int_distribution<int> dist_sel(0, valid_entries);
    for (int i = valid_entries; i < n; ++i) {
        int sel = dist_sel(gen);
        x_pool[i] = x_pool[sel];
        y_pool[i] = y_pool[sel];
        z_pool[i] = z_pool[sel];
    }

    // Shuffle the keys
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Put into the keys vector
    for (int i = 0; i < n; ++i) {
        int sel = indices[i];

        int x = x_pool[sel];
        int y = y_pool[sel];
        int z = z_pool[sel];

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

template <int C>
void run(int n,
         int runs,
         double density,
         bool debug,
         const core::HashmapBackend backend = core::HashmapBackend::StdGPU) {
    using T = int_blob<C>;
    using iterator_t =
            typename stdgpu::unordered_map<int3, T, hash_int3>::iterator;

    auto kv = GenerateKVVector<C>(n, density);
    utility::LogInfo("n {}", n);
    utility::LogInfo("c {}", C);
    utility::LogInfo("density {}", density);

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
                              device, backend);
        core::Tensor t_addrs, t_masks;
        hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
        hashmap.Find(t_keys, t_addrs, t_masks);
        cudaDeviceSynchronize();
    }

    utility::Timer timer;
    stdgpu::index_t threads = 128;
    stdgpu::index_t blocks = (n + threads - 1) / threads;

    {
        // Insert: ours
        // Note: exclude construction and destruction for fairness
        double insert_time = 0;
        for (int i = 0; i < runs; ++i) {
            core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                                  core::SizeVector{3, 1}, core::SizeVector{C},
                                  device, backend);
            cudaDeviceSynchronize();

            timer.Start();
            core::Tensor t_addrs, t_masks;
            hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
            cudaDeviceSynchronize();
            timer.Stop();
            insert_time += timer.GetDuration();
        }
        utility::LogInfo("ours.insertion {}", insert_time / runs);

        // stdgpu
        int* d_keys = static_cast<int*>(t_keys.GetDataPtr());
        T* d_values = static_cast<T*>(t_values.GetDataPtr());
        insert_time = 0;
        for (int i = 0; i < runs; ++i) {
            auto map = stdgpu::unordered_map<int3, T,
                                             hash_int3>::createDeviceObject(n);
            cudaDeviceSynchronize();

            timer.Start();
            bool* d_masks = createDeviceArray<bool>(n);
            iterator_t* d_output = createDeviceArray<iterator_t>(n);

            insert_int3<T><<<static_cast<unsigned int>(blocks),
                             static_cast<unsigned int>(threads)>>>(
                    d_keys, d_values, d_masks, d_output, n, map);

            destroyDeviceArray<bool>(d_masks);
            destroyDeviceArray<iterator_t>(d_output);
            cudaDeviceSynchronize();
            timer.Stop();
            insert_time += timer.GetDuration();

            stdgpu::unordered_map<int3, T, hash_int3>::destroyDeviceObject(map);
        }
        utility::LogInfo("stdgpu.insertion {}", insert_time / runs);
    }

    {
        // Ablation: activation
        double activate_time = 0;
        for (int i = 0; i < runs; ++i) {
            core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                                  core::SizeVector{3, 1}, core::SizeVector{C},
                                  device, backend);
            cudaDeviceSynchronize();

            timer.Start();
            core::Tensor t_addrs, t_masks;
            hashmap.Activate(t_keys, t_addrs, t_masks);
            cudaDeviceSynchronize();
            timer.Stop();
            activate_time += timer.GetDuration();
        }
        utility::LogInfo("ours.activate {}", activate_time / runs);
    }

    {
        // Find experiments
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{3, 1}, core::SizeVector{C},
                              device, backend);
        core::Tensor t_addrs, t_masks;
        hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
        hashmap.Find(t_keys, t_addrs, t_masks);
        cudaDeviceSynchronize();

        bool saved = false;
        if (debug && !saved) {
            auto query_keys = t_keys.IndexGet({t_masks});
            auto query_values = hashmap.GetValueTensor().IndexGet(
                    {t_addrs.To(core::Dtype::Int64).IndexGet({t_masks})});
            if (!query_keys.Sum({1}).AllClose(query_values.T()[0].T())) {
                utility::LogError("Not all equal, query failed.");
            } else {
                utility::LogInfo("Check passed");
            }

            query_keys.Save("keys.npy");
            query_values.Save("values.npy");
            saved = true;
        }

        double find_time = 0;
        for (int i = 0; i < runs; ++i) {
            timer.Start();
            hashmap.Find(t_keys, t_addrs, t_masks);
            cudaDeviceSynchronize();
            timer.Stop();
            find_time += timer.GetDuration();
        }
        utility::LogInfo("ours.find {}", find_time / runs);

        // stdgpu
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

        find_time = 0;
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

        utility::LogInfo("stdgpu.find {}", find_time / runs);
        stdgpu::unordered_map<int3, T, hash_int3>::destroyDeviceObject(map);
    }
}

int main(int argc, char** argv) {
    using namespace open3d;

    stdgpu::index_t n =
            utility::GetProgramOptionAsInt(argc, argv, "--n", 10000);
    int runs = utility::GetProgramOptionAsInt(argc, argv, "--runs", 10);
    int channels =
            utility::GetProgramOptionAsInt(argc, argv, "--channels", 1024);
    double density =
            utility::GetProgramOptionAsDouble(argc, argv, "--density", 0.99);
    std::string backend_str = utility::GetProgramOptionAsString(
            argc, argv, "--backend", "stdgpu");

    core::HashmapBackend backend;
    if (backend_str == "stdgpu") {
        backend = core::HashmapBackend::StdGPU;
    } else if (backend_str == "slab") {
        backend = core::HashmapBackend::Slab;
    } else {
        utility::LogError("Unsupported backend {}", backend_str);
    }
    bool debug = utility::ProgramOptionExists(argc, argv, "--debug");

    if (channels == 1) {
        run<1>(n, runs, density, debug, backend);
    } else if (channels == 2) {
        run<2>(n, runs, density, debug, backend);
    } else if (channels == 4) {
        run<4>(n, runs, density, debug, backend);
    } else if (channels == 8) {
        run<8>(n, runs, density, debug, backend);
    } else if (channels == 16) {
        run<16>(n, runs, density, debug, backend);
    } else if (channels == 32) {
        run<32>(n, runs, density, debug, backend);
    } else if (channels == 64) {
        run<64>(n, runs, density, debug, backend);
    } else if (channels == 128) {
        run<128>(n, runs, density, debug, backend);
    } else if (channels == 256) {
        run<256>(n, runs, density, debug, backend);
    } else if (channels == 512) {
        run<512>(n, runs, density, debug, backend);
    } else if (channels == 1024) {
        run<1024>(n, runs, density, debug, backend);
    } else if (channels == 2048) {
        run<2048>(n, runs, density, debug, backend);
    } else if (channels == 4096) {
        run<4096>(n, runs, density, debug, backend);
    } else {
        utility::LogInfo("C({}) not dispatched", channels);
    }

    return 0;
}
