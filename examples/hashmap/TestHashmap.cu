#include <stdgpu/iterator.h>  // device_begin, device_end
#include <stdgpu/memory.h>    // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>  // STDGPU_HOST_DEVICE
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <iostream>
#include <stdgpu/unordered_map.cuh>  // stdgpu::unordered_map

#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/utility/Timer.h"

__global__ void insert_numbers(const int* d_keys,
                               const int* d_values,
                               const stdgpu::index_t n,
                               stdgpu::unordered_map<int, int> map) {
    stdgpu::index_t i =
            static_cast<stdgpu::index_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n) return;
    map.emplace(d_keys[i], d_values[i]);
}

int main() {
    //
    // EXAMPLE DESCRIPTION
    // -------------------
    // This example demonstrates how stdgpu::unordered_map is used to compute a
    // duplicate-free set of numbers.
    //
    using namespace open3d;

    stdgpu::index_t n = 1000000;
    int counts = 100;

    // Ours
    core::Tensor t_keys = core::Tensor::Arange(0, n, 1, core::Dtype::Int32,
                                               core::Device("CUDA:0"));
    core::Tensor t_values = core::Tensor::Arange(0, n, 1, core::Dtype::Int32,
                                                 core::Device("CUDA:0"));

    utility::Timer timer;

    double total_time = 0;
    for (int i = 0; i < counts; ++i) {
        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{1}, core::SizeVector{1},
                              core::Device("CUDA:0"));
        core::Tensor t_addrs, t_masks;

        timer.Start();
        hashmap.Insert(t_keys, t_values, t_addrs, t_masks);
        cudaDeviceSynchronize();
        timer.Stop();
        total_time += timer.GetDuration();
        if (hashmap.Size() != n) {
            utility::LogError("ours: incorrect insertion");
        }
    }
    utility::LogInfo("ours takes {} on average", total_time / counts);

    // stdgpu
    int* d_keys = createDeviceArray<int>(n);
    thrust::sequence(stdgpu::device_begin(d_keys), stdgpu::device_end(d_keys),
                     0);
    int* d_values = createDeviceArray<int>(n);
    thrust::sequence(stdgpu::device_begin(d_values),
                     stdgpu::device_end(d_values), 0);

    total_time = 0;
    for (int i = 0; i < counts; ++i) {
        stdgpu::unordered_map<int, int> map =
                stdgpu::unordered_map<int, int>::createDeviceObject(n);
        stdgpu::index_t threads = 128;
        stdgpu::index_t blocks = (n + threads - 1) / threads;

        timer.Start();
        insert_numbers<<<static_cast<unsigned int>(blocks),
                         static_cast<unsigned int>(threads)>>>(d_keys, d_values,
                                                               n, map);
        cudaDeviceSynchronize();
        timer.Stop();
        total_time += timer.GetDuration();

        if (map.size() != n) {
            utility::LogError("stdgpu: incorrect insertion");
        }

        stdgpu::unordered_map<int, int>::destroyDeviceObject(map);
    }
    utility::LogInfo("stdgpu takes {} on average", total_time / counts);
    destroyDeviceArray<int>(d_keys);
    destroyDeviceArray<int>(d_values);
}
