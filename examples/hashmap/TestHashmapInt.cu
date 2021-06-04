
/*
 * Copyright 2019 Saman Ashkiani
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "gpu_hash_table.cuh"
#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Timer.h"

std::pair<std::vector<int>, std::vector<int>> GenerateKVVector(
        int n, double density = 0.9) {
    std::vector<int> k(n);
    std::vector<int> v(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-10 * n, 10 * n);

    int valid_entries = n * density;
    std::vector<int> x_pool(n);

    // Generate random keys
    for (int i = 0; i < valid_entries; ++i) {
        x_pool[i] = dist(gen);
    }

    // Reuse generated keys to ensure density
    std::uniform_int_distribution<int> dist_sel(0, valid_entries);
    for (int i = valid_entries; i < n; ++i) {
        int sel = dist_sel(gen);
        x_pool[i] = x_pool[sel];
    }

    // Shuffle the keys
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Put into the keys vector
    for (int i = 0; i < n; ++i) {
        int sel = indices[i];

        int x = x_pool[sel];

        k[i] = x;
        v[i] = x * 10;
    }
    return std::make_pair(k, v);
}

int main(int argc, char** argv) {
    using namespace open3d;
    auto backend = core::HashmapBackend::Slab;

    int n = utility::GetProgramOptionAsInt(argc, argv, "--n", 10000);

    // Slabhash setup
    uint32_t num_keys = n;
    uint32_t num_queries = n;
    float expected_chain = 0.6f;
    uint32_t num_elements_per_unit = 15;
    uint32_t expected_elements_per_bucket =
            expected_chain * num_elements_per_unit;
    uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) /
                           expected_elements_per_bucket;

    int runs = utility::GetProgramOptionAsInt(argc, argv, "--runs", 10);
    double density =
            utility::GetProgramOptionAsDouble(argc, argv, "--density", 0.99);

    auto kv = GenerateKVVector(n, density);

    core::Device host("CPU:0");
    core::Device device("CUDA:0");
    core::Tensor t_keys = core::Tensor(kv.first, {n}, core::Dtype::Int32, host);
    core::Tensor t_values =
            core::Tensor(kv.second, {n}, core::Dtype::Int32, host);

    {  // Warm up

        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{1}, core::SizeVector{1}, device,
                              backend);
        core::Tensor t_addrs, t_masks;
        hashmap.Insert(t_keys.To(device), t_values.To(device), t_addrs,
                       t_masks);
        cudaDeviceSynchronize();
    }

    {
        // Insertion: slab ours
        utility::Timer timer;
        double insert_time = 0;
        for (int i = 0; i < runs; ++i) {
            core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                                  core::SizeVector{1}, core::SizeVector{1},
                                  device, backend);
            cudaDeviceSynchronize();

            timer.Start();
            core::Tensor t_addrs, t_masks;
            hashmap.Insert(t_keys.To(device), t_values.To(device), t_addrs,
                           t_masks);
            cudaDeviceSynchronize();
            timer.Stop();
            insert_time += timer.GetDuration();
        }
        utility::LogInfo("ours.insertion {}", insert_time / runs);
    }

    {  // Insertion: slab
        utility::Timer timer;

        double insert_time = 0;
        for (int i = 0; i < runs; ++i) {
            gpu_hash_table<int, int, SlabHashTypeT::ConcurrentMap> hash_table(
                    num_keys, num_buckets, /*DEVICE_ID=*/0, /*seed=*/1);
            cudaDeviceSynchronize();

            timer.Start();
            hash_table.hash_build(kv.first.data(), kv.second.data(), num_keys);
            cudaDeviceSynchronize();
            timer.Stop();
            insert_time += timer.GetDuration();
        }
        utility::LogInfo("slab.insertion {}", insert_time / runs);
    }

    {  // Find: ours
        utility::Timer timer;
        core::Device device("CUDA:0");

        core::Hashmap hashmap(n, core::Dtype::Int32, core::Dtype::Int32,
                              core::SizeVector{1}, core::SizeVector{1}, device,
                              backend);
        core::Tensor t_addrs, t_masks;
        hashmap.Insert(t_keys.To(device), t_values.To(device), t_addrs,
                       t_masks);
        // Check, and warm up for finding
        hashmap.Find(t_keys.To(device), t_addrs, t_masks);
        auto query_keys = t_keys.IndexGet({t_masks}).To(device);
        auto query_values = hashmap.GetValueTensor().IndexGet(
                {t_addrs.To(core::Dtype::Int64).IndexGet({t_masks})});
        if (!query_keys.View({-1, 1}).AllClose(query_values / 10)) {
            utility::LogError("Not all equal, query failed.");
        }
        cudaDeviceSynchronize();

        double find_time = 0;
        for (int i = 0; i < runs; ++i) {
            timer.Start();
            core::Tensor t_addrs, t_masks;
            hashmap.Find(t_keys.To(device), t_addrs, t_masks);
            cudaDeviceSynchronize();
            timer.Stop();
            find_time += timer.GetDuration();
        }
        utility::LogInfo("ours.find {}", find_time / runs);
    }

    {
        gpu_hash_table<int, int, SlabHashTypeT::ConcurrentMap> hash_table(
                num_keys, num_buckets, /*DEVICE_ID=*/0, /*seed=*/1);
        hash_table.hash_build(kv.first.data(), kv.second.data(), num_keys);

        // Check, and warm up for finding
        std::vector<int> h_result(n);
        hash_table.hash_search(kv.first.data(), h_result.data(), num_keys);
        for (int i = 0; i < n; ++i) {
            if (kv.first[i] * 10 != h_result[i]) {
                utility::LogInfo("Slab find fails!");
            }
        }
        cudaDeviceSynchronize();

        utility::Timer timer;
        double find_time = 0;

        for (int i = 0; i < runs; ++i) {
            timer.Start();
            hash_table.hash_search(kv.first.data(), h_result.data(),
                                   num_queries);
            cudaDeviceSynchronize();
            timer.Stop();
            find_time += timer.GetDuration();
        }
        utility::LogInfo("slab.find {}", find_time / runs);
    }
}
