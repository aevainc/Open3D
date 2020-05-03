#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
#include "Hashmap.h"
#include "Open3D/Utility/Timer.h"

using namespace open3d;
hash_t dummy_hash_fn;

template <typename Key, typename Value, typename HashFunc = std::hash<Key>>
bool Compare(iterator_t* ret_iterators,
             uint8_t* ret_masks,
             int num_queries,
             std::vector<Key>& query_keys,
             std::unordered_map<Key, Value, HashFunc>& hashmap_gt) {
    auto iterators =
            std::vector<iterator_t>(ret_iterators, ret_iterators + num_queries);
    auto masks = std::vector<uint8_t>(ret_masks, ret_masks + num_queries);

    for (int i = 0; i < num_queries; ++i) {
        auto iterator_gt = hashmap_gt.find(query_keys[i]);
        if (iterator_gt == hashmap_gt.end()) {  /// Not found
            if (masks[i] != 0) {
                utility::LogError(
                        "keys[{}] should be not found in cpu hashmap.");
            }
        } else {  /// Found
            iterator_t iterator = iterators[i];
            Key key = *(reinterpret_cast<Key*>(iterator));
            Value val = *(reinterpret_cast<Value*>(iterator + sizeof(Key)));
            if (!(key == iterator_gt->first)) {
                utility::LogError("key[{}] is not equal to ground truth", i);
            }
            if (!(val == iterator_gt->second)) {
                utility::LogError("value[{}] is not equal to ground truth", i);
            }
        }
    }

    return true;
}

void TEST_SIMPLE() {
    /// C++ ground truth generation
    const int max_keys = 10;
    std::unordered_map<int, int> hashmap_gt;
    std::vector<int> insert_keys = {1, 3, 5};
    std::vector<int> insert_vals = {100, 300, 500};
    for (int i = 0; i < insert_keys.size(); ++i) {
        hashmap_gt[insert_keys[i]] = insert_vals[i];
    }
    std::vector<int> query_keys = {1, 2, 3, 4, 5};

    /// CPU data generation: use thrust for simplicity
    std::vector<int> insert_keys_cpu = insert_keys;
    std::vector<int> insert_vals_cpu = insert_vals;
    std::vector<int> query_keys_cpu = query_keys;

    /// CPU data ptr conversion
    uint8_t* insert_keys_ptr_cpu =
            reinterpret_cast<uint8_t*>(insert_keys_cpu.data());
    uint8_t* insert_vals_ptr_cpu =
            reinterpret_cast<uint8_t*>(insert_vals_cpu.data());
    uint8_t* query_keys_ptr_cpu =
            reinterpret_cast<uint8_t*>(query_keys_cpu.data());

    /// Hashmap creation
    auto hashmap = CreateHashmap<DefaultHash>(
            max_keys, sizeof(int), sizeof(int), open3d::Device("CPU:0"));

    /// Hashmap insertion
    hashmap->Insert(insert_keys_ptr_cpu, insert_vals_ptr_cpu,
                    insert_keys_cpu.size());

    /// Hashmap search
    iterator_t* ret_iterators;
    uint8_t* ret_masks;
    std::tie(ret_iterators, ret_masks) =
            hashmap->Search(query_keys_ptr_cpu, query_keys_cpu.size());

    /// Result parsing
    Compare<int, int>(ret_iterators, ret_masks, query_keys.size(), query_keys,
                      hashmap_gt);
    utility::LogInfo("TEST_SIMPLE() passed");
}

int main() { TEST_SIMPLE(); }
